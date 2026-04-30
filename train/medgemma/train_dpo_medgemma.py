"""
MedGemma DPO fine-tuning for radiology report generation.

Identical DPO algorithm to the original MMed-RAG pipeline
(sigmoid loss, same β, LoRA r=128 α=256, noise-augmented rejected),
adapted for PaliGemmaForConditionalGeneration + AutoProcessor.

Key differences from the LLaVA-Med version:
  - Model: google/medgemma-4b-it  (SigLIP + Gemma-3 4B)
  - Processor: AutoProcessor  (unified image + text tokenisation)
  - LoRA targets: Gemma-3 language-model linear layers only
  - Separate forward passes for chosen/rejected (different pixel_values)
  - Reference model = base model with LoRA adapters disabled (no 2nd copy)
  - QLoRA (4-bit NF4) for GPU-memory efficiency on Kaggle T4/A100
"""

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import wandb
from datasets import Dataset as HFDataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from trl import DPOConfig, DPOTrainer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from medgemma_dataset import DPODataCollator, MedGemmaDPODataset

IGNORE_INDEX = -100
MODEL_ID = "google/medgemma-4b-it"

# LoRA target modules for Gemma-3 language model (leave SigLIP frozen)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=MODEL_ID)
    lora_checkpoint_path: Optional[str] = field(default=None)
    use_4bit: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    # max_length lives in DPOTrainingArguments (DPOConfig.max_length) to avoid
    # argparse conflicts — both classes would otherwise register --max_length.
    # Remap original image_root values baked into the JSON to local paths.
    # Format: comma-separated "old_prefix:new_prefix" pairs, e.g.:
    #   "/home/wenhao/Datasets/med/rad/iu_xray/images:/mnt/d/iu_xray/images,..."
    image_root_remap: Optional[str] = field(default=None)
    # Fallback image folders to search if image not found in primary root.
    # Format: comma-separated paths, e.g.: "/path/to/iu_xray,/path/to/mimic_cxr"
    fallback_image_folders: Optional[str] = field(default=None)


@dataclass
class DPOTrainingArguments(DPOConfig):
    # Override DPOConfig.max_length default (None) to 1024: 256 image tokens + ~768 text tokens
    max_length: Optional[int] = field(default=1024)
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=64)       # 128 → 64: halves LoRA + optimizer memory
    lora_alpha: int = field(default=128)  # keep ratio lora_alpha/lora_r = 2
    lora_dropout: float = field(default=0.05)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_bnb_8bit")  # 8-bit optimizer: saves ~1 GB vs fp32 AdamW


# ---------------------------------------------------------------------------
# Custom DPO trainer
# ---------------------------------------------------------------------------

class MedGemmaDPOTrainer(DPOTrainer):
    """
    DPO trainer for MedGemma (Gemma3ForConditionalGeneration + SigLIP).

    Inherits from trl.DPOTrainer for proper DPO infrastructure: loss variants
    (sigmoid/hinge/ipo/…), null_ref_context for PEFT reference model, and
    store_metrics/log integration.  compute_loss is overridden for the
    multimodal-specific forward pass: SigLIP runs once per image (no_grad +
    detach), then 4 LM-only passes (policy chosen/rejected, ref chosen/rejected).

    Dataset is pre-tokenised (MedGemmaDPODataset), so __init__ bypasses TRL's
    .map()-based tokenisation by passing a dummy empty HF Dataset, then swapping
    in the real dataset after super().__init__() completes.
    """

    def __init__(self, image_token_id: int = None, train_dataset=None, **kwargs):
        # TRL's DPOTrainer.__init__ calls train_dataset.map(...) which requires a
        # HuggingFace datasets.Dataset.  Our MedGemmaDPODataset is a pre-tokenised
        # torch.utils.data.Dataset, so we pass an empty HF Dataset to let TRL
        # complete its attribute setup, then swap in the real dataset.
        fake_ds = HFDataset.from_dict({"prompt": [], "chosen": [], "rejected": []})
        super().__init__(train_dataset=fake_ds, **kwargs)
        self.train_dataset = train_dataset
        self.image_token_id = image_token_id

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
                "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
                "chosen_pixel_values", "rejected_pixel_values",
            ]

    def log(self, logs: Dict[str, float], start_time=None) -> None:
        # TRL 0.12.1's DPOTrainer.log() doesn't accept start_time, but newer
        # transformers passes it.  Replicate TRL's metrics-flush logic here and
        # call Trainer.log directly to stay compatible with both versions.
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        if start_time is not None:
            super(DPOTrainer, self).log(logs, start_time)
        else:
            super(DPOTrainer, self).log(logs)

    # ------------------------------------------------------------------
    # Log-probability computation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != IGNORE_INDEX
        labels[labels == IGNORE_INDEX] = 0
        per_token_logps = torch.gather(
            logits.to(torch.float32).log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        # Normalize by number of response tokens so rewards don't scale with
        # sequence length — prevents sigmoid saturation and grad underflow.
        n_tokens = loss_mask.sum(-1).clamp(min=1)
        return (per_token_logps * loss_mask).sum(-1) / n_tokens

    # ------------------------------------------------------------------
    # Vision feature pre-computation + inputs_embeds construction
    # ------------------------------------------------------------------

    def _build_inputs_embeds(
        self,
        base_model: nn.Module,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace image-placeholder tokens in the token-embedding sequence with
        pre-computed SigLIP features.  Mirrors what Gemma3ForConditionalGeneration
        does internally (masked_scatter on image_token_id positions), but lets us
        skip the vision encoder on every call.

        image_features: [B, n_img, lm_hidden]  (n_img == 256 for MedGemma)
        """
        inputs_embeds = base_model.get_input_embeddings()(input_ids)
        mask = (
            (input_ids == self.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        # masked_scatter expects a 1-D source with exactly as many elements as
        # True entries in the mask: B * n_img * lm_hidden
        flat_features = image_features.to(inputs_embeds.dtype).reshape(-1)
        return inputs_embeds.masked_scatter(mask, flat_features)

    # ------------------------------------------------------------------
    # LM-only forward (vision features already baked into inputs_embeds)
    # ------------------------------------------------------------------

    def _forward(
        self,
        model: nn.Module,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.FloatTensor:
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # accelerate's convert_to_fp32 upcasts logits to fp32:
        # [1, 1024, 256128] × fp32 = 1 GB per pass.  Downcast to bf16 immediately
        # (512 MB) — sufficient precision for DPO log-ratio differences.
        logits = out.logits.to(torch.bfloat16)
        del out
        return self._get_batch_logps(logits, labels)

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple]:

        unwrapped = self.accelerator.unwrap_model(model)
        # Navigate PeftModel → LoraModel → Gemma3ForConditionalGeneration
        base_model = unwrapped.base_model.model

        # ── SigLIP: run ONCE per image, always without gradient ─────────
        # SigLIP is entirely frozen (not in LoRA targets), so its output is
        # identical with adapters on or off.  Computing it once (no_grad +
        # detach) instead of 4× eliminates the ~1 GB attention peak per pass.
        with torch.no_grad():
            chosen_features = base_model.get_image_features(
                inputs["chosen_pixel_values"].to(torch.bfloat16)
            ).detach()
            rejected_features = base_model.get_image_features(
                inputs["rejected_pixel_values"].to(torch.bfloat16)
            ).detach()
        # Flush CUDA allocator after the large SigLIP attention peaks so
        # reserved-but-unallocated memory is returned before the LM passes.
        torch.cuda.empty_cache()

        # ── Build inputs_embeds (image features substituted) ────────────
        chosen_embeds = self._build_inputs_embeds(
            base_model, inputs["chosen_input_ids"], chosen_features
        )
        rejected_embeds = self._build_inputs_embeds(
            base_model, inputs["rejected_input_ids"], rejected_features
        )

        # ── Policy: chosen + rejected (LoRA active, grads through LM) ───
        policy_chosen_logps = self._forward(
            model, chosen_embeds,
            inputs["chosen_attention_mask"], inputs["chosen_labels"],
        )
        policy_rejected_logps = self._forward(
            model, rejected_embeds,
            inputs["rejected_attention_mask"], inputs["rejected_labels"],
        )

        # ── Reference: chosen + rejected (LoRA disabled via null_ref_context) ─
        with torch.no_grad(), self.null_ref_context():
            ref_chosen_logps = self._forward(
                model, chosen_embeds.detach(),
                inputs["chosen_attention_mask"], inputs["chosen_labels"],
            )
            ref_rejected_logps = self._forward(
                model, rejected_embeds.detach(),
                inputs["rejected_attention_mask"], inputs["rejected_labels"],
            )

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )
        loss = losses.mean()

        # ── Metrics — stored via TRL's store_metrics/log pipeline ───────
        self.store_metrics({
            "rewards/chosen": chosen_rewards.mean().item(),
            "rewards/rejected": rejected_rewards.mean().item(),
            "rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().item(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
        }, train_eval="train")

        return (loss, {}) if return_outputs else loss


# ---------------------------------------------------------------------------
# Model + LoRA setup
# ---------------------------------------------------------------------------

def _patch_init_weights_for_qlora():
    """
    Patch weight initialisation to handle bitsandbytes 4-bit (QLoRA).

    transformers calls self.apply(self._initialize_weights) inside
    _initialize_missing_keys, which tries normal_() on uint8 'Byte' weights:
      NotImplementedError: "normal_kernel_cpu" not implemented for 'Byte'

    The bnb quantization happens *during* from_pretrained so the Linear4bit
    layers do not exist yet when _initialize_missing_keys is first entered —
    we cannot detect them there.  Instead we patch _initialize_weights (the
    per-module wrapper) to silently skip any module with a Byte weight.
    This is safe because bitsandbytes has already initialised those weights.
    """
    import torch
    from transformers.modeling_utils import PreTrainedModel

    _orig = PreTrainedModel._initialize_weights

    def _safe_initialize_weights(self, module):
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dtype == torch.uint8:
            return  # bitsandbytes quantized weight — already initialised
        return _orig(self, module)

    PreTrainedModel._initialize_weights = _safe_initialize_weights


def load_medgemma(model_args: ModelArguments, training_args: DPOTrainingArguments):
    # Use AutoModelForImageTextToText so the correct class (Gemma3ForConditionalGeneration
    # or similar) is selected from the checkpoint's model_type field.
    # Loading MedGemma (type "gemma3") with PaliGemmaForConditionalGeneration fails because
    # the multi_modal_projector is newly-initialised with wrong hidden-size dimensions,
    # causing the image-token count check to fail at inference time.
    from transformers import AutoModelForImageTextToText

    _patch_init_weights_for_qlora()

    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # With DDP each process owns one GPU — device_map must point to that GPU.
    # device_map="auto" is incompatible with DDP and must not be used there.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        device_map = "auto"          # single-GPU: let accelerate place layers
    else:
        device_map = {"": local_rank}  # DDP: this process owns exactly one GPU

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False

    if training_args.lora_enable:
        # Always enable gradient checkpointing for QLoRA on constrained hardware.
        # Four DPO forward passes per step make this essential on T4/A100.
        #
        # use_reentrant=False is required when the same LoRA parameters are used
        # in multiple forward passes per step (chosen + rejected).  The old-style
        # reentrant checkpointing causes DDP to fire "mark variable ready" hooks
        # twice for the same parameter, crashing with:
        #   "Expected to mark a variable ready only once."
        gc_kwargs = {"use_reentrant": False}
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs=gc_kwargs,
            )
        except TypeError:
            # Older PEFT versions don't accept gradient_checkpointing_kwargs
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

        training_args.gradient_checkpointing = True            # keep Trainer in sync
        training_args.gradient_checkpointing_kwargs = gc_kwargs  # Trainer passes this to enable()
        
        if model_args.lora_checkpoint_path:
            logger.info(f"Resuming LoRA from {model_args.lora_checkpoint_path}")
            model = PeftModel.from_pretrained(
                model, model_args.lora_checkpoint_path, is_trainable=True
            )
        else:
            lora_cfg = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        # prepare_model_for_kbit_training enables gradient checkpointing on the LM
        # but not always on the vision tower.  Enable it explicitly on SigLIP so
        # the large 896×896 → 4096-patch ViT doesn't OOM during the DPO forward.
        try:
            base = model.base_model.model
            if hasattr(base, "vision_tower"):
                base.vision_tower.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gc_kwargs
                )
                logger.info("Vision tower gradient checkpointing enabled (use_reentrant=False).")
        except Exception as e:
            logger.warning(f"Could not enable vision tower gradient checkpointing: {e}")

    return model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        os.environ["WANDB_DISABLED"] = "true"
        logger.warning("WANDB_API_KEY not set — W&B logging disabled.")

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        token=os.environ.get("HF_TOKEN"),
    )

    model = load_medgemma(model_args, training_args)
    print(processor.tokenizer.additional_special_tokens)
    # Resolve the image-placeholder token ID used by Gemma3 in input_ids.
    # We need this to substitute pre-computed SigLIP features into inputs_embeds
    # instead of re-running the vision encoder on every DPO forward pass.
    _base_for_cfg = getattr(getattr(model, "base_model", model), "model", model)
    image_token_id = getattr(_base_for_cfg.config, "image_token_id", None)
    if image_token_id is None:
        tok_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
        unk_id = processor.tokenizer.unk_token_id
        if tok_id != unk_id:
            image_token_id = tok_id
    if image_token_id is None:
    # Fallback: some versions might still use the prompt-level tag <img> 
    # but for feature substitution, you almost always want the soft token.
        tok_id = processor.tokenizer.convert_tokens_to_ids("<img>")
        if tok_id != unk_id:
            image_token_id = tok_id

    if image_token_id is None:
        raise RuntimeError(
            "Cannot find image_token_id in model config or tokenizer vocab. "
            "Check that the processor has a '<image_soft>' token."
        )
    logger.info(f"Image token ID for inputs_embeds substitution: {image_token_id}")

    # Parse path remapping (old_prefix:new_prefix,old2:new2,...)
    root_remap: dict = {}
    if data_args.image_root_remap:
        for pair in data_args.image_root_remap.split(","):
            if ":" in pair:
                old, new = pair.split(":", 1)
                root_remap[old.strip()] = new.strip()

    # Parse fallback image folders (comma-separated list)
    fallback_roots: list = []
    if data_args.fallback_image_folders:
        fallback_roots = [p.strip() for p in data_args.fallback_image_folders.split(",") if p.strip()]

    dataset = MedGemmaDPODataset(
        data_path=data_args.data_path,
        processor=processor,
        image_folder=data_args.image_folder,
        max_length=training_args.max_length,
        root_remap=root_remap,
        fallback_image_folders=fallback_roots,
    )
    logger.info(f"Training samples: {len(dataset)}")

    collator = DPODataCollator(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    trainer = MedGemmaDPOTrainer(
        image_token_id=image_token_id,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        ref_model=None,  # PEFT model → null_ref_context uses disable_adapter()
    )

    logger.info("Starting DPO training ...")
    trainer.train()

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved → {training_args.output_dir}")


if __name__ == "__main__":
    train()
