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
import torch.nn.functional as F
import transformers
import wandb
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutput

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
    max_length: int = field(default=512)   # 256 image tokens + ~256 text tokens
    # Remap original image_root values baked into the JSON to local paths.
    # Format: comma-separated "old_prefix:new_prefix" pairs, e.g.:
    #   "/home/wenhao/Datasets/med/rad/iu_xray/images:/mnt/d/iu_xray/images,..."
    image_root_remap: Optional[str] = field(default=None)
    # Fallback image folders to search if image not found in primary root.
    # Format: comma-separated paths, e.g.: "/path/to/iu_xray,/path/to/mimic_cxr"
    fallback_image_folders: Optional[str] = field(default=None)


@dataclass
class DPOTrainingArguments(TrainingArguments):
    beta: float = field(default=0.1)
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)
    remove_unused_columns: bool = field(default=False)


# ---------------------------------------------------------------------------
# Custom DPO trainer
# ---------------------------------------------------------------------------

class MedGemmaDPOTrainer(Trainer):
    """
    DPO sigmoid-loss trainer for MedGemma (Gemma3ForConditionalGeneration + SigLIP).

    Memory-efficient design: SigLIP (896×896 → 4096 patches) runs exactly TWICE
    per step (once for chosen, once for rejected), always inside torch.no_grad()
    and with the result detached.  The 4 log-prob computations (policy × 2,
    reference × 2) then run as LM-only passes with pre-built inputs_embeds,
    avoiding the 1024 MB SigLIP attention allocation that causes OOM on T4s.

    Reference model = same PeftModel with LoRA adapters disabled — no 2nd copy.
    """

    def __init__(self, beta: float = 0.1, image_token_id: int = None, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.image_token_id = image_token_id

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
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)

    # ------------------------------------------------------------------
    # DPO loss
    # ------------------------------------------------------------------

    def _dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards

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
        return self._get_batch_logps(out.logits.float(), labels)

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

        # ── Reference: chosen + rejected (LoRA disabled, no grad) ───────
        with torch.no_grad():
            with unwrapped.disable_adapter():
                ref_chosen_logps = self._forward(
                    model, chosen_embeds.detach(),
                    inputs["chosen_attention_mask"], inputs["chosen_labels"],
                )
                ref_rejected_logps = self._forward(
                    model, rejected_embeds.detach(),
                    inputs["rejected_attention_mask"], inputs["rejected_labels"],
                )

        losses, chosen_rewards, rejected_rewards = self._dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )
        loss = losses.mean()

        # ── Metrics ─────────────────────────────────────────────────────
        if self.accelerator.is_main_process:
            self.log({
                "train/loss": loss.item(),
                "train/chosen_rewards": chosen_rewards.mean().item(),
                "train/rejected_rewards": rejected_rewards.mean().item(),
                "train/reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
                "train/reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
            })

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
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        training_args.gradient_checkpointing = True  # keep Trainer in sync

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
                base.vision_tower.gradient_checkpointing_enable()
                logger.info("Vision tower gradient checkpointing enabled.")
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

    # Resolve the image-placeholder token ID used by Gemma3 in input_ids.
    # We need this to substitute pre-computed SigLIP features into inputs_embeds
    # instead of re-running the vision encoder on every DPO forward pass.
    _base_for_cfg = getattr(getattr(model, "base_model", model), "model", model)
    image_token_id = getattr(_base_for_cfg.config, "image_token_id", None)
    if image_token_id is None:
        tok_id = processor.tokenizer.convert_tokens_to_ids("<image_soft>")
        unk_id = processor.tokenizer.unk_token_id
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
        max_length=data_args.max_length,
        root_remap=root_remap,
        fallback_image_folders=fallback_roots,
    )
    logger.info(f"Training samples: {len(dataset)}")

    collator = DPODataCollator(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    trainer = MedGemmaDPOTrainer(
        beta=training_args.beta,
        image_token_id=image_token_id,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("Starting DPO training ...")
    trainer.train()

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved → {training_args.output_dir}")


if __name__ == "__main__":
    train()
