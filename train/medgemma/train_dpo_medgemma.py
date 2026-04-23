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
    max_length: int = field(default=1024)
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
    Trainer that implements the DPO sigmoid loss for MedGemma.

    Does TWO separate forward passes per step (chosen + rejected) because
    chosen uses the real image while rejected uses a Gaussian-noise image —
    they cannot be batched into a single concatenated forward.

    The reference model is the same PeftModel with adapters disabled,
    so we never need to load a second copy of the 4B weights.
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    # ------------------------------------------------------------------
    # Log-probability computation (identical to original pipeline)
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
    # DPO loss (same sigmoid formula as original)
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
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.FloatTensor:
        # pixel_values must be in bfloat16 for SigLIP — model.dtype returns uint8
        # for QLoRA storage and must not be used here.
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.to(torch.bfloat16),
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

        # ── Policy: chosen ──────────────────────────────────────────────
        policy_chosen_logps = self._forward(
            model,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"],
            inputs["chosen_pixel_values"],
            inputs["chosen_labels"],
        )

        # ── Policy: rejected ────────────────────────────────────────────
        policy_rejected_logps = self._forward(
            model,
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"],
            inputs["rejected_pixel_values"],
            inputs["rejected_labels"],
        )

        # ── Reference model (base weights, LoRA disabled) ──────────────
        with torch.no_grad():
            with self.accelerator.unwrap_model(model).disable_adapter():
                ref_chosen_logps = self._forward(
                    model,
                    inputs["chosen_input_ids"],
                    inputs["chosen_attention_mask"],
                    inputs["chosen_pixel_values"],
                    inputs["chosen_labels"],
                )
                ref_rejected_logps = self._forward(
                    model,
                    inputs["rejected_input_ids"],
                    inputs["rejected_attention_mask"],
                    inputs["rejected_pixel_values"],
                    inputs["rejected_labels"],
                )

        losses, chosen_rewards, rejected_rewards = self._dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
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
    Patch PaliGemma's _init_weights to skip uint8 (4-bit NF4) tensors.

    Some transformers versions call _initialize_missing_keys → _init_weights on ALL
    submodules including already-quantized ones, then fail with:
      NotImplementedError: "normal_kernel_cpu" not implemented for 'Byte'
    Skipping Byte-dtype weights is safe — they are already initialised by bnb.
    """
    from transformers.models.paligemma import modeling_paligemma as _pg

    _orig = _pg.PaliGemmaForConditionalGeneration._init_weights

    def _safe_init(self, module):
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dtype == torch.uint8:
            return  # already quantised — skip
        try:
            _orig(self, module)
        except (NotImplementedError, RuntimeError):
            pass  # other quantised sub-modules (e.g. bnb Linear4bit)

    _pg.PaliGemmaForConditionalGeneration._init_weights = _safe_init


def load_medgemma(model_args: ModelArguments, training_args: DPOTrainingArguments):
    from transformers import PaliGemmaForConditionalGeneration

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

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False

    if training_args.lora_enable:
        if model_args.lora_checkpoint_path:
            logger.info(f"Resuming LoRA from {model_args.lora_checkpoint_path}")
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            model = PeftModel.from_pretrained(
                model, model_args.lora_checkpoint_path, is_trainable=True
            )
        else:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
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
