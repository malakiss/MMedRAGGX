"""
MedGemma SFT (Supervised Fine-Tuning) — Stage 1 before DPO.

Trains on the CHOSEN responses only (standard cross-entropy).
Run 1 epoch, save the LoRA adapter, then use it as the starting
point for DPO via --lora_checkpoint_path.

Usage (Kaggle 2×T4):
    accelerate launch --num_processes 2 --mixed_precision bf16 \
        train/medgemma/train_sft_medgemma.py \
        --data_path      /kaggle/input/.../radiology_report.json \
        --image_folder   /kaggle/input/... \
        --output_dir     /kaggle/working/medgemma_sft \
        --num_train_epochs 1
"""

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from medgemma_dataset import DPODataCollator, MedGemmaDPODataset

IGNORE_INDEX = -100
MODEL_ID = "google/medgemma-4b-it"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=MODEL_ID)
    use_4bit: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    max_length: int = field(default=1024)
    image_root_remap: Optional[str] = field(default=None)
    fallback_image_folders: Optional[str] = field(default=None)


@dataclass
class SFTTrainingArguments(TrainingArguments):
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_bnb_8bit")


# ---------------------------------------------------------------------------
# SFT collator — extracts chosen fields, renames to standard model inputs
# ---------------------------------------------------------------------------

class SFTCollator:
    """
    Wraps DPODataCollator output: discards rejected_* fields and renames
    chosen_* to the field names Gemma3ForConditionalGeneration expects.
    """
    def __init__(self, pad_token_id: int):
        self._dpo_collator = DPODataCollator(pad_token_id=pad_token_id)

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = self._dpo_collator(instances)
        return {
            "input_ids":      batch["chosen_input_ids"],
            "attention_mask": batch["chosen_attention_mask"],
            "labels":         batch["chosen_labels"],
            "pixel_values":   batch["chosen_pixel_values"],
        }


# ---------------------------------------------------------------------------
# SFT trainer — single forward pass, CE loss on chosen responses
# ---------------------------------------------------------------------------

class MedGemmaSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = out.loss
        if self.accelerator.is_main_process:
            self.log({"train/sft_loss": loss.item()})
        return (loss, out) if return_outputs else loss


# ---------------------------------------------------------------------------
# Model loading (same as DPO script)
# ---------------------------------------------------------------------------

def _patch_init_weights_for_qlora():
    from transformers.modeling_utils import PreTrainedModel
    _orig = PreTrainedModel._initialize_weights
    def _safe(self, module):
        w = getattr(module, "weight", None)
        if w is not None and w.dtype == torch.uint8:
            return
        return _orig(self, module)
    PreTrainedModel._initialize_weights = _safe


def load_medgemma(model_args: ModelArguments, training_args: SFTTrainingArguments):
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

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank != -1 else "auto"

    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.use_cache = False

    gc_kwargs = {"use_reentrant": False}
    try:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs=gc_kwargs,
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = gc_kwargs

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
# Main
# ---------------------------------------------------------------------------

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        os.environ["WANDB_DISABLED"] = "true"

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        token=os.environ.get("HF_TOKEN"),
    )

    model = load_medgemma(model_args, training_args)

    root_remap: dict = {}
    if data_args.image_root_remap:
        for pair in data_args.image_root_remap.split(","):
            if ":" in pair:
                old, new = pair.split(":", 1)
                root_remap[old.strip()] = new.strip()

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
    logger.info(f"SFT training samples: {len(dataset)}")

    collator = SFTCollator(pad_token_id=processor.tokenizer.pad_token_id)

    trainer = MedGemmaSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("Starting SFT ...")
    trainer.train()

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"SFT adapter saved → {training_args.output_dir}")


if __name__ == "__main__":
    train()
