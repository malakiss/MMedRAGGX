"""
DPO dataset for MedGemma fine-tuning.

Each sample returns tokenized chosen + rejected pairs where:
  - chosen  uses the real X-ray image + RAG context (RadGraph entities)
  - rejected uses a Gaussian-noise image (if rejected_noised==1) + no context
"""

import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import ProcessorMixin

# Allow loading of truncated/incomplete JPEG images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IGNORE_INDEX = -100

# Matches LLaVA-style image token that must be stripped for MedGemma
_IMAGE_TOKEN_RE = re.compile(r"<image>\s*\n?")

# Matches numbered reference lists: "1. text\n2. text ..."
_NUMBERED_REF_RE = re.compile(r"\d+\.\s+(.*?)(?=\s*\n\d+\.|$)", re.DOTALL)


def generate_gaussian_noise(image: Image.Image, stddev: float = 1.0) -> Image.Image:
    arr = np.random.normal(0.0, stddev, np.array(image).shape)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


@dataclass
class DPODataCollator:
    pad_token_id: int

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        keys = [
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
        ]
        batch = {}
        for k in keys:
            seqs = [inst[k] for inst in instances]
            pad_val = IGNORE_INDEX if "labels" in k else (
                0 if "attention" in k else self.pad_token_id
            )
            max_len = max(s.shape[0] for s in seqs)
            padded = torch.stack([
                torch.nn.functional.pad(s, (0, max_len - s.shape[0]), value=pad_val)
                for s in seqs
            ])
            batch[k] = padded

        # pixel_values: (B, C, H, W)  — already tensors from processor
        for key in ("chosen_pixel_values", "rejected_pixel_values"):
            batch[key] = torch.stack([inst[key] for inst in instances])

        return batch


class MedGemmaDPODataset(Dataset):
    """
    Reads radiology_report.json (or the RadGraph-enriched variant) and
    yields tokenized DPO pairs for MedGemma.

    JSON schema per item (as produced by the original MMed-RAG pipeline):
        id, image, image_root, conversations, rejected_conversations,
        rejected_noised (int, optional), radgraph_context (str, optional)

    Path remapping
    --------------
    The JSON stores the original machine's absolute ``image_root`` paths
    (e.g. /home/wenhao/Datasets/...).  Pass ``root_remap`` to translate
    those to paths that exist on the current machine:

        root_remap={
          "/home/wenhao/Datasets/med/rad/iu_xray/images":    "/mnt/d/iu_xray/images",
          "/home/wenhao/Datasets/med/rad/mimic-cxr-jpg/...": "/mnt/d/mimic_cxr/files",
        }

    If no remap matches, ``image_folder`` is used as a universal fallback root.
    """

    def __init__(
        self,
        data_path: str,
        processor: ProcessorMixin,
        image_folder: str,
        max_length: int = 1024,
        root_remap: Optional[Dict[str, str]] = None,
        fallback_image_folders: Optional[List[str]] = None,
    ):
        self.data = json.load(open(data_path))
        self.processor = processor
        self.image_folder = image_folder
        self.max_length = max_length
        self.root_remap = root_remap or {}
        self.fallback_image_folders = fallback_image_folders or []

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_image_path(self, item: dict) -> str:
        """
        Resolve the full image path.
        Priority:
          1. root_remap match (full path remapping)
          2. image_folder + full image path
          3. fallback_image_folders + basename only (last component after /)
        """
        original_root = item.get("image_root", "")
        image_name = item["image"]
        basename = os.path.basename(image_name)  # just the filename

        candidates = []

        # 1. Try remapped roots first (full path)
        for old_root, new_root in self.root_remap.items():
            if original_root.startswith(old_root):
                candidates.append(os.path.join(new_root, image_name))

        # 2. Primary image folder (full path)
        candidates.append(os.path.join(self.image_folder, image_name))

        # 3. Fallback folders search by basename only
        for fallback_root in self.fallback_image_folders:
            candidates.append(os.path.join(fallback_root, basename))

        # Return first candidate that exists, else return first candidate
        for path in candidates:
            if os.path.isfile(path):
                return path

        # If none exist, return first candidate anyway (will error at load time with useful message)
        return candidates[0] if candidates else os.path.join(self.image_folder, image_name)

    @staticmethod
    def _strip_image_token(text: str) -> str:
        """Remove LLaVA-style <image> token — MedGemma uses the processor instead."""
        return _IMAGE_TOKEN_RE.sub("", text).strip()

    def _build_question(self, item: dict) -> str:
        """
        Build the text prompt for MedGemma (no <image> token — that is
        injected by the processor via {"type": "image"} in the message).

        If ``radgraph_context`` is present (from preprocess_radgraph_alignment.py),
        use structured entity summaries.  Otherwise fall back to the raw
        reference-report text as embedded in the original human turn.
        """
        if "radgraph_context" in item and item["radgraph_context"].strip():
            context_block = item["radgraph_context"]
            # Guard against very long RAG contexts that push the prompt past
            # max_length, leaving no room for the response (all labels → IGNORE_INDEX).
            # ~1500 chars ≈ 375 tokens; with image (256) + system (~80) the prompt
            # stays well under 1024, reserving ≥300 tokens for the response.
            if len(context_block) > 1500:
                context_block = context_block[:1500].rsplit(" ", 1)[0] + " ..."
            return (
                "You are a professional radiologist. "
                "Below are structured findings extracted from similar chest X-ray reports:\n\n"
                f"{context_block}\n\n"
                "Please generate a report based on the image. "
                "The reference findings are for comparison only and cannot be directly "
                "used as a diagnosis. Include only the report content in your response."
            )
        else:
            # Strip the <image> token that the original pipeline embedded in the text
            raw = self._strip_image_token(item["conversations"][0]["value"])
            return raw

    def _tokenize_pair(
        self, image: Image.Image, question: str, response: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (input_ids, attention_mask, pixel_values, labels).
        Labels are -100 for the prompt tokens; actual ids only for response.
        """
        messages_full = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": response},
        ]

        messages_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text_full = self.processor.apply_chat_template(
            messages_full, add_generation_prompt=False
        )
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, add_generation_prompt=True
        )

        full_enc = self.processor(
            text=text_full,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        prompt_enc = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )
        # Cap to max_length: if the untruncated prompt exceeds the budget,
        # the full_enc only has max_length tokens; indexing beyond would be
        # a no-op but prompt_len > max_length means the response was truncated
        # away entirely.  The context cap in _build_question prevents this, but
        # this is a second line of defence.
        prompt_len = min(prompt_enc["input_ids"].shape[1], self.max_length)

        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]
        pixel_values = full_enc["pixel_values"][0]

        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        labels[input_ids == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

        return input_ids, attention_mask, pixel_values, labels

    # ------------------------------------------------------------------

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        item = self.data[i]

        image_path = self._resolve_image_path(item)
        image = Image.open(image_path).convert("RGB")

        rejected_image = (
            generate_gaussian_noise(image)
            if item.get("rejected_noised") == 1
            else copy.deepcopy(image)
        )

        question = self._build_question(item)
        chosen_response = item["conversations"][1]["value"]
        rejected_response = item["rejected_conversations"][1]["value"]

        c_ids, c_mask, c_pv, c_labels = self._tokenize_pair(image, question, chosen_response)
        r_ids, r_mask, r_pv, r_labels = self._tokenize_pair(rejected_image, question, rejected_response)

        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "chosen_pixel_values": c_pv,
            "chosen_labels": c_labels,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
            "rejected_pixel_values": r_pv,
            "rejected_labels": r_labels,
        }
