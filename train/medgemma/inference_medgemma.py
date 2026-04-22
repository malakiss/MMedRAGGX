"""
Inference script for fine-tuned MedGemma.

Reads the RadGraph-enriched retrieval output (from retrieve_clip_radgraph.py),
builds a structured prompt, and generates a radiology report per image.

Usage:
    python inference_medgemma.py \
        --checkpoint   /path/to/dpo_checkpoint \
        --retrieved    /path/to/output_rag_radgraph.json \
        --img_root     /path/to/iu_xray/images \
        --output       results_medgemma.json \
        --use_4bit
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a board-certified radiologist. "
    "Given the chest X-ray image and structured reference findings, "
    "write a concise, accurate radiology report."
)


def build_prompt(radgraph_context: str) -> str:
    if radgraph_context:
        return (
            "Below are structured findings extracted from similar radiology reports:\n\n"
            f"{radgraph_context}\n\n"
            "Based on the image and these reference findings, "
            "generate a detailed radiology report."
        )
    return "Based on the chest X-ray image, generate a detailed radiology report."


def load_model(checkpoint: str, base_model_id: str, use_4bit: bool):
    from transformers import BitsAndBytesConfig

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )

    # Load LoRA adapter if checkpoint differs from base
    if Path(checkpoint).is_dir() and (Path(checkpoint) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.merge_and_unload()
    else:
        model = base

    model.eval()
    return model


@torch.inference_mode()
def generate_report(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=0.9,
    )
    # decode only the newly generated tokens
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens[0], skip_special_tokens=True).strip()


def main(args):
    print("Loading processor and model ...")
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        token=os.environ.get("HF_TOKEN"),
    )
    model = load_model(args.checkpoint, args.base_model, args.use_4bit)

    print(f"Reading retrieval output from {args.retrieved} ...")
    samples = []
    with open(args.retrieved) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    results = []
    for sample in tqdm(samples, desc="Generating reports"):
        image_path = os.path.join(args.img_root, sample["image"])
        image = Image.open(image_path).convert("RGB")

        radgraph_ctx = sample.get("radgraph_context", "")
        prompt_text = build_prompt(radgraph_ctx)

        generated = generate_report(
            model, processor, image, prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        results.append({
            "id": sample.get("id", ""),
            "image": sample["image"],
            "ground_truth": sample.get("report", ""),
            "radgraph_context": radgraph_ctx,
            "generated_report": generated,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to DPO-fine-tuned LoRA checkpoint dir")
    parser.add_argument("--base_model", default="google/medgemma-4b-it")
    parser.add_argument("--retrieved", required=True,
                        help="JSONL from retrieve_clip_radgraph.py")
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--output", default="results_medgemma.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    main(parser.parse_args())
