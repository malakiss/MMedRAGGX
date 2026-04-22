"""
Radiology report retrieval with OpenCLIP (top-3) + RadGraph-XL entity extraction.

Replaces retrieve_clip_report.py for the MedGemma pipeline.
Key changes vs the original:
  - Default --fixed_k 3 (instead of 10)
  - Runs RadGraph-XL on the top-3 retrieved reports
  - Writes a ``radgraph_context`` field alongside ``reference_reports``

Output JSONL schema (one JSON object per line):
  {
    "id": "...",
    "report": "...",                   # ground-truth report
    "image": "relative/path.png",
    "reference_reports": [...],        # top-3 raw report strings
    "radgraph_context": "...",         # entity summary for MedGemma prompt
    "retrieve_k": 3
  }

Usage (mirrors retrieve_clip_report.sh):
    python retrieve_clip_radgraph.py \
        --img_root       /path/to/iu_xray/images \
        --train_json     /path/to/rad_iu.json \
        --eval_json      /path/to/mimic_test.json \
        --model_name_or_path  hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
        --checkpoint_path     /path/to/epoch_360.pt \
        --output_path    /path/to/output_rag_radgraph.json \
        --fixed_k 3 \
        --radgraph_model radgraph-xl \
        --radgraph_batch 16
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import open_clip
from training.data import IUXrayDataset

KEEP_LABELS = {"OBS-DP", "OBS-U", "OBS-DA", "ANAT-DP"}


# ---------------------------------------------------------------------------
# RadGraph helpers
# ---------------------------------------------------------------------------

def build_radgraph_context(reports: list[str], annotations: list[dict]) -> str:
    parts = []
    for i, ann in enumerate(annotations):
        entities = []
        for e in ann.get("entities", {}).values():
            if e.get("label") in KEEP_LABELS and e.get("tokens", "").strip():
                entities.append(e["tokens"])
        seen, uniq = set(), []
        for ent in entities:
            low = ent.lower()
            if low not in seen:
                seen.add(low)
                uniq.append(ent)
        parts.append(
            f"Reference {i + 1} findings: "
            f"{', '.join(uniq[:20]) if uniq else 'no findings'}"
        )
    return "\n".join(parts)


def run_radgraph_batched(
    rg,
    reports: list[str],
    batch_size: int = 16,
) -> list[dict]:
    results = []
    for i in tqdm(range(0, len(reports), batch_size), desc="RadGraph", leave=False):
        batch = reports[i : i + batch_size]
        results.extend(rg(batch))
    return results


# ---------------------------------------------------------------------------
# CLIP retrieval (unchanged from original)
# ---------------------------------------------------------------------------

def retrieve_topk_per_image(logits, val_k_list, clip_threshold=""):
    if not clip_threshold:
        preds = []
        for i, k in enumerate(val_k_list):
            if k == 0:
                preds.append(torch.tensor([-1]))
            elif k == 1:
                preds.append(logits["image_to_text"][i].argmax(dim=0, keepdim=True))
            else:
                _, topk_idx = logits["image_to_text"][i].topk(k, dim=0)
                preds.append(topk_idx)
        return preds
    else:
        thresh = float(clip_threshold)
        preds = []
        for i, k in enumerate(val_k_list):
            if k == 0:
                preds.append(torch.tensor([-1]))
            else:
                vals = logits["image_to_text"][i]
                top1 = vals.max()
                sorted_vals, sorted_idx = torch.sort(vals, descending=True)
                ratios = top1 / sorted_vals
                sel = sorted_idx[ratios < thresh]
                if sel.size(0) > k:
                    sel = sel[:k]
                preds.append(sel)
        return preds


def get_logits(image_features, text_features, logit_scale):
    lpi = (logit_scale * image_features @ text_features.t()).detach().cpu()
    return {"image_to_text": lpi, "text_to_image": lpi.t().detach().cpu()}


def clean_data_info(data_info):
    out = {}
    for k, v in data_info.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            out[k] = v.item()
        elif isinstance(v, list) and len(v) == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


def split_and_clean_data_infos(batch_data_infos):
    n = len(next(iter(batch_data_infos.values())))
    result = []
    for i in range(n):
        item = {}
        for k, v in batch_data_infos.items():
            if isinstance(v, torch.Tensor):
                item[k] = v[i]
            elif isinstance(v, list) and len(v) == n:
                item[k] = v[i]
            else:
                item[k] = v
        result.append(clean_data_info(item))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading OpenCLIP model from {args.model_name_or_path} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name_or_path,
        pretrained=args.checkpoint_path,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model_name_or_path)

    train_dataset = IUXrayDataset(
        args.img_root,
        json_file=args.train_json,
        transforms=preprocess,
        tokenizer=tokenizer,
        load_include_path=True,
    )
    eval_dataset = IUXrayDataset(
        args.img_root,
        json_file=args.eval_json,
        transforms=preprocess,
        tokenizer=tokenizer,
        load_include_path=True,
        load_include_k=True,
        retrieval_k=int(args.fixed_k) if args.fixed_k else 3,
    )

    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=False,
                          num_workers=8, pin_memory=True)
    eval_dl  = DataLoader(eval_dataset,  batch_size=64, shuffle=False,
                          num_workers=8, pin_memory=True)
    train_dl.num_samples = len(train_dataset)
    eval_dl.num_samples  = len(eval_dataset)

    # ── Feature extraction ───────────────────────────────────────────
    train_img_feats, train_txt_feats = [], []
    val_img_feats, val_txt_feats     = [], []
    val_k_list, data_infos_list      = [], []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, texts, _ in tqdm(train_dl, desc="Train features"):
            images = images.to(device=device, dtype=input_dtype)
            texts  = texts.to(device)
            img_f, txt_f, logit_scale = model(images, texts)
            train_img_feats.append(img_f.cpu())
            train_txt_feats.append(txt_f.cpu())

        for images, texts, _, ks, data_infos in tqdm(eval_dl, desc="Eval features"):
            images = images.to(device=device, dtype=input_dtype)
            texts  = texts.to(device)
            img_f, txt_f, logit_scale = model(images, texts)
            val_img_feats.append(img_f.cpu())
            val_txt_feats.append(txt_f.cpu())
            data_infos_list.extend(split_and_clean_data_infos(data_infos))
            val_k_list.extend([int(k) for k in ks])

    logit_scale = logit_scale.mean().cpu()
    logits = get_logits(
        torch.cat(val_img_feats),
        torch.cat(train_txt_feats),
        logit_scale,
    )

    if args.fixed_k:
        val_k_list = [int(args.fixed_k)] * len(val_k_list)
    print(f"Using k={val_k_list[0]} per image")

    pred_per_image = retrieve_topk_per_image(
        logits, val_k_list, clip_threshold=args.clip_threshold
    )
    true_k_list = [len(idx) for idx in pred_per_image]

    # ── Collect reference reports ────────────────────────────────────
    print("Collecting reference reports ...")
    all_reference_reports: list[list[str]] = []
    for topk_indices in pred_per_image:
        refs = []
        for idx in topk_indices:
            if idx.item() == -1:
                break
            refs.append(train_dl.dataset.image_report_pairs[idx.item()][1])
        all_reference_reports.append(refs)

    # ── RadGraph entity extraction ───────────────────────────────────
    print(f"Loading RadGraph model: {args.radgraph_model} ...")
    from radgraph import RadGraph
    rg = RadGraph(model_type=args.radgraph_model)

    # Flatten all reports for batched inference
    flat_reports: list[str] = []
    sample_spans: list[tuple[int, int]] = []
    for refs in all_reference_reports:
        start = len(flat_reports)
        flat_reports.extend(refs)
        sample_spans.append((start, start + len(refs)))

    flat_annotations = run_radgraph_batched(rg, flat_reports, batch_size=args.radgraph_batch)

    # ── Write output ─────────────────────────────────────────────────
    print(f"Writing output to {args.output_path} ...")
    with open(args.output_path, "w") as f:
        for i, (data_info, k) in enumerate(zip(data_infos_list, true_k_list)):
            refs  = all_reference_reports[i]
            start, end = sample_spans[i]
            anns  = flat_annotations[start:end]
            ctx   = build_radgraph_context(refs, anns)

            record = {
                "id":               data_info.get("id", ""),
                "report":           data_info.get("report", ""),
                "image":            data_info.get("image_path", ""),
                "reference_reports": refs,
                "radgraph_context": ctx,
                "retrieve_k":       k,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Done. Wrote {len(data_infos_list)} records → {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenCLIP top-3 retrieval + RadGraph-XL entity extraction"
    )
    parser.add_argument("--img_root",          type=str, required=True)
    parser.add_argument("--train_json",        type=str, required=True)
    parser.add_argument("--eval_json",         type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str,
                        default="hf-hub:thaottn/OpenCLIP-resnet50-CC12M")
    parser.add_argument("--checkpoint_path",   type=str, default="")
    parser.add_argument("--output_path",       type=str, required=True)
    parser.add_argument("--clip_threshold",    type=str, default="")
    parser.add_argument("--fixed_k",           type=str, default="3")
    parser.add_argument("--radgraph_model",    type=str, default="radgraph-xl",
                        choices=["radgraph", "radgraph-xl"])
    parser.add_argument("--radgraph_batch",    type=int, default=16)
    main(parser.parse_args())
