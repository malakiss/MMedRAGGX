"""
Preprocessing script: enrich the DPO alignment JSON with RadGraph entities.

Reads  : data/training/alignment/radiology/radiology_vqa_iu.json
Writes : data/training/alignment/radiology/radiology_vqa_iu_radgraph.json

Each item gets a new field  ``radgraph_context``  which replaces the raw
reference-report block in the prompt with a compact, entity-level summary
produced by RadGraph-XL.

Usage:
    python preprocess_radgraph_alignment.py \
        --input  ../../data/training/alignment/radiology/radiology_vqa_iu.json \
        --output ../../data/training/alignment/radiology/radiology_vqa_iu_radgraph.json \
        --model_type radgraph-xl \
        --batch_size 8
"""

import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

KEEP_LABELS = {"OBS-DP", "OBS-U", "OBS-DA", "ANAT-DP"}

# "Reference N: text"  style  (radiology_vqa_iu.json)
_REFERENCE_RE = re.compile(
    r"Reference\s+\d+:\s*(.*?)(?=Reference\s+\d+:|Question:|$)",
    re.DOTALL,
)
# "N. text"  style  (radiology_report.json)
_NUMBERED_RE = re.compile(r"\d+\.\s+(.*?)(?=\s*\n\d+\.|$)", re.DOTALL)


def extract_raw_references(human_text: str) -> list[str]:
    # Strip LLaVA image token
    text = re.sub(r"<image>\s*\n?", "", human_text)
    # Try "Reference N: ..." first
    refs = [m.group(1).strip() for m in _REFERENCE_RE.finditer(text)]
    if refs:
        return refs
    # Fall back to "N. ..." style; extract only the numbered-list section
    ref_section = text.split("\nPlease")[0].split("report(s):")[-1]
    refs = [m.group(1).strip() for m in _NUMBERED_RE.finditer(ref_section)]
    return refs if refs else [text]


def build_radgraph_context(reports: list[str], annotations: list[dict]) -> str:
    parts = []
    for i, ann in enumerate(annotations):
        entities = [
            e["tokens"]
            for e in ann.get("entities", {}).values()
            if e.get("label") in KEEP_LABELS and e.get("tokens", "").strip()
        ]
        # deduplicate preserving order
        seen, uniq = set(), []
        for ent in entities:
            low = ent.lower()
            if low not in seen:
                seen.add(low)
                uniq.append(ent)
        parts.append(f"Reference {i + 1} findings: {', '.join(uniq[:25]) or 'no findings'}")
    return "\n".join(parts)


def main(args):
    data = json.load(open(args.input))
    print(f"Loaded {len(data)} samples from {args.input}")

    from radgraph import RadGraph
    rg = RadGraph(model_type=args.model_type)

    results = []
    batch_size = args.batch_size

    # Collect all unique reports across the whole dataset to avoid redundant inference
    all_reports: list[str] = []
    sample_ref_indices: list[list[int]] = []

    for item in tqdm(data, desc="Collecting reports"):
        human_text = item["conversations"][0]["value"]
        refs = extract_raw_references(human_text)
        indices = []
        for r in refs:
            if r not in all_reports:
                all_reports.append(r)
            indices.append(all_reports.index(r))
        sample_ref_indices.append(indices)

    print(f"Running RadGraph on {len(all_reports)} unique reports ...")
    all_annotations: list[dict] = []
    for i in tqdm(range(0, len(all_reports), batch_size), desc="RadGraph"):
        batch = all_reports[i : i + batch_size]
        ann = rg(batch)
        all_annotations.extend(ann)

    for item, ref_idx in zip(data, sample_ref_indices):
        refs = [all_reports[i] for i in ref_idx]
        anns = [all_annotations[i] for i in ref_idx]
        item["radgraph_context"] = build_radgraph_context(refs, anns)
        results.append(item)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved enriched dataset → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_type", default="radgraph-xl",
                        choices=["radgraph", "radgraph-xl"])
    parser.add_argument("--batch_size", type=int, default=8)
    main(parser.parse_args())
