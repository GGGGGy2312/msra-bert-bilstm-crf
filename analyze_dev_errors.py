"""Audit entity errors on the fixed development split of a saved checkpoint.

This script does not tune the checkpoint or read MSRA test.tsv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_bilstm_crf import (BertBiLstmCrf, TAGS, collate, encode, entities,
                              read_msra)
from data import ensure_file, md5
from download_model import ensure_model


ROOT = Path(__file__).resolve().parent
LENGTH_BUCKETS = ((1, 2, "1-2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, None, "9+"))


def length_bucket(length: int) -> str:
    for lower, upper, name in LENGTH_BUCKETS:
        if length >= lower and (upper is None or length <= upper):
            return name
    raise ValueError("entity spans must have positive length")


def classify_miss(gold: tuple[str, int, int], predicted: set[tuple[str, int, int]]) -> str:
    """Give each missed gold entity one diagnostic category, in priority order."""
    label, start, end = gold
    if any(p_start == start and p_end == end and p_label != label
           for p_label, p_start, p_end in predicted):
        return "type_error"
    if any(p_label == label and p_start < end and start < p_end
           for p_label, p_start, p_end in predicted):
        return "boundary_error"
    if any(p_start < end and start < p_end for _, p_start, p_end in predicted):
        return "boundary_and_type"
    return "not_detected"


def summarize(rows: list[tuple[str, set[tuple[str, int, int]],
                               set[tuple[str, int, int]]]]) -> dict:
    by_type = Counter()
    by_length = Counter()
    errors = Counter()
    false_positives = Counter()
    for text, predicted, gold in rows:
        if any(not 0 <= start < end <= len(text) for _, start, end in predicted | gold):
            raise ValueError("entity offset is outside its source text")
        for label, start, end in gold:
            by_type[(label, "gold")] += 1
            by_length[(length_bucket(end - start), "gold")] += 1
            if (label, start, end) in predicted:
                by_type[(label, "correct")] += 1
                by_length[(length_bucket(end - start), "correct")] += 1
            else:
                errors[(label, classify_miss((label, start, end), predicted))] += 1
        for label, start, end in predicted - gold:
            false_positives[label] += 1
    total_gold = sum(count for (label, key), count in by_type.items() if key == "gold")
    total_correct = sum(count for (label, key), count in by_type.items() if key == "correct")
    total_predicted = total_correct + sum(false_positives.values())
    def scored(counts: Counter, names: tuple[str, ...]) -> dict:
        return {name: {"gold": counts[(name, "gold")],
                       "correct": counts[(name, "correct")],
                       "recall": (counts[(name, "correct")] / counts[(name, "gold")]
                                  if counts[(name, "gold")] else None)}
                for name in names}
    type_scores = {}
    for label in ("PER", "ORG", "LOC"):
        correct = by_type[(label, "correct")]
        gold = by_type[(label, "gold")]
        predicted = correct + false_positives[label]
        type_scores[label] = {"gold": gold, "predicted": predicted, "correct": correct,
                              "precision": correct / predicted if predicted else 0.0,
                              "recall": correct / gold if gold else 0.0,
                              "f1": 2 * correct / (predicted + gold)
                              if predicted + gold else 0.0}
    return {"sentences": len(rows),
            "overall": {"gold": total_gold, "predicted": total_predicted,
                        "correct": total_correct,
                        "precision": total_correct / total_predicted if total_predicted else 0.0,
                        "recall": total_correct / total_gold if total_gold else 0.0,
                        "f1": (2 * total_correct / (total_predicted + total_gold)
                               if total_predicted + total_gold else 0.0)},
            "by_type": type_scores,
            "by_length": scored(by_length, tuple(bucket[2] for bucket in LENGTH_BUCKETS)),
            "missed_gold_categories": {
                label: {category: errors[(label, category)] for category in
                        ("type_error", "boundary_error", "boundary_and_type", "not_detected")}
                for label in ("PER", "ORG", "LOC")},
            "false_positives_by_type": dict(sorted(false_positives.items()))}


def analyze(directory: Path, output: Path, threads: int) -> dict:
    torch.set_num_threads(threads)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    checkpoint = directory / "best.pt"
    source = ensure_file("train.tsv")
    if md5(source) != run["train_md5"]:
        raise ValueError("training data checksum differs from checkpoint metadata")
    by_line = {row["line"]: row for row in read_msra(source, run["max_chars"])}
    lines = run["dev_lines"]
    if len(set(lines)) != len(lines) or any(line not in by_line for line in lines):
        raise ValueError("saved development rows are missing or duplicated")
    if set(lines) & set(run["train_lines"]):
        raise ValueError("saved training and development rows overlap")
    model_dir = ensure_model(run["model_key"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    encoded = [encode(by_line[line], tokenizer, run["max_chars"]) for line in lines]
    loader = DataLoader(encoded, batch_size=16,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    model = BertBiLstmCrf(model_dir, hidden=run["hidden"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    observations = []
    with torch.inference_mode():
        for batch in loader:
            paths = model.crf.decode(model.emissions(batch), batch["mask"])
            for text, path, tags, mask in zip(batch["texts"], paths,
                                              batch["labels"], batch["mask"]):
                gold_tags = [TAGS[int(tag)] for tag in tags[:int(mask.sum())]]
                observations.append((text, entities(text, [TAGS[tag] for tag in path]),
                                     entities(text, gold_tags)))
    result = {"source": "fixed train.tsv development split, not MSRA test.tsv",
              "train_md5": md5(source), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "dev_lines_sha256": hashlib.sha256(json.dumps(lines).encode()).hexdigest(),
              **summarize(observations)}
    selected_dev = run.get("selected_dev")
    if selected_dev and abs(result["overall"]["f1"] - selected_dev["overall"]["f1"]) > 1e-10:
        raise ValueError("checkpoint F1 does not match saved selected_dev metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / ".cache" / "bert_bilstm_crf")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments" / "active_dev_errors.json")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    result = analyze(args.checkpoint_dir, args.output, args.threads)
    print(json.dumps(result, ensure_ascii=False, indent=2))
