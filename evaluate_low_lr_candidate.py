"""Evaluate the low-learning-rate candidate against the saved prior demo.

The holdout is drawn from train.tsv rows outside the original model's
training/development split. The other mode replays a previously published
MSRA test.tsv subset as a supplemental matched comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_bilstm_crf import BertBiLstmCrf, collate, encode, evaluate, read_msra
from data import ensure_file, md5
from download_model import ensure_model


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / ".cache" / "mini_before_low_lr"
CANDIDATE = ROOT / ".cache" / "mini_low_lr_10k"
REPORTS = ROOT / "experiments"


def report(mode: str) -> dict:
    torch.set_num_threads(8)
    original = json.loads((BASELINE / "run.json").read_text(encoding="utf-8"))
    candidate = json.loads((CANDIDATE / "run.json").read_text(encoding="utf-8"))
    if candidate["dev_lines"] != original["dev_lines"] or not set(
            candidate["train_lines"]).issubset(original["train_lines"]):
        raise ValueError("candidate does not preserve the original development split")
    if candidate["history"][-1]["dev"]["overall"]["f1"] <= \
            original["selected_dev"]["overall"]["f1"]:
        raise ValueError("candidate did not improve development F1")
    if mode == "holdout":
        source = ensure_file("train.tsv")
        excluded = set(original["train_lines"]) | set(original["dev_lines"])
        pool = [row for row in read_msra(source, original["max_chars"])
                if row["line"] not in excluded]
        prior_holdout = {row["line"] for row in random.Random(2042).sample(pool, 500)}
        rows = random.Random(2050).sample(
            [row for row in pool if row["line"] not in prior_holdout], 500)
        source_note = "train.tsv unused by original training/dev; excludes prior lexicon audit"
    else:
        source = ensure_file("test.tsv")
        published = json.loads((REPORTS / "blend_fresh_test.json").read_text(encoding="utf-8"))
        by_line = {row["line"]: row for row in read_msra(source, original["max_chars"])}
        rows = [by_line[line] for line in published["test_lines"]]
        source_note = "previously published MSRA test.tsv subset; supplemental reused test"
    model_dir = ensure_model(original["model_key"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    encoded = [encode(row, tokenizer, original["max_chars"]) for row in rows]
    loader = DataLoader(encoded, batch_size=16,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    results = {}
    for name, directory in (("baseline", BASELINE), ("candidate", CANDIDATE)):
        model = BertBiLstmCrf(model_dir, hidden=original["hidden"])
        model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=True))
        model.eval()
        results[name] = evaluate(model, loader)
        print(f"{name} {mode} entity F1={results[name]['overall']['f1']:.4%}", flush=True)
    output = {"source_note": source_note, "source_md5": md5(source),
              "test_lines": [row["line"] for row in rows],
              "baseline_checkpoint_sha256": hashlib.sha256((BASELINE / "best.pt").read_bytes()).hexdigest(),
              "candidate_checkpoint_sha256": hashlib.sha256((CANDIDATE / "best.pt").read_bytes()).hexdigest(),
              "dev_baseline_f1": original["selected_dev"]["overall"]["f1"],
              "dev_candidate_f1": candidate["history"][-1]["dev"]["overall"]["f1"],
              "results": results}
    target = REPORTS / f"low_lr_{mode}.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("holdout", "reused"))
    report(parser.parse_args().mode)
