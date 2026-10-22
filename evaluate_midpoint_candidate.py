"""Compare the dev-selected step checkpoint on a reserved MSRA train holdout."""

from __future__ import annotations

import hashlib
import json
import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_bilstm_crf import BertBiLstmCrf, collate, encode, evaluate, read_msra
from data import ensure_file, md5
from download_model import ensure_model

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / ".cache" / "bert_bilstm_crf"
CANDIDATE = ROOT / ".cache" / "mini_low_lr_round3_midpoint"
ORIGINAL = ROOT / ".cache" / "mini_before_low_lr"
REPORT = ROOT / "experiments" / "midpoint_holdout.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(mode: str) -> None:
    torch.set_num_threads(8)
    original = json.loads((ORIGINAL / "run.json").read_text(encoding="utf-8"))
    baseline = json.loads((BASELINE / "run.json").read_text(encoding="utf-8"))
    candidate = json.loads((CANDIDATE / "run.json").read_text(encoding="utf-8"))
    for key in ("model_key", "hidden", "max_chars", "dev_lines"):
        if original[key] != baseline[key] or original[key] != candidate[key]:
            raise ValueError(f"split or architecture changed: {key}")
    if not set(baseline["train_lines"]).issubset(original["train_lines"]) or not set(
            candidate["train_lines"]).issubset(original["train_lines"]):
        raise ValueError("continued training left the original training pool")
    baseline_dev = baseline["selected_dev"]["overall"]["f1"]
    selected = max(candidate["history"], key=lambda row: row["dev"]["overall"]["f1"])
    candidate_dev = selected["dev"]["overall"]["f1"]
    if candidate["best_step"] != selected["step"] or candidate_dev <= baseline_dev:
        raise ValueError("candidate was not selected by a higher dev F1")
    if mode == "holdout":
        source = ensure_file("train.tsv")
        excluded = set(original["train_lines"]) | set(original["dev_lines"])
        pool = [row for row in read_msra(source, original["max_chars"])
                if row["line"] not in excluded]
        prior_lexicon = {row["line"] for row in random.Random(2042).sample(pool, 500)}
        after_lexicon = [row for row in pool if row["line"] not in prior_lexicon]
        prior_low_lr = {row["line"] for row in random.Random(2050).sample(after_lexicon, 500)}
        available = [row for row in after_lexicon if row["line"] not in prior_low_lr]
        if len(available) < 500:
            raise ValueError(f"only {len(available)} unused rows remain")
        rows = random.Random(2052).sample(available, 500)
        source_note = "MSRA train.tsv rows outside original train/dev and two earlier holdouts"
    else:
        source = ensure_file("test.tsv")
        prior = json.loads((ROOT / "experiments" / "blend_fresh_test.json").read_text(encoding="utf-8"))
        by_line = {row["line"]: row for row in read_msra(source, original["max_chars"])}
        rows = [by_line[line] for line in prior["test_lines"]]
        source_note = "reused previously published MSRA test.tsv subset; supplemental only"
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
    output = {"source": source_note,
              "source_md5": md5(source), "selection": "highest F1 on unchanged 1024-row dev split",
              "candidate_best_step": candidate["best_step"],
              "dev_baseline_f1": baseline_dev, "dev_candidate_f1": candidate_dev,
              "selection_seed": 2052 if mode == "holdout" else None,
              "test_lines": [row["line"] for row in rows],
              "baseline_checkpoint_sha256": sha256(BASELINE / "best.pt"),
              "candidate_checkpoint_sha256": sha256(CANDIDATE / "best.pt"),
              "results": results}
    target = REPORT if mode == "holdout" else ROOT / "experiments" / "midpoint_reused.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("holdout", "reused"), nargs="?", default="holdout")
    main(parser.parse_args().mode)
