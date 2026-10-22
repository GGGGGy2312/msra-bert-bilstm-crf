"""Paired sentence bootstrap for the 500-row low-LR holdout comparison."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_bilstm_crf import BertBiLstmCrf, TAGS, collate, encode, entities, read_msra
from data import ensure_file
from download_model import ensure_model


ROOT = Path(__file__).resolve().parent
SEED = 2051
RESAMPLES = 5000


def counts_for_model(model_dir: Path, checkpoint: Path, run: dict,
                     loader: DataLoader) -> list[tuple[int, int, int]]:
    model = BertBiLstmCrf(model_dir, hidden=run["hidden"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    counts = []
    with torch.no_grad():
        for batch in loader:
            paths = model.crf.decode(model.emissions(batch), batch["mask"])
            for text, path, gold_ids in zip(batch["texts"], paths, batch["labels"]):
                predicted = entities(text, [TAGS[i] for i in path])
                gold = entities(text, [TAGS[int(i)] for i in gold_ids[:len(text)]])
                counts.append((len(predicted & gold), len(predicted - gold),
                               len(gold - predicted)))
    return counts


def f1(counts: list[tuple[int, int, int]], indices: list[int]) -> float:
    correct = sum(counts[i][0] for i in indices)
    false_positive = sum(counts[i][1] for i in indices)
    missed = sum(counts[i][2] for i in indices)
    return 2 * correct / (2 * correct + false_positive + missed)


def main() -> None:
    torch.set_num_threads(8)
    published = json.loads((ROOT / "experiments" / "low_lr_holdout.json").read_text(encoding="utf-8"))
    run = json.loads((ROOT / ".cache" / "mini_before_low_lr" / "run.json").read_text(encoding="utf-8"))
    by_line = {row["line"]: row for row in read_msra(ensure_file("train.tsv"), run["max_chars"])}
    rows = [by_line[line] for line in published["test_lines"]]
    model_dir = ensure_model(run["model_key"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    loader = DataLoader([encode(row, tokenizer, run["max_chars"]) for row in rows],
                        batch_size=16,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    baseline = counts_for_model(model_dir, ROOT / ".cache" / "mini_before_low_lr" / "best.pt",
                                run, loader)
    candidate = counts_for_model(model_dir, ROOT / ".cache" / "mini_low_lr_10k" / "best.pt",
                                 run, loader)
    all_rows = list(range(len(rows)))
    observed = f1(candidate, all_rows) - f1(baseline, all_rows)
    if abs(observed - (published["results"]["candidate"]["overall"]["f1"] -
                       published["results"]["baseline"]["overall"]["f1"])) > 1e-12:
        raise ValueError("bootstrap input does not match published overall F1")
    rng = random.Random(SEED)
    differences = []
    for _ in range(RESAMPLES):
        chosen = [rng.randrange(len(rows)) for _ in rows]
        differences.append(f1(candidate, chosen) - f1(baseline, chosen))
    differences.sort()
    result = {"method": "paired_sentence_bootstrap_percentile", "seed": SEED,
              "resamples": RESAMPLES, "sample_size": len(rows),
              "observed_difference_percentage_points": observed * 100,
              "interval_95_percentage_points":
                  [differences[int(0.025 * RESAMPLES)] * 100,
                   differences[int(0.975 * RESAMPLES) - 1] * 100]}
    target = ROOT / "experiments" / "low_lr_bootstrap.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
