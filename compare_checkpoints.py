"""Evaluate two frozen checkpoints on the same candidate held-out test rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_bilstm_crf import BertBiLstmCrf, MODEL_DIR, collate, encode, evaluate, read_msra
from data import ensure_file


def score(checkpoint_dir: Path, test_rows: list[dict], tokenizer, max_chars: int) -> dict:
    metadata = json.loads((checkpoint_dir / "run.json").read_text(encoding="utf-8"))
    if metadata["max_chars"] != max_chars:
        raise ValueError("checkpoints use different maximum text lengths")
    model = BertBiLstmCrf(MODEL_DIR, hidden=metadata["hidden"])
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True))
    rows = [encode(row, tokenizer, max_chars) for row in test_rows]
    loader = DataLoader(rows, batch_size=8,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    return evaluate(model, loader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    candidate = json.loads((args.candidate_dir / "run.json").read_text(encoding="utf-8"))
    selected = set(candidate["test_lines"])
    test_rows = [row for row in read_msra(ensure_file("test.tsv"), candidate["max_chars"])
                 if row["line"] in selected]
    if len(test_rows) != len(selected):
        raise ValueError("candidate test rows are missing")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    result = {"test_lines": sorted(selected),
              "baseline": score(args.baseline_dir, test_rows, tokenizer, candidate["max_chars"]),
              "candidate": score(args.candidate_dir, test_rows, tokenizer, candidate["max_chars"])}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("baseline", "candidate"):
        print(f"{name} F1: {result[name]['overall']['f1']:.2%}")


if __name__ == "__main__":
    main()
