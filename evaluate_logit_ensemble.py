"""Recheck a development-selected two-checkpoint logit ensemble.

This experiment does not replace the active single-checkpoint demo: the
published untouched test subset showed no overall gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_bilstm_crf import BertBiLstmCrf, collate, encode, evaluate, read_msra
from data import ensure_file
from download_model import ensure_model


ROOT = Path(__file__).resolve().parent
OLD = ROOT / ".cache" / "mini_previous" / "best.pt"
NEW = ROOT / ".cache" / "mini_before_blend" / "best.pt"
CURRENT = ROOT / ".cache" / "bert_bilstm_crf" / "best.pt"
RUN = ROOT / ".cache" / "bert_bilstm_crf" / "run.json"
DEV_REPORT = ROOT / "experiments" / "ensemble_dev_sweep.json"
TEST_REPORT = ROOT / "experiments" / "ensemble_fresh_test.json"
ALPHAS = (0.25, 0.5, 0.75)


class LogitEnsemble(nn.Module):
    def __init__(self, old: BertBiLstmCrf, new: BertBiLstmCrf,
                 current: BertBiLstmCrf, alpha: float):
        super().__init__()
        self.old = old
        self.new = new
        self.crf = current.crf
        self.alpha = alpha

    def emissions(self, batch: dict) -> torch.Tensor:
        return torch.lerp(self.old.emissions(batch), self.new.emissions(batch), self.alpha)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    torch.set_num_threads(8)
    run = json.loads(RUN.read_text(encoding="utf-8"))
    model_dir = ensure_model(run["model_key"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    def load(path: Path) -> BertBiLstmCrf:
        model = BertBiLstmCrf(model_dir, hidden=run["hidden"])
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model

    return run, tokenizer, load(OLD), load(NEW), load(CURRENT)


def make_loader(rows: list[dict], tokenizer, max_chars: int) -> DataLoader:
    encoded = [encode(row, tokenizer, max_chars) for row in rows]
    return DataLoader(encoded, batch_size=16,
                      collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))


def dev() -> None:
    run, tokenizer, old, new, current = prepare()
    pool = {row["line"]: row for row in read_msra(ensure_file("train.tsv"), run["max_chars"])}
    lines = run["dev_lines"]
    if len(set(lines)) != len(lines) or any(line not in pool for line in lines):
        raise ValueError("development rows missing or duplicated")
    loader = make_loader([pool[line] for line in lines], tokenizer, run["max_chars"])
    scores = []
    for alpha in ALPHAS:
        metrics = evaluate(LogitEnsemble(old, new, current, alpha), loader)
        scores.append({"alpha_new": alpha, "metrics": metrics})
        print(f"alpha={alpha:.2f} dev entity F1={metrics['overall']['f1']:.4%}", flush=True)
    chosen = max(scores, key=lambda row: (row["metrics"]["overall"]["f1"], row["alpha_new"]))
    report = {"method": "logit_ensemble_with_current_crf", "dev_lines": lines,
              "checkpoint_sha256": {"old": sha256(OLD), "new": sha256(NEW),
                                    "current": sha256(CURRENT)},
              "scores": scores, "chosen_alpha_new": chosen["alpha_new"]}
    DEV_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected alpha={chosen['alpha_new']:.2f}")


def verify() -> None:
    report = json.loads(TEST_REPORT.read_text(encoding="utf-8"))
    dev_report = json.loads(DEV_REPORT.read_text(encoding="utf-8"))
    if report["alpha_new"] != dev_report["chosen_alpha_new"]:
        raise ValueError("published test alpha does not match development selection")
    digests = {"old": sha256(OLD), "new": sha256(NEW), "current": sha256(CURRENT)}
    if digests != dev_report["checkpoint_sha256"] or digests != report["checkpoint_sha256"]:
        raise ValueError("checkpoint hashes changed")
    run, tokenizer, old, new, current = prepare()
    pool = {row["line"]: row for row in read_msra(ensure_file("test.tsv"), run["max_chars"])}
    lines = report["test_lines"]
    if len(set(lines)) != len(lines) or any(line not in pool for line in lines):
        raise ValueError("published test rows missing or duplicated")
    loader = make_loader([pool[line] for line in lines], tokenizer, run["max_chars"])
    for name, model in (("current", current),
                        ("ensemble_075", LogitEnsemble(old, new, current, report["alpha_new"]))):
        observed = evaluate(model, loader)
        if observed != report["results"][name]:
            raise ValueError(f"{name} published result does not reproduce")
        print(f"verified {name} entity F1={observed['overall']['f1']:.4%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dev", "verify"))
    args = parser.parse_args()
    (dev if args.mode == "dev" else verify)()
