"""Evaluate interpolations of adjacent Mini checkpoints without test-set tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_bilstm_crf import (BertBiLstmCrf, collate, encode, evaluate,
                              read_msra)
from data import ensure_file
from download_model import ensure_model


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / ".cache" / "mini_previous"
CURRENT = ROOT / ".cache" / "mini_before_blend"
DEV_RESULT = ROOT / ".cache" / "blend_dev_sweep.json"
TEST_RESULT = ROOT / ".cache" / "blend_fresh_test.json"
PUBLISHED_TEST = ROOT / "experiments" / "blend_fresh_test.json"
CANDIDATE = ROOT / ".cache" / "mini_blend_075"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TEST_SEED = 2031
TEST_SIZE = 500


def metadata(path: Path) -> dict:
    return json.loads((path / "run.json").read_text(encoding="utf-8"))


def load_sources() -> tuple[dict, dict, dict, dict]:
    previous, current = metadata(PREVIOUS), metadata(CURRENT)
    for key in ("model_key", "hidden", "max_chars", "train_lines", "dev_lines"):
        if previous[key] != current[key]:
            raise ValueError(f"checkpoints do not match on {key}")
    old_state = torch.load(PREVIOUS / "best.pt", map_location="cpu", weights_only=True)
    new_state = torch.load(CURRENT / "best.pt", map_location="cpu", weights_only=True)
    if old_state.keys() != new_state.keys():
        raise ValueError("checkpoint parameters differ")
    return previous, current, old_state, new_state


def blend(old_state: dict, new_state: dict, alpha: float) -> dict:
    result = {}
    for key, old in old_state.items():
        new = new_state[key]
        if old.shape != new.shape:
            raise ValueError(f"incompatible tensor {key}")
        if old.is_floating_point():
            result[key] = torch.lerp(old, new, alpha)
        else:
            if not torch.equal(old, new):
                raise ValueError(f"non-floating buffer differs: {key}")
            result[key] = new
    return result


def loader(rows: list[dict], tokenizer, max_chars: int) -> DataLoader:
    encoded = [encode(row, tokenizer, max_chars) for row in rows]
    return DataLoader(encoded, batch_size=16,
                      collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))


def build_model(current: dict):
    model_dir = ensure_model(current["model_key"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = BertBiLstmCrf(model_dir, hidden=current["hidden"])
    return model, tokenizer


def dev_sweep(current: dict, old_state: dict, new_state: dict) -> None:
    by_line = {row["line"]: row for row in read_msra(ensure_file("train.tsv"),
                                                    current["max_chars"])}
    dev_lines = current["dev_lines"]
    if len(set(dev_lines)) != len(dev_lines) or any(line not in by_line for line in dev_lines):
        raise ValueError("development lines are missing or duplicated")
    model, tokenizer = build_model(current)
    dev_loader = loader([by_line[line] for line in dev_lines], tokenizer,
                        current["max_chars"])
    results = []
    for alpha in ALPHAS:
        model.load_state_dict(blend(old_state, new_state, alpha))
        metrics = evaluate(model, dev_loader)
        results.append({"alpha_current": alpha, "metrics": metrics})
        print(f"alpha={alpha:.2f} dev entity F1={metrics['overall']['f1']:.4%}", flush=True)
    chosen = max(results, key=lambda row: (row["metrics"]["overall"]["f1"],
                                           row["alpha_current"]))
    output = {"note": "Development sweep; test data not scored", "alphas": results,
              "chosen_alpha_current": chosen["alpha_current"],
              "dev_lines_sha256": hashlib.sha256(json.dumps(dev_lines).encode()).hexdigest(),
              "previous_checkpoint_sha256": hashlib.sha256((PREVIOUS / "best.pt").read_bytes()).hexdigest(),
              "current_checkpoint_sha256": hashlib.sha256((CURRENT / "best.pt").read_bytes()).hexdigest()}
    DEV_RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected alpha={chosen['alpha_current']:.2f}; saved {DEV_RESULT}")


def fresh_test(current: dict, old_state: dict, new_state: dict) -> None:
    if PUBLISHED_TEST.exists():
        raise ValueError("fresh test has already been published; use verify to replay it")
    sweep = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
    if sweep["previous_checkpoint_sha256"] != hashlib.sha256((PREVIOUS / "best.pt").read_bytes()).hexdigest() or \
            sweep["current_checkpoint_sha256"] != hashlib.sha256((CURRENT / "best.pt").read_bytes()).hexdigest():
        raise ValueError("checkpoints changed after development selection")
    alpha = sweep["chosen_alpha_current"]
    current_dev = next(row["metrics"]["overall"]["f1"] for row in sweep["alphas"]
                       if row["alpha_current"] == 1.0)
    chosen_dev = next(row["metrics"]["overall"]["f1"] for row in sweep["alphas"]
                      if row["alpha_current"] == alpha)
    if alpha == 1.0 or chosen_dev <= current_dev:
        raise ValueError("no development gain; fresh test will not be used")
    used = set()
    for path in (ROOT / "experiments").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        used.update(record.get("test_lines", []))
    available = [row for row in read_msra(ensure_file("test.tsv"), current["max_chars"])
                 if row["line"] not in used]
    if len(available) < TEST_SIZE:
        raise ValueError(f"only {len(available)} previously unused test rows remain")
    selected = random.Random(TEST_SEED).sample(available, TEST_SIZE)
    selected_lines = [row["line"] for row in selected]
    model, tokenizer = build_model(current)
    test_loader = loader(selected, tokenizer, current["max_chars"])
    scores = {}
    for name, weight in (("current", 1.0), ("candidate", alpha)):
        model.load_state_dict(blend(old_state, new_state, weight))
        scores[name] = evaluate(model, test_loader)
        print(f"{name} fresh-test entity F1={scores[name]['overall']['f1']:.4%}", flush=True)
    output = {"note": "One fresh test comparison after development selection",
              "chosen_alpha_current": alpha, "current_dev_f1": current_dev,
              "candidate_dev_f1": chosen_dev, "test_seed": TEST_SEED,
              "excluded_prior_test_lines": len(used), "test_lines": selected_lines,
              "previous_checkpoint_sha256": sweep["previous_checkpoint_sha256"],
              "current_checkpoint_sha256": sweep["current_checkpoint_sha256"],
              **scores}
    TEST_RESULT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {TEST_RESULT}")


def verify_published_test(current: dict, old_state: dict, new_state: dict) -> None:
    """Replay the published comparison without drawing or selecting test rows."""
    report = json.loads(PUBLISHED_TEST.read_text(encoding="utf-8"))
    for source, key in ((PREVIOUS, "previous_checkpoint_sha256"),
                        (CURRENT, "current_checkpoint_sha256")):
        if hashlib.sha256((source / "best.pt").read_bytes()).hexdigest() != report[key]:
            raise ValueError(f"source checkpoint changed: {source}")
    selected = report["test_lines"]
    by_line = {row["line"]: row for row in read_msra(ensure_file("test.tsv"),
                                                    current["max_chars"])}
    if len(set(selected)) != len(selected) or any(line not in by_line for line in selected):
        raise ValueError("published test lines are missing or duplicated")
    model, tokenizer = build_model(current)
    test_loader = loader([by_line[line] for line in selected], tokenizer,
                         current["max_chars"])
    for name, weight in (("current", 1.0),
                         ("candidate", report["chosen_alpha_current"])):
        model.load_state_dict(blend(old_state, new_state, weight))
        observed = evaluate(model, test_loader)
        if observed != report[name]:
            raise ValueError(f"published {name} result does not reproduce")
        print(f"verified {name} entity F1={observed['overall']['f1']:.4%}")


def materialize(current: dict, old_state: dict, new_state: dict) -> None:
    """Write the dev-selected, fresh-test-verified blend in a separate directory."""
    sweep = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
    comparison_path = TEST_RESULT if TEST_RESULT.exists() else PUBLISHED_TEST
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    alpha = sweep["chosen_alpha_current"]
    if alpha != comparison["chosen_alpha_current"] or alpha == 1.0:
        raise ValueError("development and test selections disagree")
    for source, key in ((PREVIOUS, "previous_checkpoint_sha256"),
                        (CURRENT, "current_checkpoint_sha256")):
        digest = hashlib.sha256((source / "best.pt").read_bytes()).hexdigest()
        if sweep[key] != digest or comparison[key] != digest:
            raise ValueError(f"source checkpoint changed: {source}")
    if comparison["candidate"]["overall"]["f1"] <= comparison["current"]["overall"]["f1"]:
        raise ValueError("candidate did not improve on the fresh matched test")
    if CANDIDATE.exists():
        raise FileExistsError(f"candidate already exists: {CANDIDATE}")
    CANDIDATE.mkdir(parents=True)
    candidate_state = blend(old_state, new_state, alpha)
    torch.save(candidate_state, CANDIDATE / "best.pt")
    verified = torch.load(CANDIDATE / "best.pt", map_location="cpu", weights_only=True)
    if any(not torch.equal(value, verified[key]) for key, value in candidate_state.items()):
        raise ValueError("saved candidate weights differ from selected blend")
    record = dict(current)
    record["checkpoint_method"] = "linear_weight_interpolation"
    record["blend_alpha_current"] = alpha
    record["blend_previous_checkpoint_sha256"] = sweep["previous_checkpoint_sha256"]
    record["blend_current_checkpoint_sha256"] = sweep["current_checkpoint_sha256"]
    record["source_test"] = current["test"]
    record["source_test_lines"] = current["test_lines"]
    record["source_training_history"] = record.pop("history", [])
    record["selected_dev"] = next(row["metrics"] for row in sweep["alphas"]
                                  if row["alpha_current"] == alpha)
    record["test"] = comparison["candidate"]
    record["test_lines"] = comparison["test_lines"]
    record["test_size"] = len(comparison["test_lines"])
    record["matched_baseline_f1"] = comparison["current"]["overall"]["f1"]
    record["test_seed"] = comparison["test_seed"]
    record["excluded_prior_test_lines"] = comparison["excluded_prior_test_lines"]
    (CANDIDATE / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"materialized {CANDIDATE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dev", "test", "verify", "materialize"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    _, current, old_state, new_state = load_sources()
    if args.mode == "dev":
        dev_sweep(current, old_state, new_state)
    elif args.mode == "test":
        fresh_test(current, old_state, new_state)
    elif args.mode == "verify":
        verify_published_test(current, old_state, new_state)
    else:
        materialize(current, old_state, new_state)


if __name__ == "__main__":
    main()
