"""Activate the dev-selected low-LR candidate after matched comparison gates."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / ".cache" / "mini_before_low_lr"
CANDIDATE = ROOT / ".cache" / "mini_low_lr_10k"
ACTIVE = ROOT / ".cache" / "bert_bilstm_crf"
EXPERIMENTS = ROOT / "experiments"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = json.loads((BASELINE / "run.json").read_text(encoding="utf-8"))
    new = json.loads((CANDIDATE / "run.json").read_text(encoding="utf-8"))
    holdout = json.loads((EXPERIMENTS / "low_lr_holdout.json").read_text(encoding="utf-8"))
    reused = json.loads((EXPERIMENTS / "low_lr_reused.json").read_text(encoding="utf-8"))
    old_hash = sha256(BASELINE / "best.pt")
    new_hash = sha256(CANDIDATE / "best.pt")
    for report in (holdout, reused):
        if (report["baseline_checkpoint_sha256"] != old_hash or
                report["candidate_checkpoint_sha256"] != new_hash):
            raise ValueError("evaluation report checkpoint hashes differ")
        if report["results"]["candidate"]["overall"]["f1"] <= \
                report["results"]["baseline"]["overall"]["f1"]:
            raise ValueError("candidate did not improve on both matched comparisons")
    if new["dev_lines"] != old["dev_lines"] or not set(new["train_lines"]).issubset(
            old["train_lines"]):
        raise ValueError("candidate training/development rows changed unexpectedly")
    if new["history"][-1]["dev"]["overall"]["f1"] <= old["selected_dev"]["overall"]["f1"]:
        raise ValueError("candidate development F1 did not improve")
    if sha256(ACTIVE / "best.pt") not in (old_hash, new_hash):
        raise ValueError("active checkpoint differs from both the backup and candidate")
    record = dict(new)
    record.update({
        "checkpoint_method": "low_lr_subset_resume",
        "init_checkpoint": str(BASELINE / "best.pt"),
        "source_train_size": old["train_size"],
        "fine_tune_size": new["train_size"],
        "source_checkpoint_sha256": old_hash,
        "candidate_checkpoint_sha256": new_hash,
        "selected_dev": new["history"][-1]["dev"],
        "evaluation_source": "previously_used_msra_test_subset",
        "evaluation_note": "Reused published MSRA test rows; supplemental comparison, not an untouched test",
        "test": reused["results"]["candidate"],
        "test_lines": reused["test_lines"],
        "test_size": len(reused["test_lines"]),
        "matched_baseline_f1": reused["results"]["baseline"]["overall"]["f1"],
        "holdout_source": holdout["source_note"],
        "holdout_size": len(holdout["test_lines"]),
        "holdout_f1": holdout["results"]["candidate"]["overall"]["f1"],
        "holdout_baseline_f1": holdout["results"]["baseline"]["overall"]["f1"],
    })
    metadata = json.dumps(record, ensure_ascii=False, indent=2)
    (EXPERIMENTS / "low_lr_run.json").write_text(metadata, encoding="utf-8")
    shutil.copy2(CANDIDATE / "best.pt", ACTIVE / "best.pt")
    temporary = ACTIVE / "run.json.tmp"
    temporary.write_text(metadata, encoding="utf-8")
    temporary.replace(ACTIVE / "run.json")
    if sha256(ACTIVE / "best.pt") != new_hash:
        raise ValueError("activated checkpoint was not copied correctly")
    print(f"activated {new_hash}; holdout F1={record['holdout_f1']:.4%}; "
          f"reused test F1={record['test']['overall']['f1']:.4%}")


if __name__ == "__main__":
    main()
