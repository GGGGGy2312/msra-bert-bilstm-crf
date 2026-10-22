"""Compare long-text boundary recovery on constructed passages from dev rows.

This is a diagnostic for segmentation only, not an independent MSRA test score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bert_bilstm_crf import OUTPUT_DIR, entities, read_msra
from data import ensure_file
from demo_server import boundary_windows, predict_rows, segment_text


FILLER = "研究团队持续开展项目工作"
ZH_TO_TAG = {"人名": "PER", "机构名": "ORG", "地名": "LOC"}


def score(predicted: set[tuple[str, int, int]],
          gold: set[tuple[str, int, int]]) -> dict[str, int]:
    return {"correct": len(predicted & gold), "predicted": len(predicted),
            "gold": len(gold)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output", type=Path,
                        default=OUTPUT_DIR.parent / "boundary_recovery_dev.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = json.loads((OUTPUT_DIR / "run.json").read_text(encoding="utf-8"))
    dev_lines = set(run["dev_lines"])
    dev_rows = [row for row in read_msra(ensure_file("train.tsv"), run["max_chars"])
                if row["line"] in dev_lines]
    totals = {name: {"correct": 0, "predicted": 0, "gold": 0, "target_hits": 0}
              for name in ("baseline", "recovered")}
    cases = []
    for row in dev_rows:
        spans = sorted(entities(row["text"], row["tags"]), key=lambda item: item[1])
        choices = [span for span in spans if span[2] - span[1] >= 2 and span[1] < 90]
        if not choices:
            continue
        target = choices[0]
        padding = run["max_chars"] - target[1] - 1
        if padding < 0:
            continue
        prefix = (FILLER * ((padding // len(FILLER)) + 1))[:padding]
        text = prefix + row["text"]
        target_global = (target[0], target[1] + padding, target[2] + padding)
        windows = boundary_windows(text, segment_text(text, run["max_chars"]),
                                   run["max_chars"])
        if not any(target_global[1] < boundary < target_global[2]
                   for boundary, _, _ in windows):
            continue
        gold = {(kind, start + padding, end + padding) for kind, start, end in spans}
        case = {"dev_line": row["line"], "target": target_global,
                "text": text, "gold": sorted(gold)}
        for name, recovery in (("baseline", False), ("recovered", True)):
            predicted = {(item["label"], item["start"], item["end"])
                         for item in predict_rows(text, recover_boundaries=recovery)
                         if item["start"] >= padding}
            predicted = {(ZH_TO_TAG[kind], start, end)
                         for kind, start, end in predicted}
            metrics = score(predicted, gold)
            metrics["target_hits"] = int(target_global in predicted)
            case[name] = {**metrics, "predictions": sorted(predicted)}
            for key in totals[name]:
                totals[name][key] += metrics[key]
        cases.append(case)
        print(f"case {len(cases)}/{args.limit}: line {row['line']} "
              f"target baseline={case['baseline']['target_hits']} "
              f"recovered={case['recovered']['target_hits']}", flush=True)
        if len(cases) >= args.limit:
            break
    if not cases:
        raise ValueError("no eligible development rows crossed a hard boundary")
    result = {"note": "Constructed hard-boundary passages from development rows; not an independent test score",
              "model": run["model_key"], "max_chars": run["max_chars"],
              "cases": cases, "totals": totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
