"""Check whether BIO-valid Viterbi paths improve the fixed development set."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import MethodType

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_bilstm_crf import BertBiLstmCrf, TAGS, collate, encode, evaluate, read_msra
from data import ensure_file
from download_model import ensure_model

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / ".cache" / "bert_bilstm_crf"


def constrained_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
    start = self.start.detach().clone()
    transition = self.transition.detach().clone()
    for target, tag in enumerate(TAGS):
        if tag.startswith("I-"):
            start[target] = -10000.0
        for source, previous in enumerate(TAGS):
            if tag.startswith("I-") and previous not in ("B-" + tag[2:], "I-" + tag[2:]):
                transition[source, target] = -10000.0
    output = []
    for scores, valid in zip(emissions, mask):
        length = int(valid.sum())
        state = start + scores[0]
        pointers = []
        for step in range(1, length):
            best, pointer = (state[:, None] + transition).max(dim=0)
            state = best + scores[step]
            pointers.append(pointer)
        path = [int((state + self.end).argmax())]
        for pointer in reversed(pointers):
            path.append(int(pointer[path[-1]]))
        output.append(path[::-1])
    return output


def main() -> None:
    torch.set_num_threads(8)
    run = json.loads((RUN_DIR / "run.json").read_text(encoding="utf-8"))
    model_dir = ensure_model(run["model_key"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    by_line = {row["line"]: row for row in read_msra(ensure_file("train.tsv"), run["max_chars"])}
    encoded = [encode(by_line[line], tokenizer, run["max_chars"]) for line in run["dev_lines"]]
    loader = DataLoader(encoded, batch_size=16,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    model = BertBiLstmCrf(model_dir, hidden=run["hidden"])
    model.load_state_dict(torch.load(RUN_DIR / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    regular = evaluate(model, loader)
    model.crf.decode = MethodType(constrained_decode, model.crf)
    constrained = evaluate(model, loader)
    output = {"checkpoint_sha256": hashlib.sha256((RUN_DIR / "best.pt").read_bytes()).hexdigest(),
              "regular": regular, "bio_constrained": constrained,
              "dev_lines": run["dev_lines"]}
    target = ROOT / "experiments" / "bio_decode_dev.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"regular dev entity F1={regular['overall']['f1']:.4%}")
    print(f"BIO-constrained dev entity F1={constrained['overall']['f1']:.4%}")
    print(f"saved {target}")


if __name__ == "__main__":
    main()
