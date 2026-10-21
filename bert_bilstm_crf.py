"""Train and run an independent MSRA NER model: Chinese BERT -> BiLSTM -> CRF."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from data import ensure_file, md5
from download_model import MODEL_DIR, MODEL_ID, REVISION, ensure_model


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / ".cache" / "bert_bilstm_crf"
TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
LABEL_ZH = {"PER": "人名", "ORG": "机构名", "LOC": "地名"}


def read_msra(path: Path, max_chars: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            columns = line.rstrip("\r\n").split("\t")
            if len(columns) != 2:
                raise ValueError(f"{path.name}:{number}: expected two TSV columns")
            chars, labels = columns[0].split("\x02"), columns[1].split("\x02")
            if len(chars) != len(labels) or any(len(c) != 1 for c in chars):
                raise ValueError(f"{path.name}:{number}: not character-aligned")
            if not (1 <= len(chars) <= max_chars):
                continue
            if any(tag not in TAG_TO_ID for tag in labels):
                raise ValueError(f"{path.name}:{number}: unknown BIO tag")
            rows.append({"text": "".join(chars), "tags": labels, "line": number})
    return rows


def sample_splits(train: list[dict], test: list[dict], train_size: int,
                  dev_size: int, test_size: int, seed: int,
                  excluded_test_lines: set[int] | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    test = [row for row in test if row["line"] not in (excluded_test_lines or set())]
    if train_size + dev_size > len(train) or test_size > len(test):
        raise ValueError("requested subset exceeds available data")
    rng = random.Random(seed)
    train_ids = rng.sample(range(len(train)), train_size + dev_size)
    test_ids = rng.sample(range(len(test)), test_size)
    return ([train[i] for i in train_ids[:train_size]],
            [train[i] for i in train_ids[train_size:]],
            [test[i] for i in test_ids])


def encode(row: dict, tokenizer, max_chars: int) -> dict:
    if len(row["text"]) > max_chars:
        raise ValueError("text exceeds --max-chars")
    encoded = tokenizer(list(row["text"]), is_split_into_words=True,
                        max_length=max_chars * 3 + 2, truncation=True)
    indices = []
    previous = None
    for index, word_id in enumerate(encoded.word_ids()):
        if word_id is not None and word_id != previous:
            indices.append(index)
        previous = word_id
    if len(indices) != len(row["text"]):
        raise ValueError(f"tokenizer alignment failed on line {row.get('line')}")
    return {"ids": encoded["input_ids"], "positions": indices,
            "labels": [TAG_TO_ID[t] for t in row.get("tags", ["O"] * len(indices))],
            "text": row["text"]}


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(row["ids"]) for row in batch)
    length = max(len(row["positions"]) for row in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    positions = torch.zeros((len(batch), length), dtype=torch.long)
    labels = torch.zeros((len(batch), length), dtype=torch.long)
    mask = torch.zeros((len(batch), length), dtype=torch.bool)
    for i, row in enumerate(batch):
        n, m = len(row["ids"]), len(row["positions"])
        ids[i, :n] = torch.tensor(row["ids"])
        attention[i, :n] = 1
        positions[i, :m] = torch.tensor(row["positions"])
        labels[i, :m] = torch.tensor(row["labels"])
        mask[i, :m] = True
    return {"ids": ids, "attention": attention, "positions": positions,
            "labels": labels, "mask": mask, "texts": [r["text"] for r in batch]}


class LinearCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        self.transition = nn.Parameter(torch.zeros(num_tags, num_tags))

    def loss(self, emissions: torch.Tensor, tags: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = emissions.shape
        first = tags[:, 0]
        gold = self.start[first] + emissions[:, 0].gather(1, first[:, None]).squeeze(1)
        alpha = self.start + emissions[:, 0]
        for step in range(1, length):
            current, previous = tags[:, step], tags[:, step - 1]
            emit = emissions[:, step].gather(1, current[:, None]).squeeze(1)
            gold += (self.transition[previous, current] + emit) * mask[:, step]
            candidate = torch.logsumexp(
                alpha[:, :, None] + self.transition[None, :, :] +
                emissions[:, step, None, :], dim=1)
            alpha = torch.where(mask[:, step, None], candidate, alpha)
        last = tags.gather(1, mask.long().sum(1)[:, None] - 1).squeeze(1)
        gold += self.end[last]
        partition = torch.logsumexp(alpha + self.end, dim=1)
        return (partition - gold).mean()

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        output = []
        for scores, valid in zip(emissions, mask):
            length = int(valid.sum())
            state = self.start + scores[0]
            pointers = []
            for step in range(1, length):
                candidates = state[:, None] + self.transition
                best, pointer = candidates.max(dim=0)
                state = best + scores[step]
                pointers.append(pointer)
            path = [int((state + self.end).argmax())]
            for pointer in reversed(pointers):
                path.append(int(pointer[path[-1]]))
            output.append(path[::-1])
        return output


class BertBiLstmCrf(nn.Module):
    def __init__(self, model_dir: Path, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_dir)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, len(TAGS))
        self.crf = LinearCRF(len(TAGS))

    def emissions(self, batch: dict) -> torch.Tensor:
        states = self.bert(input_ids=batch["ids"],
                           attention_mask=batch["attention"]).last_hidden_state
        gather = batch["positions"].unsqueeze(-1).expand(-1, -1, states.size(-1))
        character_states = states.gather(1, gather)
        lengths = batch["mask"].sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(character_states, lengths,
                                                   batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True,
                                                     total_length=character_states.size(1))
        return self.classifier(self.dropout(output))


def entities(text: str, tags: list[str]) -> set[tuple[str, int, int]]:
    found = set()
    kind, start = None, 0
    for i, tag in enumerate(tags + ["O"]):
        prefix, label = (tag.split("-", 1) if tag != "O" else ("O", None))
        if kind is not None and (prefix != "I" or label != kind):
            found.add((kind, start, i))
            kind = None
        if prefix == "B" or (prefix == "I" and kind is None):
            kind, start = label, i
    return found


@torch.no_grad()
def evaluate(model: BertBiLstmCrf, loader: DataLoader) -> dict:
    model.eval()
    tp = fp = fn = 0
    details = {name: [0, 0, 0] for name in LABEL_ZH}
    for batch in loader:
        decoded = model.crf.decode(model.emissions(batch), batch["mask"])
        for text, pred_ids, gold_ids, mask in zip(batch["texts"], decoded,
                                                  batch["labels"], batch["mask"]):
            gold_tags = [TAGS[int(x)] for x in gold_ids[:int(mask.sum())]]
            pred = entities(text, [TAGS[x] for x in pred_ids])
            gold = entities(text, gold_tags)
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
            for label in LABEL_ZH:
                p, g = {x for x in pred if x[0] == label}, {x for x in gold if x[0] == label}
                detail = details[label]
                detail[0] += len(p & g)
                detail[1] += len(p - g)
                detail[2] += len(g - p)
    def metrics(a, b, c):
        precision = a / (a + b) if a + b else 0.0
        recall = a / (a + c) if a + c else 0.0
        return {"correct": a, "false_positive": b, "missed": c,
                "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {"overall": metrics(tp, fp, fn),
            "per_class": {name: metrics(*values) for name, values in details.items()}}


def train(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    train_file = ensure_file("train.tsv")
    test_file = ensure_file("test.tsv")
    ensure_model()
    train_pool = read_msra(train_file, args.max_chars)
    test_pool = read_msra(test_file, args.max_chars)
    excluded = set()
    if args.exclude_test_lines:
        previous = json.loads(args.exclude_test_lines.read_text(encoding="utf-8"))
        excluded = set(previous["test_lines"])
    training, development, testing = sample_splits(
        train_pool, test_pool, args.train_size, args.dev_size, args.test_size,
        args.seed, excluded)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("fast tokenizer is required for word alignment")
    encode_rows = lambda rows: [encode(row, tokenizer, args.max_chars) for row in rows]
    train_rows, dev_rows, test_rows = map(encode_rows, (training, development, testing))
    make_loader = lambda rows, shuffle=False: DataLoader(
        rows, batch_size=args.batch_size, shuffle=shuffle,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    train_loader = make_loader(train_rows, True)
    dev_loader = make_loader(dev_rows)
    test_loader = make_loader(test_rows)
    model = BertBiLstmCrf(MODEL_DIR, hidden=args.hidden)
    if args.freeze_bert:
        for parameter in model.bert.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.bert.parameters() if p.requires_grad], "lr": args.bert_lr},
        {"params": list(model.lstm.parameters()) + list(model.classifier.parameters()) +
         list(model.crf.parameters()), "lr": args.head_lr},
    ], weight_decay=0.01)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            loss = model.crf.loss(model.emissions(batch), batch["labels"], batch["mask"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 20 == 0 or step == len(train_loader):
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss {sum(losses)/len(losses):.4f}", flush=True)
        dev = evaluate(model, dev_loader)
        f1 = dev["overall"]["f1"]
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev": dev})
        print(f"epoch {epoch} dev entity F1={f1:.3%}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best.pt", map_location="cpu", weights_only=True))
    test = evaluate(model, test_loader)
    metadata = {"architecture": "Chinese BERT -> BiLSTM -> linear -> CRF",
                "pretrained": MODEL_ID, "pretrained_revision": REVISION,
                "freeze_bert": args.freeze_bert,
                "seed": args.seed, "train_size": len(training), "dev_size": len(development),
                "test_size": len(testing), "max_chars": args.max_chars,
                "batch_size": args.batch_size, "epochs": args.epochs,
                "hidden": args.hidden, "bert_lr": args.bert_lr, "head_lr": args.head_lr,
                "train_md5": md5(train_file), "test_md5": md5(test_file),
                "train_lines": [r["line"] for r in training],
                "dev_lines": [r["line"] for r in development],
                "test_lines": [r["line"] for r in testing],
                "excluded_prior_test_lines": sorted(excluded),
                "history": history, "test": test}
    (OUTPUT_DIR / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# MSRA BERT-BiLSTM-CRF 小规模复现实验", "",
              "模型：UER 中文 RoBERTa-Tiny（BERT 结构，2 层、隐层 128）→ 双向 LSTM → 线性层 → CRF。",
              "数据：PaddleNLP 提供的 MSRA NER；仅从 train.tsv 划出训练集和开发集，test.tsv 只在模型选择完成后评估一次。",
              f"训练/开发/测试：{len(training)}/{len(development)}/{len(testing)} 句；长度 ≤ {args.max_chars} 字；种子 {args.seed}。",
              f"epoch={args.epochs}，batch={args.batch_size}，BERT 冻结={args.freeze_bert}。",
              "实体类别及字符起止位置完全一致才算正确。以下是本机小样本结果，不是完整数据集或论文复现分数。", "",
              "| 类别 | 正确 | 误报 | 漏报 | Precision | Recall | F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in [("总体", test["overall"])] + [(LABEL_ZH[k], v) for k, v in test["per_class"].items()]:
        report.append(f"| {name} | {values['correct']} | {values['false_positive']} | {values['missed']} | {values['precision']:.1%} | {values['recall']:.1%} | {values['f1']:.1%} |")
    report += ["", "## 各轮开发集", ""]
    report += [f"- Epoch {row['epoch']}：loss {row['train_loss']:.4f}，F1 {row['dev']['overall']['f1']:.1%}" for row in history]
    report += ["", "运行配置和抽样行号保存在 `.cache/bert_bilstm_crf/run.json`。", ""]
    (ROOT / "bert_bilstm_crf_report.md").write_text("\n".join(report), encoding="utf-8")
    print("TEST", json.dumps(test["overall"], ensure_ascii=False), flush=True)


def predict(args) -> None:
    torch.set_num_threads(args.threads)
    if not (OUTPUT_DIR / "best.pt").exists():
        raise FileNotFoundError("run the train command first")
    metadata = json.loads((OUTPUT_DIR / "run.json").read_text(encoding="utf-8"))
    ensure_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = BertBiLstmCrf(MODEL_DIR, hidden=metadata["hidden"])
    model.load_state_dict(torch.load(OUTPUT_DIR / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    row = encode({"text": args.text}, tokenizer, metadata["max_chars"])
    batch = collate([row], tokenizer.pad_token_id)
    with torch.no_grad():
        path = model.crf.decode(model.emissions(batch), batch["mask"])[0]
    found = sorted(entities(args.text, [TAGS[i] for i in path]), key=lambda x: x[1])
    print(json.dumps([{"text": args.text[start:end], "label": LABEL_ZH[kind],
                       "start": start, "end": end} for kind, start, end in found], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--text", default="张三在北京大学学习，后来去了上海。")
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--dev-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--bert-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exclude-test-lines", type=Path,
                        help="JSON run file whose earlier test lines must be excluded")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze-bert", action="store_true", help="faster CPU teaching run")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
