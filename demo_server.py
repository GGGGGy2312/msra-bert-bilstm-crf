"""Dependency-free local web UI for the trained MSRA BERT-BiLSTM-CRF model."""

from __future__ import annotations

import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bert_bilstm_crf import (BertBiLstmCrf, LABEL_ZH, OUTPUT_DIR,
                              TAGS, collate, encode, entities)
from download_model import ensure_model


ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
COMPONENTS = None
MAX_INPUT_CHARS = 2000


def segment_text(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split non-whitespace runs while retaining offsets in the original text."""
    chunks = []
    for match in re.finditer(r"\S+", text):
        run = match.group()
        offset = match.start()
        while run:
            end = min(len(run), max_chars)
            if end < len(run):
                # Prefer sentence endings; use commas only when no ending is nearby.
                lower = max(1, end // 2)
                for marks in ("。！？!?；;", "，,、"):
                    cuts = [i + 1 for i, char in enumerate(run[:end])
                            if char in marks and i + 1 >= lower]
                    if cuts:
                        end = cuts[-1]
                        break
            chunks.append((offset, run[:end]))
            offset += end
            run = run[end:]
    return chunks


def boundary_windows(text: str, segments: list[tuple[int, str]],
                     max_chars: int) -> list[tuple[int, int, str]]:
    """Add context only where a hard cut interrupts a non-whitespace run."""
    windows = []
    punctuation = "。！？!?；;，,、"
    for (left_start, left), (right_start, _) in zip(segments, segments[1:]):
        boundary = left_start + len(left)
        if boundary != right_start or left[-1] in punctuation:
            continue
        run_start = left_start
        while run_start and not text[run_start - 1].isspace():
            run_start -= 1
        run_end = right_start
        while run_end < len(text) and not text[run_end].isspace():
            run_end += 1
        start = max(run_start, boundary - max_chars // 2)
        end = min(run_end, start + max_chars)
        start = max(run_start, end - max_chars)
        if start < boundary < end:
            windows.append((boundary, start, text[start:end]))
    return windows


def merge_boundary_rows(rows: list[dict], recovered: list[dict]) -> list[dict]:
    """Replace partial, overlapping spans only when a context window finds a crossing entity."""
    selected = []
    for candidate in sorted(recovered, key=lambda row: (row["start"], -row["end"])):
        if not any(candidate["start"] < row["end"] and row["start"] < candidate["end"]
                   for row in selected):
            selected.append(candidate)
    kept = [row for row in rows if not any(row["start"] < candidate["end"] and
                                             candidate["start"] < row["end"]
                                             for candidate in selected)]
    return sorted(kept + selected, key=lambda row: row["start"])


def decode_segments(text: str, segments: list[tuple[int, str]], model,
                    tokenizer, max_chars: int,
                    include_segment_index: bool = False) -> list[dict]:
    rows = []
    with torch.no_grad():
        for i in range(0, len(segments), 8):
            group = segments[i:i + 8]
            batch = collate([encode({"text": part}, tokenizer, max_chars)
                             for _, part in group], tokenizer.pad_token_id)
            paths = model.crf.decode(model.emissions(batch), batch["mask"])
            for j, ((offset, part), path) in enumerate(zip(group, paths)):
                for kind, start, end in entities(part, [TAGS[tag] for tag in path]):
                    row = {"text": text[offset + start:offset + end],
                           "label": LABEL_ZH[kind], "start": offset + start,
                           "end": offset + end}
                    if include_segment_index:
                        row["segment_index"] = i + j
                    rows.append(row)
    return rows


def predict_rows(text: str, recover_boundaries: bool = True) -> list[dict]:
    global COMPONENTS
    with LOCK:
        if COMPONENTS is None:
            checkpoint = OUTPUT_DIR / "best.pt"
            metadata_file = OUTPUT_DIR / "run.json"
            if not checkpoint.exists() or not metadata_file.exists():
                raise ValueError("尚未训练模型，请先运行 python bert_bilstm_crf.py train")
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            model_dir = ensure_model(metadata.get("model_key", "tiny"))
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
            model = BertBiLstmCrf(model_dir, hidden=metadata["hidden"])
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            COMPONENTS = model, tokenizer, metadata
        model, tokenizer, metadata = COMPONENTS
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"最多输入 {MAX_INPUT_CHARS} 字")
        if not text.strip():
            raise ValueError("请输入中文文本")
        segments = segment_text(text, metadata["max_chars"])
        rows = decode_segments(text, segments, model, tokenizer, metadata["max_chars"])
        if recover_boundaries:
            windows = boundary_windows(text, segments, metadata["max_chars"])
            contextual = decode_segments(text, [(start, part) for _, start, part in windows],
                                         model, tokenizer, metadata["max_chars"], True)
            recovered = []
            for row in contextual:
                boundary = windows[row.pop("segment_index")][0]
                if row["start"] < boundary < row["end"]:
                    recovered.append(row)
            rows = merge_boundary_rows(rows, recovered)
        return sorted(rows, key=lambda row: row["start"])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            content = (ROOT / "demo.html").read_bytes()
            self.send_bytes(200, content, "text/html; charset=utf-8")
        elif self.path == "/api/summary":
            path = OUTPUT_DIR / "run.json"
            if not path.exists():
                self.send_json(200, {"trained": False})
                return
            run = json.loads(path.read_text(encoding="utf-8"))
            self.send_json(200, {"trained": True,
                                 "model": run.get("model_key", "tiny"),
                                 "train_size": run["train_size"],
                                 "dev_size": run["dev_size"], "test_size": run["test_size"],
                                 "f1": run["test"]["overall"]["f1"],
                                 "matched_baseline_f1": run.get("matched_baseline_f1")})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/ner":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 32000:
                raise ValueError("请求过长或为空")
            payload = json.loads(self.rfile.read(size))
            text = payload.get("text", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("请输入中文文本")
            self.send_json(200, {"text": text, "entities": predict_rows(text)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(500, {"error": f"识别失败：{exc}"})

    def send_json(self, status: int, payload: dict) -> None:
        self.send_bytes(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8")

    def send_bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = "http://127.0.0.1:8766/"
    server = ThreadingHTTPServer(("127.0.0.1", 8766), Handler)
    print(f"Demo: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()
