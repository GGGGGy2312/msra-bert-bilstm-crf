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
BASELINE_COMPONENTS = None
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


def load_components(directory: Path) -> tuple:
    checkpoint = directory / "best.pt"
    metadata_file = directory / "run.json"
    if not checkpoint.exists() or not metadata_file.exists():
        raise ValueError(f"本机缺少模型权重：{directory.name}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    model_dir = ensure_model(metadata.get("model_key", "tiny"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = BertBiLstmCrf(model_dir, hidden=metadata["hidden"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model, tokenizer, metadata


def predict_with_components(text: str, components: tuple,
                            recover_boundaries: bool = True) -> list[dict]:
    model, tokenizer, metadata = components
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


def predict_rows(text: str, recover_boundaries: bool = True) -> list[dict]:
    global COMPONENTS
    with LOCK:
        if COMPONENTS is None:
            COMPONENTS = load_components(OUTPUT_DIR)
        return predict_with_components(text, COMPONENTS, recover_boundaries)


def compare_rows(text: str) -> dict:
    global COMPONENTS, BASELINE_COMPONENTS
    with LOCK:
        if COMPONENTS is None:
            COMPONENTS = load_components(OUTPUT_DIR)
        baseline_name = ("mini_before_low_lr" if COMPONENTS[2].get("checkpoint_method") ==
                         "low_lr_subset_resume" else "mini_before_blend")
        baseline_dir = ROOT / ".cache" / baseline_name
        if BASELINE_COMPONENTS is None:
            BASELINE_COMPONENTS = load_components(baseline_dir)
        current_metadata = COMPONENTS[2]
        baseline_metadata = BASELINE_COMPONENTS[2]
        for key in ("model_key", "hidden", "max_chars", "dev_lines"):
            if current_metadata[key] != baseline_metadata[key]:
                raise ValueError(f"对照权重不兼容：{key}")
        if current_metadata.get("checkpoint_method") == "low_lr_subset_resume":
            compatible_train = set(current_metadata["train_lines"]).issubset(
                baseline_metadata["train_lines"])
        else:
            compatible_train = current_metadata["train_lines"] == baseline_metadata["train_lines"]
        if not compatible_train:
            raise ValueError("对照权重不兼容：train_lines")
        current = predict_with_components(text, COMPONENTS)
        baseline = predict_with_components(text, BASELINE_COMPONENTS)
    key = lambda row: (row["label"], row["start"], row["end"])
    current_keys = {key(row) for row in current}
    baseline_keys = {key(row) for row in baseline}
    return {"text": text, "current_count": len(current), "baseline_count": len(baseline),
            "common": len(current_keys & baseline_keys),
            "current_only": [row for row in current if key(row) not in baseline_keys],
            "baseline_only": [row for row in baseline if key(row) not in current_keys]}


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
            low_lr = run.get("checkpoint_method") == "low_lr_subset_resume"
            baseline_name = "mini_before_low_lr" if low_lr else "mini_before_blend"
            self.send_json(200, {"trained": True,
                                 "model": run.get("model_key", "tiny"),
                                 "checkpoint_method": run.get("checkpoint_method"),
                                 "blend_alpha_current": run.get("blend_alpha_current"),
                                 "train_size": run["train_size"],
                                 "dev_size": run["dev_size"], "test_size": run["test_size"],
                                 "f1": run["test"]["overall"]["f1"],
                                 "matched_baseline_f1": run.get("matched_baseline_f1"),
                                 "source_train_size": run.get("source_train_size"),
                                 "fine_tune_size": run.get("fine_tune_size"),
                                 "evaluation_source": run.get("evaluation_source"),
                                 "holdout_f1": run.get("holdout_f1"),
                                 "holdout_baseline_f1": run.get("holdout_baseline_f1"),
                                 "comparison_available":
                                 (ROOT / ".cache" / baseline_name / "best.pt").exists()})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path not in ("/api/ner", "/api/compare"):
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 32000:
                raise ValueError("请求过长或为空")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是 JSON 对象")
            text = payload.get("text", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("请输入中文文本")
            if self.path == "/api/compare":
                self.send_json(200, compare_rows(text))
            else:
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
