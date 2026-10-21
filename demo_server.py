"""Dependency-free local web UI for the trained MSRA BERT-BiLSTM-CRF model."""

from __future__ import annotations

import json
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


def predict_rows(text: str) -> list[dict]:
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
        if len(text) > metadata["max_chars"]:
            raise ValueError(f"最多输入 {metadata['max_chars']} 字")
        batch = collate([encode({"text": text}, tokenizer, metadata["max_chars"])],
                        tokenizer.pad_token_id)
        with torch.no_grad():
            path = model.crf.decode(model.emissions(batch), batch["mask"])[0]
        return [{"text": text[start:end], "label": LABEL_ZH[kind],
                 "start": start, "end": end}
                for kind, start, end in sorted(entities(text, [TAGS[i] for i in path]),
                                               key=lambda item: item[1])]


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
                                 "f1": run["test"]["overall"]["f1"]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/ner":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 1 <= size <= 2048:
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
