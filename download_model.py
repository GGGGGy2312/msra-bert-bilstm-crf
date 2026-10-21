"""Fetch the pinned UER Chinese BERT-architecture checkpoint."""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / ".cache" / "uer-bert-tiny-chinese"
MODEL_ID = "uer/chinese_roberta_L-2_H-128"
REVISION = "374ab7ade2513a8ecfeada0348ae8bb59d52ce1c"
MODELS = {
    "tiny": {"id": MODEL_ID, "revision": REVISION, "dir": MODEL_DIR},
    "mini": {"id": "uer/chinese_roberta_L-4_H-256",
             "revision": "68d674689f3c9d425e5436149c8acece077187f1",
             "dir": ROOT / ".cache" / "uer-bert-mini-chinese"},
}


def ensure_model(key: str = "tiny") -> Path:
    spec = MODELS[key]
    model_dir = spec["dir"]
    if (model_dir / "pytorch_model.bin").exists() and (model_dir / "config.json").exists():
        return model_dir
    snapshot_download(repo_id=spec["id"], revision=spec["revision"], local_dir=model_dir,
                      allow_patterns=["config.json", "vocab.txt", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json",
                                      "pytorch_model.bin"])
    if not (model_dir / "pytorch_model.bin").exists():
        raise FileNotFoundError("UER Chinese BERT-architecture weights were not downloaded")
    return model_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="tiny")
    print(ensure_model(parser.parse_args().model))
