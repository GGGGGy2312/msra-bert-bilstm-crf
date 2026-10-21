"""Fetch the pinned UER Chinese BERT-architecture checkpoint."""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / ".cache" / "uer-bert-tiny-chinese"
MODEL_ID = "uer/chinese_roberta_L-2_H-128"
REVISION = "374ab7ade2513a8ecfeada0348ae8bb59d52ce1c"


def ensure_model() -> Path:
    if (MODEL_DIR / "pytorch_model.bin").exists() and (MODEL_DIR / "config.json").exists():
        return MODEL_DIR
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
                      allow_patterns=["config.json", "vocab.txt", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json",
                                      "pytorch_model.bin"])
    if not (MODEL_DIR / "pytorch_model.bin").exists():
        raise FileNotFoundError("UER Chinese BERT-architecture weights were not downloaded")
    return MODEL_DIR


if __name__ == "__main__":
    print(ensure_model())
