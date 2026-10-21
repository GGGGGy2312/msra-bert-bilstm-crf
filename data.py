"""Download and verify the public MSRA NER copy distributed by PaddleNLP."""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / ".cache" / "msra"
URL = "https://bj.bcebos.com/paddlenlp/datasets/msra_ner.tar.gz"
ARCHIVE_MD5 = "f1aadbbf328ea2fa50c9c2b56db0d31e"
FILE_MD5 = {"train.tsv": "e5b4b734ef91861384f441456ad995dd",
            "test.tsv": "40b26ae09b63af78ea3a91ac8b8ae303"}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_file(filename: str) -> Path:
    if filename not in FILE_MD5:
        raise ValueError(f"unsupported dataset file: {filename}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / filename
    if target.exists() and md5(target) == FILE_MD5[filename]:
        return target
    archive = DATA_DIR / "msra_ner.tar.gz"
    if not archive.exists() or md5(archive) != ARCHIVE_MD5:
        urllib.request.urlretrieve(URL, archive)
    if md5(archive) != ARCHIVE_MD5:
        raise ValueError("MSRA archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as bundle:
        source = bundle.extractfile(f"msra_ner/{filename}")
        if source is None:
            raise ValueError(f"{filename} missing from archive")
        with target.open("wb") as dest:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                dest.write(chunk)
    if md5(target) != FILE_MD5[filename]:
        target.unlink(missing_ok=True)
        raise ValueError(f"{filename} checksum mismatch")
    return target
