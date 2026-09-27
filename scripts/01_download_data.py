#!/usr/bin/env python3
"""
01_download_data.py
-------------------
Downloads the LibriSpeech test-clean and test-other archives from OpenSLR
(SLR12) and extracts them into data/raw/.

LibriSpeech is released under CC-BY-4.0, which allows redistribution and
commercial use with attribution: https://www.openslr.org/12/

We only need the two test splits:
  * test-clean  (~337 MB, 2,620 utterances / 5.4 h) - read speech, low noise
  * test-other  (~314 MB, 2,864 utterances / 5.1 h) - harder recordings:
                noisier channels, stronger accents, more disfluency

Usage:
    python scripts/01_download_data.py
"""

import os
import tarfile
import urllib.request

BASE = "https://www.openslr.org/resources/12"
FILES = ["test-clean.tar.gz", "test-other.tar.gz"]
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download(fname: str) -> str:
    dest = os.path.join(RAW_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
        print(f"[skip] {fname} already downloaded")
        return dest
    url = f"{BASE}/{fname}"
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[ok  ] {fname}  ({os.path.getsize(dest)/1e6:.1f} MB)")
    return dest


def extract(archive: str) -> None:
    with tarfile.open(archive) as tf:
        tf.extractall(RAW_DIR)
    print(f"[xtr ] {os.path.basename(archive)}")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    for f in FILES:
        extract(download(f))
    print("done.")
