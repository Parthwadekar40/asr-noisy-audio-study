#!/usr/bin/env python3
"""
02_prepare_audio.py
-------------------
Builds the evaluation set used for the benchmark.

We take utterances from two LibriSpeech splits:

  * test-clean : 5 speakers x 4 utterances  = 20 files   (primary track)
  * test-other : 4 speakers x 4 utterances  = 16 files   (hard-real-world track)

Every selected utterance is emitted in four conditions:

  clean       - untouched original
  pink10      - stationary pink noise mixed at 10 dB SNR ("office hum / AC")
  babble10    - speech babble (5 talkers) mixed at 10 dB SNR ("busy call floor")
  babble5     - speech babble (5 talkers) mixed at 5 dB SNR  ("very noisy")

Babble noise is created by summing five random utterances spoken by people
who are NOT in the evaluation set, which is the standard way to approximate
a noisy call-centre floor without shipping a proprietary noise corpus.

Everything is written as 16 kHz mono WAV (what all three models expect) and a
manifest.json records reference transcripts and per-condition SNR targets so
the whole pipeline is reproducible.
"""

import json
import os
import random

import numpy as np
import soundfile as sf

random.seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RAW = os.path.join(ROOT, "data", "raw", "LibriSpeech")
EVAL = os.path.join(ROOT, "data", "eval")
SR = 16000

CONDITIONS = ["clean", "pink10", "babble10", "babble5"]


# ---------------------------------------------------------------- corpus walk
def walk_split(split: str):
    """Yield (spk, utt_id, wav_path, transcript) for a LibriSpeech split."""
    split_dir = os.path.join(RAW, split)
    for spk in sorted(os.listdir(split_dir)):
        spk_dir = os.path.join(split_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        for chap in sorted(os.listdir(spk_dir)):
            chap_dir = os.path.join(spk_dir, chap)
            if not os.path.isdir(chap_dir):
                continue
            tr = {}
            for line in open(os.path.join(chap_dir, f"{spk}-{chap}.trans.txt")):
                uid, text = line.strip().split(" ", 1)
                tr[uid] = text
            for uid in sorted(tr):
                yield spk, uid, os.path.join(chap_dir, f"{uid}.flac"), tr[uid]


def pick(split: str, n_spk: int, per_spk: int, exclude_spk=frozenset()):
    """Pick n_spk speakers x per_spk utterances, 4-12 s long, 6-45 words."""
    by_spk = {}
    for spk, uid, wav, text in walk_split(split):
        if spk in exclude_spk:
            continue
        info = sf.info(wav)
        dur = info.frames / info.samplerate
        nwords = len(text.split())
        if 4.0 <= dur <= 12.0 and 6 <= nwords <= 45:
            by_spk.setdefault(spk, []).append((uid, wav, text, dur))
    picked = []
    for spk in random.sample(sorted(by_spk), n_spk):
        utts = random.sample(by_spk[spk], min(per_spk, len(by_spk[spk])))
        picked.extend([(spk,) + u for u in utts])
    return picked


# ---------------------------------------------------------------- noise types
def pink_noise(n: int) -> np.ndarray:
    """Pink noise (1/f) via the standard Paul Kellet filter, unit RMS."""
    b = np.zeros(7)
    white = np.random.randn(n)
    pink = np.zeros(n)
    for k in range(n):
        w = white[k]
        b[0] = 0.99886 * b[0] + w * 0.0555179
        b[1] = 0.99332 * b[1] + w * 0.0750759
        b[2] = 0.96900 * b[2] + w * 0.1538520
        b[3] = 0.86650 * b[3] + w * 0.3104856
        b[4] = 0.55000 * b[4] + w * 0.5329522
        b[5] = -0.7616 * b[5] - w * 0.0168980
        pink[k] = (b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362) * 0.11
        b[6] = w * 0.115926
    pink -= pink.mean()
    return pink / (np.sqrt(np.mean(pink**2)) + 1e-9)


def babble(n: int, pool: list) -> np.ndarray:
    """Sum 5 random non-eval utterances into a crowd-noise track."""
    tracks = []
    for _ in range(5):
        wav, _, _ = random.choice(pool)
        x, _ = sf.read(wav)
        if len(x) < n:
            reps = int(np.ceil(n / len(x)))
            x = np.tile(x, reps)
        start = random.randint(0, max(0, len(x) - n))
        x = x[start : start + n].astype(np.float64)
        x /= np.sqrt(np.mean(x**2)) + 1e-9  # per-talker RMS normalise
        tracks.append(x)
    y = np.sum(tracks, axis=0)
    return y / (np.sqrt(np.mean(y**2)) + 1e-9)


def mix_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise into clean speech at the requested SNR (dB), clip-guarded."""
    c_rms = np.sqrt(np.mean(clean.astype(np.float64) ** 2)) + 1e-9
    n_rms = np.sqrt(np.mean(noise**2)) + 1e-9
    scale = c_rms / (n_rms * 10 ** (snr_db / 20.0))
    mixed = clean.astype(np.float64) + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed *= 0.99 / peak
    return mixed.astype(np.float32)


# --------------------------------------------------------------------- main
def main() -> None:
    for c in CONDITIONS:
        os.makedirs(os.path.join(EVAL, c), exist_ok=True)

    eval_rows = pick("test-clean", n_spk=5, per_spk=4)
    other_rows = pick("test-other", n_spk=4, per_spk=4)
    eval_spk = {r[0] for r in eval_rows} | {r[0] for r in other_rows}

    # babble pool: utterances from speakers that are NOT in the eval set
    pool = [
        (wav, spk, uid)
        for spk, uid, wav, text in walk_split("test-other")
        if spk not in eval_spk and random.random() < 0.05
    ]
    print(f"babble pool: {len(pool)} utterances from non-eval speakers")

    manifest = []
    for track, rows in (("test-clean", eval_rows), ("test-other", other_rows)):
        for spk, uid, wav, text, dur in rows:
            clean, _ = sf.read(wav)
            clean = clean.astype(np.float32)
            n = len(clean)

            out = {
                "uid": uid,
                "speaker": spk,
                "track": track,
                "ref": text,
                "duration_s": round(dur, 2),
                "files": {},
            }
            sf.write(os.path.join(EVAL, "clean", f"{uid}.wav"), clean, SR)
            out["files"]["clean"] = f"clean/{uid}.wav"

            noise_tracks = {
                "pink10": (pink_noise(n), 10.0),
                "babble10": (babble(n, pool), 10.0),
                "babble5": (babble(n, pool), 5.0),
            }
            for cond, (noise, snr) in noise_tracks.items():
                mixed = mix_snr(clean, noise, snr)
                sf.write(os.path.join(EVAL, cond, f"{uid}.wav"), mixed, SR)
                out["files"][cond] = f"{cond}/{uid}.wav"

            manifest.append(out)

    with open(os.path.join(EVAL, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(r["duration_s"] for r in manifest)
    print(f"manifest: {len(manifest)} utterances, {total/60:.1f} min of audio, "
          f"4 conditions -> {total*4/60:.1f} min to transcribe per model")


if __name__ == "__main__":
    main()
