#!/usr/bin/env python3
"""
03_benchmark.py
---------------
Runs every candidate ASR model over every utterance x condition in
data/eval/manifest.json and records:

  * hypothesis text            -> results/hypotheses/<model>__<uid>__<cond>.txt
  * inference wall time        -> per utterance, seconds
  * real-time factor (RTF)     -> decode time / audio duration  (lower = faster)
  * peak process RSS           -> sampled every 50 ms in a background thread
  * word error rate            -> jiwer, after standard ASR text normalisation

Models benchmarked:
  1. faster-whisper base.en          (int8, CTranslate2, CPU)
  2. faster-distil-small.en          (int8, CTranslate2, CPU)
  3. wav2vec2-base-960h              (fp32, HuggingFace transformers, CPU)

Usage:
    python scripts/03_benchmark.py                 # all models
    python scripts/03_benchmark.py --models whisper_base wav2vec2
"""

import argparse
import json
import os
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
EVAL = os.path.join(ROOT, "data", "eval")
RESULTS = os.path.join(ROOT, "results")
HYP_DIR = os.path.join(RESULTS, "hypotheses")

try:  # torch is only needed by the wav2vec2 runner
    import torch  # noqa: E402

    torch.set_num_threads(2)
except ImportError:
    torch = None


# ------------------------------------------------------------ text normalise
import re  # noqa: E402

_PUNCT = re.compile(r"[^\w\s']")


def normalise(text: str) -> str:
    """Standard ASR scoring normalisation: lowercase, strip punctuation."""
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    text = text.replace("'", "")
    return " ".join(text.split())


# ------------------------------------------------------------- memory probe
class MemoryProbe:
    """Samples process RSS from a daemon thread; keeps the max."""

    def __init__(self):
        import psutil

        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=1)
        self.peak = max(self.peak, self.proc.memory_info().rss)


# ----------------------------------------------------------------- runners
def run_faster_whisper(model_id: str, wav: str) -> str:
    global _FW_MODEL
    if "_FW_MODEL" not in globals():
        from faster_whisper import WhisperModel

        _FW_MODEL = WhisperModel(model_id, device="cpu", compute_type="int8")
    segments, _info = _FW_MODEL.transcribe(
        wav,
        beam_size=5,
        condition_on_previous_text=False,  # utterances are independent
    )
    return " ".join(s.text.strip() for s in segments)


def run_wav2vec2(model_id: str, wav: str) -> str:
    global _W2V
    if "_W2V" not in globals():
        from transformers import AutoProcessor, AutoModelForCTC

        _W2V = (
            AutoProcessor.from_pretrained(model_id),
            AutoModelForCTC.from_pretrained(model_id),
        )
    import soundfile as sf

    proc, model = _W2V
    audio, sr = sf.read(wav)
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)[0]
    return proc.decode(ids)


MODELS = {
    "whisper_base": {
        "label": "Whisper base.en (faster-whisper, int8)",
        "runner": lambda wav: run_faster_whisper("base.en", wav),
    },
    "distil_small": {
        "label": "Distil-Whisper small.en (faster-whisper, int8)",
        "runner": lambda wav: run_faster_whisper("Systran/faster-distil-whisper-small.en", wav),
    },
    "wav2vec2_base": {
        "label": "Wav2Vec2 base 960h (transformers, fp32)",
        "runner": lambda wav: run_wav2vec2("facebook/wav2vec2-base-960h", wav),
    },
}


# --------------------------------------------------------------------- main
def benchmark(model_key: str, manifest: list) -> dict:
    from jiwer import wer as jiwer_wer

    cfg = MODELS[model_key]
    os.makedirs(HYP_DIR, exist_ok=True)
    print(f"\n=== {cfg['label']} ===", flush=True)

    # warm the model up on one utterance so lazy init / JIT does not pollute
    # the timing of the first measured file; memory probe covers load + decode
    rows = []
    with MemoryProbe() as probe:
        t0 = time.time()
        cfg["runner"](os.path.join(EVAL, manifest[0]["files"]["clean"]))
        load_s = time.time() - t0
        print(f"  load+warmup: {load_s:.1f}s", flush=True)

        for item in manifest:
            for cond in ("clean", "pink10", "babble10", "babble5"):
                wav = os.path.join(EVAL, item["files"][cond])
                t0 = time.time()
                text = cfg["runner"](wav)
                dt = time.time() - t0
                ref_n, hyp_n = normalise(item["ref"]), normalise(text)
                e = jiwer_wer(ref_n, hyp_n)
                rows.append(
                    {
                        "uid": item["uid"],
                        "track": item["track"],
                        "cond": cond,
                        "duration_s": item["duration_s"],
                        "decode_s": round(dt, 3),
                        "rtf": round(dt / item["duration_s"], 3),
                        "wer": round(e * 100, 2),
                        "hyp": text,
                    }
                )
                with open(
                    os.path.join(HYP_DIR, f"{model_key}__{item['uid']}__{cond}.txt"), "w"
                ) as f:
                    f.write(text)
            print(f"  {item['uid']} done", flush=True)
    peak_mb = probe.peak / 1e6
    print(f"  peak RSS: {peak_mb:.0f} MB", flush=True)

    total_audio = sum(r["duration_s"] for r in rows)
    total_decode = sum(r["decode_s"] for r in rows)
    return {
        "model": model_key,
        "label": cfg["label"],
        "load_s": round(load_s, 1),
        "peak_rss_mb": round(peak_mb),
        "rows": rows,
        "total_audio_s": round(total_audio, 1),
        "total_decode_s": round(total_decode, 1),
        "overall_rtf": round(total_decode / total_audio, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(EVAL, "manifest.json")))
    os.makedirs(RESULTS, exist_ok=True)

    out_path = os.path.join(RESULTS, "raw_results.json")
    all_results = {}
    if os.path.exists(out_path):
        all_results = json.load(open(out_path))

    for key in args.models:
        res = benchmark(key, manifest)
        all_results[key] = res
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)  # checkpoint after each model
        print(f"  -> saved {key}", flush=True)

    print("\nall done. results in", out_path)


if __name__ == "__main__":
    main()
