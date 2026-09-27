#!/usr/bin/env python3
"""
04_analyze.py
-------------
Aggregates results/raw_results.json into:

  * results/summary.csv          - one row per model x track x condition
  * results/summary.md           - the same numbers as markdown tables

It reads the raw per-utterance rows written by 03_benchmark.py, so any single
number in the tables can be traced back to one decode in
results/hypotheses/.
"""

import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")

CONDS = ["clean", "pink10", "babble10", "babble5"]
COND_LABEL = {
    "clean": "Clean",
    "pink10": "Pink noise 10 dB",
    "babble10": "Babble 10 dB",
    "babble5": "Babble 5 dB",
}
TRACKS = ["test-clean", "test-other"]


def aggregate(raw: dict):
    """mean WER / mean RTF per model x track x condition."""
    agg = defaultdict(lambda: defaultdict(list))
    for key, res in raw.items():
        for row in res["rows"]:
            agg[key][(row["track"], row["cond"])].append(row)
    out = {}
    for key, cells in agg.items():
        out[key] = {
            tc: {
                "wer": sum(r["wer"] for r in rows) / len(rows),
                "rtf": sum(r["rtf"] for r in rows) / len(rows),
                "n": len(rows),
            }
            for tc, rows in cells.items()
        }
    return out


def write_csv(raw, agg):
    path = os.path.join(RESULTS, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["model", "label", "track", "condition", "n_utt", "mean_wer_pct",
             "mean_rtf", "peak_rss_mb", "load_s"]
        )
        for key, res in raw.items():
            for track in TRACKS:
                for cond in CONDS:
                    a = agg[key][(track, cond)]
                    w.writerow(
                        [key, res["label"], track, cond, a["n"],
                         round(a["wer"], 2), round(a["rtf"], 3),
                         res["peak_rss_mb"], res["load_s"]]
                    )
    print("wrote", path)


def write_md(raw, agg):
    lines = []
    lines.append("## Mean WER (%), test-clean track\n")
    header = "| Model | " + " | ".join(COND_LABEL[c] for c in CONDS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(CONDS) + 1))
    for key, res in raw.items():
        row = [res["label"].split(" (")[0]]
        row += [f"{agg[key][('test-clean', c)]['wer']:.1f}" for c in CONDS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## Mean WER (%), test-other track\n")
    lines.append(header)
    lines.append("|" + "---|" * (len(CONDS) + 1))
    for key, res in raw.items():
        row = [res["label"].split(" (")[0]]
        row += [f"{agg[key][('test-other', c)]['wer']:.1f}" for c in CONDS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## Speed and memory (both tracks combined)\n")
    lines.append("| Model | Mean RTF | Peak RSS (MB) | Load+warmup (s) |")
    lines.append("|---|---|---|---|")
    for key, res in raw.items():
        rtfs = [r["rtf"] for r in res["rows"]]
        lines.append(
            f"| {res['label'].split(' (')[0]} | {sum(rtfs)/len(rtfs):.3f} "
            f"| {res['peak_rss_mb']} | {res['load_s']} |"
        )
    path = os.path.join(RESULTS, "summary.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)

def main():
    raw = json.load(open(os.path.join(RESULTS, "raw_results.json")))
    agg = aggregate(raw)
    write_csv(raw, agg)
    write_md(raw, agg)


if __name__ == "__main__":
    main()
