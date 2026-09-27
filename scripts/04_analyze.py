#!/usr/bin/env python3
"""
04_analyze.py
-------------
Aggregates results/raw_results.json into:

  * results/summary.csv          - one row per model x track x condition
  * results/summary.md           - markdown tables (paste into the report)
  * results/chart_wer.svg        - WER degradation under noise (grouped bars)
  * results/chart_speed.svg      - RTF / decode speed comparison
  * results/chart_memory.svg     - peak RSS comparison

The SVG charts are hand-rolled on purpose: the presentation website must
work offline with zero external dependencies, so no chart library.
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


# --------------------------------------------------------------------- SVG
PALETTE = ["#2563eb", "#059669", "#d97706", "#dc2626"]


def bar_chart(path, title, categories, series, ymax, fmt="{:.0f}", unit=""):
    """series: list of (name, [values])"""
    W, H = 760, 380
    L, R, T, B = 60, 20, 46, 64
    pw, ph = W - L - R, H - T - B
    n_cat, n_ser = len(categories), len(series)
    group = pw / n_cat
    bw = group * 0.8 / n_ser
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="26" text-anchor="middle" font-size="15" '
        f'font-weight="600" fill="#111">{title}</text>',
    ]
    # gridlines
    for i in range(6):
        y = T + ph - ph * i / 5
        v = ymax * i / 5
        parts.append(
            f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
            f'fill="#6b7280">{fmt.format(v)}</text>'
        )
    # bars
    for ci, cat in enumerate(categories):
        gx = L + ci * group + group * 0.1
        for si, (name, vals) in enumerate(series):
            v = min(vals[ci], ymax)
            h = ph * v / ymax
            x = gx + si * bw
            y = T + ph - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-3:.1f}" '
                f'height="{h:.1f}" fill="{PALETTE[si % len(PALETTE)]}" rx="2"/>'
            )
            parts.append(
                f'<text x="{x+(bw-3)/2:.1f}" y="{y-4:.1f}" text-anchor="middle" '
                f'font-size="10" fill="#374151">{fmt.format(vals[ci])}</text>'
            )
        parts.append(
            f'<text x="{L+ci*group+group/2:.1f}" y="{H-B+20}" '
            f'text-anchor="middle" font-size="12" fill="#111">{cat}</text>'
        )
    # legend
    lx = L
    for si, (name, _) in enumerate(series):
        parts.append(
            f'<rect x="{lx}" y="{H-24}" width="12" height="12" rx="2" '
            f'fill="{PALETTE[si % len(PALETTE)]}"/>'
        )
        parts.append(
            f'<text x="{lx+17}" y="{H-14}" font-size="11" fill="#374151">{name}</text>'
        )
        lx += 17 + 8 * len(name) + 24
    if unit:
        parts.append(
            f'<text x="{L-8}" y="{T-10}" text-anchor="end" font-size="10" '
            f'fill="#6b7280">{unit}</text>'
        )
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))
    print("wrote", path)


def main():
    raw = json.load(open(os.path.join(RESULTS, "raw_results.json")))
    agg = aggregate(raw)
    write_csv(raw, agg)
    write_md(raw, agg)

    labels = [res["label"].split(" (")[0] for res in raw.values()]

    # WER vs noise, test-clean track, log-ish capped axis
    series = []
    for key in raw:
        series.append(
            (labels[list(raw).index(key)],
             [agg[key][("test-clean", c)]["wer"] for c in CONDS])
        )
    ymax = max(v for _, vals in series for v in vals) * 1.15
    bar_chart(
        os.path.join(RESULTS, "chart_wer.svg"),
        "Word Error Rate under noise — LibriSpeech test-clean speakers",
        [COND_LABEL[c] for c in CONDS], series, ymax, unit="WER %",
    )

    # speed: mean RTF (lower is better)
    vals = [sum(r["rtf"] for r in raw[k]["rows"]) / len(raw[k]["rows"]) for k in raw]
    ymax = max(vals) * 1.35
    bar_chart(
        os.path.join(RESULTS, "chart_speed.svg"),
        "Real-time factor on 2-core CPU (lower is faster)",
        labels, [("mean RTF", vals)], ymax, fmt="{:.2f}",
    )

    # memory
    vals = [raw[k]["peak_rss_mb"] for k in raw]
    ymax = max(vals) * 1.25
    bar_chart(
        os.path.join(RESULTS, "chart_memory.svg"),
        "Peak process memory on CPU (MB)",
        labels, [("peak RSS", vals)], ymax, unit="MB",
    )

    # dump machine-readable summary for the website
    site = {
        "models": {
            k: {
                "label": raw[k]["label"],
                "peak_rss_mb": raw[k]["peak_rss_mb"],
                "load_s": raw[k]["load_s"],
                "mean_rtf": round(
                    sum(r["rtf"] for r in raw[k]["rows"]) / len(raw[k]["rows"]), 3
                ),
                "wer": {
                    f"{t}|{c}": round(agg[k][(t, c)]["wer"], 2)
                    for t in TRACKS for c in CONDS
                },
            }
            for k in raw
        },
        "conditions": CONDS,
        "cond_labels": COND_LABEL,
        "tracks": TRACKS,
    }
    with open(os.path.join(RESULTS, "site_data.json"), "w") as f:
        json.dump(site, f, indent=2)
    print("wrote results/site_data.json")


if __name__ == "__main__":
    main()
