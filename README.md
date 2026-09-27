# Comparative Study of Speech-to-Text Models for Noisy Real-World Audio

Benchmark and research behind a model recommendation for a voice AI assistant
on **noisy customer-support calls**. Part of the Research–Technical (AI/ML)
assignment: research recent ASR models, benchmark three of them, analyse,
recommend.

**Headline result:** on clean audio the three models sit within 2.5 WER
points of each other; at 5 dB speech babble they spread from 14.6% to 87.6%.
**Distil-Whisper small.en wins on noise robustness** and is the recommended
engine. Full numbers in [`results/summary.md`](results/summary.md).

## Repository layout

```
├── scripts/
│   ├── 01_download_data.py   # LibriSpeech test-clean + test-other (OpenSLR)
│   ├── 02_prepare_audio.py   # eval set + synthetic noise at fixed SNR
│   ├── 03_benchmark.py       # WER / RTF / peak-RSS harness for the 3 models
│   ├── 04_analyze.py         # summary tables from raw results
│   ├── 06_make_pdfs.py       # markdown -> PDF (report deliverables)
│   └── 07_make_docx.py       # markdown -> Word (editable versions)
├── results/
│   ├── raw_results.json      # every hypothesis + per-utterance timings
│   ├── hypotheses/           # one .txt per model x utterance x condition
│   └── summary.csv / .md     # model x track x condition aggregates
├── report/
│   ├── technical_report.md / .pdf   # the technical report
│   └── executive_summary.md / .pdf  # 1-page summary
├── references.md             # annotated bibliography
└── requirements.txt
```

## Reproducing the benchmark

Python 3.10+, CPU only, ~2 GB disk for the audio. The full run took under an
hour on a 2-core machine.

```bash
pip install -r requirements.txt

python scripts/01_download_data.py    # ~650 MB from openslr.org
python scripts/02_prepare_audio.py    # builds data/eval/{clean,pink10,babble10,babble5}
python scripts/03_benchmark.py        # runs all 3 models
python scripts/04_analyze.py          # writes results/summary.{csv,md}
```

## Benchmark design in one paragraph

36 LibriSpeech utterances (20 from test-clean speakers, 16 from test-other
speakers with harder accents) x 4 conditions: clean, pink noise @ 10 dB SNR,
5-talker babble @ 10 dB and @ 5 dB, mixed at verified SNRs from non-eval
speakers. Models: Whisper base.en and Distil-Whisper small.en (int8 via
faster-whisper / CTranslate2) and Wav2Vec2 base 960h (transformers, fp32).
Scored with `jiwer` after lowercase + punctuation-stripping normalisation.
Timings are wall-clock decode on a 2-core CPU; memory is peak RSS sampled at
50 ms. See `report/technical_report.md` §4 for methodology, §8 for what the
setup does *not* prove.

## Licenses

- LibriSpeech: CC-BY-4.0 (https://www.openslr.org/12/)
- Whisper / Distil-Whisper: MIT · Wav2Vec2 base 960h: Apache 2.0
- Benchmark code in this repo: MIT
