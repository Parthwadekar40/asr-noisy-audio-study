# References & Documents Researched

Every source consulted for this study, grouped by what it was used for, with
a one-line note on what we actually took from it. Accessed September 2026.

## A. Core papers (architecture & training)

1. **Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"
   (Whisper)** — arXiv:2212.04356 (Dec 2022).
   *What we took:* the 680k-hour weakly-supervised recipe and its split
   (438k h English ASR / 117k h multilingual / 126k h translation); the
   encoder–decoder transformer on 80-bin log-Mel, 30 s windows; multi-task
   tokens (language ID, translation, timestamps); the paper's own evidence
   that scale + diversity of weak labels → robustness to accents/noise
   without task-specific fine-tuning. Also the origin of the known
   hallucination-on-silence failure mode of the autoregressive decoder.

2. **Baevski et al., "wav2vec 2.0"** — NeurIPS 2020, arXiv:2006.11477.
   *What we took:* the self-supervised contrastive pretraining design (CNN
   feature encoder + transformer context net + quantised targets); why
   `base-960h` fine-tuned on 960 h of clean LibriSpeech is fast (CTC,
   non-autoregressive, 95 M params) but brittle off-distribution — our
   benchmark quantifies that brittleness under noise.

3. **Gandhi et al., "Distil-Whisper: Robust Knowledge Distillation via
   Large-Scale Pseudo Labelling"** — arXiv:2311.00430 (2023).
   *What we took:* distillation recipe — frozen Whisper encoder, 2-layer
   decoder (init from first+last teacher layers), KL + cross-entropy on
   ~19k h pseudo-labelled audio; published claim 6× faster / 49% smaller /
   within 1% WER OOD. Our noise benchmark tests exactly that "within 1%"
   claim under conditions the paper did not stress.

4. **Xu et al., "Efficient Sequence Transduction by Jointly Predicting
   Tokens and Durations" (TDT)** — ICML 2023, arXiv:2304.06795.
   *What we took:* the TDT head predicts token + duration → frame-skipping
   decode, up to 2.82× faster than RNN-T with equal/better WER; **Fig. 6:
   TDT degrades more gracefully than RNN-T as noise increases and its
   inference time is flat across SNRs** — a direct argument for transducer
   models (Parakeet) in our noisy-telephony scenario.

5. **"Moonshine: Speech Recognition for Live Transcription and Voice
   Commands"** — arXiv:2410.15608 (2024).
   *What we took:* architecture table (tiny 27.1 M / base 61.5 M params vs
   Whisper tiny.en 37.8 M / base.en 72.6 M); design choices for short
   utterances on edge devices; claim of lower WER than same-size Whisper
   with up to 3× latency reduction on OpenASR-leaderboard datasets.

6. **Srivastav et al., "Open ASR Leaderboard: Towards Reproducible and
   Transparent Multilingual and Long-Form Speech Recognition Evaluation"** —
   arXiv:2510.06961 (2025).
   *What we took:* how the leaderboard is built (8 English short-form
   datasets + multilingual and long-form tracks, unified harness, RTFx on
   A100 batch 64) — important for reading its numbers honestly and for the
   long-form caveat (Parakeet v3: 6.3% short-form but 10.7% long-form).

## B. Leaderboard & model cards (2026 SOTA survey)

7. **Hugging Face Open ASR Leaderboard** (huggingface.co/spaces/hf-audio/open_asr_leaderboard)
   plus leaderboard mirrors/summaries (codesota.com/speech/stt-leaderboard,
   marktechpost.com 2026-07-23).
   *What we took:* the Sep-2026 ranking — Granite Speech 4.1 2B (5.33%),
   Cohere Transcribe (5.42%), Granite 4.0 1B (5.52%), Canary-Qwen-2.5B
   (5.63%), Parakeet TDT 0.6B v2 (6.05%) / v3 (6.32%), Voxtral Small 24B
   (6.62%), Whisper large-v3 (7.44%); RTFx figures; license column.

8. **NVIDIA model cards** — `nvidia/parakeet-tdt-0.6b-v2`, `v3`,
   `nvidia/canary-qwen-2.5b`, `nvidia/canary-1b-v2` (HF).
   *What we took:* FastConformer-TDT architecture details; v3 = 25 European
   languages with auto language ID, CC-BY-4.0, RTFx ≈ 3,333, word-level
   timestamps + punctuation out of the box, up to ~24 min full-attention /
   ~3 h local-attention audio; Canary-Qwen = FastConformer encoder +
   Qwen3-1.7B LLM + LoRA ("SALM"), 234k h training, English-only, RTFx ≈ 418.

9. **IBM Granite Speech** — github.com/ibm-granite/granite-speech-models +
   IBM Research blog (Jun 2025) + Granite Speech 4.1 tech report.
   *What we took:* two-pass design (transcribe, then optionally LLM-process);
   acoustic encoder + modality adapter + LoRA on Granite LLM; Apache 2.0;
   3.3-2b/8b trained mid-2025 on open corpora; 4.1 2B = 174k h, 6 languages
   + speech translation, keyword-list biasing, RTFx ≈ 231 — the accuracy
   leader as of Sep 2026, per IBM's own balanced-sampling and
   block-self-attention Conformer explanation.

10. **Cohere Transcribe model card & launch coverage** (Mar 2026, e.g.
    winbuzzer.com 2026-03-27).
    *What we took:* 2B params, Apache 2.0, 14 languages, no auto language
    detection; per-dataset table (LS clean 1.25 / LS other 2.37 / AMI 8.13 /
    Earnings22 10.86 …); human-eval win rates (64% vs Whisper large-v3 in
    English). First non-IBM/NVIDIA open model to top the leaderboard.

11. **Mistral Voxtral** — model card + MarkTechPost comparison (2026-07).
    *What we took:* audio-language model family (Mini 3B/4B, Small 24B);
    Apache 2.0 (Mini/Small open weights); strength is audio understanding
    (summarise/answer over speech), transcription mid-pack (Mini ≈ 7.05%);
    relevant if the assistant needs comprehension, not just transcripts.

## C. Commercial APIs (deployment alternative)

12. **Deepgram Nova-3** — deepgram.com docs + independent reviews (2026).
    *What we took:* streaming-first design, sub-150 ms partials, published
    noise-robustness figures (noise robustness 9/10 in diyai.io review;
    AA-WER 5.2% per Telnyx), ~36+ languages, pricing ~$0.0043–0.007/min
    tiers; Flux variant built for voice agents (turn-taking).

13. **AssemblyAI Universal-2 / Universal-3.5 Pro** — assemblyai.com blog
    "Best speech-to-text APIs for startups" (Aug 2026).
    *What we took:* Universal-3.5 Pro = 5.6% mean / 4.9% median English WER,
    hallucination rate ~30% below Whisper, LLM-based decoder for entities,
    19 languages realtime + code-switching, Voice Focus (isolates primary
    speaker in noise), $0.15/h Universal-2 batch (Jul 2026 price drop).

14. **OpenAI gpt-4o-transcribe / gpt-4o-mini-transcribe** — OpenAI platform
    docs + Telnyx benchmark table (2026).
    *What we took:* 4.0% AA-WER (Artificia, noisy/multi-domain), 32.5×
    speed factor; hosted-only; the vendor claims it beats Whisper on noisy
    audio — we could not verify independently (no leaderboard submission).

15. **Google Cloud STT v2 (Chirp 2), Speechmatics Ursa-2, ElevenLabs
    Scribe v2, AWS Transcribe, Azure AI Speech** — vexascribe.com
    13-API comparison (Aug 2026), transcribetube.com (Jun 2026).
    *What we took:* language coverage extremes (Google 125+), HIPAA options
    (AWS/Azure BAA), ElevenLabs 150 ms first-word streaming + diarization
    accuracy claims, Speechmatics accent/dialect emphasis — the menu of
    managed options if self-hosting is ruled out.

## D. Datasets, tools & infrastructure

16. **LibriSpeech** — Panayotov et al., ICASSP 2015; openslr.org/12.
    *What we took:* the corpus itself (test-clean 2,620 utts / test-other
    2,864 utts), CC-BY-4.0 licensing, per-chapter transcripts used as refs.
17. **jiwer** — github.com/jitsi/jiwer. *What we took:* WER computation;
    forced us to fix a normalisation convention (lowercase + strip
    punctuation) so Whisper's punctuated output isn't unfairly penalised.
18. **CTranslate2 / faster-whisper** — github.com/OpenNMT/CTranslate2,
    github.com/SYSTRAN/faster-whisper. *What we took:* int8 CPU inference
    (~4× faster than reference Whisper), the `condition_on_previous_text`
    knob, beam-size defaults — the practical deployment layer for Whisper.
19. **parakeet.cpp** — github.com/mudler/parakeet.cpp (2026).
    *What we took:* ggml port of the whole Parakeet family running on CPU,
    validated WER-0 against NeMo, faster than whisper.cpp at equal size —
    evidence that NVIDIA models are becoming viable without GPUs (flagged
    as follow-up work in the report).
20. **Speechmatics engineering blog, "Token Duration Transducer (TDT)
    Explained"** (Mar 2026). *What we took:* an independent, vendor-side
    walkthrough of TDT inference mechanics confirming the paper's speedup
    figures; used to sanity-check our reading of arXiv:2304.06795.

## E. Noise-robustness background

21. **CHiME challenge series** (chimechallenge.org) — background reading on
    real far-field/multi-talker ASR evaluation; confirms babble + SNR
    sweeps are the standard way to approximate such conditions when the
    real corpus (MUSAN, CHiME) is impractical to ship in a small repo.
22. **Noise-augmentation practice** (SpecAugment / speed-perturbation /
    multi-condition training literature, e.g. Ko et al. 2015 "Study of
    Data Augmentation for Noise-robust ASR", ICASSP).
    *What we took:* the fine-tuning recipe recommended in §7 — mixing
    noise into training audio at random SNRs is the cheapest large win for
    a call-centre domain.
