# Executive Summary — Picking a Speech-to-Text Engine for Noisy Support Calls

**Assignment 1 · Research–Technical (AI/ML) · Immverse AI**
Parth Wadekar · parthwadekar40@gmail.com · September 2026

**The question.** Which ASR model should power a voice AI assistant on
customer-support calls? These calls are noisy by nature: busy floors, mixed
accents, telephony audio. Clean-audio leaderboard scores say very little
about that world, so I did not rely on them alone.

**What I did.** I surveyed the 2026 field (Open ASR Leaderboard, arXiv
papers, vendor docs — IBM Granite Speech, Cohere Transcribe, NVIDIA
Canary/Parakeet, Voxtral, Whisper large-v3, and the commercial APIs), then
benchmarked three models myself on one 2-core CPU box: **Whisper base.en**
and **Distil-Whisper small.en** (both int8 via faster-whisper) and **Wav2Vec2
base-960h**. Each transcribed the same 36 call-sized LibriSpeech utterances
in four conditions: clean, office hum at 10 dB, crowd babble at 10 dB, and
babble at 5 dB. I measured word error rate, inference time, memory and setup
complexity. The scripts are in the repo; anyone can re-run them.

**What I found.** Clean WER misleads: on clean audio all three sit within
2.5 points (4.3–5.3%), but at 5 dB babble the spread is 73 points —
Wav2Vec2 87.6%, Whisper base.en 25.6%, **Distil-Whisper 14.6%**. Training
data explains it: models that saw messy web audio (the Whisper family, 680k
hours) held up; the model fine-tuned only on clean read speech fell apart,
sometimes inventing more words than the sentence contained. Distil-Whisper
also used the least memory (811 MB) and still decoded at 2.5× real time on
two CPU cores.

**Recommendation.** Ship **Distil-Whisper small.en via faster-whisper in
int8**. It was the most robust of the three on both speaker tracks, it fits
a 1 GB container, it needs one pip install under an MIT license, and the same
runtime serves larger distilled models when GPUs arrive. Put a VAD in front,
stream partial transcripts, route low-confidence calls to humans, and LoRA
fine-tune on 50–200 hours of real calls with noise and telephony-codec
augmentation — the biggest cheap win. Wav2Vec2 is out despite being the
fastest engine tested; Whisper base.en is the latency-first fallback.
Upgrade path on GPU: distil-large-v3 or NVIDIA Parakeet TDT; managed
fallback: Deepgram Nova-3. Re-check the leaderboard quarterly — its
leadership changed twice in six months.

**Caveats.** LibriSpeech approximates the acoustics of a call, not its
language. 36 utterances is directional, not statistical, though the gaps
here (10–70 WER points) are far past noise. CPU speed rankings will shift on
GPUs. Before launch I would re-run the same harness on 200+ real call
recordings.

*Deliverables: technical report, this summary, and the benchmark repo with
scripts and README.*
