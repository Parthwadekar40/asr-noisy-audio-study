## Mean WER (%), test-clean track

| Model | Clean | Pink noise 10 dB | Babble 10 dB | Babble 5 dB |
|---|---|---|---|---|
| Whisper base.en | 4.3 | 6.7 | 8.7 | 25.6 |
| Distil-Whisper small.en | 4.8 | 7.3 | 5.5 | 14.6 |
| Wav2Vec2 base 960h | 5.3 | 16.7 | 37.5 | 87.6 |

## Mean WER (%), test-other track

| Model | Clean | Pink noise 10 dB | Babble 10 dB | Babble 5 dB |
|---|---|---|---|---|
| Whisper base.en | 8.0 | 15.2 | 19.3 | 43.8 |
| Distil-Whisper small.en | 8.3 | 12.6 | 16.3 | 28.7 |
| Wav2Vec2 base 960h | 7.3 | 24.7 | 48.8 | 94.4 |

## Speed and memory (both tracks combined)

| Model | Mean RTF | Peak RSS (MB) | Load+warmup (s) |
|---|---|---|---|
| Whisper base.en | 0.168 | 852 | 4.6 |
| Distil-Whisper small.en | 0.405 | 811 | 4.9 |
| Wav2Vec2 base 960h | 0.072 | 1344 | 12.7 |
