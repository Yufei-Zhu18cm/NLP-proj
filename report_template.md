# ID_name Project Report (Chinese-English Machine Translation)

Repo URL: (put your GitHub link here)

## 1. Task & Dataset
- Describe the dataset jsonl format, split sizes, preprocessing.

## 2. RNN-based NMT
- Architecture: 2-layer uni-directional encoder/decoder (GRU/LSTM).
- Attention: dot / general (multiplicative) / additive.
- Training: teacher forcing vs free running (scheduled sampling).
- Decoding: greedy vs beam search (beam size, length penalty).
- Provide experiment table (BLEU + qualitative examples) and analysis.

## 3. Transformer-based NMT (from scratch)
- Encoder-decoder Transformer, training details.
- Ablations:
  - Position encoding: sinusoidal vs learned vs rotary
  - Normalization: LayerNorm vs RMSNorm
- Hyperparameter sensitivity:
  - batch size / lr / model size
- Provide BLEU + training curve (loss vs steps) + qualitative analysis.

## 4. Pretrained LM (T5) Fine-tuning
- Fine-tuning setup and comparison with scratch models.

## 5. Comparative Analysis & Discussion
Compare RNN vs Transformer:
- architecture (sequential vs parallel)
- training efficiency
- translation performance (BLEU, fluency/adequacy)
- scalability (long sentences)
- practical trade-offs (size/latency)

## 6. Visualization-based Analysis
- attention heatmaps / loss curves / BLEU vs steps (choose what you can provide)

## 7. Reflection
- What worked, what didn’t, future improvements
