# Chinese-English NMT Project (RNN + Transformer + T5)

This repo matches the course project requirements:
- RNN-based NMT: 2-layer uni-directional encoder/decoder, attention (dot/general/additive),
  teacher forcing vs free running (scheduled sampling), greedy vs beam search (beam_size configurable).
- Transformer-based NMT: from scratch encoder-decoder; ablations: position embedding + norm.
- From pretrained LM: fine-tune T5.
- Provide one-click inference script: inference.py

## 0) Data
Put dataset JSONL files under: data/
Expected 4 jsonl: train_10k.jsonl, train_100k.jsonl, valid.jsonl, test.jsonl
Each line: a dict containing a parallel pair.
Common keys: ("zh","en") or ("src","tgt") or ("source","target") etc.
If keys unknown, the loader will pick the first two string fields.

## 1) Train SentencePiece (recommended)
Train separate SPM for zh and en:
python scripts/train_spm.py --train_jsonl data/train_10k.jsonl --out_dir artifacts/spm --vocab_size 16000

## 2) Train RNN (examples)
# 2.1 LSTM + additive attention + scheduled sampling (TF from 1.0 -> 0.0) + beam-search (beam=5, length_penalty=0.6)
python train.py --model rnn \
  --train_jsonl data/train_10k.jsonl --valid_jsonl data/valid.jsonl \
  --spm_src artifacts/spm/spm_zh.model --spm_tgt artifacts/spm/spm_en.model \
  --rnn_cell lstm --attn additive \
  --epochs 8 --batch_size 32 --lr 3e-4 \
  --tf_start 1.0 --tf_end 0.0 \
  --decode_strategy beam --beam_size 5 --length_penalty 0.6 \
  --save_dir checkpoints/rnn_lstm_add_sched_beam

# 2.2 纯 Teacher Forcing + greedy 作对比
python train.py --model rnn \
  --train_jsonl data/train_10k.jsonl --valid_jsonl data/valid.jsonl \
  --spm_src artifacts/spm/spm_zh.model --spm_tgt artifacts/spm/spm_en.model \
  --rnn_cell lstm --attn additive \
  --epochs 8 --batch_size 32 --lr 3e-4 \
  --tf_start 1.0 --tf_end 1.0 \
  --decode_strategy greedy \
  --save_dir checkpoints/rnn_lstm_add_tf_greedy

## 3) Train Transformer (scratch)
python train.py --model transformer \
  --train_jsonl data/train_10k.jsonl --valid_jsonl data/valid.jsonl \
  --spm_src artifacts/spm/spm_zh.model --spm_tgt artifacts/spm/spm_en.model \
  --pos_encoding sinusoidal --norm layernorm \
  --d_model 256 --n_heads 4 --n_layers 4 \
  --epochs 5 --batch_size 32 --lr 3e-4 \
  --save_dir checkpoints/tfm_sin_ln

Ablation examples:
# position encodings
python train.py --model transformer ... --pos_encoding learned --norm layernorm --save_dir checkpoints/tfm_learn_ln
python train.py --model transformer ... --pos_encoding rotary  --norm layernorm --save_dir checkpoints/tfm_rope_ln
# norms
python train.py --model transformer ... --pos_encoding sinusoidal --norm rmsnorm --save_dir checkpoints/tfm_sin_rms

## 4) Fine-tune T5
python train.py --model t5 \
  --train_jsonl data/train_10k.jsonl --valid_jsonl data/valid.jsonl \
  --epochs 3 --batch_size 8 --lr 1e-4 \
  --save_dir checkpoints/t5_finetune

## 5) One-click inference
# RNN greedy vs beam
python inference.py --ckpt_dir checkpoints/rnn_lstm_add_sched_beam --input "今天 天气 很好" --decode_strategy greedy
python inference.py --ckpt_dir checkpoints/rnn_lstm_add_sched_beam --input "今天 天气 很好" --decode_strategy beam --beam_size 5 --length_penalty 0.6

# Transformer
python inference.py --ckpt_dir checkpoints/tfm_sin_ln --input "今天 天气 很好"

# T5
python inference.py --ckpt_dir checkpoints/t5_finetune --input "今天 天气 很好"

## 6) Evaluate BLEU on test
python evaluate.py --ckpt_dir checkpoints/rnn_lstm_add_sched_beam --test_jsonl data/test.jsonl
python evaluate.py --ckpt_dir checkpoints/tfm_sin_ln              --test_jsonl data/test.jsonl
python evaluate.py --ckpt_dir checkpoints/t5_finetune             --test_jsonl data/test.jsonl

only tfm_100k_sin_ln_ep40 and rnn_100k_add_tf_beam_ep40 save the best .pt
