import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os, argparse
import sentencepiece as spm
from nmt.utils import read_jsonl, pick_parallel_fields, clean_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=16000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    zh_txt = os.path.join(args.out_dir, "train.zh.txt")
    en_txt = os.path.join(args.out_dir, "train.en.txt")
    with open(zh_txt, "w", encoding="utf-8") as fzh, open(en_txt, "w", encoding="utf-8") as fen:
        for obj in read_jsonl(args.train_jsonl):
            zh,en = pick_parallel_fields(obj)
            fzh.write(clean_text(zh) + "\n")
            fen.write(clean_text(en) + "\n")

    # Train separate models
    spm.SentencePieceTrainer.Train(
        input=zh_txt,
        model_prefix=os.path.join(args.out_dir, "spm_zh"),
        vocab_size=args.vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=0, bos_id=1, eos_id=2, unk_id=3
    )
    spm.SentencePieceTrainer.Train(
        input=en_txt,
        model_prefix=os.path.join(args.out_dir, "spm_en"),
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=0, bos_id=1, eos_id=2, unk_id=3
    )
    print("Saved:", os.path.join(args.out_dir, "spm_zh.model"), os.path.join(args.out_dir, "spm_en.model"))

if __name__ == "__main__":
    main()
