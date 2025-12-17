# train.py
import os, json, argparse, time, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nmt.dataset import JsonlParallelDataset, collate_batch
from nmt.tokenizers import SpmTokenizer, has_spm
from nmt.bleu import corpus_bleu
from nmt.models.rnn_attn import Seq2SeqRNN
from nmt.models.transformer_scratch import TransformerScratch

BOS_ID, EOS_ID, PAD_ID = 1, 2, 0


def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def greedy_decode_transformer(model, src, max_len=128):
    model.eval()
    B = src.size(0)
    ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=src.device)
    for _ in range(max_len - 1):
        logits = model(src, ys)  # (B,T,V)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if (next_tok == EOS_ID).all():
            break
    return ys


@torch.no_grad()
def greedy_decode_rnn(model, src, src_lens, max_len=128):
    model.eval()
    enc_out, enc_state = model.encode(src, src_lens)
    dec_state = enc_state
    src_mask = (src != 0)

    prev = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
    out = [prev]
    for _ in range(max_len - 1):
        emb = model.emb_tgt(prev)
        dec_out, dec_state = model.decoder(emb, dec_state)
        h = dec_out.squeeze(1)
        ctx, _ = model.attn(h, enc_out, key_mask=src_mask[:, :enc_out.size(1)])
        logit = model.out(torch.cat([h, ctx], dim=-1))
        prev = logit.argmax(dim=-1, keepdim=True)
        out.append(prev)
        if (prev == EOS_ID).all():
            break
    return torch.cat(out, dim=1)


@torch.no_grad()
def beam_decode_rnn(model, src, src_lens, beam_size=5, max_len=128, length_penalty=0.6):
    """
    单样本 beam-search（eval_bleu 内会逐个样本调用）。
    """
    model.eval()
    device = src.device
    enc_out, enc_state = model.encode(src, src_lens)
    src_mask = (src != 0)

    if isinstance(enc_state, tuple):
        h, c = enc_state
        init_state = (h.clone(), c.clone())
    else:
        init_state = enc_state.clone()

    init_tokens = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    beams = [(0.0, init_state, init_tokens, False)]  # (logp, state, tokens, done)

    for _ in range(max_len - 1):
        new_beams = []
        for logp, state, tokens, done in beams:
            if done:
                new_beams.append((logp, state, tokens, True))
                continue

            prev = tokens[:, -1:]  # (1,1)
            emb = model.emb_tgt(prev)

            dec_out, new_state = model.decoder(emb, state)
            h_t = dec_out.squeeze(1)  # (1,H)
            ctx, _ = model.attn(h_t, enc_out, key_mask=src_mask[:, :enc_out.size(1)])
            logit = model.out(torch.cat([h_t, ctx], dim=-1))  # (1,V)
            log_probs = torch.log_softmax(logit, dim=-1).squeeze(0)  # (V,)

            topk_logp, topk_ids = torch.topk(log_probs, beam_size)
            for k in range(beam_size):
                tid = int(topk_ids[k].item())
                lp = logp + float(topk_logp[k].item())
                new_tokens = torch.cat([tokens, torch.tensor([[tid]], device=device)], dim=1)
                done_k = (tid == EOS_ID)
                new_beams.append((lp, new_state, new_tokens, done_k))

        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

        if all(b[3] for b in beams):
            break

    def score_fn(b):
        logp, _, tokens, _ = b
        length = tokens.size(1) - 1  # exclude BOS
        if length_penalty > 0:
            lp = ((5 + length) ** length_penalty) / ((5 + 1) ** length_penalty)
        else:
            lp = 1.0
        return logp / lp

    best = max(beams, key=score_fn)
    _, _, tokens, _ = best
    ids = tokens[0].tolist()
    if EOS_ID in ids:
        ids = ids[1: ids.index(EOS_ID)]
    else:
        ids = ids[1:]
    return ids


@torch.no_grad()
def eval_bleu(model, tok_tgt, dl, device, model_type, max_len=128):
    hyps, refs = [], []
    for src, tgt, src_lens, tgt_lens in dl:
        src, tgt, src_lens = src.to(device), tgt.to(device), src_lens.to(device)

        if model_type == "transformer":
            pred = greedy_decode_transformer(model, src, max_len=max_len)
        else:
            decode_strategy = getattr(model, "decode_strategy", "greedy")
            if decode_strategy == "beam":
                outs = []
                for i in range(src.size(0)):
                    s = src[i:i+1]
                    l = src_lens[i:i+1]
                    ids = beam_decode_rnn(
                        model, s, l,
                        beam_size=getattr(model, "beam_size", 5),
                        max_len=max_len,
                        length_penalty=getattr(model, "length_penalty", 0.6),
                    )
                    outs.append(torch.tensor([BOS_ID] + ids + [EOS_ID], device=src.device, dtype=torch.long))

                max_len_b = max(x.size(0) for x in outs)
                pred = torch.full((len(outs), max_len_b), PAD_ID, dtype=torch.long, device=src.device)
                for i, t in enumerate(outs):
                    pred[i, :t.size(0)] = t
            else:
                pred = greedy_decode_rnn(model, src, src_lens, max_len=max_len)

        for i in range(pred.size(0)):
            hyp_ids = pred[i].tolist()
            ref_ids = tgt[i].tolist()

            hyp_ids = hyp_ids[1: (hyp_ids.index(EOS_ID) if EOS_ID in hyp_ids else len(hyp_ids))]
            ref_ids = ref_ids[1: (ref_ids.index(EOS_ID) if EOS_ID in ref_ids else len(ref_ids))]

            hyps.append(tok_tgt.decode(hyp_ids))
            refs.append(tok_tgt.decode(ref_ids))

    return corpus_bleu(hyps, refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rnn", "transformer", "t5"], required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--valid_jsonl", required=True)
    ap.add_argument("--spm_src", default=None)
    ap.add_argument("--spm_tgt", default=None)
    ap.add_argument("--save_dir", required=True)

    # TensorBoard: 固定写到 ./run，方便你 tensorboard --logdir=./run
    ap.add_argument("--tb_logdir", default="./run")

    # shared
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # rnn
    ap.add_argument("--rnn_cell", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--attn", choices=["dot", "general", "additive"], default="additive")
    ap.add_argument("--teacher_forcing", type=float, default=1.0)
    ap.add_argument("--decode_strategy", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=0.6)
    # scheduled sampling: linearly decay TF from tf_start -> tf_end
    ap.add_argument("--tf_start", type=float, default=1.0)
    ap.add_argument("--tf_end", type=float, default=0.0)

    # transformer
    ap.add_argument("--pos_encoding", choices=["sinusoidal", "learned", "rotary", "none"], default="sinusoidal")
    ap.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)

    # t5 (offline)
    ap.add_argument("--t5_dir", default="./t5_base_offline")

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- TensorBoard writer: ./run/<save_dir_basename>/events...
    exp_name = os.path.basename(os.path.normpath(args.save_dir))
    tb_dir = os.path.join(args.tb_logdir, exp_name)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # 记录超参（在 TB 的 hparams/文本里更容易追踪）
    try:
        writer.add_text("hparams", json.dumps(vars(args), ensure_ascii=False, indent=2))
    except Exception:
        pass

    # ------------------- T5 branch -------------------
    if args.model == "t5":
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        from nmt.utils import read_jsonl, pick_parallel_fields, clean_text

        tok = T5Tokenizer.from_pretrained(args.t5_dir)
        model = T5ForConditionalGeneration.from_pretrained(args.t5_dir).to(device)

        train_pairs = []
        for obj in read_jsonl(args.train_jsonl):
            zh, en = pick_parallel_fields(obj)
            train_pairs.append(("translate Chinese to English: " + clean_text(zh), clean_text(en)))

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        def batchify(pairs, bs):
            for i in range(0, len(pairs), bs):
                yield pairs[i:i + bs]

        global_step = 0
        best = -1.0

        for ep in range(1, args.epochs + 1):
            model.train()
            ep_loss_sum, ep_tokens, ep_batches = 0.0, 0, 0
            start_t = time.time()

            pbar = tqdm(batchify(train_pairs, args.batch_size), desc=f"t5 ep{ep}")
            for batch in pbar:
                src_text = [x for x, _ in batch]
                tgt_text = [y for _, y in batch]

                enc = tok(
                    src_text, return_tensors="pt",
                    padding=True, truncation=True, max_length=args.max_len
                ).to(device)

                labels = tok(
                    tgt_text, return_tensors="pt",
                    padding=True, truncation=True, max_length=args.max_len
                ).input_ids.to(device)
                labels = labels.masked_fill(labels == tok.pad_token_id, -100)

                out = model(**enc, labels=labels)
                loss = out.loss

                opt.zero_grad()
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                opt.step()

                bs_tokens = int(labels.ne(-100).sum().item())
                ep_loss_sum += float(loss.item())
                ep_tokens += bs_tokens
                ep_batches += 1

                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/tokens", bs_tokens, global_step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)
                global_step += 1

                pbar.set_postfix(loss=float(loss.item()))

            dur = time.time() - start_t
            if ep_batches > 0:
                writer.add_scalar("train/avg_loss_epoch", ep_loss_sum / ep_batches, ep)
            if dur > 0 and ep_tokens > 0:
                writer.add_scalar("train/tokens_per_sec_epoch", ep_tokens / dur, ep)

            # 保存微调模型
            model.save_pretrained(args.save_dir)
            tok.save_pretrained(args.save_dir)

        writer.close()
        print("Saved T5 to", args.save_dir)
        return

    # ------------------- RNN / Transformer branch -------------------
    assert has_spm(args.spm_src) and has_spm(args.spm_tgt), "Please train SPM first (scripts/train_spm.py)."
    tok_src = SpmTokenizer(args.spm_src)
    tok_tgt = SpmTokenizer(args.spm_tgt)

    train_ds = JsonlParallelDataset(args.train_jsonl, tok_src, tok_tgt, max_len=args.max_len)
    valid_ds = JsonlParallelDataset(args.valid_jsonl, tok_src, tok_tgt, max_len=args.max_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    if args.model == "rnn":
        model = Seq2SeqRNN(tok_src.vocab_size, tok_tgt.vocab_size,
                           rnn_cell=args.rnn_cell, attn=args.attn).to(device)
        # 供 eval_bleu / inference 使用
        model.decode_strategy = args.decode_strategy
        model.beam_size = args.beam_size
        model.length_penalty = args.length_penalty
    else:
        model = TransformerScratch(tok_src.vocab_size, tok_tgt.vocab_size,
                                   d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                                   pos_encoding=args.pos_encoding, norm=args.norm).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best = -1.0
    global_step = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss_sum, ep_tokens, ep_batches = 0.0, 0, 0
        start_t = time.time()

        # scheduled sampling teacher forcing ratio
        if args.model == "rnn":
            if args.epochs > 1:
                ratio = (ep - 1) / max(1, args.epochs - 1)
                tf_now = args.tf_start + (args.tf_end - args.tf_start) * ratio
            else:
                tf_now = args.teacher_forcing
            tf_now = max(0.0, min(1.0, float(tf_now)))
            writer.add_scalar("train/teacher_forcing_ratio", tf_now, ep)

        pbar = tqdm(train_dl, desc=f"{args.model} ep{ep}")
        for src, tgt, src_lens, tgt_lens in pbar:
            src, tgt, src_lens = src.to(device), tgt.to(device), src_lens.to(device)

            if args.model == "rnn":
                logits = model(src, src_lens, tgt, teacher_forcing=tf_now)  # (B,T-1,V)
                gold = tgt[:, 1:]
            else:
                logits = model(src, tgt[:, :-1])  # (B,T-1,V)
                gold = tgt[:, 1:]

            loss = loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))

            opt.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()

            batch_tokens = int(gold.ne(PAD_ID).sum().item())
            ep_loss_sum += float(loss.item())
            ep_tokens += batch_tokens
            ep_batches += 1

            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)
            writer.add_scalar("train/tokens", batch_tokens, global_step)
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)
            global_step += 1

            pbar.set_postfix(loss=float(loss.item()))

        dur = time.time() - start_t
        if ep_batches > 0:
            writer.add_scalar("train/avg_loss_epoch", ep_loss_sum / ep_batches, ep)
        if dur > 0 and ep_tokens > 0:
            writer.add_scalar("train/tokens_per_sec_epoch", ep_tokens / dur, ep)

        bleu = eval_bleu(model, tok_tgt, valid_dl, device, model_type=args.model, max_len=args.max_len)
        print(f"Epoch {ep} valid BLEU: {bleu:.2f}")
        writer.add_scalar("valid/BLEU", bleu, ep)

        if bleu > best:
            best = bleu
            ckpt = {
                "args": vars(args),
                "model_state": model.state_dict(),
                "spm_src": args.spm_src,
                "spm_tgt": args.spm_tgt,
            }
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            with open(os.path.join(args.save_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"best_bleu": best, "args": vars(args)}, f, ensure_ascii=False, indent=2)
            print("Saved best.pt")

        # 让 TensorBoard 更快刷出来
        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
