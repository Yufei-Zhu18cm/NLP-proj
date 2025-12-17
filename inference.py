import os, argparse
import torch

from nmt.tokenizers import SpmTokenizer
from nmt.models.rnn_attn import Seq2SeqRNN
from nmt.models.transformer_scratch import TransformerScratch

BOS_ID, EOS_ID = 1,2

@torch.no_grad()
def translate_transformer(model, tok_src, tok_tgt, text, device, max_len=128):
    model.eval()
    src_ids = [BOS_ID] + tok_src.encode(text)[:max_len-2] + [EOS_ID]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_len-1):
        logits = model(src, ys)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        if int(nxt.item()) == EOS_ID:
            break
    out = ys[0].tolist()
    if EOS_ID in out: out = out[1:out.index(EOS_ID)]
    else: out = out[1:]
    return tok_tgt.decode(out)

@torch.no_grad()
def translate_rnn(model, tok_src, tok_tgt, text, device, max_len=128):
    model.eval()
    src_ids = [BOS_ID] + tok_src.encode(text)[:max_len-2] + [EOS_ID]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_lens = torch.tensor([src.size(1)], dtype=torch.long, device=device)
    enc_out, enc_state = model.encode(src, src_lens)
    dec_state = enc_state
    src_mask = (src != 0)
    prev = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    out = []
    for _ in range(max_len-1):
        emb = model.emb_tgt(prev)
        dec_out, dec_state = model.decoder(emb, dec_state)
        h = dec_out.squeeze(1)
        ctx,_ = model.attn(h, enc_out, key_mask=src_mask[:, :enc_out.size(1)])
        logit = model.out(torch.cat([h, ctx], dim=-1))
        prev = logit.argmax(dim=-1, keepdim=True)
        tid = int(prev.item())
        if tid == EOS_ID:
            break
        out.append(tid)
    return tok_tgt.decode(out)

@torch.no_grad()
def translate_rnn_beam(model, tok_src, tok_tgt, text, device, beam_size=5, max_len=128, length_penalty=0.6):
    model.eval()
    src_ids = [BOS_ID] + tok_src.encode(text)[:max_len-2] + [EOS_ID]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_lens = torch.tensor([src.size(1)], dtype=torch.long, device=device)
    enc_out, enc_state = model.encode(src, src_lens)
    src_mask = (src != 0)

    if isinstance(enc_state, tuple):
        h, c = enc_state
        init_state = (h.clone(), c.clone())
    else:
        init_state = enc_state.clone()

    init_token = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    beams = [(0.0, init_state, init_token, False)]

    for _ in range(max_len - 1):
        new_beams = []
        for logp, state, tokens, done in beams:
            if done:
                new_beams.append((logp, state, tokens, True))
                continue
            prev = tokens[:, -1:].to(device)
            emb = model.emb_tgt(prev)
            if isinstance(state, tuple):
                dec_out, new_state = model.decoder(emb, state)
            else:
                dec_out, new_state = model.decoder(emb, state)
            h = dec_out.squeeze(1)
            ctx,_ = model.attn(h, enc_out, key_mask=src_mask[:, :enc_out.size(1)])
            logit = model.out(torch.cat([h, ctx], dim=-1))
            log_probs = torch.log_softmax(logit, dim=-1).squeeze(0)

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
        length = tokens.size(1) - 1
        if length_penalty > 0:
            lp = ((5 + length) ** length_penalty) / ((5 + 1) ** length_penalty)
        else:
            lp = 1.0
        return logp / lp

    best = max(beams, key=score_fn)
    _, _, tokens, _ = best
    ids = tokens[0].tolist()
    if EOS_ID in ids:
        ids = ids[1:ids.index(EOS_ID)]
    else:
        ids = ids[1:]
    return tok_tgt.decode(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Directory containing best.pt or T5 weights")
    ap.add_argument("--input", required=True, help="A Chinese sentence")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--decode_strategy", choices=["greedy","beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=0.6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # T5 case
    if os.path.exists(os.path.join(args.ckpt_dir, "config.json")):
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tok = T5Tokenizer.from_pretrained(args.ckpt_dir)
        model = T5ForConditionalGeneration.from_pretrained(args.ckpt_dir).to(device)
        prompt = "translate Chinese to English: " + args.input.strip()
        enc = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(device)
        out = model.generate(**enc, max_new_tokens=args.max_len)
        print(tok.decode(out[0], skip_special_tokens=True))
        return

    best_pt = os.path.join(args.ckpt_dir, "best.pt")
    ckpt = torch.load(best_pt, map_location="cpu")
    cfg = ckpt["args"]
    tok_src = SpmTokenizer(ckpt["spm_src"])
    tok_tgt = SpmTokenizer(ckpt["spm_tgt"])

    if cfg["model"] == "rnn":
        model = Seq2SeqRNN(tok_src.vocab_size, tok_tgt.vocab_size,
                           rnn_cell=cfg["rnn_cell"], attn=cfg["attn"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        if args.decode_strategy == "beam":
            out = translate_rnn_beam(
                model, tok_src, tok_tgt, args.input, device,
                beam_size=args.beam_size, max_len=args.max_len,
                length_penalty=args.length_penalty
            )
        else:
            out = translate_rnn(model, tok_src, tok_tgt, args.input, device, max_len=args.max_len)
        print(out)
    else:
        model = TransformerScratch(tok_src.vocab_size, tok_tgt.vocab_size,
                                  d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                                  pos_encoding=cfg["pos_encoding"], norm=cfg["norm"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(translate_transformer(model, tok_src, tok_tgt, args.input, device, max_len=args.max_len))

if __name__ == "__main__":
    main()
