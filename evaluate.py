import os, argparse, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nmt.dataset import JsonlParallelDataset, collate_batch
from nmt.tokenizers import SpmTokenizer
from nmt.bleu import corpus_bleu

from nmt.models.rnn_attn import Seq2SeqRNN
from nmt.models.transformer_scratch import TransformerScratch

BOS_ID, EOS_ID, PAD_ID = 1,2,0

@torch.no_grad()
def greedy_decode_transformer(model, src, max_len=128):
    model.eval()
    ys = torch.full((src.size(0),1), BOS_ID, dtype=torch.long, device=src.device)
    for _ in range(max_len-1):
        logits = model(src, ys)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        if (nxt == EOS_ID).all(): break
    return ys

@torch.no_grad()
def greedy_decode_rnn(model, src, src_lens, max_len=128):
    model.eval()
    enc_out, enc_state = model.encode(src, src_lens)
    dec_state = enc_state
    src_mask = (src != 0)
    prev = torch.full((src.size(0),1), BOS_ID, dtype=torch.long, device=src.device)
    outs = [prev]
    for _ in range(max_len-1):
        emb = model.emb_tgt(prev)
        dec_out, dec_state = model.decoder(emb, dec_state)
        h = dec_out.squeeze(1)
        ctx,_ = model.attn(h, enc_out, key_mask=src_mask[:, :enc_out.size(1)])
        logit = model.out(torch.cat([h, ctx], dim=-1))
        prev = logit.argmax(dim=-1, keepdim=True)
        outs.append(prev)
        if (prev == EOS_ID).all(): break
    return torch.cat(outs, dim=1)

@torch.no_grad()
def beam_decode_rnn(model, src, src_lens, beam_size=5, max_len=128, length_penalty=0.6):
    model.eval()
    device = src.device
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
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # T5 evaluate
    if os.path.exists(os.path.join(args.ckpt_dir, "config.json")):
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        from nmt.utils import read_jsonl, pick_parallel_fields, clean_text
        tok = T5Tokenizer.from_pretrained(args.ckpt_dir)
        model = T5ForConditionalGeneration.from_pretrained(args.ckpt_dir).to(device)
        hyps, refs = [], []
        for obj in read_jsonl(args.test_jsonl):
            zh,en = pick_parallel_fields(obj)
            prompt = "translate Chinese to English: " + clean_text(zh)
            enc = tok([prompt], return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
            out = model.generate(**enc, max_new_tokens=args.max_len)
            hyps.append(tok.decode(out[0], skip_special_tokens=True))
            refs.append(clean_text(en))
        print("BLEU:", corpus_bleu(hyps, refs))
        return

    from nmt.utils import read_jsonl, pick_parallel_fields, clean_text
    ckpt = torch.load(os.path.join(args.ckpt_dir, "best.pt"), map_location="cpu")
    cfg = ckpt["args"]
    tok_src = SpmTokenizer(ckpt["spm_src"])
    tok_tgt = SpmTokenizer(ckpt["spm_tgt"])

    ds = JsonlParallelDataset(args.test_jsonl, tok_src, tok_tgt, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_batch)

    decode_strategy = cfg.get("decode_strategy", "greedy")
    beam_size = cfg.get("beam_size", 5)
    length_penalty = cfg.get("length_penalty", 0.6)

    if cfg["model"] == "rnn":
        model = Seq2SeqRNN(tok_src.vocab_size, tok_tgt.vocab_size,
                          rnn_cell=cfg["rnn_cell"], attn=cfg["attn"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model_type = "rnn"
    else:
        model = TransformerScratch(tok_src.vocab_size, tok_tgt.vocab_size,
                                  d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                                  pos_encoding=cfg["pos_encoding"], norm=cfg["norm"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model_type = "transformer"

    hyps, refs = [], []
    for src, tgt, src_lens, tgt_lens in tqdm(dl, desc="eval"):
        src, tgt, src_lens = src.to(device), tgt.to(device), src_lens.to(device)
        if model_type == "transformer":
            pred = greedy_decode_transformer(model, src, max_len=args.max_len)
        else:
            if decode_strategy == "beam":
                outs = []
                for i in range(src.size(0)):
                    s = src[i:i+1]
                    l = src_lens[i:i+1]
                    ids = beam_decode_rnn(model, s, l, beam_size=beam_size,
                                          max_len=args.max_len, length_penalty=length_penalty)
                    outs.append(torch.tensor([BOS_ID] + ids + [EOS_ID], device=src.device))
                max_len_b = max(x.size(0) for x in outs)
                pred = torch.full((len(outs), max_len_b), PAD_ID, dtype=torch.long, device=src.device)
                for i,t in enumerate(outs):
                    pred[i,:t.size(0)] = t
            else:
                pred = greedy_decode_rnn(model, src, src_lens, max_len=args.max_len)

        for i in range(pred.size(0)):
            hyp = pred[i].tolist()
            ref = tgt[i].tolist()
            hyp = hyp[1:(hyp.index(EOS_ID) if EOS_ID in hyp else len(hyp))]
            ref = ref[1:(ref.index(EOS_ID) if EOS_ID in ref else len(ref))]
            hyps.append(tok_tgt.decode(hyp))
            refs.append(tok_tgt.decode(ref))

    print("BLEU:", corpus_bleu(hyps, refs))

if __name__ == "__main__":
    main()
