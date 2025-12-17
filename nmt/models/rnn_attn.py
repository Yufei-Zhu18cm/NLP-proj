import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    # alignment: dot | general | additive
    def __init__(self, hidden_size, alignment="additive"):
        super().__init__()
        self.alignment = alignment
        if alignment == "general":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif alignment == "additive":
            self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, key_mask=None):
        # query: (B,H), keys: (B,T,H)
        if self.alignment == "dot":
            scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # (B,T)
        elif self.alignment == "general":
            q = self.W(query)  # (B,H)
            scores = torch.bmm(keys, q.unsqueeze(-1)).squeeze(-1)
        else:  # additive (Bahdanau)
            q = self.Wq(query).unsqueeze(1)  # (B,1,H)
            k = self.Wk(keys)                # (B,T,H)
            scores = self.v(torch.tanh(q + k)).squeeze(-1)  # (B,T)

        if key_mask is not None:
            scores = scores.masked_fill(~key_mask, -1e9)

        attn = F.softmax(scores, dim=-1)                 # (B,T)
        ctx = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)  # (B,H)
        return ctx, attn

class Seq2SeqRNN(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb=256, hidden=256, num_layers=2,
                 rnn_cell="lstm", attn="additive", dropout=0.1):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.emb_src = nn.Embedding(src_vocab, emb, padding_idx=0)
        self.emb_tgt = nn.Embedding(tgt_vocab, emb, padding_idx=0)
        RNN = nn.LSTM if rnn_cell.lower() == "lstm" else nn.GRU
        self.encoder = RNN(emb, hidden, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout if num_layers>1 else 0.0)
        self.decoder = RNN(emb, hidden, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout if num_layers>1 else 0.0)
        self.attn = Attention(hidden, alignment=attn)
        self.out = nn.Linear(hidden + hidden, tgt_vocab)
        self.dropout = nn.Dropout(dropout)
        self.rnn_cell = rnn_cell.lower()

    def encode(self, src, src_lens):
        emb = self.dropout(self.emb_src(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        enc_out, enc_state = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)  # (B,T,H)
        return enc_out, enc_state

    def _dec_init(self, enc_state):
        return enc_state  # same hidden size

    def forward(self, src, src_lens, tgt_inp, teacher_forcing=1.0):
        # tgt_inp includes BOS ... EOS
        B, Tt = tgt_inp.shape
        enc_out, enc_state = self.encode(src, src_lens)
        dec_state = self._dec_init(enc_state)

        src_mask = (src != 0)
        logits = []
        prev = tgt_inp[:, 0:1]  # BOS
        for t in range(1, Tt):
            emb = self.dropout(self.emb_tgt(prev))
            dec_out, dec_state = self.decoder(emb, dec_state)  # (B,1,H)
            h = dec_out.squeeze(1)
            ctx, _ = self.attn(h, enc_out, key_mask=src_mask[:, :enc_out.size(1)])
            logit = self.out(torch.cat([h, ctx], dim=-1))  # (B,V)
            logits.append(logit.unsqueeze(1))

            use_tf = (torch.rand(1).item() < teacher_forcing)
            if use_tf:
                prev = tgt_inp[:, t:t+1]
            else:
                prev = logit.argmax(dim=-1, keepdim=True)

        return torch.cat(logits, dim=1)  # (B, Tt-1, V)
