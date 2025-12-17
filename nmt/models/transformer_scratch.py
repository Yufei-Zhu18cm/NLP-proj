import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # x: (..., d)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.scale

def make_norm(kind, d):
    kind = kind.lower()
    if kind == "rmsnorm":
        return RMSNorm(d)
    return nn.LayerNorm(d)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, kind="sinusoidal"):
        super().__init__()
        self.kind = kind.lower()
        if self.kind == "learned":
            self.pe = nn.Embedding(max_len, d_model)
        elif self.kind == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe_buf", pe, persistent=False)

    def forward(self, x):
        # x: (B,T,D)
        if self.kind == "none":
            return x
        B,T,D = x.shape
        if self.kind == "learned":
            pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            return x + self.pe(pos)
        # sinusoidal
        return x + self.pe_buf[:T].unsqueeze(0).to(x.device)

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        # x: (B, heads, T, dim)
        T = x.size(-2)
        t = torch.arange(T, device=x.device).float()
        freqs = torch.einsum("t,d->td", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return (x * cos) + (rotate_half(x) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_rope=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.d_head) if use_rope else None

    def forward(self, x, kv=None, attn_mask=None, key_padding_mask=None):
        # x: (B,T,D), kv: (B,S,D) if cross-attn else None
        if kv is None:
            kv = x
        B,T,D = x.shape
        S = kv.size(1)

        q = self.qkv(x)[..., :D]
        k = self.qkv(kv)[..., D:2*D]
        v = self.qkv(kv)[..., 2*D:]

        def split(t, L):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1,2)  # (B,h,L,dh)
        q = split(q, T); k = split(k, S); v = split(v, S)

        if self.use_rope and self.rope is not None and (kv is x):
            q = self.rope(q); k = self.rope(k)

        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_head)  # (B,h,T,S)

        if attn_mask is not None:
            scores = scores + attn_mask  # broadcastable

        if key_padding_mask is not None:
            # key_padding_mask: (B,S) True for keep? We'll treat 1 for keep.
            mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B,h,T,dh)
        out = out.transpose(1,2).contiguous().view(B,T,D)
        return self.proj(out)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm="layernorm", use_rope=False):
        super().__init__()
        self.norm1 = make_norm(norm, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_rope=use_rope)
        self.norm2 = make_norm(norm, d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h, kv=None, key_padding_mask=src_key_padding_mask))
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm="layernorm", use_rope=False):
        super().__init__()
        self.norm1 = make_norm(norm, d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_rope=use_rope)
        self.norm2 = make_norm(norm, d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, use_rope=False)
        self.norm3 = make_norm(norm, d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_attn_mask=None, tgt_key_padding_mask=None, src_key_padding_mask=None):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, kv=None, attn_mask=tgt_attn_mask, key_padding_mask=tgt_key_padding_mask))
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, kv=mem, key_padding_mask=src_key_padding_mask))
        h = self.norm3(x)
        x = x + self.dropout(self.ffn(h))
        return x

class TransformerScratch(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, n_heads=4, n_layers=4, d_ff=1024,
                 dropout=0.1, pos_encoding="sinusoidal", norm="layernorm"):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.d_model = d_model
        use_rope = (pos_encoding.lower() == "rotary")
        self.pos = PositionalEncoding(d_model, kind=("none" if use_rope else pos_encoding))
        self.emb_src = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.emb_tgt = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.enc = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, norm=norm, use_rope=use_rope) for _ in range(n_layers)])
        self.dec = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout, norm=norm, use_rope=use_rope) for _ in range(n_layers)])
        self.norm_out = make_norm(norm, d_model)
        self.proj = nn.Linear(d_model, tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, T, device):
        # additive mask: (1,1,T,T)
        m = torch.full((T,T), float("-inf"), device=device)
        m = torch.triu(m, diagonal=1)
        return m.unsqueeze(0).unsqueeze(0)

    def encode(self, src, src_key_padding_mask):
        x = self.emb_src(src) * math.sqrt(self.d_model)
        x = self.dropout(self.pos(x))
        for layer in self.enc:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def forward(self, src, tgt_inp):
        # src: (B,S), tgt_inp: (B,T) includes BOS.. (shifted input)
        src_kpm = (src != 0)
        tgt_kpm = (tgt_inp != 0)
        mem = self.encode(src, src_kpm)
        x = self.emb_tgt(tgt_inp) * math.sqrt(self.d_model)
        x = self.dropout(self.pos(x))
        mask = self._causal_mask(tgt_inp.size(1), tgt_inp.device)
        for layer in self.dec:
            x = layer(x, mem, tgt_attn_mask=mask, tgt_key_padding_mask=tgt_kpm, src_key_padding_mask=src_kpm)
        x = self.norm_out(x)
        return self.proj(x)
