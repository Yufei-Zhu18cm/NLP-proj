import torch
from torch.utils.data import Dataset
from typing import Optional
from nmt.utils import read_jsonl, pick_parallel_fields, clean_text

BOS_ID = 1
EOS_ID = 2
PAD_ID = 0

class JsonlParallelDataset(Dataset):
    def __init__(self, path: str, tok_src, tok_tgt, max_len=128, limit: Optional[int]=None):
        self.samples = []
        for i,obj in enumerate(read_jsonl(path)):
            if limit is not None and i >= limit: break
            src, tgt = pick_parallel_fields(obj)
            src, tgt = clean_text(src), clean_text(tgt)
            self.samples.append((src, tgt))
        self.tok_src, self.tok_tgt = tok_src, tok_tgt
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def _encode(self, tok, text: str):
        ids = tok.encode(text)
        ids = ids[: self.max_len - 2]
        return [BOS_ID] + ids + [EOS_ID]

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        return torch.tensor(self._encode(self.tok_src, src), dtype=torch.long), \
               torch.tensor(self._encode(self.tok_tgt, tgt), dtype=torch.long)

def collate_batch(batch):
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(x) for x in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgts], dtype=torch.long)
    src_max = int(src_lens.max().item())
    tgt_max = int(tgt_lens.max().item())

    src_pad = torch.full((len(batch), src_max), 0, dtype=torch.long)
    tgt_pad = torch.full((len(batch), tgt_max), 0, dtype=torch.long)

    for i,(s,t) in enumerate(zip(srcs,tgts)):
        src_pad[i,:len(s)] = s
        tgt_pad[i,:len(t)] = t
    return src_pad, tgt_pad, src_lens, tgt_lens
