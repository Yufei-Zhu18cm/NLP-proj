from collections import Counter
from dataclasses import dataclass
from typing import List, Dict

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

@dataclass
class Vocab:
    itos: List[str]
    stoi: Dict[str,int]

    @classmethod
    def build(cls, token_lists: List[List[str]], max_size=50000, min_freq=2):
        cnt = Counter()
        for toks in token_lists: cnt.update(toks)
        specials = [PAD, BOS, EOS, UNK]
        words = [w for w,f in cnt.most_common() if f >= min_freq and w not in specials]
        itos = specials + words[: max(0, max_size - len(specials))]
        stoi = {w:i for i,w in enumerate(itos)}
        return cls(itos=itos, stoi=stoi)

    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi[UNK]) for t in toks]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else UNK for i in ids]
