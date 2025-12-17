import os
from dataclasses import dataclass
from typing import List
import sentencepiece as spm

try:
    import jieba
except Exception:
    jieba = None

@dataclass
class SpmTokenizer:
    model_path: str

    def __post_init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.model_path)

    @property
    def vocab_size(self): return self.sp.GetPieceSize()
    def encode(self, text: str) -> List[int]: return self.sp.EncodeAsIds(text)
    def decode(self, ids: List[int]) -> str: return self.sp.DecodeIds([int(x) for x in ids])

class FallbackZhTokenizer:
    # jieba word seg -> tokens, then use an on-the-fly vocab in training (not used if SPM provided)
    def encode_tokens(self, text: str) -> List[str]:
        if jieba is None:
            return list(text.strip())
        return [t for t in jieba.lcut(text) if t.strip()]

class FallbackEnTokenizer:
    def encode_tokens(self, text: str) -> List[str]:
        return [t for t in text.strip().split() if t]

def has_spm(path: str) -> bool:
    return path is not None and os.path.exists(path)
