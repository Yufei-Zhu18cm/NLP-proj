import json, re
from typing import Dict, Tuple

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def pick_parallel_fields(obj: Dict) -> Tuple[str,str]:
    # Try common key pairs
    candidates = [
        ("zh","en"), ("cn","en"),
        ("src","tgt"), ("source","target"),
        ("chinese","english"),
    ]
    for a,b in candidates:
        if a in obj and b in obj and isinstance(obj[a], str) and isinstance(obj[b], str):
            return obj[a], obj[b]
    # Fallback: first two string fields
    strings = [(k,v) for k,v in obj.items() if isinstance(v, str)]
    if len(strings) >= 2:
        return strings[0][1], strings[1][1]
    raise ValueError(f"Cannot find parallel fields in: {obj}")

def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s
