
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert TTA term-definition JSONL to SFT JSONL for instruction-tuning.

Input:
  data/cleaned/tta_terms.jsonl
    {"term": "...", "definition": "..."}

Output:
  data/cleaned/tta_sft.jsonl
    {"instruction": "...", "input": "", "output": "..."}

Notes:
- Create multiple instruction templates for variety (definition / 핵심 요지 / 특징)
- Keep answers concise (1~3 sentences).
"""
import json, random
from pathlib import Path

random.seed(42)

ROOT = Path(".")
SRC = ROOT / "data" / "cleaned" / "tta_terms.jsonl"
OUT = ROOT / "data" / "cleaned" / "tta_sft.jsonl"

TEMPLATES = [
    "다음 용어를 한국어로 간결하게 정의하시오: {term}",
    "아래 용어의 의미를 한두 문장으로 설명하시오: {term}",
    "다음 보안 용어의 핵심 요지를 요약하시오: {term}",
    "보안 전문가의 관점에서 다음 용어를 설명하시오: {term}",
]

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"not found: {SRC}")
    cnt_in, cnt_out = 0, 0
    with OUT.open("w", encoding="utf-8") as fout:
        for line in SRC.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            obj = json.loads(line)
            term = obj.get("term","").strip()
            definition = obj.get("definition","").strip()
            if not term or not definition: 
                continue
            inst = random.choice(TEMPLATES).format(term=term)
            rec = {"instruction": inst, "input": "", "output": definition}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cnt_in += 1; cnt_out += 1
    print(f"[OK] {cnt_out} SFT records written -> {OUT}")

if __name__ == "__main__":
    main()
