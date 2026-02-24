#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dapt_jsonl.py
- 입력: cleaned/*.cleaned_lines.txt
- 출력: /mnt/data/dapt_laws.jsonl (한 줄 = {"text": "..."} )
- 약 1,800자 단위로 라인들을 묶어 청크 생성

python make_dapt_jsonl.py
"""
from pathlib import Path
import json, re

ROOT = Path("./data")
CLEAN_DIR = ROOT / "cleaned"
OUT_JSONL = ROOT / "./cleaned/dapt_laws.jsonl"

def load_lines(p: Path):
    txt = p.read_text(encoding="utf-8")
    txt = re.sub(r"\n{2,}", "\n", txt)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return lines

def main():
    files = sorted(CLEAN_DIR.glob("*.cleaned_lines.txt"))
    if not files:
        raise FileNotFoundError(f"No *.cleaned_lines.txt under {CLEAN_DIR}")
    char_limit = 1800
    written = 0
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for p in files:
            buf, cur_len = [], 0
            for ln in load_lines(p):
                ln_len = len(ln) + 1
                if cur_len + ln_len > char_limit and buf:
                    text = " ".join(buf).strip()
                    if text:
                        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        written += 1
                    buf, cur_len = [ln], len(ln)
                else:
                    buf.append(ln); cur_len += ln_len
            if buf:
                text = " ".join(buf).strip()
                if text:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
    print(f"[OK] wrote {written} docs -> {OUT_JSONL}")

if __name__ == "__main__":
    main()