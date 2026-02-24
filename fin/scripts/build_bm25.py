#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a BM25 index from cleaned law corpus (excluding TTA).
- Input : data/cleaned/*.cleaned_lines.txt
- Exclude: files containing "tta" in filename
- Output: data/index/bm25_laws.npz  (docs + corpus tokenized, object dtype)
         data/index/bm25_laws_meta.jsonl (per-line source metadata)

Dependency: pip install rank-bm25
"""
import re, json, numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

ROOT = Path(".")
SRC_DIR = ROOT / "data" / "cleaned"
OUT_DIR = ROOT / "data" / "index"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "bm25_laws.npz"
META_PATH = OUT_DIR / "bm25_laws_meta.jsonl"

def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    toks = [re.sub(r"^[^0-9a-z가-힣]+|[^0-9a-z가-힣]+$", "", t) for t in text.split()]
    return [t for t in toks if t]

def main():
    files = sorted(p for p in SRC_DIR.glob("*.cleaned_lines.txt") if "tta" not in p.stem.lower())
    if not files:
        raise FileNotFoundError(f"No law files found under {SRC_DIR}")

    docs, meta = [], []
    for p in files:
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            docs.append(ln)
            meta.append({"source": p.name})

    # Tokenize once here; retriever가 corpus로 BM25를 재구성합니다.
    corpus = [simple_tokenize(d) for d in docs]
    _ = BM25Okapi(corpus)  # sanity check만. 결과는 저장하지 않음.

    # 중요: object dtype으로 저장 (list of list)
    np.savez_compressed(
        OUT_PATH,
        docs=np.array(docs, dtype=object),
        corpus=np.array(corpus, dtype=object),
    )
    with META_PATH.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[OK] BM25 index saved -> {OUT_PATH}")
    print(f"[OK] Meta saved        -> {META_PATH}")
    print(f"[Info] #lines indexed  : {len(docs)} | #files: {len(files)}")

if __name__ == "__main__":
    main()