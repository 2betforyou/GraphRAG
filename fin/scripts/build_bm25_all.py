#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge BM25 corpora: Laws (npz+meta) + MITRE (jsonl) → one unified index.

Inputs:
  - data/index/bm25_laws.npz             # {docs: object[], corpus: object[]}
  - data/index/bm25_laws_meta.jsonl      # {"source": "..."} per line
  - data/cleaned/mitre_bm25_ko.jsonl     # {"text": "...", "source": "...", "tactic": "..."} per line

Outputs:
  - data/index/bm25_all.npz              # merged docs+corpus (object dtype)
  - data/index/bm25_all_meta.jsonl       # merged meta; aligned with docs index

Usage:
  python scripts/build_bm25_all.py [--dedup]

Notes:
  - --dedup: exact-text deduplication (recommended).
  - Tokenizer must match the one used in build_bm25.py (simple_tokenize).
"""

import argparse, json, re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi  # sanity check only (not persisted)

# -------- Paths --------
ROOT = Path(".")
LAWS_NPZ  = ROOT / "data" / "index"   / "bm25_laws.npz"
LAWS_META = ROOT / "data" / "index"   / "bm25_laws_meta.jsonl"
MITRE_JL  = ROOT / "data" / "cleaned" / "mitre_bm25_ko.jsonl"

OUT_DIR   = ROOT / "data" / "index"
OUT_NPZ   = OUT_DIR / "bm25_all.npz"
OUT_META  = OUT_DIR / "bm25_all_meta.jsonl"

# -------- Tokenizer (must mirror build_bm25.py) --------
def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    toks = [re.sub(r"^[^0-9a-z가-힣]+|[^0-9a-z가-힣]+$", "", t) for t in text.split()]
    return [t for t in toks if t]

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup", action="store_true", help="exact-text deduplication")
    args = ap.parse_args()

    assert LAWS_NPZ.exists(), f"missing: {LAWS_NPZ}"
    assert LAWS_META.exists(), f"missing: {LAWS_META}"
    assert MITRE_JL.exists(),  f"missing: {MITRE_JL}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load Laws (npz + meta) ----
    npz = np.load(LAWS_NPZ, allow_pickle=True)
    laws_docs   = list(npz["docs"].tolist())
    laws_corpus = list(npz["corpus"].tolist())
    laws_meta   = list(read_jsonl(LAWS_META))

    if not (len(laws_docs) == len(laws_corpus) == len(laws_meta)):
        raise RuntimeError(f"laws lengths mismatch: docs={len(laws_docs)} corpus={len(laws_corpus)} meta={len(laws_meta)}")

    # ---- Load MITRE (jsonl) ----
    mitre_docs, mitre_corpus, mitre_meta = [], [], []
    for obj in read_jsonl(MITRE_JL):
        txt = (obj.get("text") or "").strip()
        if not txt:
            continue
        mitre_docs.append(txt)
        mitre_corpus.append(simple_tokenize(txt))
        # keep tactic if present
        mitre_meta.append({"source": obj.get("source","mitre/enterprise-attack"), "tactic": obj.get("tactic", "")})

    # ---- Merge ----
    merged_docs   = []
    merged_corpus = []
    merged_meta   = []

    # exact dedup (optional)
    seen = set() if args.dedup else None

    def add_block(docs, corpus, meta):
        for d, tks, m in zip(docs, corpus, meta):
            if seen is not None:
                if d in seen:
                    continue
                seen.add(d)
            merged_docs.append(d)
            merged_corpus.append(tks)
            # ensure at least 'source' key exists
            if "source" not in m:
                m = {"source": m.get("source","unknown")}
            merged_meta.append(m)

    add_block(laws_docs,  laws_corpus, laws_meta)
    add_block(mitre_docs, mitre_corpus, mitre_meta)

    # ---- Sanity: build BM25 (not saved) ----
    _ = BM25Okapi(merged_corpus)

    # ---- Save ----
    np.savez_compressed(
        OUT_NPZ,
        docs=np.array(merged_docs, dtype=object),
        corpus=np.array(merged_corpus, dtype=object),
    )
    with OUT_META.open("w", encoding="utf-8") as f:
        for m in merged_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[OK] wrote -> {OUT_NPZ} (docs={len(merged_docs)})")
    print(f"[OK] meta  -> {OUT_META}")
    if args.dedup:
        print("[Info] dedup enabled")

if __name__ == "__main__":
    main()