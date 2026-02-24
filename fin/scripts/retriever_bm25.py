
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight BM25 retriever API:
- load_index(path) -> retriever
- retriever.search(query, topk=5) -> list of dicts: {"text","score","source"}
"""
import re, json, numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List
from rank_bm25 import BM25Okapi

def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    toks = [re.sub(r"^[^0-9a-z가-힣]+|[^0-9a-z가-힣]+$", "", t) for t in text.split()]
    return [t for t in toks if t]

@dataclass
class BM25Retriever:
    docs: list
    corpus: list
    bm25: BM25Okapi
    sources: list

    def search(self, query: str, topk: int = 5):
        q = simple_tokenize(query)
        scores = self.bm25.get_scores(q)
        idxs = np.argsort(scores)[::-1][:topk]
        return [{"text": self.docs[i], "score": float(scores[i]), "source": self.sources[i]} for i in idxs]

# retriever_bm25.py (발췌)
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

DEFAULT_NPZ  = Path("data/index/bm25_all.npz")          # ← 통합 인덱스
DEFAULT_META = Path("data/index/bm25_all_meta.jsonl")

def load_index(base_dir="."):
    npz_path  = Path(base_dir) / "data/index/bm25_all.npz"
    meta_path = Path(base_dir) / "data/index/bm25_all_meta.jsonl"

    arr = np.load(npz_path, allow_pickle=True)
    docs   = arr["docs"].tolist()
    corpus = arr["corpus"].tolist()
    bm25   = BM25Okapi(corpus)

    # (선택) 메타 로드
    meta = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    def _search(query, topk=5):
        q_toks = simple_tokenize(query)
        scores = bm25.get_scores(q_toks)
        top_idx = np.argsort(scores)[::-1][:topk]
        out = []
        for i in top_idx:
            out.append({"text": docs[i], "score": float(scores[i]), "meta": meta[i] if i < len(meta) else {}})
        return out

    class Retriever:
        def search(self, q, topk=5):
            return _search(q, topk)

    return Retriever()
