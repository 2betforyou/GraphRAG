# -*- coding: utf-8 -*-
# Build a unified RAG index from .txt and .jsonl
# - Embeddings: sentence-transformers (E5)
# - Vector index: FAISS
# - (Optional) BM25: rank_bm25
import json, argparse, pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from utils import read_text_file, iter_jsonl, make_chunks, clean_text, require_cuda_or_fail

def glob_sources(patterns: List[str]) -> List[Path]:
    files = []
    for p in patterns:
        for x in Path().glob(p):
            if x.is_file() and x.suffix.lower() in {".txt", ".jsonl"}:
                files.append(x.resolve())
    return sorted(set(files))

def build_corpus(files: List[Path], chunk_chars: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    corpus = []
    uid = 0
    for f in files:
        if f.suffix.lower() == ".txt":
            try:
                raw = read_text_file(str(f))
                chunks = make_chunks(raw, chunk_chars, chunk_overlap)
                for ch, a, b in chunks:
                    corpus.append({
                        "id": f"{f.name}:::{uid}",
                        "text": ch,
                        "source": str(f),
                        "start": int(a),
                        "end": int(b),
                        "extra": {}
                    })
                    uid += 1
            except Exception as e:
                print(f"[WARN] txt load fail: {f} → {e}")
        elif f.suffix.lower() == ".jsonl":
            for j in iter_jsonl(str(f)):
                text = j.get("text") or j.get("passage") or j.get("content") or ""
                text = clean_text(str(text))
                if not text:
                    continue
                corpus.append({
                    "id": f"{f.name}:::{uid}",
                    "text": text,
                    "source": j.get("source", str(f)),
                    "start": -1,
                    "end": -1,
                    "extra": {k:v for k,v in j.items() if k not in {"text"}}
                })
                uid += 1
    return corpus

def embed_corpus(corpus_texts: List[str], model_name: str, device: str) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    enc_texts = [f"passage: {t}" for t in corpus_texts]
    emb = model.encode(
        enc_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")
    return emb

def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)   # cosine via inner product on normalized embeddings
    index.add(embeddings)
    return index

def build_bm25(corpus_texts: List[str]):
    from rank_bm25 import BM25Okapi
    def tok(s: str):
        # 간단 토크나이저: whitespace (Konlpy 미설치 환경 호환)
        return s.split()
    tokenized = [tok(t) for t in corpus_texts]
    return BM25Okapi(tokenized)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="Glob patterns for .txt/.jsonl")
    ap.add_argument("--index-dir", type=str, required=True)
    ap.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-base")
    ap.add_argument("--chunk-chars", type=int, default=900)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--bm25", action="store_true", help="Build BM25 index additionally")
    ap.add_argument("--gpu-only", action="store_true", help="Require CUDA, fail if not available")
    args = ap.parse_args()

    require_cuda_or_fail(args.gpu_only)

    out_dir = Path(args.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = glob_sources(args.sources)
    if not files:
        raise SystemExit("No source files matched.")

    print(f"[BUILD] {len(files)} files → build corpus ...")
    corpus = build_corpus(files, args.chunk_chars, args.chunk_overlap)
    print(f"[BUILD] chunks: {len(corpus)}")

    texts = [c["text"] for c in corpus]

    # Embeddings
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EMBED] model={args.embedding_model}, device={device}")
    emb = embed_corpus(texts, args.embedding_model, device)

    # FAISS
    print("[FAISS] building index ...")
    index = build_faiss(emb)

    # Save
    np.save(out_dir / "embeddings.npy", emb)
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    faiss.write_index(index, str(out_dir / "faiss.index"))

    meta = {
        "embedding_model": args.embedding_model,
        "embed_dim": int(emb.shape[1]),
        "chunk_chars": args.chunk_chars,
        "chunk_overlap": args.chunk_overlap,
        "count": int(len(corpus)),
        "bm25": bool(args.bm25),
    }
    with open(out_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.bm25:
        print("[BM25] building ...")
        bm25 = build_bm25(texts)
        import pickle
        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)
        print("[BM25] saved.")

    print("[DONE] index saved to:", out_dir)

if __name__ == "__main__":
    main()