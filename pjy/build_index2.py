# -*- coding: utf-8 -*-
# Build two separate RAG indices:
#  - Laws index from .txt files
#  - MITRE index from .jsonl
# Embeddings: sentence-transformers (E5)
# Vector index: FAISS (FlatIP; normalized cos)
# (Optional) BM25: rank_bm25
import json, argparse, pickle, re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from utils import read_text_file, iter_jsonl, make_chunks, clean_text, require_cuda_or_fail

LAW_NAME_PAT = re.compile(r"(법|시행령|시행규칙)")
LAW_ARTICLE_PAT = re.compile(r"(제\s?\d+\s?조)(?:\s*제?\s?\d+\s?항)?")
MITRE_TECH_PAT = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
MITRE_TAC_PAT = re.compile(r"\bTA\d{4}\b")

def glob_sources(patterns: List[str]) -> List[Path]:
    files = []
    for p in patterns:
        for x in Path().glob(p):
            if x.is_file() and x.suffix.lower() in {".txt", ".jsonl"}:
                files.append(x.resolve())
    return sorted(set(files))

def infer_domain_meta(text: str, source_hint: str) -> Dict[str, Any]:
    meta = {"domain": "other"}
    s = f"{source_hint}\n{text}".lower()

    # 1) MITRE / enterprise-attack 우선
    if any(k in s for k in [
        "attack.mitre.org", "enterprise-attack", "ics-attack", "mobile-attack",
        "mitre att&ck", "mitre attack", "mitre-attack"
    ]):
        meta["domain"] = "mitre"
        techs = list(set(MITRE_TECH_PAT.findall(text)))
        tacs  = list(set(MITRE_TAC_PAT.findall(text)))
        if techs: meta["techniques"] = techs
        if tacs:  meta["tactics"] = tacs
        return meta

    # 2) 본문 기반 MITRE 패턴
    techs = list(set(MITRE_TECH_PAT.findall(text)))
    tacs  = list(set(MITRE_TAC_PAT.findall(text)))
    if techs or tacs or "att&ck" in s or "technique" in s:
        meta["domain"] = "mitre"
        if techs: meta["techniques"] = techs
        if tacs:  meta["tactics"] = tacs
        return meta

    # 3) 법령 키워드
    if any(k in s for k in ["법", "시행령", "시행규칙", "대통령령", "총리령", "부령"]):
        meta["domain"] = "law"
        arts = list(set(re.findall(r"(제\s?\d+\s?조(?:\s*제?\s?\d+\s?항)?)", text)))
        if arts: meta["articles"] = [a.replace(" ", "") for a in arts]
        return meta

    return meta

def build_corpus_laws(files: List[Path], chunk_chars: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    corpus, uid = [], 0
    for f in files:
        if f.suffix.lower() != ".txt":
            continue
        try:
            raw = read_text_file(str(f))
            chunks = make_chunks(raw, chunk_chars, chunk_overlap)
            for ch, a, b in chunks:
                base = {
                    "id": f"{f.name}:::{uid}",
                    "text": clean_text(ch),
                    "source": str(f),
                    "start": int(a),
                    "end": int(b),
                }
                base["extra"] = infer_domain_meta(ch, str(f)) | {"domain": "law"}
                corpus.append(base); uid += 1
        except Exception as e:
            print(f"[WARN] txt load fail: {f} → {e}")
    return corpus

def build_corpus_mitre(jsonl_files: List[Path]) -> List[Dict[str, Any]]:
    corpus, uid = [], 0
    for f in jsonl_files:
        if f.suffix.lower() != ".jsonl":
            continue
        for j in iter_jsonl(str(f)):
            text = j.get("text") or j.get("passage") or j.get("content") or ""
            text = clean_text(str(text))
            if not text:
                continue
            base = {
                "id": f"{f.name}:::{uid}",
                "text": text,
                "source": j.get("source", str(f)),
                "start": -1,
                "end": -1,
            }
            extra = {k:v for k,v in j.items() if k not in {"text"}}
            extra.update(infer_domain_meta(text, base["source"]))
            extra["domain"] = "mitre"
            base["extra"] = extra
            corpus.append(base); uid += 1
    return corpus

def embed_corpus(corpus_texts: List[str], model_name: str, device: str) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    enc_texts = [f"passage: {t}" for t in corpus_texts]  # E5 규약
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
    def tok(s: str): return s.split()  # 간단 공백 토크나이저
    tokenized = [tok(t) for t in corpus_texts]
    return BM25Okapi(tokenized)

def save_index(out_dir: Path, emb: np.ndarray, index, corpus: list, embedding_model: str, chunk_chars: int, chunk_overlap: int, bm25_obj=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", emb)
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    faiss.write_index(index, str(out_dir / "faiss.index"))
    meta = {
        "embedding_model": embedding_model,
        "embed_dim": int(emb.shape[1]),
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "count": int(len(corpus)),
        "bm25": bool(bm25_obj is not None),
        "schema_version": 2,
    }
    with open(out_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if bm25_obj is not None:
        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25_obj, f)

def main():
    ap = argparse.ArgumentParser()
    # 입력
    ap.add_argument("--laws_glob", nargs="+", required=True, help="법령 .txt 글롭 패턴 (예: ../dataset/laws/cleaned/*.txt)")
    ap.add_argument("--mitre_glob", nargs="+", required=True, help="MITRE .jsonl 글롭 패턴 (예: ../dataset/all/mitre_bm25_ko_cleaned2.jsonl)")
    # 출력
    ap.add_argument("--laws_index_dir", type=str, required=True, help="법령 인덱스 출력 폴더 (예: ../dataset/index/law_rag)")
    ap.add_argument("--mitre_index_dir", type=str, required=True, help="MITRE 인덱스 출력 폴더 (예: ../dataset/index/mitre_rag)")
    # 공통 설정
    ap.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-base")
    ap.add_argument("--chunk-chars", type=int, default=900)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--bm25", action="store_true", help="BM25 인덱스도 함께 생성")
    ap.add_argument("--gpu-only", action="store_true", help="CUDA 필요 시 강제")
    args = ap.parse_args()

    require_cuda_or_fail(args.gpu_only)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 1) Laws =====
    law_files = [p for p in glob_sources(args.laws_glob) if p.suffix.lower()==".txt"]
    if not law_files:
        raise SystemExit("[Laws] No .txt matched.")
    print(f"[Laws] {len(law_files)} files → build corpus ...")
    laws_corpus = build_corpus_laws(law_files, args.chunk_chars, args.chunk_overlap)
    print(f"[Laws] chunks: {len(laws_corpus)}")

    print(f"[Laws-EMBED] model={args.embedding-model if hasattr(args,'embedding-model') else args.embedding_model}, device={device}")
    # 위 줄은 하이픈 속성 접근 이슈를 피하기 위해 다음 줄로 대체
    print(f"[Laws-EMBED] model={args.embedding_model}, device={device}")
    laws_texts = [c["text"] for c in laws_corpus]
    laws_emb = embed_corpus(laws_texts, args.embedding_model, device)
    print("[Laws-FAISS] building index ...")
    laws_index = build_faiss(laws_emb)
    laws_bm25 = build_bm25(laws_texts) if args.bm25 else None
    save_index(Path(args.laws_index_dir), laws_emb, laws_index, laws_corpus,
               args.embedding_model, args.chunk_chars, args.chunk_overlap, laws_bm25)
    print("[Laws] DONE →", args.laws_index_dir)

    # ===== 2) MITRE =====
    mitre_files = [p for p in glob_sources(args.mitre_glob) if p.suffix.lower()==".jsonl"]
    if not mitre_files:
        raise SystemExit("[MITRE] No .jsonl matched.")
    print(f"[MITRE] {len(mitre_files)} files → build corpus ...")
    mitre_corpus = build_corpus_mitre(mitre_files)
    print(f"[MITRE] chunks: {len(mitre_corpus)}")

    print(f"[MITRE-EMBED] model={args.embedding_model}, device={device}")
    mitre_texts = [c["text"] for c in mitre_corpus]
    mitre_emb = embed_corpus(mitre_texts, args.embedding_model, device)
    print("[MITRE-FAISS] building index ...")
    mitre_index = build_faiss(mitre_emb)
    mitre_bm25 = build_bm25(mitre_texts) if args.bm25 else None
    save_index(Path(args.mitre_index_dir), mitre_emb, mitre_index, mitre_corpus,
               args.embedding_model, args.chunk_chars, args.chunk_overlap, mitre_bm25)
    print("[MITRE] DONE →", args.mitre_index_dir)

    print("[DONE] Both indices saved.")


if __name__ == "__main__":
    main()