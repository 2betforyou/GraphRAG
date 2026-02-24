#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/rag_laws.py
- data/raw/laws/*.txt 에 있는 법령 텍스트로 **로컬 RAG** 인덱스를 구축하고 검색/질의 가능
- 외부 서비스/클라우드 의존성 없이 sentence-transformers + NumPy 로 동작 (FAISS 미사용)
- 한국어 지원 다국어 임베딩(e5) 사용: intfloat/multilingual-e5-base

[설치]
pip install -r requirements_rag.txt
# 또는
pip install sentence-transformers torch tqdm numpy

[예시]
# 1) 인덱스 구축
python scripts/rag_laws.py index \
  --laws_dir data/raw/laws \
  --index_dir data/index/laws_e5 \
  --chunk_size 1200 --overlap 200

# 2) 검색만 (상위 5개 청크)
python scripts/rag_laws.py search \
  --index_dir data/index/laws_e5 \
  --question "전자문서의 법적 효력은?" \
  --k 5 --mmr

# 3) 콘솔 Q&A (컨텍스트만 생성하여 출력)
python scripts/rag_laws.py qa \
  --index_dir data/index/laws_e5 \
  --question "전자금융거래의 정의를 설명해줘." \
  --k 5 --max_context_chars 1800

# 4) 파이썬에서 사용(예: inference 코드와 연동)
from scripts.rag_laws import LawRAG
rag = LawRAG(laws_dir="data/raw/laws", index_dir="data/index/laws_e5")
rag.build_index_if_needed(chunk_size=1200, overlap=200)
ctx, hits = rag.retrieve("개인정보의 정의는 무엇인가?", k=5, mmr=True)
prompt = f"질문: ...\n\n[참고 문맥]\n{ctx}"

[디자인 노트]
- 이전에 'Killed' 문제가 있었다면, 전체 파일을 한 번에 메모리에 올리지 않고, **파일 단위로 스트리밍** 처리/임베딩합니다.
- 인덱스는 (embeddings.npy, meta.jsonl) 두 파일로 구성되어 가볍고, NumPy 행렬 곱으로 빠르게 검색합니다.
- E5 계열 권장 프롬프트를 적용합니다: query는 'query: ...', passage는 'passage: ...' 전처리.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer


# ----------------------------
# 유틸: 텍스트 청크 분할
# ----------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[Tuple[int, int, str]]:
    """
    긴 텍스트를 문자 길이 기준으로 겹치게 분할.
    return: (start_idx, end_idx, chunk_str) 리스트
    """
    text = text.replace("\u200b", "").replace("\ufeff", "")
    n = len(text)
    if n == 0:
        return []

    chunks = []
    step = max(1, max_chars - overlap)
    for start in range(0, n, step):
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= n:
            break
    return chunks


# ----------------------------
# 인덱스 메타 구조
# ----------------------------
@dataclass
class ChunkMeta:
    doc_id: int
    file: str
    chunk_id: int
    start: int
    end: int
    text: str


# ----------------------------
# RAG 본체
# ----------------------------
class LawRAG:
    def __init__(self, laws_dir: str | Path, index_dir: str | Path,
                 model_name: str = "intfloat/multilingual-e5-base", device: str | None = None):
        self.laws_dir = Path(laws_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)

        self.emb_path = self.index_dir / "embeddings.npy"
        self.meta_path = self.index_dir / "meta.jsonl"
        self._emb: np.ndarray | None = None
        self._meta: List[ChunkMeta] | None = None

    # ---------- Private IO ----------
    def _save_index(self, embeddings: np.ndarray, metas: List[ChunkMeta]) -> None:
        np.save(self.emb_path, embeddings.astype(np.float32))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    def _load_index(self) -> Tuple[np.ndarray, List[ChunkMeta]]:
        if self._emb is None:
            self._emb = np.load(self.emb_path, mmap_mode="r")
        if self._meta is None:
            metas: List[ChunkMeta] = []
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    metas.append(ChunkMeta(**obj))
            self._meta = metas
        return self._emb, self._meta

    # ---------- Build ----------
    def build_index(self, chunk_size: int = 1200, overlap: int = 200, batch_size: int = 64) -> None:
        """
        laws_dir 의 *.txt 를 파일 단위로 스트리밍하며 임베딩 → 인덱스 구성.
        embeddings.npy (float32, shape=[N, D])
        meta.jsonl     (한 줄 한 청크 메타)
        """
        files = sorted(self.laws_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in: {self.laws_dir}")

        all_embeds: List[np.ndarray] = []
        all_metas: List[ChunkMeta] = []
        dim = None
        doc_id = 0

        for path in tqdm(files, desc="Indexing laws (stream by file)"):
            try:
                text = Path(path).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = Path(path).read_text(encoding="utf-8-sig", errors="ignore")

            chunks = chunk_text(text, max_chars=chunk_size, overlap=overlap)
            if not chunks:
                continue

            # E5 권장 포맷: "passage: ..."
            passages = [f"passage: {c[2]}" for c in chunks]

            # 배치 임베딩
            embeds = self.model.encode(passages, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
            if dim is None:
                dim = embeds.shape[1]

            for j, (start, end, chunk_str) in enumerate(chunks):
                all_metas.append(ChunkMeta(
                    doc_id=doc_id, file=str(path), chunk_id=j, start=start, end=end, text=chunk_str
                ))
            all_embeds.append(embeds)
            doc_id += 1

        if not all_embeds:
            raise RuntimeError("No embeddings produced. Check input texts.")

        emb_mat = np.vstack(all_embeds).astype(np.float32)
        self._save_index(emb_mat, all_metas)
        # 캐시 메모리 초기화
        self._emb, self._meta = None, None

    def build_index_if_needed(self, **kwargs) -> None:
        if self.emb_path.exists() and self.meta_path.exists():
            return
        self.build_index(**kwargs)

    # ---------- Search ----------
    def _encode_query(self, question: str) -> np.ndarray:
        q = f"query: {question}"
        q_emb = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        return q_emb

    def _cosine_topk(self, q_emb: np.ndarray, emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # emb, q_emb 는 모두 L2 정규화되어 있으므로 내적 == cosine
        sims = emb @ q_emb  # [N]
        idx = np.argpartition(-sims, kth=min(k, sims.shape[0]-1))[:k]
        # 점수 내림차순 정렬
        order = idx[np.argsort(-sims[idx])]
        return order, sims[order]

    def _mmr(self, q_emb: np.ndarray, emb: np.ndarray, k: int, lambda_mult: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        # 간단한 MMR 구현 (중복 최소화)
        sims = emb @ q_emb  # [N]
        selected: List[int] = []
        candidates = np.arange(emb.shape[0]).tolist()

        if len(candidates) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        # 첫 번째: 가장 유사한 것
        first = int(np.argmax(sims))
        selected.append(first)
        candidates.remove(first)

        # 사전 계산
        emb_sel = [emb[first]]
        while len(selected) < min(k, emb.shape[0]):
            # 각 후보에 대해: lambda*sim(q, d) - (1-lambda)*max_sim(d, selected)
            cand_mat = emb[candidates]
            # 유사도 계산
            rel = cand_mat @ q_emb  # [C]
            red = np.max(cand_mat @ np.stack(emb_sel, axis=0).T, axis=1)  # [C]
            mmr_scores = lambda_mult * rel - (1.0 - lambda_mult) * red
            best_i = int(np.argmax(mmr_scores))
            best_global_idx = candidates[best_i]
            selected.append(best_global_idx)
            emb_sel.append(emb[best_global_idx])
            candidates.pop(best_i)

        selected = np.array(selected, dtype=int)
        return selected, sims[selected]

    def retrieve(self, question: str, k: int = 5, mmr: bool = True, lambda_mult: float = 0.5) -> Tuple[str, List[Dict[str, Any]]]:
        emb, metas = self._load_index()
        q_emb = self._encode_query(question)

        if mmr:
            idx, scores = self._mmr(q_emb, emb, k=k, lambda_mult=lambda_mult)
        else:
            idx, scores = self._cosine_topk(q_emb, emb, k=k)

        hits = []
        for rank, (i, s) in enumerate(zip(idx, scores), start=1):
            m = metas[int(i)]
            hits.append({
                "rank": rank,
                "score": float(s),
                "file": m.file,
                "chunk_id": m.chunk_id,
                "start": m.start,
                "end": m.end,
                "text": m.text
            })

        context = "\n\n".join([f"[{h['rank']}] ({Path(h['file']).name} #{h['chunk_id']})\n{h['text']}" for h in hits])
        return context, hits

    # ---------- Formatting ----------
    @staticmethod
    def format_context_for_prompt(context: str, max_chars: int = 2000) -> str:
        """
        모델 프롬프트에 바로 붙이기 좋은 형태로 컨텍스트 길이를 제한하여 반환
        """
        if len(context) <= max_chars:
            return context
        # 너무 길면 앞부분 위주로 (법령은 조문 구조이므로 앞쪽 맥락도 유용)
        return context[:max_chars].rstrip() + "\n…(생략)"


# ----------------------------
# CLI
# ----------------------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local RAG over data/raw/laws/*.txt (no FAISS)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="인덱스 구축")
    p_index.add_argument("--laws_dir", type=str, default="data/raw/laws")
    p_index.add_argument("--index_dir", type=str, default="data/index/laws_e5")
    p_index.add_argument("--chunk_size", type=int, default=1200)
    p_index.add_argument("--overlap", type=int, default=200)
    p_index.add_argument("--batch_size", type=int, default=64)

    p_search = sub.add_parser("search", help="문서 검색(컨텍스트 미리보기)")
    p_search.add_argument("--index_dir", type=str, default="data/index/laws_e5")
    p_search.add_argument("--question", type=str, required=True)
    p_search.add_argument("--k", type=int, default=5)
    p_search.add_argument("--mmr", action="store_true", help="MMR 사용 (중복 최소화)")
    p_search.add_argument("--no-mmr", dest="mmr", action="store_false", help="MMR 비활성화")
    p_search.set_defaults(mmr=True)

    p_qa = sub.add_parser("qa", help="컨텍스트만 생성(프롬프트에 붙여 쓰기)")
    p_qa.add_argument("--index_dir", type=str, default="data/index/laws_e5")
    p_qa.add_argument("--question", type=str, required=True)
    p_qa.add_argument("--k", type=int, default=5)
    p_qa.add_argument("--mmr", action="store_true", help="MMR 사용 (중복 최소화)")
    p_qa.add_argument("--no-mmr", dest="mmr", action="store_false", help="MMR 비활성화")
    p_qa.add_argument("--max_context_chars", type=int, default=2000)
    p_qa.set_defaults(mmr=True)

    return p


def main():
    parser = build_cli()
    args = parser.parse_args()

    if args.cmd == "index":
        rag = LawRAG(laws_dir=args.laws_dir, index_dir=args.index_dir)
        print(f"[INFO] Building index from {args.laws_dir} -> {args.index_dir}")
        rag.build_index(chunk_size=args.chunk_size, overlap=args.overlap, batch_size=args.batch_size)
        print("[INFO] Done. Files:", rag.emb_path, rag.meta_path)

    elif args.cmd == "search":
        rag = LawRAG(laws_dir=".", index_dir=args.index_dir)  # laws_dir는 필요 없음
        if not (Path(args.index_dir) / "embeddings.npy").exists():
            print("[WARN] Index not found. Try building with the 'index' command first.")
            return
        context, hits = rag.retrieve(args.question, k=args.k, mmr=args.mmr)
        print("=== Top-K Chunks ===")
        for h in hits:
            print(f"[{h['rank']}] score={h['score']:.4f} file={h['file']} chunk={h['chunk_id']} "
                  f"({h['start']}:{h['end']})")
        print("\n--- Context Preview ---\n")
        print(context)

    elif args.cmd == "qa":
        rag = LawRAG(laws_dir=".", index_dir=args.index_dir)
        if not (Path(args.index_dir) / "embeddings.npy").exists():
            print("[WARN] Index not found. Try building with the 'index' command first.")
            return
        context, hits = rag.retrieve(args.question, k=args.k, mmr=args.mmr)
        ctx = LawRAG.format_context_for_prompt(context, max_chars=args.max_context_chars)
        print(ctx)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()