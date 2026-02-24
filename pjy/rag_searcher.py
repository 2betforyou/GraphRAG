# -*- coding: utf-8 -*-
import json, math, pickle, heapq, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# (선택) Cross-Encoder 재랭커가 있으면 활성화
HAS_RERANKER = False
try:
    from FlagEmbedding import FlagReranker
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False

E5_QUERY_PREFIX = "query: "
LAW_KEYWORDS = re.compile(r"(법|시행령|시행규칙|제\s?\d+\s?조|대통령령|총리령|부령)")
MITRE_KEYS     = re.compile(r"\b(TA\d{4}|T\d{4}(?:\.\d{3})?|ATT&CK)\b", re.I)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, cand_ids: List[int], k: int, lambda_: float=0.65) -> List[int]:
    """Maximal Marginal Relevance for 다양화"""
    selected, used = [], set()
    if not len(cand_ids): return selected

    # 시작점: 최고 점수
    selected.append(cand_ids[0]); used.add(cand_ids[0])
    while len(selected) < min(k, len(cand_ids)):
        best, best_score = None, -1e9
        for idx in cand_ids:
            if idx in used: 
                continue
            # 유사도: 쿼리-문서
            s1 = float(np.dot(query_vec, cand_vecs[idx]))
            # 중복도: 문서-선택셋 최대
            s2 = max(float(np.dot(cand_vecs[idx], cand_vecs[s])) for s in selected) if selected else 0.0
            score = lambda_ * s1 - (1 - lambda_) * s2
            if score > best_score:
                best, best_score = idx, score
        selected.append(best); used.add(best)
    return selected

class RagSearcher:
    def __init__(self, index_dir: str, device: Optional[str] = None):
        self.index_dir = Path(index_dir)
        with open(self.index_dir / "index_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.embed_dim = int(meta["embed_dim"])
        self.embedding_model_name = meta["embedding_model"]
        self.has_bm25 = bool(meta.get("bm25", False))

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(self.embedding_model_name, device=self.device)

        self.faiss = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.emb = np.load(self.index_dir / "embeddings.npy")
        assert self.emb.shape[1] == self.embed_dim, "Embed dim mismatch."

        self.meta: List[Dict[str, Any]] = load_jsonl(str(self.index_dir / "meta.jsonl"))

        if self.has_bm25:
            with open(self.index_dir / "bm25.pkl", "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            self.bm25 = None

        # (선택) Cross-Encoder 재랭커
        self.reranker = None
        if HAS_RERANKER:
            try:
                self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
            except Exception:
                self.reranker = None

    # -------- 라우팅 --------
    def route(self, query: str) -> str:
        if MITRE_KEYS.search(query): return "mitre"
        if LAW_KEYWORDS.search(query): return "law"
        # 메타에 기반한 소프트 필터는 검색 시 적용
        return "auto"

    # -------- 임베딩 쿼리 --------
    def encode_query(self, q: str) -> np.ndarray:
        q = E5_QUERY_PREFIX + q
        vec = self.encoder.encode([q], normalize_embeddings=True)
        return vec.astype("float32")[0]

    # -------- 하이브리드 1차 검색 --------
    def search_hybrid(self, query: str, topk: int = 50, domain: str = "auto") -> List[Tuple[int, float, float]]:
        qv = self.encode_query(query)
        # FAISS (cosine on normalized)
        D, I = self.faiss.search(np.expand_dims(qv, 0), topk)
        dense = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

        # 라우팅: domain soft-filter → 점수 가산
        def soft_domain_bonus(i: int) -> float:
            ex = self.meta[i].get("extra", {})
            d = ex.get("domain", "other")
            if domain in ("law", "mitre") and d == domain:
                return 0.05  # 작은 보너스
            return 0.0

        # BM25 (있으면) 결합
        sparse = {}
        if self.bm25 is not None:
            toks = query.split()
            bm_scores = self.bm25.get_scores(toks)
            # 상위 topk만 취득
            top_idx = np.argsort(bm_scores)[::-1][:topk]
            for i in top_idx:
                sparse[int(i)] = float(bm_scores[i])

        # 스코어 결합 (가중치는 데이터셋에 맞춰 미세조정)
        alpha = 0.7 if self.bm25 is not None else 1.0
        cand = {}
        for i, dscore in dense:
            cand[i] = cand.get(i, 0.0) + alpha * dscore + soft_domain_bonus(i)
        if self.bm25 is not None:
            for i, sscore in sparse.items():
                cand[i] = cand.get(i, 0.0) + (1 - alpha) * (sscore / (sscore + 10.0))  # 안정화 스케일링

        # 정렬 후 상위 반환
        merged = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [(i, cand[i], float(np.dot(qv, self.emb[i]))) for i, _ in merged]  # (id, fused, dense_raw)

    # -------- 재랭킹(선택) & MMR 다양화 --------
    def rerank_and_diversify(self, query: str, cands: List[Tuple[int, float, float]], k: int = 5) -> List[int]:
        if not cands: return []

        ids = [i for i, _, _ in cands]
        if self.reranker is not None:
            pairs = [(query, self.meta[i]["text"]) for i in ids]
            scores = self.reranker.compute_score(pairs, normalize=True)
            order = np.argsort(scores)[::-1].tolist()
            ids = [ids[o] for o in order]

        # MMR
        qv = self.encode_query(query)
        chosen = mmr(qv, self.emb, ids, k=k, lambda_=0.65)
        return chosen

    # -------- 최종 검색 API --------
    def retrieve(self, query: str, k: int = 3) -> Dict[str, Any]:
        route = self.route(query)
        cands = self.search_hybrid(query, topk=50, domain=route)
        if not cands:
            return {"ok": False, "contexts": [], "route": route, "reason": "no-candidate"}

        # 신뢰도 게이팅: 1위 후보의 dense_raw가 너무 낮으면 OFF
        top_dense = cands[0][2]
        if top_dense < 0.25:  # 데이터별로 0.2~0.35 사이 조정
            return {"ok": False, "contexts": [], "route": route, "reason": f"low-dense={top_dense:.3f}"}

        ids = self.rerank_and_diversify(query, cands, k=k)
        ctxs = []
        for i in ids:
            m = self.meta[i]
            head = []
            ex = m.get("extra", {})
            if ex.get("domain") == "law":
                arts = ", ".join(ex.get("articles", [])[:3]) if ex.get("articles") else ""
                head.append(f"[LAW] {arts}".strip())
            elif ex.get("domain") == "mitre":
                tacs = ",".join(ex.get("tactics", [])[:3]) if ex.get("tactics") else ""
                tech = ",".join(ex.get("techniques", [])[:3]) if ex.get("techniques") else ""
                head.append(f"[ATT&CK] {tacs} {tech}".strip())
            else:
                head.append("[REF]")
            head.append(Path(m["source"]).name)
            header = " | ".join([h for h in head if h])

            # 간단 요약(여기서는 앞문장 2~3줄만; 실제는 추출 요약기로 대체 가능)
            text = m["text"]
            sentences = re.split(r"(?<=[.!?。])\s+|\n", text)
            summary = " ".join(sentences[:3]).strip()
            ctxs.append({
                "id": m["id"], "header": header, "summary": summary, "text": text,
                "source": m["source"], "extra": ex
            })
        return {"ok": True, "contexts": ctxs, "route": route}