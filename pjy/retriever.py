# from rank_bm25 import BM25Okapi 
import re, json
import os, glob



# (A) 간단 BM25 리트리버 유틸 < RAG 추가
def _tok_ko_en(s: str):
    # 한국어/영문/숫자 토큰 단순 분할
    return re.findall(r"[가-힣A-Za-z0-9]+", (s or "").lower())

class SimpleBM25:
    """외부 라이브러리 없이 동작하는 매우 간단한 BM25 유사 구현 (TF-IDF 가중 + length norm)"""
    def __init__(self, docs):
        self.docs = docs
        self.doc_tokens = [ _tok_ko_en(d) for d in docs ]
        self.df = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.docs)

    def _idf(self, term):
        # BM25-ish idf
        df = self.df.get(term, 0)
        return max(0.0, ( (self.N - df + 0.5) / (df + 0.5) ))  # 로그 생략(안정성)

    def get_scores(self, query):
        q = _tok_ko_en(query)
        scores = [0.0] * self.N
        for qi in q:
            idf = self._idf(qi)
            if idf == 0: 
                continue
            for i, toks in enumerate(self.doc_tokens):
                tf = toks.count(qi)
                if tf == 0: 
                    continue
                # TF 가중 (아주 단순화)
                scores[i] += idf * tf / (len(toks) ** 0.25)  # 길이 보정 약하게
        return scores

    def search(self, query, k=3):
        scores = self.get_scores(query)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, scores[i], self.docs[i]) for i in top]

def chunk_text(text: str, max_chars=1800, overlap=200):
    """간단 char기준 청킹 (토크나이저 없이 가볍게)"""
    out, n = [], len(text)
    i = 0
    while i < n:
        j = min(i + max_chars, n)
        out.append(text[i:j])
        i = max(j - overlap, i + 1)
    return out

def load_corpus_and_build_retriever(paths_or_glob: list[str], max_chars=1800, overlap=200):
    texts = []
    # 경로/글롭 리스트 처리
    for p in paths_or_glob:
        for fp in glob.glob(p):
            if os.path.isdir(fp):
                for f in glob.glob(os.path.join(fp, "*.txt")):
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        texts.append(fh.read())
            else:
                # 단일 파일
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    texts.append(fh.read())
    # 비었으면 빈 리트리버 리턴
    if not texts:
        return SimpleBM25([""]), []

    # 문서 → 청크
    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t, max_chars=max_chars, overlap=overlap))
    retr = SimpleBM25(chunks)
    return retr, chunks
