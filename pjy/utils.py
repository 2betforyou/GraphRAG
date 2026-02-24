# -*- coding: utf-8 -*-
import re, json
from typing import List, Tuple

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def clean_text(s: str) -> str:
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def make_chunks(text: str, chunk_chars: int = 900, overlap: int = 200) -> List[Tuple[str, int, int]]:
    # 슬라이딩 윈도우 방식으로 글자 기준 청크 생성.
    text = clean_text(text)
    n = len(text)
    chunks = []
    i = 0
    while i < n:
        j = min(i + chunk_chars, n)
        chunk = text[i:j]
        chunks.append((chunk, i, j))
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mx, mn = max(scores), min(scores)
    if mx == mn:
        return [0.0 for _ in scores]
    return [(s - mn) / (mx - mn + 1e-12) for s in scores]

def require_cuda_or_fail(gpu_only: bool = False):
    import torch
    if gpu_only and not torch.cuda.is_available():
        raise RuntimeError("GPU-only 모드이나 CUDA를 감지하지 못했습니다. GPU 환경을 확인하세요.")

def get_device(gpu_only: bool = False) -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if gpu_only:
        raise RuntimeError("GPU-only 모드이나 CUDA를 감지하지 못했습니다.")
    return "cpu"