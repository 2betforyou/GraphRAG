# -*- coding: utf-8 -*-
# Query the RAG index and generate an answer with EXAONE-3.5-7.8B-Instruct
import json, argparse, pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import normalize_scores, get_device, require_cuda_or_fail

def load_index(index_dir: str):
    p = Path(index_dir)
    emb = np.load(p / "embeddings.npy")
    meta = []
    with open(p / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    fa = faiss.read_index(str(p / "faiss.index"))
    with open(p / "index_meta.json", "r", encoding="utf-8") as f:
        idx_meta = json.load(f)

    if emb.shape[1] != fa.d:
        raise RuntimeError(f"FAISS dim ({fa.d}) != embeddings dim ({emb.shape[1]})")

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        fa = faiss.index_cpu_to_gpu(res, 0, fa)
    return emb, meta, fa, idx_meta

def load_bm25(index_dir: str):
    p = Path(index_dir) / "bm25.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def embed_query(q: str, model: SentenceTransformer, use_e5_prefix=True) -> np.ndarray:
    q_enc = f"query: {q}" if use_e5_prefix else q
    v = model.encode([q_enc], normalize_embeddings=True).astype("float32")[0]
    return v

def normalize(vals):
    if len(vals) == 0:
        return vals
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return [0.0 for _ in vals]
    return [(v - mn) / (mx - mn + 1e-12) for v in vals]

def search(q: str, faiss_index, meta: List[Dict[str, Any]], model: SentenceTransformer,
           topk: int = 6, bm25=None, alpha: float = 0.2) -> List[Dict[str, Any]]:
    # Vector search
    qv = embed_query(q, model)
    D, I = faiss_index.search(qv.reshape(1, -1), topk * 4)
    vec_hits = [{"idx": int(i), "score": float(s)} for i, s in zip(I[0], D[0]) if i != -1]

    # BM25 (optional)
    bm25_hits = []
    if bm25 is not None and alpha > 0:
        scores = bm25.get_scores(q.split())
        top_idx = np.argsort(scores)[::-1][:topk*4]
        bm25_hits = [{"idx": int(i), "score": float(scores[i])} for i in top_idx]

    # Fuse
    fused = {}
    if vec_hits:
        vs = normalize([h["score"] for h in vec_hits])
        for h, ns in zip(vec_hits, vs):
            fused[h["idx"]] = fused.get(h["idx"], 0.0) + (1.0 - alpha) * ns
    if bm25_hits and alpha > 0:
        bs = normalize([h["score"] for h in bm25_hits])
        for h, ns in zip(bm25_hits, bs):
            fused[h["idx"]] = fused.get(h["idx"], 0.0) + alpha * ns

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:topk]
    out = []
    for i, sc in ranked:
        m = meta[i]
        out.append({
            "rank": len(out)+1,
            "score": float(sc),
            "text": m["text"],
            "source": m["source"],
            "span": [m.get("start", -1), m.get("end", -1)],
            "id": m["id"]
        })
    return out

def format_contexts(ctxs: List[Dict[str, Any]]) -> str:
    lines = []
    for c in ctxs:
        src = Path(c["source"]).name
        span = "" if c["span"][0] < 0 else f"{c['span'][0]}-{c['span'][1]}"
        head = f"[{c['rank']}] ({src}{':' + span if span else ''})"
        body = c["text"].strip().replace("\n", " ")
        lines.append(head + "\n" + body)
    return "\n\n".join(lines)

def build_messages(question: str, contexts: str) -> list:
    system = (
        "당신은 한국 법령/지침 전문 RAG 어시스턴트입니다. "
        "오직 제공된 컨텍스트를 근거로 답하고, 근거가 부족하면 모른다고 답하세요. "
        "모든 답변은 한국어로 작성합니다. 반드시 문장 끝에 [근거: 파일명:범위] 인용을 포함하세요."
    )
    user = (
        f"질문:\n{question}\n\n"
        f"관련 컨텍스트 발췌:\n{contexts}\n\n"
        "요청:\n- 위 컨텍스트만 근거로 간결하고 정확하게 답하세요.\n"
        "- 각 주장마다 대괄호로 출처 인용을 붙이세요.\n"
        "- 불확실하면 '제공된 근거로는 단정하기 어렵습니다.'라고 답하세요."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages

def load_lm(base_model: str, gpu_only: bool = False):
    device = get_device(gpu_only)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

def generate_answer(tok, model, messages: list, max_new_tokens=320, temperature=0.2, top_p=0.9):
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
            repetition_penalty=1.02,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=str, required=True)
    ap.add_argument("--base-model", type=str, default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    ap.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-base")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.2, help="0=vector only, 1=BM25 only")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--question", type=str, default=None, help="질문을 직접 문자열로 전달")
    args = ap.parse_args()

    require_cuda_or_fail(args.gpu_only)

    emb, meta, fa, idx_meta = load_index(args.index_dir)
    bm25 = None if args.no_bm25 else load_bm25(args.index_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_encoder = SentenceTransformer(args.embedding_model, device=device)

    tok, model = load_lm(args.base_model, gpu_only=args.gpu_only)

    if args.question:
        questions = [args.question]
    else:
        print("질문을 입력하세요. (종료: 빈 줄)")
        questions = []
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q:
                break
            questions.append(q)

    for q in questions:
        print("="*80)
        print("[Q]", q)
        ctxs = search(q, fa, meta, q_encoder, topk=args.topk, bm25=bm25, alpha=args.alpha)
        print(f"[RAG] top-{len(ctxs)} contexts")
        for c in ctxs:
            print(f"  - [{c['rank']}] {Path(c['source']).name} ({c['span'][0]}-{c['span'][1]}) score={c['score']:.3f} id={c['id']}")

        context_text = format_contexts(ctxs)
        messages = build_messages(q, context_text)
        ans = generate_answer(tok, model, messages, args.max_new_tokens, args.temperature, args.top_p)

        print("\n[MODEL OUTPUT]\n", ans)
        print()

if __name__ == "__main__":
    main()