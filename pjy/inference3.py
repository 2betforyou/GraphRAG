#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.csv → submission.csv (RAG + 생성 제어; EXAONE + LoRA + E5/FAISS RAG)
- 프롬프트: load.py의 make_prompt_auto(text) 사용 (messages 형식 가정)
- 객관식: 첫 토큰 숫자 강제(1~N)
- 주관식: 간결·정확 생성, 후처리
- RAG(E5+FAISS, +선택 BM25 결합):
  * 필요 파일: {index_dir}/faiss.index, embeddings.npy, meta.jsonl, (선택) bm25.pkl
  * 검색 질의 강화: 객관식은 "질문 + 보기" 동시 검색
  * 벡터×BM25 가중 합성(alpha, 기본 0.2)
  * 실패/불일치 시 로깅 후 미적용
  * 컨텍스트는 system 또는 user 프리픽스로 주입 선택 가능
- 로깅: RAG 상태, (옵션) Top1 타이틀/미리보기 프린트
"""

import os
import re
import json
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
import tqdm as tq  # 진행바와 공존하는 로그 출력용 (tq.tqdm.write)

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    LogitsProcessor, LogitsProcessorList
)
from peft import PeftModel

# 반드시 프로젝트의 load.py 경로가 PYTHONPATH에 있어야 합니다.
from load import is_multiple_choice, extract_question_and_choices, make_prompt_auto
from transformers import StoppingCriteria, StoppingCriteriaList
import re 

class RegexStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, tail_tokens=120, max_symbol_run=6, max_noise_lines=3,
                 max_english_ratio=0.25):
        self.tok = tokenizer
        self.tail = tail_tokens
        self.max_symbol_run = max_symbol_run
        self.max_noise_lines = max_noise_lines
        self.max_english_ratio = max_english_ratio

        # 기호 연속(예: /***///, @@@@@, ###### 등)
        self.re_symbol_run = re.compile(
            rf"([/\*@#%=\-\|_~\\]{{{max_symbol_run},}})"
        )

        # 공백/기호만으로 이뤄진 “노이즈 라인”
        self.re_noise_line = re.compile(
            r"^[\s/\*\@\#\%\=\-\|\_\~\\\.\,\:\;\^\+\!\?\[\]\(\)\{\}\<\>\'\"\\]+$"
        )

        # 영어 비율 체크
        self.re_eng = re.compile(r"[A-Za-z]")

    def __call__(self, input_ids, scores, **kwargs):
        tail_ids = input_ids[0][-self.tail:].tolist()
        text = self.tok.decode(tail_ids, skip_special_tokens=True)

        if self.re_symbol_run.search(text):
            return True

        noise_lines = [ln for ln in text.splitlines() if self.re_noise_line.match(ln)]
        if len(noise_lines) >= self.max_noise_lines:
            return True 

        return False

# =========================
# 유틸: 텍스트 정리
# =========================
def clean_text(s: str) -> str:
    s = s.replace("〈", "").replace("〉", "").replace("<", "").replace(">", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\.{2,}", "…", s)  # ... → …
    return s


def extract_answer_only(generated_text: str, original_question: str) -> str:
    """모델 출력에서 실답 텍스트만 추출"""
    text = generated_text.split("답변:")[-1].strip() if "답변:" in generated_text else generated_text.strip()
    if not text:
        return "미응답"
    if is_multiple_choice(original_question):
        m = re.search(r"([1-9][0-9]?)", text)
        return m.group(1) if m else "0"
    return clean_text(text)


# =========================
# 숫자 강제: 1~N만 허용(첫 토큰)
# =========================
class FirstDigitOnly(LogitsProcessor):
    """
    첫 생성 스텝에서 허용된 숫자들만 나오도록 제한.
    - '1', ' 1', '\n1' 같은 변형이나 '10'이 한 토큰인 경우도 처리
    """
    def __init__(self, tokenizer, allowed_digits: list[str]):
        self.tokenizer = tokenizer
        self.allowed_ids = set()
        self.step = 0

        def add_single_token_ids(s: str):
            for pre in ["", " ", "\n"]:
                ids = tokenizer.encode(pre + s, add_special_tokens=False)
                if len(ids) == 1:
                    self.allowed_ids.add(ids[0])

        for d in allowed_digits:
            add_single_token_ids(d)
            # '10' 등의 첫 자리 '1'만 한 토큰인 케이스도 허용
            add_single_token_ids(d[0])

        self.allowed_ids = list(self.allowed_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.step == 0:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.allowed_ids] = scores[:, self.allowed_ids]
            scores = mask
        self.step += 1
        return scores


# =========================
# RAG: FAISS + E5 (+선택 BM25) 검색기
# =========================
class FaissRAG:
    """
    {index_dir}/faiss.index + embeddings.npy + meta.jsonl (+선택 bm25.pkl)을 사용하는 RAG.
    - 벡터 검색: sentence-transformers(E5) 정규화 벡터 + FAISS inner product(코사인)
    - BM25(선택): rank_bm25 점수와 min-max 정규화 후 가중 결합(alpha)
    - search()는 융합 상위 K 청크를 반환, build_context()는 질의응답용 컨텍스트 문자열 구성
    """
    def __init__(self, index_dir: str, topk: int = 5, max_chars: int = 1200,
                 alpha: float = 0.2, embed_model: str = "intfloat/multilingual-e5-base"):
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index_dir = Path(index_dir)
        self.topk = int(topk)
        self.max_chars = int(max_chars)
        self.alpha = float(alpha)
        self.embed_model_name = embed_model

        # ----- 메타/임베딩/FAISS 로드 -----
        emb_path = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "meta.jsonl"
        faiss_path = self.index_dir / "faiss.index"

        if not (emb_path.exists() and meta_path.exists() and faiss_path.exists()):
            raise FileNotFoundError(
                f"필수 인덱스 파일이 없습니다. 다음 3개가 필요합니다:\n"
                f" - {emb_path}\n - {meta_path}\n - {faiss_path}"
            )

        self.emb = np.load(emb_path)  # (N, D)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f]

        self.faiss_index = faiss.read_index(str(faiss_path))

        # GPU로 올리기 (가능하면). faiss-cpu 환경이면 안전하게 패스.
        try:
            if torch.cuda.is_available() and hasattr(faiss, "index_cpu_to_gpu"):
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        except Exception:
            pass  # CPU 유지

        # E5 임베딩 모델 (질의용)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_encoder = SentenceTransformer(self.embed_model_name, device=device)

        # (선택) BM25 로드
        self.bm25 = None
        bm25_path = self.index_dir / "bm25.pkl"
        if bm25_path.exists():
            import pickle
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)

    @staticmethod
    def _normalize(vals):
        if not vals:
            return vals
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.0] * len(vals)
        return [(v - mn) / (mx - mn + 1e-12) for v in vals]

    def _embed_query(self, q: str):
        import numpy as np
        q_enc = f"query: {q}"
        v = self.q_encoder.encode([q_enc], normalize_embeddings=True).astype("float32")[0]
        return v

    def _extract_window(self, text: str, query: str, win: int = 300) -> str:
        """질의 토큰이 등장하는 주변으로 스니펫 추출(없으면 앞부분)."""
        if not text:
            return ""
        # 간단 토크나이즈
        toks = [t for t in re.findall(r"[가-힣A-Za-z0-9]+", query) if t]
        for t in toks:
            try:
                m = re.search(re.escape(t), text, flags=re.IGNORECASE)
            except re.error:
                m = None
            if m:
                s = max(0, m.start() - win)
                e = min(len(text), m.end() + win)
                return (("…" if s > 0 else "") + text[s:e] + ("…" if e < len(text) else ""))
        return text[: self.max_chars] + ("…" if len(text) > self.max_chars else "")

    def search(self, query: str):
        import numpy as np

        if not query or not query.strip():
            return []

        # 1) 벡터 검색 (topk*4 넓게)
        qv = self._embed_query(query)
        D, I = self.faiss_index.search(qv.reshape(1, -1), max(self.topk * 4, self.topk))
        vec_hits = [{"idx": int(i), "score": float(s)} for i, s in zip(I[0], D[0]) if i != -1]

        # 2) BM25 (선택)
        bm25_hits = []
        if self.bm25 is not None and self.alpha > 0:
            scores = self.bm25.get_scores(query.split())
            top_idx = np.argsort(scores)[::-1][:max(self.topk * 4, self.topk)]
            bm25_hits = [{"idx": int(i), "score": float(scores[i])} for i in top_idx]

        # 3) 점수 융합 (min-max 정규화 후 alpha 가중)
        fused = {}
        if vec_hits:
            vs = self._normalize([h["score"] for h in vec_hits])
            for h, ns in zip(vec_hits, vs):
                fused[h["idx"]] = fused.get(h["idx"], 0.0) + (1.0 - self.alpha) * ns
        if bm25_hits and self.alpha > 0:
            bs = self._normalize([h["score"] for h in bm25_hits])
            for h, ns in zip(bm25_hits, bs):
                fused[h["idx"]] = fused.get(h["idx"], 0.0) + self.alpha * ns

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: self.topk]
        results = []
        for rank, (i, sc) in enumerate(ranked, 1):
            m = self.meta[i]
            src = Path(m.get("source", "")).name or m.get("id", f"doc{i}")
            text = m.get("text", "")
            snippet = self._extract_window(text, query, win=self.max_chars // 2)
            # span 정보가 있으면 노출
            a, b = m.get("start", -1), m.get("end", -1)
            span = f"{a}-{b}" if isinstance(a, int) and a >= 0 and isinstance(b, int) and b >= 0 else ""
            head = f"[{rank}] ({src}{':' + span if span else ''})"
            results.append({
                "rank": rank,
                "score": float(sc),
                "source": src,
                "span": span,
                "text": text,
                "snippet": snippet
            })
        return results

    def build_context(self, question_or_query: str) -> str:
        """질의응답 컨텍스트 문자열 구성(시스템 주입용)."""
        hits = self.search(question_or_query)
        if not hits:
            return ""
        parts = []
        guide = (
            "다음은 법령/규정/지침/정의 관련 참조 문맥입니다. "
            "정답은 문맥과 질문에 근거하여 간결하고 정확하게 작성하세요. "
            "출처 표시는 하지 말고, 최종 답만 제시하세요."
        )
        parts.append(guide)
        for h in hits:
            line1 = f"(#{h['rank']}) [{h['source']}{(':' + h['span']) if h['span'] else ''}]"
            body = (h["snippet"] or h["text"]).replace("\n", " ")
            parts.append(line1 + "\n" + body)
        return "\n\n".join(parts)


# =========================
# 메인
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    parser.add_argument("--adapter", type=str, required=True, help="학습한 LoRA 어댑터 경로 (예: out-exaone-law-qlora/20250816_032220/adapter)")
    parser.add_argument("--test_csv", type=str, default="../dataset/test.csv")
    parser.add_argument("--sample_sub", type=str, default="./sample_submission.csv")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--max_choices", type=int, default=6, help="객관식 최대 선택지 번호(기본 6)")
    parser.add_argument("--bf16", action="store_true", help="가능하면 bf16 사용 (기본은 fp16)")
    parser.add_argument("--min_new_tokens_subjective", type=int, default=128, help="주관식 최소 생성 토큰")
    parser.add_argument("--max_new_tokens_subjective", type=int, default=2048, help="주관식 최대 생성 토큰")

    # ----- RAG 옵션 -----
    parser.add_argument("--rag", type=str, choices=["on","off"], default="on")
    parser.add_argument("--index_dir", type=str, default="../dataset/index")
    parser.add_argument("--rag_topk", type=int, default=5)
    parser.add_argument("--rag_max_chars", type=int, default=1200)
    parser.add_argument("--rag_alpha", type=float, default=0.2, help="0=벡터만, 1=BM25만 (융합 가중)")
    parser.add_argument("--rag_embed_model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--rag_injection", choices=["system", "user"], default="system",
                        help="RAG 컨텍스트 주입 위치 (system 우선, 템플릿 호환성 문제 시 user로)")
    parser.add_argument("--rag_log_top1", action="store_true", help="RAG 상위1 미리보기 로깅")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"baseline_submission_{ts}.csv")

    # 1) 모델/토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id

    # 2) RAG 초기화 (FAISS + E5, +선택 BM25)
    rag = None
    if args.rag == "on":
        try:
            rag = FaissRAG(
                index_dir=args.index_dir,
                topk=args.rag_topk,
                max_chars=args.rag_max_chars,
                alpha=args.rag_alpha,
                embed_model=args.rag_embed_model,
            )
            print(f"[RAG] 활성화됨 — topk={args.rag_topk}, max_chars={args.rag_max_chars}, alpha={args.rag_alpha}")
        except Exception as e:
            print(f"[RAG] 경고: 초기화 실패 → 비활성화합니다. 이유: {e}")
            rag = None
    else:
        print("[RAG] OFF")

    # (선택) 간단 캐시
    rag_cache = {}

    # 3) 데이터 로드
    test = pd.read_csv(args.test_csv)
    preds = []

    # 4) 추론 루프
    for q in tqdm(test["Question"], desc="Inference"):
        # load.py 기반 메시지 (Chat 템플릿 메시지 리스트여야 함)
        messages = make_prompt_auto(q)

        # --- RAG 컨텍스트 삽입 ---
        if rag is not None:
            # 객관식은 "질문 + 보기"로 검색 강화
            query_text = q
            if is_multiple_choice(q):
                try:
                    _, options = extract_question_and_choices(q)
                    query_text = q + "\n" + "\n".join(options)
                except Exception:
                    query_text = q

            # 캐시 활용
            if query_text in rag_cache:
                ctx = rag_cache[query_text]
            else:
                try:
                    ctx = rag.build_context(query_text)
                    rag_cache[query_text] = ctx
                except Exception as e:
                    print(f"[RAG] 검색 실패 → 이번 문항은 RAG 미적용. 이유: {e}")
                    ctx = ""

            if ctx:
                if args.rag_injection == "system":
                    messages = [{"role": "system", "content": ctx}] + messages
                else:
                    messages = [{"role": "user", "content": f"[참고문맥]\n{ctx}"}] + messages

            # Top1 로그
            if args.rag_log_top1:
                try:
                    hits = rag.search(query_text)
                    if hits:
                        head = f"{hits[0]['source']}{(':'+hits[0]['span']) if hits[0]['span'] else ''}"
                        preview = (hits[0].get("snippet") or hits[0].get("text") or "").replace("\n", " ")
                        tq.tqdm.write(f"[RAG-Top1] {head} | {preview}")
                    else:
                        tq.tqdm.write("[RAG-Top1] (no hits)")
                except Exception as e:
                    tq.tqdm.write(f"[RAG-Top1] ERROR: {e}")

        # 토크나이저 적용
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        # 일부 모델은 token_type_ids 없음
        if isinstance(inputs, dict):
            inputs.pop("token_type_ids", None)

        # --- 생성 ---
        if is_multiple_choice(q):
            # 허용 숫자 = 1~max_choices, 선택지 파싱 실패에도 안전
            allowed = [str(i) for i in range(1, args.max_choices + 1)]

            # 선택지 안에 동그라미 숫자만 있을 가능성 보완
            circled_map = {"①":"1","②":"2","③":"3","④":"4","⑤":"5","⑥":"6",
                           "⑦":"7","⑧":"8","⑨":"9","⑩":"10","⑪":"11","⑫":"12"}
            try:
                _, options = extract_question_and_choices(q)
                opt_text = "\n".join(options)
                for k, v in circled_map.items():
                    opt_text = opt_text.replace(k, v)
                # 실제 등장 숫자와 교집합
                cands = sorted(set(re.findall(r"^\s*([1-9][0-9]?)", opt_text, re.M)))
                cands = [x for x in cands if 1 <= int(x) <= args.max_choices]
                nums = cands if cands else allowed
            except Exception:
                nums = allowed

            processors = LogitsProcessorList([FirstDigitOnly(tokenizer, nums)])

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    min_new_tokens=1,
                    max_new_tokens=4,  # 숫자 + 개행/공백 정도
                    logits_processor=processors,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            m = re.search(r"([1-9][0-9]?)", gen)
            pred = m.group(1) if m else "0"

            print(f"[MCQ]\n{q}\n[A] {pred}\n{'-'*50}")

        else:
            stops = StoppingCriteriaList([
                RegexStoppingCriteria(tokenizer, tail_tokens=120, max_symbol_run=6,
                                    max_noise_lines=2)
            ])

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0.3,       # 살짝 낮춰 안정화
                    top_p=0.9,
                    min_new_tokens=int(args.min_new_tokens_subjective),
                    max_new_tokens=int(args.max_new_tokens_subjective),
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.07,   # 약간 상향해 중복 감소
                    eos_token_id=int(tokenizer.eos_token_id),
                    pad_token_id=int(tokenizer.pad_token_id),
                    # logits_processor=ko_processors,     # ★ 추가
                    stopping_criteria=stops             # ★ 추가
                )
            gen_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            pred = extract_answer_only(gen_text, q)

            print(f"[SUBJ]\n{q}\n[Gen]\n{gen_text}\n[A] {pred}\n{'-'*50}")

        preds.append(pred)

    # 5) 제출 파일 저장 — sample_submission 스키마/순서 준수
    sub = pd.read_csv(args.sample_sub)
    if "Answer" not in sub.columns or "ID" not in sub.columns:
        raise ValueError("sample_submission.csv에는 'ID'와 'Answer' 컬럼이 모두 있어야 합니다.")

    if len(sub) != len(preds):
        raise ValueError(f"예측 개수({len(preds)})와 sample 행 수({len(sub)})가 다릅니다.")

    # ID 정렬 유지: sample_submission 순서대로 채움
    sub = sub.sort_values("ID").reset_index(drop=True)

    # preds 는 test의 원래 순서와 1:1 대응이므로 test['ID']와 함께 merge
    out_map = pd.DataFrame({"ID": test["ID"].tolist(), "Answer": preds})
    out_df = sub[["ID"]].merge(out_map, on="ID", how="left")

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()