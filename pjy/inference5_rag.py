#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.csv → submission.csv (RAG + 생성 제어)
- 프롬프트: load.py의 make_prompt_auto(text) 사용 (messages 형식 가정)
- 객관식: 첫 토큰 숫자 강제(1~N)
- 주관식: 간결·정확 생성, 후처리
- RAG(BM25): ../dataset/index/bm25_all_meta.jsonl + bm25_all.npz
  * 검색 질의 강화: 객관식은 "질문 + 보기" 동시 검색
  * 사전계산 인덱스(npz) 사용 시 CSC 변환으로 열 접근 최적화
  * 실패/불일치 시 메모리 Fallback BM25
  * 컨텍스트는 system 또는 user 프리픽스로 주입 선택 가능
- 로깅: RAG 상태, (옵션) Top1 타이틀 프린트
"""

import os
import re
import json
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import pandas as pd
from tqdm import tqdm
import tqdm as tq  # ← 진행바와 공존하는 로그 출력용 (tq.tqdm.write)

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    LogitsProcessor, LogitsProcessorList
)
from peft import PeftModel

# 반드시 프로젝트의 load.py 경로가 PYTHONPATH에 있어야 합니다.
from load import is_multiple_choice, extract_question_and_choices, make_prompt_auto


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

def _normalize(self, s: str) -> str:
    # 동그라미 숫자 → 아라비아 숫자
    circ = {"①":"1","②":"2","③":"3","④":"4","⑤":"5","⑥":"6","⑦":"7","⑧":"8","⑨":"9","⑩":"10","⑪":"11","⑫":"12"}
    for k, v in circ.items():
        s = s.replace(k, v)
    # 전각 → 반각 (필요 시)
    s = s.replace("／","/").replace("－","-")
    # 일반 소문자화
    return s.lower()

def _basic_tokenize(self, text: str):
    text = self._normalize(text)
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text)
    return toks

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
            add_single_token_ids(d[0])  # 한 자리 분해 케이스

        self.allowed_ids = list(self.allowed_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.step == 0:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.allowed_ids] = scores[:, self.allowed_ids]
            scores = mask
        self.step += 1
        return scores


# =========================
# RAG: BM25 검색기 (docs/corpus npz 스키마 지원)
# =========================
class BM25RAG:
    """
    index_dir 안의 bm25_all_meta.jsonl + bm25_all.npz 사용.
    - npz에 'docs'/'corpus'가 있으면 rank_bm25.BM25Okapi로 프리빌트 사용
    - meta에 text/title이 없으면 docs로 보강하여 정렬을 맞춤
    - 실패 시 간단 BM25(Fallback)로 동작
    """
    def __init__(self, index_dir: str, topk: int = 5, max_chars: int = 1200,
                 meta_text_field=None, meta_title_field=None, meta_force_text=False):
        self.index_dir = index_dir
        self.meta_path = os.path.join(index_dir, "bm25_all_meta.jsonl")
        self.npz_path  = os.path.join(index_dir, "bm25_all.npz")
        self.topk = topk
        self.max_chars = max_chars

        # 확장 옵션
        self.meta_text_field  = meta_text_field
        self.meta_title_field = meta_title_field
        self.meta_force_text  = meta_force_text

        # 내부 상태
        self.meta = []             # [{'title':..., 'text':..., 'source':...}, ...]
        self.use_bm25okapi = False # docs/corpus 기반 prebuilt
        self.bm25 = None           # BM25Okapi 인스턴스
        self.corpus_tokens = None  # 토큰화 된 문서 리스트
        self.docs = None           # 문자열 본문 리스트

        # Fallback용 역색인
        self.fb_ready = False

        self._load_npz_and_meta()

        if not self.use_bm25okapi:
            self._build_fallback_index()
    
    # ---------- 공통 유틸 ----------
    def _normalize(self, s: str) -> str:
        if s is None:
            return ""
        if not isinstance(s, str):
            s = str(s)
        circ = {"①":"1","②":"2","③":"3","④":"4","⑤":"5","⑥":"6","⑦":"7","⑧":"8","⑨":"9","⑩":"10","⑪":"11","⑫":"12"}
        for k, v in circ.items():
            s = s.replace(k, v)
        return s

    def _basic_tokenize(self, text: str):
        text = self._normalize(text).lower()
        return re.findall(r"[가-힣A-Za-z0-9]+", text)

    # ---------- 로딩 ----------
    def _load_npz_and_meta(self):
        import json, gzip, numpy as np
        from rank_bm25 import BM25Okap

        # 1) meta 로드 (JSONL/JSON/텍스트 모두 수용)
        if not os.path.isfile(self.meta_path):
            raise FileNotFoundError(f"메타 파일을 찾을 수 없습니다: {self.meta_path}")

        def open_any(path):
            if path.endswith(".gz"):
                return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
            return open(path, "r", encoding="utf-8", errors="ignore")

        raw_meta = []
        with open_any(self.meta_path) as f:
            head = f.read(4096)
        lead = head.lstrip()
        is_array = lead.startswith("[")
        is_object = lead.startswith("{")

        def add_meta_obj(obj_or_text):
            if isinstance(obj_or_text, dict):
                # 지정 키 우선, 없으면 후보 키
                text = obj_or_text.get(self.meta_text_field) if self.meta_text_field else None
                title = obj_or_text.get(self.meta_title_field) if self.meta_title_field else None
                if text is None:
                    for k in ["text","contents","content","body","doc","document","passage"]:
                        if k in obj_or_text:
                            text = obj_or_text.get(k); break
                if title is None:
                    for k in ["title","section","heading","headline","name"]:
                        if k in obj_or_text:
                            title = obj_or_text.get(k); break
                rec = dict(obj_or_text)
                rec["text"]  = self._normalize(text or "")
                rec["title"] = self._normalize(title or "")
                raw_meta.append(rec)
            else:
                # 라인 텍스트
                raw_meta.append({"title": "", "text": self._normalize(obj_or_text)})

        if self.meta_force_text:
            with open_any(self.meta_path) as f:
                for line in f:
                    s = line.strip()
                    if s: add_meta_obj(s)
        else:
            if is_array or is_object:
                with open_any(self.meta_path) as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for key in ["data","documents","docs","items","results","records"]:
                                if key in data and isinstance(data[key], list):
                                    data = data[key]; break
                        if not isinstance(data, list):
                            data = [data]
                        for obj in data:
                            add_meta_obj(obj if isinstance(obj, dict) else str(obj))
                    except Exception:
                        with open_any(self.meta_path) as f2:
                            for line in f2:
                                s = line.strip()
                                if not s: continue
                                try:
                                    obj = json.loads(s)
                                    add_meta_obj(obj if isinstance(obj, dict) else s)
                                except Exception:
                                    add_meta_obj(s)
            else:
                with open_any(self.meta_path) as f:
                    for line in f:
                        s = line.strip()
                        if not s: continue
                        if s.startswith("{") and s.endswith("}"):
                            try:
                                obj = json.loads(s)
                                add_meta_obj(obj if isinstance(obj, dict) else s)
                            except Exception:
                                add_meta_obj(s)
                        else:
                            add_meta_obj(s)

        # 2) npz 로드 — docs/corpus 지원
        if not os.path.isfile(self.npz_path):
            # npz가 없다면 meta만으로 진행(나중 Fallback 인덱싱)
            self.use_bm25okapi = False
            # meta가 text를 포함해야 함
            self.meta = [{"title": m.get("title",""), "text": m.get("text","")} for m in raw_meta if m.get("text","")]
            if not self.meta:
                raise ValueError("메타 데이터를 비어 있게 읽었습니다. 파일 내용/필드명을 확인하세요.")
            return

        npz = np.load(self.npz_path, allow_pickle=True)
        npz_keys = set(npz.files)

        if {"docs","corpus"}.issubset(npz_keys):
            # ===== rank_bm25 경로 =====
            self.docs = list(npz["docs"].tolist())            # 본문 텍스트
            self.corpus_tokens = list(npz["corpus"].tolist()) # 토큰 리스트
            # meta 보강: 길이 일치 / text 주입
            if len(raw_meta) != len(self.docs):
                # 길이 불일치라도 최소한 docs 길이에 맞춰 자르기/패딩
                n = len(self.docs)
                raw_meta = (raw_meta[:n]) if len(raw_meta) >= n else (raw_meta + [{}]*(n-len(raw_meta)))
            self.meta = []
            for i, m in enumerate(raw_meta):
                m = dict(m)
                m_text = m.get("text") or self.docs[i] or ""
                m_title = m.get("title") or m.get("source","") or ""
                self.meta.append({"title": self._normalize(m_title), "text": self._normalize(m_text)})
            # BM25Okapi 인덱스
            self.bm25 = BM25Okapi(self.corpus_tokens)
            self.use_bm25okapi = True
        else:
            # 기대 스키마가 아니면 meta 기반 Fallback로 진행
            self.use_bm25okapi = False
            self.meta = [{"title": m.get("title",""), "text": m.get("text","")} for m in raw_meta if m.get("text","")]
            if not self.meta:
                raise ValueError("메타 데이터를 비어 있게 읽었습니다. 파일 내용/필드명을 확인하세요.")

    # ---------- Fallback 인덱스 ----------
    def _build_fallback_index(self):
        from collections import Counter, defaultdict
        texts = [m["text"] for m in self.meta]
        toks_list = [self._basic_tokenize(t) for t in texts]
        self.fb_N = len(toks_list)
        self.fb_dl = [len(ts) for ts in toks_list]
        self.fb_avgdl = sum(self.fb_dl)/max(1, self.fb_N)
        self.fb_df = Counter()
        self.fb_post = defaultdict(list)  # w -> list of (doc_id, tf)
        for di, tokens in enumerate(toks_list):
            tf = Counter(tokens)
            for w, c in tf.items():
                self.fb_post[w].append((di, c))
                self.fb_df[w] += 1
        self.fb_k1 = 1.5
        self.fb_b  = 0.75
        self.fb_ready = True

    # ---------- 스니펫 추출 ----------
    def _extract_window(self, text: str, query_tokens, win=300):
        if not text:
            return ""
        for t in query_tokens:
            try:
                m = re.search(re.escape(t), text, flags=re.IGNORECASE)
            except re.error:
                m = None
            if m:
                s = max(0, m.start() - win)
                e = min(len(text), m.end() + win)
                return (("…" if s>0 else "") + text[s:e] + ("…" if e<len(text) else ""))
        return text[: self.max_chars] + ("…" if len(text) > self.max_chars else "")

    # ---------- 점수화 ----------
    def _score_bm25_fb(self, query_tokens):
        import math
        scores = [0.0]*self.fb_N
        for w in set(query_tokens):
            df = self.fb_df.get(w, 0)
            if df == 0: continue
            idf = math.log((self.fb_N - df + 0.5)/(df + 0.5) + 1.0)
            for di, tf in self.fb_post.get(w, []):
                denom = tf + self.fb_k1*(1 - self.fb_b + self.fb_b*self.fb_dl[di]/self.fb_avgdl)
                scores[di] += idf * (tf*(self.fb_k1+1))/max(1e-9, denom)
        idx = sorted(range(self.fb_N), key=lambda i: scores[i], reverse=True)[:self.topk]
        return [(i, scores[i]) for i in idx if scores[i] > 0]

    # ---------- 검색 ----------
    def search(self, query: str):
        if not query or not query.strip():
            return []
        q_toks = self._basic_tokenize(query)
        if not q_toks:
            return []

        hits = []
        try:
            if self.use_bm25okapi and self.bm25 is not None:
                scores = self.bm25.get_scores(q_toks)
                # 상위 k 인덱스
                import numpy as np
                k = min(self.topk, len(scores))
                top = np.argpartition(-scores, k-1)[:k]
                top = top[np.argsort(-scores[top])]
                hits = [(int(i), float(scores[i])) for i in top if scores[i] > 0]
                # no-hits면 Fallback로 백오프
                if not hits and self.fb_ready:
                    hits = self._score_bm25_fb(q_toks)
            else:
                hits = self._score_bm25_fb(q_toks)
        except Exception:
            if self.fb_ready:
                hits = self._score_bm25_fb(q_toks)

        results = []
        for di, _ in hits:
            full = self.meta[di]["text"]
            title = self.meta[di]["title"]
            snippet = self._extract_window(full, q_toks, win=self.max_chars // 2)
            results.append({"title": title, "text": snippet})
        return results

    def build_context(self, question_or_query: str) -> str:
        docs = self.search(question_or_query)
        if not docs:
            return ""
        chunks = []
        for i, d in enumerate(docs, 1):
            title = f"[{d['title']}]" if d['title'] else ""
            chunks.append(f"(#{i}) {title} {d['text']}")
        guide = (
            "다음은 법령/규정/용어 관련 참조 문맥입니다. "
            "정답은 문맥과 질문에 근거하여 간결하고 정확하게 작성하세요. "
            "출처 표시는 하지 말고, 최종 답만 제시하세요."
        )
        return f"{guide}\n\n" + "\n\n".join(chunks)


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
    parser.add_argument("--min_new_tokens_subjective", type=int, default=80, help="주관식 최소 생성 토큰")
    parser.add_argument("--max_new_tokens_subjective", type=int, default=2048, help="주관식 최대 생성 토큰")

    # ----- RAG 옵션 -----
    parser.add_argument("--rag", type=str, choices=["on","off"], default="on")
    parser.add_argument("--index_dir", type=str, default="../dataset/index")
    parser.add_argument("--rag_topk", type=int, default=5)
    parser.add_argument("--rag_max_chars", type=int, default=1200)
    parser.add_argument("--rag_injection", choices=["system", "user"], default="system",
                        help="RAG 컨텍스트 주입 위치 (system 우선, 템플릿 호환성 문제 시 user로)")
    parser.add_argument("--rag_log_top1", action="store_true", help="RAG 상위1 제목 로깅")

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

    # 2) RAG 초기화
    rag = None
    if args.rag == "on":
        try:
            rag = BM25RAG(index_dir=args.index_dir, topk=args.rag_topk, max_chars=args.rag_max_chars)
            print(f"[RAG] 활성화됨 — topk={args.rag_topk}, max_chars={args.rag_max_chars}, prebuilt={rag.use_bm25okapi}")
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
        # load.py 기반 메시지
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

            # ↑ 주입은 그대로 유지하고, 로그는 컨텍스트 유무와 무관하게 찍어줍니다.
            if args.rag_log_top1:
                try:
                    hits = rag.search(query_text)
                    if hits:
                        title = (hits[0].get("title") or "").strip()
                        preview = (hits[0].get("text") or "")[:120].replace("\n", " ")
                        line = title if title else preview if preview else "(empty)"
                        tq.tqdm.write(f"[RAG-Top1] {line}")
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
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0.0,
                    top_p=0.9,
                    min_new_tokens=int(args.min_new_tokens_subjective),
                    max_new_tokens=int(args.max_new_tokens_subjective),
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.05,
                    eos_token_id=int(tokenizer.eos_token_id),
                    pad_token_id=int(tokenizer.pad_token_id),
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

    # preds 는 test의 원래 순서와 1:1 대응이므로 test["ID"]와 함께 merge
    out_map = pd.DataFrame({"ID": test["ID"].tolist(), "Answer": preds})

    # sample 기준으로 안전 병합
    out_df = sub[["ID"]].merge(out_map, on="ID", how="left")

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()