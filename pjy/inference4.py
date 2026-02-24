#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.csv → submission.csv (Dual RAG + Domain Router + Continuation MCQ Scoring + IO 로깅)
- 모델: EXAONE + LoRA (peft)
- RAG: 두 인덱스(A=law, B=mitre) + 도메인 라우팅(hard/soft)
- MCQ: 연속 토큰 로그확률(숫자 공백/개행/구두점 변형 포함)으로 스코어링 → ModelOut과 Answer 정렬
- 주관식: 단일/듀얼 생성 선택 가능
- IO: --show_io 로 문제/선택지/컨텍스트 헤더/모델 출력 표시

필수 의존:
- transformers, peft, pandas, torch, tqdm
- 프로젝트의 load.py (make_prompt_auto, is_multiple_choice, extract_question_and_choices)
- rag_searcher.py (RagSearcher: retrieve(query, k) -> {"ok": bool, "contexts": [...]})
"""

import os, re, argparse, math
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import pandas as pd
from tqdm import tqdm
import tqdm as tq

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    LogitsProcessor, LogitsProcessorList,
    StoppingCriteria, StoppingCriteriaList
)
from peft import PeftModel

from load import is_multiple_choice, extract_question_and_choices, make_prompt_auto
from rag_searcher import RagSearcher


# ----------------------
# Stopping Criteria (항상 bool 반환)
# ----------------------
class RegexStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, tail_tokens=120, max_symbol_run=12, max_noise_lines=6):
        import regex as re
        self.tok = tokenizer
        self.tail = int(tail_tokens)
        self.max_symbol_run = int(max_symbol_run)
        self.max_noise_lines = int(max_noise_lines)
        self.re_symbol_run = re.compile(rf"([\/\*\@\#\%\=\-\|\_\~\\]{{{self.max_symbol_run},}})")
        self.re_noise_line = re.compile(r"^[\s\/\*\@\#\%\=\-\|\_\~\\\.\,\:\;\^\+\!\?\[\]\(\)\{{\}}\<\>\'\"\\]+$")

    def __call__(self, input_ids, scores, **kwargs):
        try:
            tail_ids = input_ids[0][-self.tail:].tolist()
            text = self.tok.decode(tail_ids, skip_special_tokens=True)
        except Exception:
            return False
        if self.re_symbol_run.search(text):
            return True
        noise_lines = [ln for ln in text.splitlines() if self.re_noise_line.match(ln)]
        if len(noise_lines) >= self.max_noise_lines:
            return True
        return False


# ----------------------
# Utils
# ----------------------
CLEAN_LEAD = re.compile(r"^\s*(\d+|[①-⑫]|[가-하]\)|[A-D]\)|[A-D]\.)[.)]?\s*")
def strip_leads(options: list[str]) -> list[str]:
    return [CLEAN_LEAD.sub("", o).strip() for o in options]

def clean_text(s: str) -> str:
    s = s.replace("〈", "").replace("〉", "").replace("<", "").replace(">", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\.{2,}", "…", s)
    return s

def extract_answer_only(generated_text: str, original_question: str) -> str:
    text = generated_text.split("답변:")[-1].strip() if "답변:" in generated_text else generated_text.strip()
    if not text:
        return "미응답"
    if is_multiple_choice(original_question):
        m = re.search(r"([1-9][0-9]?)", text)
        return m.group(1) if m else "0"
    return clean_text(text)

class FirstDigitOnly(LogitsProcessor):
    def __init__(self, tokenizer, allowed_digits: list[str]):
        self.tokenizer = tokenizer
        self.allowed_ids, self.step = set(), 0
        def add_single(s: str):
            for pre in ["", " ", "\n"]:
                ids = tokenizer.encode(pre + s, add_special_tokens=False)
                if len(ids) == 1:
                    self.allowed_ids.add(ids[0])
        for d in allowed_digits:
            add_single(d); add_single(d[0])
        self.allowed_ids = list(self.allowed_ids)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.step == 0:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.allowed_ids] = scores[:, self.allowed_ids]
            scores = mask
        self.step += 1
        return scores


# ----------------------
# Domain Router (룰 기반 휴리스틱)
# ----------------------
LAW_PATTERNS = [
    r"\b제\s?\d+\s?조\b", r"시행령", r"시행규칙", r"대통령령", r"총리령", r"부령",
    r"금융소비자", r"자본시장", r"금융투자업", r"투자매매업", r"투자중개업", r"투자자문업",
    r"금융위원회", r"개인정보보호법", r"정보통신망", r"전자금융거래", r"신용정보법"
]
MITRE_PATTERNS = [
    r"\bT\d{4}(?:\.\d{3})?\b", r"\bTA\d{4}\b", r"ATT&CK", r"MITRE", r"전술", r"기법",
    r"탐지", r"행위자", r"랜섬웨어", r"백도어", r"기반시설", r"명령제어", r"권한상승",
    r"지속성", r"측면이동", r"방어회피"
]

def detect_domain(question: str, options_clean: list[str] | None = None) -> str:
    """'law' | 'mitre' | 'unknown'"""
    text = question + "\n" + ("\n".join(options_clean) if options_clean else "")
    law_hit = any(re.search(p, text, flags=re.IGNORECASE) for p in LAW_PATTERNS)
    mitre_hit = any(re.search(p, text, flags=re.IGNORECASE) for p in MITRE_PATTERNS)
    if law_hit and not mitre_hit:
        return "law"
    if mitre_hit and not law_hit:
        return "mitre"
    if law_hit and mitre_hit:
        if re.search(r"\b제\s?\d+\s?조\b", text) or re.search(r"자본시장|금융투자업", text):
            return "law"
        return "mitre"
    return "unknown"


# ----------------------
# RAG helpers
# ----------------------
def build_context_block(ctxs: list[dict]) -> str:
    if not ctxs: return ""
    blocks = []
    for c in ctxs:
        head = c.get("header") or "[REF]"
        snippet = (c.get("summary") or c.get("text","")[:400]).replace("\n", " ")
        blocks.append(f"{head}\n{snippet}")
    guide = ("다음은 법령/규정/지침/정의/공격기법 관련 참조 문맥입니다. "
             "정답은 문맥과 질문에 근거하여 간결하고 정확하게 작성하세요. "
             "출처 표시는 하지 말고, 최종 답만 제시하세요.")
    return guide + "\n\n" + "\n\n".join(blocks)

def token_overlap_score(q: str, ctx_block: str) -> float:
    q_toks = set(re.findall(r"[가-힣A-Za-z0-9]+", q))
    c_toks = set(re.findall(r"[가-힣A-Za-z0-9]+", ctx_block))
    if not q_toks or not c_toks: return 0.0
    return len(q_toks & c_toks) / (len(q_toks) ** 0.5 * len(c_toks) ** 0.5 + 1e-6)


# ----------------------
# MCQ Continuation scoring (연속 토큰 로그확률 합)
# ----------------------
@torch.no_grad()
def score_options_continuation(model, tokenizer, question: str, options: list[str], contexts: list[dict]):
    """
    후보(정답 번호)의 '연속 토큰' 로그확률 합으로 스코어링.
    숫자 변형: '', ' ', '\\n' prefix × [d, d.], 그리고 ' d번' (한국어 스타일)까지 평가.
    각 옵션에 대해 최고 점수를 그 옵션의 점수로 사용.
    """
    clean_opts = strip_leads(options)
    ctx_block = build_context_block(contexts) if contexts else ""

    prompt = (
        "당신은 금융보안 전문가입니다. 아래 문항에 대해 정답 번호만 출력하세요.\n\n"
        f"[질문]\n{question}\n\n[선택지]\n" +
        "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(clean_opts)]) +
        (f"\n\n[근거]\n{ctx_block}\n" if ctx_block else "\n") +
        "\n정답:"
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_ids = enc["input_ids"][0]  # [L]

    def seq_logprob(cand_ids: torch.Tensor) -> float:
        full = torch.cat([prompt_ids, cand_ids], dim=0)  # [L + C]
        inp = full[:-1].unsqueeze(0)
        tgt = full[1:].unsqueeze(0)
        out = model(input_ids=inp)
        logp = torch.log_softmax(out.logits, dim=-1)      # [1, L+C-1, V]
        gather = torch.gather(logp, 2, tgt.unsqueeze(-1)).squeeze(-1)  # [1, L+C-1]
        cand_len = cand_ids.size(0)
        return gather[0, -cand_len:].sum().item()

    scores = []
    for i in range(len(clean_opts)):
        d = str(i+1)
        variants = []
        for pre in ["", " ", "\n"]:
            variants.append(pre + d)
            variants.append(pre + d + ".")
        variants.append(" " + d + "번")

        best = -1e30
        for v in variants:
            cand_ids = tokenizer(v, add_special_tokens=False, return_tensors="pt").to(model.device)["input_ids"][0]
            score = seq_logprob(cand_ids)
            if score > best:
                best = score
        scores.append(best)

    ans = int(torch.tensor(scores).argmax().item()) + 1
    return ans, scores, clean_opts


@torch.no_grad()
def quick_mcq_generate(model, tokenizer, question: str, clean_opts: list[str], ctxs: list[dict]) -> str:
    ctx_block = build_context_block(ctxs) if ctxs else ""
    prompt = (
        "당신은 금융보안 전문가입니다. 아래 문항에 대해 정답 번호만 출력하세요.\n\n"
        f"[질문]\n{question}\n\n[선택지]\n" +
        "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(clean_opts)]) +
        (f"\n\n[근거]\n{ctx_block}\n" if ctx_block else "\n") +
        "\n정답:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, do_sample=False, min_new_tokens=1, max_new_tokens=8,
        eos_token_id=int(tokenizer.eos_token_id), pad_token_id=int(tokenizer.pad_token_id),
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


# ----------------------
# Main
# ----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    p.add_argument("--adapter", type=str, required=True)
    p.add_argument("--test_csv", type=str, default="../dataset/test.csv")
    p.add_argument("--sample_sub", type=str, default="./sample_submission.csv")
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--max_choices", type=int, default=6)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--min_new_tokens_subjective", type=int, default=64)
    p.add_argument("--max_new_tokens_subjective", type=int, default=2048)

    # RAG
    p.add_argument("--rag", choices=["on","off"], default="on")
    p.add_argument("--index_dir_a", type=str, default=None)  # law
    p.add_argument("--index_dir_b", type=str, default=None)  # mitre
    p.add_argument("--rag_topk", type=int, default=3)
    p.add_argument("--rag_injection", choices=["system","user"], default="system")
    p.add_argument("--rag_log_top1", action="store_true")
    p.add_argument("--rag_strategy", choices=["concat","ensemble"], default="ensemble")

    # Domain router
    p.add_argument("--route_mode", choices=["hard","soft"], default="soft")
    p.add_argument("--route_bonus", type=float, default=0.15, help="soft 모드에서 선호 도메인 보정(+bonus×overlap)")

    # Subjective dual
    p.add_argument("--subj_dual_generate", action="store_true")

    # Decoding
    p.add_argument("--mcq_force_digit", action="store_true")

    # IO
    p.add_argument("--show_io", action="store_true")

    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"baseline_submission_{ts}.csv")

    # Model / Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval(); model.config.use_cache = True; model.config.pad_token_id = tok.pad_token_id

    # RAG init
    searcher_a = searcher_b = None
    if args.rag == "on":
        if args.index_dir_a:
            try:
                searcher_a = RagSearcher(args.index_dir_a, device=("cuda" if torch.cuda.is_available() else "cpu"))
                print(f"[RAG-A] {args.index_dir_a}")
            except Exception as e: print(f"[RAG-A] 실패: {e}")
        if args.index_dir_b:
            try:
                searcher_b = RagSearcher(args.index_dir_b, device=("cuda" if torch.cuda.is_available() else "cpu"))
                print(f"[RAG-B] {args.index_dir_b}")
            except Exception as e: print(f"[RAG-B] 실패: {e}")

    test = pd.read_csv(args.test_csv)
    preds = []

    for q in tqdm(test["Question"], desc="Inference"):
        messages = make_prompt_auto(q)

        # 옵션 파싱 + 검색 질의 강화
        options, clean_opts = None, None
        query_text = q
        if is_multiple_choice(q):
            try:
                _, options = extract_question_and_choices(q)
                clean_opts = strip_leads(options)
                query_text = q + "\n" + "\n".join(clean_opts)
            except Exception:
                pass

        # --- Domain routing ---
        preferred = detect_domain(q, clean_opts)
        ctxs_a = ctxs_b = []
        if args.rag == "on":
            try:
                if args.route_mode == "hard":
                    if preferred in ("law","unknown"):
                        if searcher_a:
                            ra = searcher_a.retrieve(query_text, k=args.rag_topk); 
                            ctxs_a = ra["contexts"] if ra.get("ok") else []
                            if not ctxs_a and searcher_b:  # 폴백
                                rb = searcher_b.retrieve(query_text, k=args.rag_topk); 
                                ctxs_b = rb["contexts"] if rb.get("ok") else []
                    else:  # mitre
                        if searcher_b:
                            rb = searcher_b.retrieve(query_text, k=args.rag_topk); 
                            ctxs_b = rb["contexts"] if rb.get("ok") else []
                            if not ctxs_b and searcher_a:  # 폴백
                                ra = searcher_a.retrieve(query_text, k=args.rag_topk); 
                                ctxs_a = ra["contexts"] if ra.get("ok") else []
                else:
                    # soft: 둘 다 검색
                    if searcher_a:
                        ra = searcher_a.retrieve(query_text, k=args.rag_topk); 
                        ctxs_a = ra["contexts"] if ra.get("ok") else []
                    if searcher_b:
                        rb = searcher_b.retrieve(query_text, k=args.rag_topk); 
                        ctxs_b = rb["contexts"] if rb.get("ok") else []
            except Exception as e:
                tq.tqdm.write(f"[RAG] 검색 실패: {e}")

        any_ctx = bool(ctxs_a or ctxs_b)

        # 공통 inputs (대화 템플릿)
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        # ---------------- MCQ ----------------
        if is_multiple_choice(q):
            if not args.mcq_force_digit and options:
                # 컨텍스트별 컨피던스 계산 (Continuation scoring)
                def mcq_conf(cx, tag):
                    ans, scores, clean_view = score_options_continuation(model, tok, q, options, cx)
                    block = build_context_block(cx)
                    conf = max(scores) + 0.10 * token_overlap_score(q, block)
                    # soft 라우팅 보너스
                    if args.route_mode == "soft" and preferred != "unknown":
                        is_law = (preferred == "law")
                        if (is_law and tag=="A") or ((not is_law) and tag=="B"):
                            conf += args.route_bonus * token_overlap_score(q, block)
                    return conf, ans, scores, clean_view, cx

                candidates = []
                if ctxs_a: candidates.append(("A",) + mcq_conf(ctxs_a, "A"))
                if ctxs_b: candidates.append(("B",) + mcq_conf(ctxs_b, "B"))

                if candidates:
                    # (tag, conf, ans, scores, clean_view, cx)
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    tag, conf, ans, scores, clean_view, chosen_ctxs = candidates[0]
                    pred = str(ans)
                    if args.rag_log_top1:
                        tq.tqdm.write(f"[MCQ-ENSEMBLE] prefer={preferred} pick={tag} conf={conf:.3f} ans={pred}")
                else:
                    # 컨텍스트 없음 → 컨텍스트 없이 스코어링
                    ans, scores, clean_view = score_options_continuation(model, tok, q, options, [])
                    pred = str(ans); chosen_ctxs = []

                # IO 및 재검증 가드: ModelOut 숫자와 불일치 시, 동일 방식으로 재평가
                gen_view = ""
                if args.show_io:
                    print("\n[MCQ-IO]")
                    print(q)
                    for i,opt in enumerate(clean_view,1): print(f"{i}. {opt}")
                    if chosen_ctxs:
                        heads = [c.get("header","[REF]") for c in chosen_ctxs[:3]]
                        if heads: print("[CtxHeaders]", " | ".join(heads))
                    gen_view = quick_mcq_generate(model, tok, q, clean_view, chosen_ctxs or [])
                    print("[ModelOut]", gen_view)

                m = re.search(r"([1-9][0-9]?)", gen_view or "")
                if m:
                    gen_digit = int(m.group(1))
                    if 1 <= gen_digit <= len(clean_view):
                        # 동일 컨텍스트로 재스코어
                        _ans2, all_scores, _ = score_options_continuation(model, tok, q, options, chosen_ctxs or [])
                        argmax_idx = int(torch.tensor(all_scores).argmax().item()) + 1
                        if gen_digit != argmax_idx and all_scores[gen_digit-1] >= all_scores[argmax_idx-1] - 1e-6:
                            pred = str(gen_digit)

                if args.show_io:
                    print("[Answer]", pred)
                    print("-"*60)

            else:
                # 숫자 강제 대체 경로
                allowed = [str(i) for i in range(1, args.max_choices + 1)]
                processors = LogitsProcessorList([FirstDigitOnly(tok, allowed)])
                out_ids = model.generate(
                    **inputs, do_sample=False, min_new_tokens=1, max_new_tokens=4,
                    logits_processor=processors, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                )
                gen = tok.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                m = re.search(r"([1-9][0-9]?)", gen); pred = m.group(1) if m else "0"
                if args.show_io:
                    print("\n[MCQ-IO]"); print(q)
                    if options:
                        for i,o in enumerate(strip_leads(options),1): print(f"{i}. {o}")
                    print("[ModelOut]", gen.strip()); print("[Answer]", pred); print("-"*60)

        # ---------------- 주관식 ----------------
        else:
            stoppers = StoppingCriteriaList([RegexStoppingCriteria(tok)])

            if args.rag_strategy == "ensemble" and args.subj_dual_generate and any_ctx:
                def gen_with(cx):
                    local = make_prompt_auto(q)
                    block = build_context_block(cx) if cx else ""
                    if block:
                        if args.rag_injection == "system":
                            local = [{"role": "system", "content": block}] + local
                        else:
                            local = [{"role": "user", "content": f"[참고문맥]\n{block}"}] + local
                    loc_inputs = tok.apply_chat_template(
                        local, add_generation_prompt=True, tokenize=True,
                        return_tensors="pt", return_dict=True
                    ).to(model.device)
                    loc_inputs.pop("token_type_ids", None)
                    outs = model.generate(
                        **loc_inputs, do_sample=False,
                        min_new_tokens=int(args.min_new_tokens_subjective),
                        max_new_tokens=int(args.max_new_tokens_subjective),
                        no_repeat_ngram_size=3, repetition_penalty=1.05,
                        eos_token_id=int(tok.eos_token_id), pad_token_id=int(tok.pad_token_id),
                        stopping_criteria=stoppers,
                    )
                    full = tok.decode(outs[0][loc_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                    ans = extract_answer_only(full, q)
                    return ans, full, block

                a_ans = a_full = a_block = ""
                b_ans = b_full = b_block = ""
                if ctxs_a: a_ans, a_full, a_block = gen_with(ctxs_a)
                if ctxs_b: b_ans, b_full, b_block = gen_with(ctxs_b)

                def score(ans, block, is_pref):
                    base = token_overlap_score(q, block or "") + 0.02 * min(len(ans), 400)
                    if args.route_mode == "soft" and is_pref:
                        base += args.route_bonus * token_overlap_score(q, block or "")
                    return base

                domain_pref_is_law = (preferred == "law")
                sa = score(a_ans, a_block, domain_pref_is_law)
                sb = score(b_ans, b_block, (preferred == "mitre"))

                if sa >= sb:
                    pred, chosen_full, chosen_ctxs, who = a_ans, a_full, ctxs_a, "A"
                else:
                    pred, chosen_full, chosen_ctxs, who = b_ans, b_full, ctxs_b, "B"

                if args.show_io:
                    print("\n[SUBJ-IO]")
                    print(q)
                    heads = [c.get("header","[REF]") for c in (chosen_ctxs or [])[:3]]
                    if heads: print("[CtxHeaders]", " | ".join(heads))
                    print(f"[Pick]{who}  scoreA={sa:.3f}  scoreB={sb:.3f}")
                    print("[ModelOut]\n", chosen_full)
                    print("[Answer]", pred)
                    print("-"*60)
            else:
                # concat 또는 컨텍스트 한쪽만 있는 경우
                block = build_context_block((ctxs_a or []) + (ctxs_b or []))
                msgs = make_prompt_auto(q)
                if block:
                    if args.rag_injection == "system":
                        msgs = [{"role": "system", "content": block}] + msgs
                    else:
                        msgs = [{"role": "user", "content": f"[참고문맥]\n{block}"}] + msgs
                loc_inputs = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True,
                    return_tensors="pt", return_dict=True
                ).to(model.device)
                loc_inputs.pop("token_type_ids", None)
                outs = model.generate(
                    **loc_inputs, do_sample=False,
                    min_new_tokens=int(args.min_new_tokens_subjective),
                    max_new_tokens=int(args.max_new_tokens_subjective),
                    no_repeat_ngram_size=3, repetition_penalty=1.05,
                    eos_token_id=int(tok.eos_token_id), pad_token_id=int(tok.pad_token_id),
                    stopping_criteria=StoppingCriteriaList([RegexStoppingCriteria(tok)]),
                )
                gen_text = tok.decode(outs[0][loc_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                pred = extract_answer_only(gen_text, q)
                if args.show_io:
                    print("\n[SUBJ-IO]")
                    print(q)
                    heads = [c.get("header","[REF]") for c in ((ctxs_a or []) + (ctxs_b or []))[:3]] if block else []
                    if heads: print("[CtxHeaders]", " | ".join(heads))
                    print("[ModelOut]\n", gen_text)
                    print("[Answer]", pred)
                    print("-"*60)

        preds.append(pred)

    # 저장
    sub = pd.read_csv(args.sample_sub)
    if "Answer" not in sub.columns or "ID" not in sub.columns:
        raise ValueError("sample_submission.csv에는 'ID'와 'Answer' 컬럼이 모두 있어야 합니다.")
    if len(sub) != len(preds):
        raise ValueError(f"예측 개수({len(preds)})와 sample 행 수({len(sub)})가 다릅니다.")
    sub = sub.sort_values("ID").reset_index(drop=True)
    out_map = pd.DataFrame({"ID": test["ID"].tolist(), "Answer": preds})
    out_df = sub[["ID"]].merge(out_map, on="ID", how="left")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()