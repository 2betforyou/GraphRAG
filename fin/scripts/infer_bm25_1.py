#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference with optional BM25 RAG context + load.make_prompt_auto
- Two-stage MCQ (optional): (1) rationale, (2) final numeric choice
- For MCQ, we can either return only a digit (default) or explanation + digit (--mcq_explain)
- For subjective, deterministic decoding with repetition control
"""
import argparse, json, re, datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    LogitsProcessor, LogitsProcessorList,
    StoppingCriteria, StoppingCriteriaList
)
from peft import PeftModel

from retriever_bm25 import load_index
from load import is_multiple_choice, extract_question_and_choices, make_prompt_auto


# ----------------- Logits processors -----------------
class KoreanOnlyLogitsProcessor(LogitsProcessor):
    """한글/숫자/기본 구두점만 허용"""
    def __init__(self, tokenizer, allow_digits=True, allow_basic_punct=True):
        self.allow_digits = allow_digits
        self.allow_basic_punct = allow_basic_punct
        self.allowed_token_ids = self._build_allowed_token_ids(tokenizer)

    def _is_koreanish(self, s: str) -> bool:
        if not s:
            return False
        for ch in s:
            oc = ord(ch)
            if ch.isspace():
                continue
            if self.allow_digits and ch.isdigit():
                continue
            if self.allow_basic_punct and ch in ".,:-()·%/~[]{}'\"?!":
                continue
            # 한글(가-힣) + 자모
            if (0xAC00 <= oc <= 0xD7A3) or (0x1100 <= oc <= 0x11FF) or (0x3130 <= oc <= 0x318F):
                continue
            return False
        return True

    def _build_allowed_token_ids(self, tokenizer):
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        allowed = []
        for tid in range(vocab_size):
            s = tokenizer.decode([tid], skip_special_tokens=True)
            if self._is_koreanish(s):
                allowed.append(tid)
        return set(allowed)

    def __call__(self, input_ids, scores, **kwargs):
        if not self.allowed_token_ids:
            return scores
        vocab_size = scores.shape[-1]
        mask = torch.ones(vocab_size, dtype=torch.bool, device=scores.device)
        mask[list(self.allowed_token_ids)] = False
        scores[:, mask] = float("-inf")
        return scores


class StopAfterDigit(StoppingCriteria):
    """허용 숫자 토큰이 생성되는 순간 멈춤"""
    def __init__(self, allowed_token_ids, prompt_len):
        self.allowed = set(allowed_token_ids)
        self.prompt_len = prompt_len
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] > self.prompt_len and input_ids[0, -1].item() in self.allowed:
            return True
        return False


class FirstFromOptionsOnly(LogitsProcessor):
    """MCQ: 첫 토큰을 보기 번호로만 제한"""
    def __init__(self, tokenizer, allowed_str_digits):
        self.allowed_ids = []
        for s in allowed_str_digits:
            ids = tokenizer(s, add_special_tokens=False)["input_ids"]
            self.allowed_ids.append(ids[0])
        self.first_step = True
    def __call__(self, input_ids, scores, **kwargs):
        if self.first_step:
            keep = torch.full_like(scores, float("-inf"))
            keep[:, self.allowed_ids] = scores[:, self.allowed_ids]
            self.first_step = False
            return keep
        return scores


# ----------------- small helpers -----------------
def build_chat_inputs(tok, messages):
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True
    )

def squash_repetition(text: str):
    # 문장 반복 축약
    lines = [l.strip() for l in re.split(r"(?<=[.!?])\s+", text)]
    seen, keep = {}, []
    for l in lines:
        if not l:
            continue
        if seen.get(l, 0) == 0:
            keep.append(l)
        seen[l] = seen.get(l, 0) + 1
    out = " ".join(keep).strip()
    out = re.sub(r"(\b.{3,10}?\b)(?:\s*\1){2,}", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out

def first_n_sentences(s: str, n=2):
    parts = re.split(r"(?<=[\.!?]|[。！？])\s+", s.strip())
    return " ".join([p for p in parts[:n] if p]).strip()

def truncate_chars(s: str, max_chars=220):
    s = s.strip()
    return s if len(s) <= max_chars else (s[:max_chars].rstrip() + "…")

def compress_hits(hits, max_hits=5, sent_per_hit=2, max_chars_per_hit=220):
    out = []
    for h in hits[:max_hits]:
        t = first_n_sentences(h["text"], n=sent_per_hit)
        t = truncate_chars(t, max_chars=max_chars_per_hit)
        if t:
            out.append(f"- {t}")
    return "\n".join(out)

def call_generate(model, inputs, lp_list, **gen_kwargs):
    """logits_processor가 비어있으면 인자 자체를 생략하여 시그니처 에러 방지"""
    use_lp = (lp_list is not None and hasattr(lp_list, "__len__") and len(lp_list) > 0)
    if use_lp:
        return model.generate(**inputs, logits_processor=lp_list, **gen_kwargs)
    else:
        return model.generate(**inputs, **gen_kwargs)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--topk", type=int, default=7)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--test_csv", type=str, default=None)
    # MCQ 설명 모드
    ap.add_argument("--mcq_explain", action="store_true", help="MCQ에서 설명 + 최종 숫자 정답을 출력")
    ap.add_argument("--mcq_explain_tokens", type=int, default=240, help="MCQ 설명 단계 최대 토큰")
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    # pad 토큰 안전 보정(미스트랄 계열 등)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    retriever = load_index(".") if args.rag else None

    def run_one(q_text: str):
        # 0) 기본 메시지
        messages = make_prompt_auto(q_text)

        # 1) RAG(선택)
        if retriever is not None:
            hits = retriever.search(q_text, topk=args.topk)
            ctx = compress_hits(hits, max_hits=5, sent_per_hit=2, max_chars_per_hit=220)
            guide = (
                "다음은 검색으로 찾은 참고 단편입니다. "
                "핵심만 근거로 삼아 **한국어**로 간결히 답하세요. "
                "같은 문장/구를 반복하지 마세요.\n" + ctx + "\n\n"
            )
            # make_prompt_auto가 [system, user]를 반환하므로, user 턴(content)에 붙이기
            messages[-1]["content"] = guide + messages[-1]["content"]

        is_mc = is_multiple_choice(q_text)

        # 2) MCQ 처리 분기
        if is_mc and args.mcq_explain:
            # --- 2-1단계: 근거/설명 생성 ---
            messages_explain = list(messages)
            messages_explain[-1]["content"] += (
                "\n\n지시: 선택지들을 비교하여 정답이 되는 **한 가지 선택지 번호**를 추론하되,"
                " 먼저 **오직 한글만을 사용해** 2~3문장으로 근거를 설명한 뒤,"
                " 마지막 줄에 `정답: N` 형식으로 숫자 하나만 제시하시오."
            )
            inputs1 = build_chat_inputs(tok, messages_explain).to(model.device)
            logits_proc1 = LogitsProcessorList([KoreanOnlyLogitsProcessor(tok)])

            with torch.no_grad():
                out1 = call_generate(
                    model, inputs1, logits_proc1,
                    do_sample=False,
                    max_new_tokens=256,
                    temperature=0.0, top_p=1.0,
                    no_repeat_ngram_size=2, repetition_penalty=1.05,
                )
            rationale = tok.decode(out1[0][inputs1["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rationale = squash_repetition(rationale)

            # --- 2-2단계: 결론 강제 (정답 숫자만) ---
            _, options = extract_question_and_choices(q_text)
            allowed_digits = []
            for line in options:
                m = re.match(r"^\s*([1-9][0-9]?)\b", line)
                if m:
                    allowed_digits.append(m.group(1)[0])
            allowed_digits = sorted(set(allowed_digits)) or ["1","2","3","4","5"]

            decision_cue = (
                "\n\n이제 위 설명을 근거로 최종 결론만 제시하시오."
                "\n형식: `정답: N` (N은 보기 번호, 숫자 하나)"
                "\n다른 텍스트를 추가하지 말 것. 정답만."
                "\n정답: "
            )
            messages_decide = [
                {"role":"system", "content":"당신은 금융보안 전문가입니다. 모든 답변은 **한국어**로 간결하고 정확해야 합니다."},
                {"role":"user", "content": q_text + "\n\n" + "설명:\n" + rationale + decision_cue}
            ]
            inputs2 = build_chat_inputs(tok, messages_decide).to(model.device)

            lp_decide = LogitsProcessorList([FirstFromOptionsOnly(tok, allowed_digits)])
            allowed_token_ids = []
            for s in allowed_digits:
                ids = tok(s, add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    allowed_token_ids.append(ids[0])
            stopping = StoppingCriteriaList([StopAfterDigit(allowed_token_ids, inputs2["input_ids"].shape[1])])

            with torch.no_grad():
                out2 = call_generate(
                    model, inputs2, lp_decide,
                    stopping_criteria=stopping,
                    do_sample=False,
                    max_new_tokens=2,
                    temperature=0.0, top_p=1.0,
                )
            tail = tok.decode(out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            m = re.search(rf"({'|'.join(map(re.escape, allowed_digits))})", tail)
            digit = m.group(1) if m else allowed_digits[0]

            pretty = f"{rationale}\n정답: {digit}"

            print("────────────────────────────")
            print(f"[Question]\n{q_text}\n")
            print(f"[Answer]\n{pretty}\n")
            return pretty, digit

        else:
            # 기존 경로 (MCQ: 숫자만 / 주관식: 서술)
            inputs = build_chat_inputs(tok, messages).to(model.device)

            gen_kwargs = dict(do_sample=False)
            stopping = None

            if is_mc:
                _, options = extract_question_and_choices(q_text)
                allowed_digits = []
                for line in options:
                    m = re.match(r"^\s*([1-9][0-9]?)\b", line)
                    if m:
                        allowed_digits.append(m.group(1)[0])
                allowed_digits = sorted(set(allowed_digits)) or ["1","2","3","4","5"]

                lp_std = LogitsProcessorList([FirstFromOptionsOnly(tok, allowed_digits)])
                gen_kwargs.update(
                    max_new_tokens=4,
                    temperature=0.0, top_p=1.0,
                    no_repeat_ngram_size=2, repetition_penalty=1.05,
                )
                allowed_token_ids = []
                for s in allowed_digits:
                    ids = tok(s, add_special_tokens=False)["input_ids"]
                    if len(ids) == 1:
                        allowed_token_ids.append(ids[0])
                stopping = StoppingCriteriaList([StopAfterDigit(allowed_token_ids, inputs["input_ids"].shape[1])])
            else:
                lp_std = LogitsProcessorList([KoreanOnlyLogitsProcessor(tok)])
                gen_kwargs.update(
                    max_new_tokens=min(args.max_new, 2048),
                    temperature=0.0, top_p=1.0,
                    no_repeat_ngram_size=3, repetition_penalty=1.15,
                    encoder_repetition_penalty=1.05
                )

            with torch.no_grad():
                outputs = call_generate(
                    model, inputs, lp_std,
                    stopping_criteria=stopping if stopping else None,
                    **gen_kwargs,
                )
            raw = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            if is_mc:
                m = re.search(r"([1-9])", raw)
                digit = m.group(1) if m else "1"
                ans = digit
            else:
                ans = squash_repetition(raw)
                digit = None

            print("────────────────────────────")
            print(f"[Question]\n{q_text}\n")
            print(f"[Answer]\n{ans}\n")
            return ans, digit

    # 배치 모드(대회 제출)
    if args.test_csv:
        import pandas as pd
        df = pd.read_csv(args.test_csv)
        preds = []
        for q in df["Question"].tolist():
            ans_text, digit = run_one(q)
            final = digit if digit is not None else ans_text
            preds.append(final)
        df["Answer"] = preds
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.test_csv).with_name(f"submission_{timestamp}.csv")
        df[["ID","Answer"]].to_csv(out_path, index=False)
        print(f"[OK] wrote -> {out_path}")
    else:
        # 대화형
        print("Type a question (Ctrl+C to quit):")
        while True:
            try:
                q = input("> ").strip()
                if not q:
                    continue
                run_one(q)
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()