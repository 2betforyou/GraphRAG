from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from zoneinfo import ZoneInfo 

from transformers import LogitsProcessorList, LogitsProcessor   # ▼ 추가: 제약 생성용
from load import make_prompt_auto, is_multiple_choice, extract_question_and_choices
from retriever import load_corpus_and_build_retriever 


# ▼ 추가: 첫 생성 스텝만 선택지 숫자(예: 1~5, 1~10 등)로 제한
class FirstDigitOnly(LogitsProcessor):
    """첫 생성 스텝에서 허용된 숫자들만 나오도록 제한"""
    def __init__(self, tokenizer, allowed_digits: list[str]):
        # allowed_digits는 ["1","2","3","4","5"] 같은 문자열 리스트
        self.allowed_ids = []
        for d in allowed_digits:
            # 첫 글자(예: '1')의 토큰만 제한하면 됨. 두 자리(10,11)는 2번째 토큰이 자유롭게 뒤따름.
            tid = tokenizer.encode(d[0], add_special_tokens=False)
            if len(tid) > 0:
                self.allowed_ids.append(tid[0])
        self.step = 0

    def __call__(self, input_ids, scores):
        if self.step == 0:
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.allowed_ids] = scores[:, self.allowed_ids]
            scores = mask
        self.step += 1
        return scores


LAW_KEYWORDS = ["법", "조문", "시행령", "시행규칙",
                "전자금융거래법","개인정보보호법","신용정보법",
                "전자서명법","정보통신망법","전자문서","전자거래","금융실명"]
def is_law_question(text: str) -> bool:
    t = (text or "")
    return any(k in t for k in LAW_KEYWORDS)


def extract_answer_only(generated_text: str, original_question: str) -> str:
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()
    if not text:
        return "미응답"
    if is_multiple_choice(original_question):
        m = re.match(r"\D*([1-9][0-9]?)", text)
        return m.group(1) if m else "0"
    return text


DEBUG_FULL_SUBJECTIVE = True 

def clean_subjective(s: str) -> str:
    s = s.replace("〈","").replace("〉","").replace("<","").replace(">","")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\.{2,}", "…", s)   # ... → …
    return s

def compress_subjective(text, max_chars=120):
    s = clean_subjective(text)
    sents = re.split(r'(?<=[.!?。…])\s+', s)
    s = " ".join(sents[:2]).strip()
    return s[:max_chars].rstrip()


def main():
    test_path = "../dataset/test.csv"
    sub_in_path = "./sample_submission.csv"

    ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    sub_out_path = f"./baseline_submission_{ts}.csv" 
    latest_out_path = "./baseline_submission.csv" 

    base = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" 
    adapter = "out-exaone-law-qlora/20250827_162808/adapter"

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # (B) 코퍼스 로드 + 인덱싱(1회)
    retriever, chunks = load_corpus_and_build_retriever([
        "../dataset/laws/*.txt",
        "../dataset/전자금융거래법.txt",
        "../dataset/개인정보보호법.txt",
        "../dataset/신용정보법.txt",
        "../dataset/전자서명법.txt",
        "../dataset/정보통신망법.txt",
        "../dataset/전자문서및전자거래기본법.txt",
        "../dataset/금융실명거래및비밀보장법.txt",
    ])
    use_rag = bool(chunks)
    print(f"[RAG] loaded chunks: {len(chunks)}")

    test = pd.read_csv(test_path)
    preds = []

    for q in tqdm(test['Question'], desc="Inference"):
        # (C-1) 검색용 쿼리 만들기
        if is_multiple_choice(q):
            question_only, options = extract_question_and_choices(q)
        else:
            question_only = q

        # (C-2) 검색 → 컨텍스트 블록 (법령 질문일 때만 RAG)
        use_rag_q = use_rag and is_law_question(question_only) 
        context_block = ""
        if use_rag_q:
            ctx = retriever.search(question_only, k=3)
            ctx_lines = [f"- {c[2][:500]}" for c in ctx if c[2].strip()]
            if ctx_lines:
                context_block = ("참고자료(참고하여 재서술, 복붙 금지, 핵심 키워드 반영):\n"
                                + "\n".join(ctx_lines) + "\n\n")

        # (C-3) 최종 프롬프트
        task_prompt = make_prompt_auto(q) 
        full_prompt = context_block + task_prompt

        messages = [
            {"role": "system", "content": "당신은 금융보안 전문 AI 어시스턴트입니다. 질문에 대해 명확하고 간결하게 답변하세요."},
            {"role": "user", "content": full_prompt}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)
        
        inputs.pop("token_type_ids", None)

        # ▼▼▼ 객관식: 숫자 제약 생성 적용 ▼▼▼
        if is_multiple_choice(q):
            # 보기에서 실제 숫자 후보 추출 (1,2,3,4,5,...,10,11 등)
            _, options = extract_question_and_choices(q)
            nums = sorted(set(re.findall(r'^\s*([1-9][0-9]?)', "\n".join(options), re.M)))
            if not nums:
                nums = ["1","2","3","4","5"]  # 파싱 실패시 기본값

            processors = LogitsProcessorList([FirstDigitOnly(tokenizer, nums)])

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    min_new_tokens=1,      # 최소 한 토큰(숫자) 생성
                    max_new_tokens=2,      # 숫자 + 개행/공백 수준
                    logits_processor=processors,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            # 안전장치: 첫 숫자만 취득 (두 자리까지)
            m = re.search(r'([1-9][0-9]?)', gen)
            pred_answer = m.group(1) if m else "0"

            # 모니터링 출력(간단)
            print(f"[Q]\n{q}\n[Gen(MCQ)] 답변: {pred_answer}\n[CTX]\n{context_block}\n[A]\n{pred_answer}\n{'-'*50}")

        else:
            # ▼▼▼ 주관식: 완결성/간결성 보장 ▼▼▼
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    min_new_tokens=80, 
                    max_new_tokens=256, 
                    no_repeat_ngram_size=3, 
                    repetition_penalty=1.05,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            
            pred_answer = extract_answer_only(gen_text, original_question=q)
            if DEBUG_FULL_SUBJECTIVE:
                pred_answer = clean_subjective(pred_answer)
            else:
                pred_answer = compress_subjective(pred_answer, max_chars=120)

            print(f"[Q]\n{q}\n[Gen]\n{gen_text}\n[CTX]\n{context_block}\n[A]\n{pred_answer}\n{'-'*50}")

        preds.append(pred_answer)

    sub = pd.read_csv(sub_in_path)
    sub['Answer'] = preds
    
    sub.to_csv(sub_out_path, index=False, encoding='utf-8-sig')
    # sub.to_csv(latest_out_path, index=False, encoding='utf-8-sig')

    print(f"Saved -> {sub_out_path}")
    # print(f"(latest copy) -> {latest_out_path}")



if __name__ == "__main__":
    main()