# -*- coding: utf-8 -*-
"""
test.csv → submission.csv
- 분기: load.py의 is_multiple_choice / extract_question_and_choices / make_prompt_auto 만 사용
- 객관식: 허용 숫자(1~K) 1토큰만 생성(AllowedTokensProcessor)
- 주관식: 간결 생성(beam search), 길이 잘림 방지용 headroom 인코딩
- 제출형식: sample_submission.csv의 컬럼 구조를 그대로 따라 생성
- 진행률: tqdm
"""
from __future__ import annotations
import re, json, inspect
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from allowed_tokens import get_digit_processor
from load import make_prompt_auto, is_multiple_choice, extract_question_and_choices
from tqdm import tqdm

# --------------------- 경로/모델 ---------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "out-causallm-qlora"
BASE = "taetae030/fin-term-model"

# --------------------- 유틸 ---------------------
def _get_model_max_len(tok, model):
    m = getattr(model.config, "max_position_embeddings", None)
    if not m or m == float("inf"):
        m = getattr(tok, "model_max_length", None)
    if not m or m > 32000:
        m = 4096
    return int(m)

def _latest_run_dir() -> Path:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"모델 출력 폴더가 없습니다: {MODEL_DIR}")
    runs = sorted([p for p in MODEL_DIR.glob("*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"실행 결과 폴더가 비어 있습니다: {MODEL_DIR}")
    return runs[-1]

def _find_peft_config_file(adapter_dir: Path) -> Path:
    cands = [adapter_dir / "adapter_config.json", adapter_dir / "peft_config.json"]
    for p in cands:
        if p.exists():
            return p
    for p in adapter_dir.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "peft_type" in data:
                return p
        except Exception:
            pass
    raise FileNotFoundError(f"PEFT 설정 파일을 찾지 못했습니다: {adapter_dir}")

def _sanitize_adapter_config(adapter_dir: Path) -> None:
    from peft import LoraConfig
    from peft.config import PeftConfig
    cfg_path = _find_peft_config_file(adapter_dir)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    def _params(cls):
        return set(k for k in inspect.signature(cls.__init__).parameters.keys() if k != "self")

    allowed = _params(LoraConfig) | _params(PeftConfig)
    meta_keep = {"peft_type", "base_model_name_or_path", "task_type", "revision", "inference_mode", "adapter_layers_pattern"}
    new_cfg = {}
    for k, v in cfg.items():
        if (k in allowed) or (k in meta_keep):
            new_cfg[k] = v
    new_cfg.pop("corda_config", None)

    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(new_cfg, f, ensure_ascii=False, indent=2)

def load_model(run_dir: Path | None = None):
    run_dir = run_dir or _latest_run_dir()
    adapter_dir = run_dir / "adapter"
    token_dir   = run_dir / "tokenizer"

    tok = AutoTokenizer.from_pretrained(str(token_dir), use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    from peft import PeftModel
    try:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    except TypeError as e:
        print(f"[warn] PEFT 어댑터 로드 실패({e}). 설정을 패치합니다...")
        _sanitize_adapter_config(adapter_dir)
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return tok, model

def encode_chat(tokenizer, model, messages, want_new_tokens: int, reserve: int = 32):
    """
    chat messages → (input_ids, attention_mask, model_max, allowed_new)
    - 입력을 truncation으로 잘라 생성 headroom 확보(잘림 방지)
    """
    model_max = _get_model_max_len(tokenizer, model)
    raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    max_input = max(8, model_max - want_new_tokens - reserve)
    enc = tokenizer(raw, return_tensors="pt", truncation=True, max_length=max_input)
    input_ids = enc["input_ids"]
    attn      = enc.get("attention_mask")
    allowed_new = max(1, model_max - input_ids.shape[1] - 1)
    gen_new     = min(want_new_tokens, allowed_new)
    return input_ids, attn, model_max, gen_new

def build_fulltext_from_row(row: pd.Series, q_col: str, opt_cols: list[str] | None) -> str:
    """
    test.csv 형태에서 '질문 + (있다면) 줄단위 보기 "1 보기..."' 텍스트 생성
    - load.py의 is_multiple_choice / make_prompt_auto 가 그대로 쓰일 수 있도록 줄바꿈 포맷 유지
    """
    q = str(row[q_col]).strip()
    if opt_cols:
        choices = [str(row[c]).strip() for c in opt_cols if pd.notna(row[c]) and str(row[c]).strip()]
        lines = [q] + [f"{i} {c}" for i, c in enumerate(choices, 1)]
        return "\n".join(lines)
    return q

def guess_columns(df: pd.DataFrame):
    # 질문 컬럼 추정
    q_candidates = ["question", "문제", "질문", "text", "내용", "problem", "prompt"]
    q_col = None
    for k in q_candidates:
        for c in df.columns:
            if k in c.lower():
                q_col = c
                break
        if q_col:
            break
    if q_col is None:
        q_col = df.columns[0]

    # 보기 컬럼 추정
    opt_cols = []
    for c in df.columns:
        cl = c.lower()
        if ("choice" in cl) or ("option" in cl) or ("보기" in cl) or ("선택" in cl):
            opt_cols.append(c)
    if opt_cols:
        def key_idx(name: str) -> int:
            m = re.search(r"(\d+)", name)
            return int(m.group(1)) if m else 999
        opt_cols = sorted(opt_cols, key=key_idx)
    else:
        opt_cols = None
    return q_col, opt_cols

# --------------------- 메인 ---------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default=str(ROOT / "data" / "test.csv"))
    parser.add_argument("--sample_csv", type=str, default=str(ROOT / "data" / "sample_submission.csv"))
    parser.add_argument("--out_csv",  type=str, default=str(ROOT / "submission.csv"))
    parser.add_argument("--run_dir",  type=str, default=None, help="특정 체크포인트 경로(선택)")
    parser.add_argument("--use_rag", action="store_true", help="(옵션) rag_index_and_search.py 사용")
    parser.add_argument("--max_gen_tokens", type=int, default=4096)
    parser.add_argument("--print_each", action="store_true", help="각 문제/출력 프린트", default=True)
    args = parser.parse_args()

    # (선택) RAG
    USE_RAG = args.use_rag
    try:
        from fin.scripts.rag_laws import RagSearcher
        searcher = RagSearcher(ROOT) if USE_RAG else None
        if USE_RAG: print("[RAG] 사용")
    except Exception:
        searcher = None
        USE_RAG = False
        print("[RAG] 미사용")

    tok, model = load_model(Path(args.run_dir) if args.run_dir else None)
    df_test = pd.read_csv(args.test_csv)
    q_col, opt_cols = guess_columns(df_test)

    # 제출 포맷 로드 (sample_submission을 그대로 복제해서 prediction만 채움)
    df_sample = pd.read_csv(args.sample_csv)
    sub_cols = list(df_sample.columns)
    # 예: ["id","prediction"] 형태라고 가정하되, 마지막 컬럼을 예측값 컬럼으로 사용
    pred_col = sub_cols[-1]

    preds: list[str] = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Infer"):
        full_text = build_fulltext_from_row(row, q_col, opt_cols)

        # (선택) RAG 컨텍스트 결합
        if USE_RAG and searcher:
            docs = searcher.search(full_text, top_k=3)
            ctx = "\n\n".join([f"{i+1}) {d.get('source','')}\n{d['text']}" for i, d in enumerate(docs)])
            full_text = f"{full_text}\n\n[참고자료]\n{ctx}"

        # load.py만으로 분기
        messages = make_prompt_auto(full_text)
        WANT_NEW = 1 if is_multiple_choice(full_text) else args.max_gen_tokens
        inputs, attn, _, gen_new = encode_chat(tok, model, messages, want_new_tokens=WANT_NEW, reserve=32)
        inputs = inputs.to(model.device)
        attn   = attn.to(model.device) if attn is not None else None

        if is_multiple_choice(full_text):
            # 보기 개수에 맞춰 허용 숫자 집합 산출
            _, options = extract_question_and_choices(full_text)
            k = max(2, min(len(options), 9)) if options else 4  # 안전 기본값 4
            digits_str = "123456789"[:k]
            digit_proc = get_digit_processor(tok, digits=digits_str)

            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs,
                    attention_mask=attn,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0.0,
                    logits_processor=[digit_proc],
                )
            gen_ids = out[0][inputs.shape[1]:]
            txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
            pred = txt[-1:] if txt else ""
            if pred not in set(digits_str):
                pred = digits_str[0]
            if args.print_each:
                print("\n[객관식]")
                print(full_text)
                print(f"→ 모델 출력(정답 번호): {pred}")
            preds.append(pred)

        else:
            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs,
                    attention_mask=attn,
                    max_new_tokens=gen_new,
                    do_sample=False,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    length_penalty=0.9,
                    early_stopping=False,
                )
            gen_ids = out[0][inputs.shape[1]:]
            ans = tok.decode(gen_ids, skip_special_tokens=True).strip()

            if args.print_each:
                print("\n[주관식]")
                print(full_text)
                print(f"→ 모델 출력(답변): {ans}")
            preds.append(ans)

    # sample_submission 형식 맞추기
    df_out = df_sample.copy()
    if len(df_out) != len(preds):
        # test와 sample 길이가 다르면, test 순서대로 예측하고 sample의 길이에 맞춰 잘라/채움
        if len(preds) >= len(df_out):
            preds = preds[:len(df_out)]
        else:
            preds = preds + [""] * (len(df_out) - len(preds))
    df_out[pred_col] = preds
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] saved: {args.out_csv}")

if __name__ == "__main__":
    main()