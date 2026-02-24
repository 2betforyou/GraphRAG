# -*- coding: utf-8 -*-
"""
test.csv → submission.csv
- 분기: load.py의 is_multiple_choice(text)만 사용
- 프롬프트: load.py의 make_prompt_auto(text)만 사용
- 객관식: 숫자 1토큰만 생성(AllowedTokensProcessor; 1~9 허용)
- 주관식: 한두 문장으로 간결 생성
- (선택) RAG: data/index/laws_e5 의 embeddings.npy + meta.jsonl 사용
- 진행률: tqdm progress bar
- 콘솔 출력: 문제와 모델 출력 프린트
"""
import os
import re
import json
import inspect
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from scripts.allowed_tokens import get_digit_processor 
from scripts.load import make_prompt_auto, is_multiple_choice  # ← 여기 두 함수만 사용

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "out-causallm-qlora"
BASE = "taetae030/fin-term-model"


# --- replace RagSearcher with this GPU-only version ---
class RagSearcher:
    """
    - Index: {root}/data/index/laws_e5/{embeddings.npy, meta.jsonl}
    - Query embedding: E5 (default: intfloat/e5-base-v2)
    - Cosine similarity on GPU only
    """
    def __init__(self, root: Path, index_subdir: str = "data/index/laws_e5",
                 e5_model: str = "intfloat/e5-base-v2", device: str | None = None):
        # 0) GPU only
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 가용하지 않습니다. GPU 전용 모드에서는 CUDA가 필요합니다.")
        self.device = torch.device(device or "cuda:0")

        self.root = Path(root)
        self.index_dir = self.root / index_subdir
        self.emb_path = self.index_dir / "embeddings.npy"
        self.meta_path = self.index_dir / "meta.jsonl"

        if not self.emb_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"RAG 인덱스 파일을 찾지 못했습니다: {self.index_dir}")

        # 1) 문서 임베딩을 GPU로, dtype 통일(대개 fp16이 가장 빠름)
        embs_np = np.load(self.emb_path).astype(np.float32, copy=False)
        self.doc_embs = torch.from_numpy(embs_np).to(self.device, dtype=torch.float16, non_blocking=True)
        # 정규화도 GPU에서
        self.doc_norm = torch.nn.functional.normalize(self.doc_embs, dim=1)

        # 2) 메타 로드
        self.meta = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.meta.append(json.loads(line))
        if len(self.meta) != self.doc_embs.shape[0]:
            raise ValueError(f"메타({len(self.meta)})와 임베딩({self.doc_embs.shape[0]}) 개수가 다릅니다.")

        # 3) E5 로더(GPU)
        self.tok = AutoTokenizer.from_pretrained(e5_model, use_fast=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(e5_model, trust_remote_code=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _embed_query(self, text: str) -> torch.Tensor:
        # E5 규약: 'query: ' 프리픽스
        qtxt = "query: " + text.strip()
        enc = self.tok(qtxt, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        # mean pooling (마스크 고려)
        last_hidden = out.last_hidden_state                        # [1, T, H] (fp32일 수 있음)
        attn = enc["attention_mask"].unsqueeze(-1).expand_as(last_hidden).float()
        summed = torch.sum(last_hidden * attn, dim=1)              # [1, H]
        counts = torch.clamp(attn.sum(dim=1), min=1e-9)            # [1, H] -> sum over T then broadcast-safe
        emb = summed / counts                                      # [1, H] (fp32)
        # GPU에서 바로 정규화 + dtype 맞춤
        emb = torch.nn.functional.normalize(emb, dim=1).to(dtype=torch.float16)  # [1, H], fp16
        return emb.squeeze(0)  # [H], fp16 @ cuda

    @torch.no_grad()
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = self._embed_query(query)  # [H], fp16 @ cuda

        # 방어적으로 장치/dtype 일치 (이론상 이미 일치하지만 혹시 몰라 한 번 더)
        if q.device != self.doc_norm.device:
            q = q.to(self.doc_norm.device, non_blocking=True)
        if q.dtype != self.doc_norm.dtype:
            q = q.to(self.doc_norm.dtype)

        # cosine sim (둘 다 L2-normalized + fp16 @ cuda)
        # self.doc_norm: [N, H], q: [H]
        sims = self.doc_norm @ q                                     # [N]
        k = int(max(1, min(top_k, sims.shape[0])))
        vals, idxs = torch.topk(sims, k, largest=True, sorted=True)

        results = []
        for score, idx in zip(vals.tolist(), idxs.tolist()):
            m = self.meta[idx]
            results.append({
                "text": self._pick_text(m),
                "source": self._pick_source(m),
                "score": float(score)
            })
        return results

    def _pick_text(self, m: dict) -> str:
        for key in ("text", "content", "chunk", "body"):
            if key in m and isinstance(m[key], str) and m[key].strip():
                return m[key]
        if isinstance(m, str):
            return m
        return json.dumps(m, ensure_ascii=False)

    def _pick_source(self, m: dict) -> str:
        for key in ("source", "path", "file"):
            if key in m and isinstance(m[key], str) and m[key].strip():
                return m[key]
        meta = m.get("meta") or {}
        for key in ("source", "path", "file"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return ""
# --- end RagSearcher ---


# ---------------- utils ----------------
def _get_model_max_len(tok, model):
    m = getattr(model.config, "max_position_embeddings", None)
    if not m or m == float("inf"):
        m = getattr(tok, "model_max_length", None)
    if not m or m > 32000:
        m = 4096
    return int(m)

def _find_peft_config_file(adapter_dir: Path) -> Path:
    candidates = [adapter_dir / "adapter_config.json", adapter_dir / "peft_config.json"]
    for p in candidates:
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

def _latest_run_dir() -> Path:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"모델 출력 폴더가 없습니다: {MODEL_DIR}")
    runs = sorted([p for p in MODEL_DIR.glob("*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"실행 결과 폴더가 비어 있습니다: {MODEL_DIR}")
    return runs[-1]

def load_model(run_dir: Path | None = None):
    run_dir = run_dir or _latest_run_dir()
    adapter_dir = run_dir / "adapter"
    token_dir   = run_dir / "tokenizer"

    tok = AutoTokenizer.from_pretrained(str(token_dir), use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
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
    - 입력을 truncation으로 재인코딩해 생성 headroom 확보
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

def pick_text_column(df: pd.DataFrame) -> str:
    # test.csv에 질문+보기 전체가 한 칼럼(예: text, question, 문제, 질문)로 들어있다는 가정
    pref = ["text", "question", "질문", "문제", "prompt", "content"]
    for p in pref:
        for c in df.columns:
            if c.lower() == p:
                return c
    # 없으면 첫 번째 컬럼 사용
    return df.columns[0]

def maybe_rag_context(searcher: RagSearcher | None, text: str, top_k: int = 3) -> str:
    if not searcher:
        return ""
    docs = searcher.search(text, top_k=top_k)
    ctx = []
    for i, d in enumerate(docs, 1):
        src = d.get("source", "")
        ctx.append(f"{i}) {src}\n{d['text']}")
    return "\n\n".join(ctx)

# ---------------- main ----------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default=str(ROOT / "data" / "test.csv"))
    parser.add_argument("--out_csv",  type=str, default=str(ROOT / "submission.csv"))
    parser.add_argument("--run_dir",  type=str, default=None, help="특정 체크포인트 경로(선택)")
    parser.add_argument("--use_rag", action="store_true", help="법령 RAG 문맥 추가")
    parser.add_argument("--max_gen_tokens", type=int, default=256)

    # RAG 옵션
    parser.add_argument("--rag_index", type=str, default="data/index/laws_e5", help="RAG 인덱스 서브디렉터리")
    parser.add_argument("--e5_model",  type=str, default="intfloat/e5-base-v2", help="쿼리 임베딩용 E5 모델 경로/이름")
    parser.add_argument("--rag_topk",  type=int, default=7, help="참고 문맥 개수")

    args = parser.parse_args()

    # (선택) RAG 세팅
    searcher = None
    if args.use_rag:
        try:
            searcher = RagSearcher(ROOT, index_subdir=args.rag_index,
                                   e5_model=args.e5_model)
            print(f"RAG: ON (index='{args.rag_index}', e5='{args.e5_model}', topk={args.rag_topk})")
        except Exception as e:
            print(f"RAG: OFF (초기화 실패: {e})")

    tok, model = load_model(Path(args.run_dir) if args.run_dir else None)

    df = pd.read_csv(args.test_csv)
    text_col = pick_text_column(df)

    preds: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Infer"):
        full_text = str(row[text_col]).strip()

        # RAG 문맥을 “텍스트 뒤에” 붙여서 그대로 make_prompt_auto에 넘김
        context = maybe_rag_context(searcher, full_text, top_k=args.rag_topk) if searcher else ""
        if context:
            full_text_for_prompt = f"{full_text}\n\n[참고자료]\n{context}"
        else:
            full_text_for_prompt = full_text

        # load.py의 두 함수만 사용: is_multiple_choice + make_prompt_auto
        messages = make_prompt_auto(full_text_for_prompt)

        if is_multiple_choice(full_text_for_prompt):
            # ---------------- 객관식: 숫자 1토큰만 생성 ----------------
            digit_proc = get_digit_processor(tok, digits="123456789")

            inputs, attn, _, _ = encode_chat(tok, model, messages, want_new_tokens=1, reserve=8)
            inputs = inputs.to(model.device)
            attn   = attn.to(model.device) if attn is not None else None

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
            txt = tok.decode(gen_ids, skip_special_tokens=True)

            m = re.search(r"[1-9]", txt)
            pred = m.group(0) if m else "3"  # fallback

            preds.append(pred)

            # 콘솔 출력
            print("\n[객관식]")
            print(f"Q:\n{full_text}")
            print(f"→ 모델 출력(정답 번호): {pred}")

        else:
            # ---------------- 주관식: 간결 응답 ----------------
            WANT_NEW = min(args.max_gen_tokens, 256)
            inputs, attn, _, gen_new = encode_chat(tok, model, messages, want_new_tokens=WANT_NEW, reserve=32)
            inputs = inputs.to(model.device)
            attn   = attn.to(model.device) if attn is not None else None

            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs,
                    attention_mask=attn,
                    max_new_tokens=gen_new,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=False,
                    no_repeat_ngram_size=2,
                    length_penalty=1.0,
                )
            gen_ids = out[0][inputs.shape[1]:]
            ans = tok.decode(gen_ids, skip_special_tokens=True).strip()

            preds.append(ans)

            # 콘솔 출력
            print("\n[주관식]")
            print(f"Q:\n{full_text}")
            print(f"→ 모델 출력(답변): {ans}")

    try:
        ids = df["ID"].astype(str).tolist()
    except KeyError:
        raise KeyError("test.csv에 'ID' 컬럼이 없습니다. 컬럼명을 확인해주세요.")

    if len(ids) != len(preds):
        print(f"[warn] ID 개수({len(ids)})와 예측 개수({len(preds)})가 다릅니다. 작은 쪽 기준으로 자릅니다.")
        n = min(len(ids), len(preds))
        ids = ids[:n]
        preds = preds[:n]

    submission = pd.DataFrame({
        "ID": ids,
        "Answer": [str(p) for p in preds]
    })

    submission.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] saved (sample_submission 형식): {args.out_csv}")

if __name__ == "__main__":
    # CUDA 없을 때 bfloat16 경고 억제(선택)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()