import os
import re
import json
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    LogitsProcessor, LogitsProcessorList, MinLengthLogitsProcessor
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
# RAG: BM25 검색기
# =========================
class BM25RAG:
    """
    ../dataset/index 안의 bm25_all_meta.jsonl, bm25_all.npz를 사용.
    - 우선 순위: 사전계산 인덱스(npz) 사용을 시도
    - 실패/불일치 시: meta.jsonl 전체를 메모리에 올려 간단한 BM25(Fallback) 수행
    """
    def __init__(self, index_dir: str, topk: int = 5, max_chars: int = 1200):
        self.index_dir = index_dir
        self.meta_path = os.path.join(index_dir, "bm25_all_meta.jsonl")
        self.npz_path  = os.path.join(index_dir, "bm25_all.npz")
        self.topk = topk
        self.max_chars = max_chars

        self.meta = []           # [{'id':..., 'text':..., 'title':...}, ...]
        self.use_prebuilt = False
        self._load_meta()
        self._try_load_prebuilt()

        # Fallback 준비: 토크나이저 및 역색인
        if not self.use_prebuilt:
            self._build_fallback_index()

    def _load_meta(self):
        if not os.path.isfile(self.meta_path):
            raise FileNotFoundError(f"메타 파일을 찾을 수 없습니다: {self.meta_path}")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # 일반화: 필드 이름 방어코드
                    text = obj.get("text") or obj.get("contents") or obj.get("body") or ""
                    title = obj.get("title") or obj.get("section") or ""
                    self.meta.append({"title": title, "text": text})
                except Exception:
                    # 라인 깨짐 방지
                    continue
        if not self.meta:
            raise ValueError("메타 데이터를 비어 있게 읽었습니다. bm25_all_meta.jsonl 내용을 확인하세요.")

    def _try_load_prebuilt(self):
        """
        사전계산 npz를 로드. 다양한 저장 스키마를 방어적으로 지원.
        - 기대하는 키 예시:
          * 'indptr','indices','data','shape','idf','avgdl','dl','vocab' (CSR + BM25 파라미터)
          * 또는 'matrix' (csr_matrix 저장), 'vocab', 'idf', 'avgdl', 'dl'
        """
        try:
            import numpy as np
            from scipy.sparse import csr_matrix

            if not os.path.isfile(self.npz_path):
                self.use_prebuilt = False
                return

            npz = np.load(self.npz_path, allow_pickle=True)

            if "matrix" in npz:
                self.M = npz["matrix"].item() if isinstance(npz["matrix"], np.ndarray) else npz["matrix"]
                if not isinstance(self.M, csr_matrix):
                    # 혹시 dense로 저장돼 있으면 csr로 변환
                    self.M = csr_matrix(self.M)
            else:
                indptr = npz.get("indptr")
                indices = npz.get("indices")
                data = npz.get("data")
                shape = tuple(npz.get("shape", []))
                if indptr is None or indices is None or data is None or not shape:
                    self.use_prebuilt = False
                    return
                self.M = csr_matrix((data, indices, indptr), shape=shape)

            self.vocab = npz.get("vocab", None)
            if isinstance(self.vocab, np.ndarray):
                self.vocab = self.vocab.item() if self.vocab.dtype == object and self.vocab.size == 1 else dict(self.vocab.tolist())
            elif self.vocab is None:
                # vocab이 없으면 질의 벡터화를 할 수 없어 Fallback로 전환
                self.use_prebuilt = False
                return

            # BM25 파라미터(있으면 사용, 없으면 합리적 상수로)
            self.idf   = npz.get("idf", None)
            self.avgdl = float(npz.get("avgdl", 1000.0))  # 문서 평균 길이
            self.dl    = npz.get("dl", None)              # 각 문서 길이
            self.k1    = 1.5
            self.b     = 0.75

            # 최소 요건: vocab + M
            if self.vocab is None or self.M is None:
                self.use_prebuilt = False
                return

            self.use_prebuilt = True

        except Exception:
            # 어떤 경우든 실패 시 Fallback
            self.use_prebuilt = False

    # --------- Fallback 구성 ----------
    def _basic_tokenize(self, text: str):
        # 한글/영문/숫자 토큰 보존
        toks = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
        return toks

    def _build_fallback_index(self):
        """
        간단한 BM25(Fallback) 인덱스 구성 (메모리 내). 대회 용량이 너무 크면 느릴 수 있으니
        npz가 안 먹힐 때만 사용.
        """
        from collections import Counter, defaultdict
        self.fb_docs = [m["text"] or "" for m in self.meta]
        self.fb_tok_docs = [self._basic_tokenize(t) for t in self.fb_docs]
        self.fb_N = len(self.fb_tok_docs)
        self.fb_df = Counter()
        self.fb_dl = []
        for tokens in self.fb_tok_docs:
            self.fb_dl.append(len(tokens))
            for w in set(tokens):
                self.fb_df[w] += 1
        self.fb_avgdl = sum(self.fb_dl)/max(1, self.fb_N)
        # 역색인
        self.fb_post = defaultdict(list)  # w -> list of (doc_id, tf)
        for di, tokens in enumerate(self.fb_tok_docs):
            tf = Counter(tokens)
            for w, c in tf.items():
                self.fb_post[w].append((di, c))
        self.fb_k1 = 1.5
        self.fb_b  = 0.75

    # --------- 검색 ----------
    def _score_bm25_fb(self, query_tokens):
        import math
        scores = [0.0]*self.fb_N
        for w in set(query_tokens):
            df = self.fb_df.get(w, 0)
            if df == 0:
                continue
            idf = math.log((self.fb_N - df + 0.5)/(df + 0.5) + 1.0)
            postings = self.fb_post.get(w, [])
            for di, tf in postings:
                denom = tf + self.fb_k1*(1 - self.fb_b + self.fb_b*self.fb_dl[di]/self.fb_avgdl)
                scores[di] += idf * (tf*(self.fb_k1+1))/max(1e-9, denom)
        # 상위 k
        idx = sorted(range(self.fb_N), key=lambda i: scores[i], reverse=True)[:self.topk]
        return [(i, scores[i]) for i in idx if scores[i] > 0]

    def _score_bm25_prebuilt(self, query_tokens):
        """
        사전계산 인덱스용 간이 스코어러:
        - vocab 기반으로 질의의 용어 인덱스를 모으고,
        - 각 용어의 열을 보는 방식이 가장 빠르지만, 여기선 간소화하여
          질의 Bag-of-Words를 점수화(가중 합)하는 접근.
        """
        import numpy as np
        from scipy.sparse import csr_matrix

        # 질의 BoW -> 어휘 인덱스 목록
        q_ids = []
        for w in set(query_tokens):
            wid = self.vocab.get(w, None)
            if wid is not None:
                q_ids.append(wid)
        if not q_ids:
            return []

        # 열 슬라이싱은 CSC가 빠르지만 M은 CSR일 가능성이 높음 → 각 wid 컬럼을 개별 추출(간단화)
        # 점수는 단순히 해당 용어의 가중치 합으로 근사 (idf가 있으면 가중)
        scores = np.zeros(self.M.shape[0], dtype=np.float32)
        for wid in q_ids:
            col = self.M[:, wid]
            # col은 csr_matrix라 인덱싱 시 희소 벡터 반환
            data = col.toarray().reshape(-1)  # 안전하지만 느릴 수 있음(상위 k만 필요하므로 대회 규모에서 OK)
            if self.idf is not None:
                try:
                    idf_w = float(self.idf[wid])
                except Exception:
                    idf_w = 1.0
            else:
                idf_w = 1.0
            scores += data * idf_w

        # 상위 k 추출
        topk_idx = np.argpartition(-scores, min(self.topk, len(scores)-1))[:self.topk]
        topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        return [(int(i), float(scores[i])) for i in topk_idx if scores[i] > 0]

    def search(self, query: str):
        """
        반환: [{"title":..., "text":...}, ...] 최대 topk
        """
        if not query or not query.strip():
            return []

        q_toks = self._basic_tokenize(query)
        if not q_toks:
            return []

        try:
            if self.use_prebuilt:
                hits = self._score_bm25_prebuilt(q_toks)
            else:
                hits = self._score_bm25_fb(q_toks)
        except Exception:
            # 어떤 예외도 안전하게 Fallback
            hits = self._score_bm25_fb(q_toks)

        results = []
        for di, _ in hits:
            rec = self.meta[di]
            text = (rec.get("text") or "").strip()
            title = (rec.get("title") or "").strip()
            if not text:
                continue
            # 너무 길면 앞쪽 요약
            if len(text) > self.max_chars:
                text = text[:self.max_chars] + "…"
            results.append({"title": title, "text": text})
        return results

    def build_context(self, question: str) -> str:
        docs = self.search(question)
        if not docs:
            return ""
        chunks = []
        for i, d in enumerate(docs, 1):
            title = f"[{d['title']}]" if d['title'] else ""
            chunks.append(f"(#{i}) {title} {d['text']}")
        ctx = "\n\n".join(chunks)
        # 모델에 주입될 시스템 보조 컨텍스트
        guide = (
            "다음은 법령/규정/용어 관련 참조 문맥입니다. "
            "정답은 문맥과 질문에 근거하여 간결하고 정확하게 작성하세요. "
            "출처 표시는 하지 말고, 최종 답만 제시하세요."
        )
        return f"{guide}\n\n{ctx}"

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

    # 2) RAG 초기화 (필요 시)
    rag = None
    if args.rag == "on":
        try:
            rag = BM25RAG(index_dir=args.index_dir, topk=args.rag_topk, max_chars=args.rag_max_chars)
            print(f"[RAG] 활성화됨 — topk={args.rag_topk}, max_chars={args.rag_max_chars} (prebuilt={rag.use_prebuilt})")
        except Exception as e:
            print(f"[RAG] 경고: 초기화 실패 → 비활성화합니다. 이유: {e}")
            rag = None
    else:
        print("[RAG] OFF")

    # 3) 데이터 로드
    test = pd.read_csv(args.test_csv)
    preds = []

    # 4) 추론 루프
    for q in tqdm(test["Question"], desc="Inference"):
        # load.py 기반 메시지 얻기
        messages = make_prompt_auto(q)

        # --- RAG 컨텍스트 삽입 ---
        if rag is not None:
            try:
                ctx = rag.build_context(q)
            except Exception as e:
                print(f"[RAG] 검색 실패 → 이번 문항은 RAG 미적용. 이유: {e}")
                ctx = ""
            if ctx:
                # system 보조 컨텍스트 앞단 삽입
                # messages는 [{"role": "...", "content": "..."}] 형태라고 가정
                messages = [{"role": "system", "content": ctx}] + messages

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        # 일부 토크나이저는 token_type_ids 없음
        inputs.pop("token_type_ids", None)

        if is_multiple_choice(q):
            # 허용 숫자 = 1~max_choices, 선택지 파싱 실패에도 안전
            allowed = [str(i) for i in range(1, args.max_choices + 1)]

            # 선택지 안에 동그라미 숫자만 있을 가능성 보완
            circled_map = {"①":"1","②":"2","③":"3","④":"4","⑤":"5","⑥":"6",
                           "⑦":"7","⑧":"8","⑨":"9","⑩":"10","⑪":"11","⑫":"12"}
            _, options = extract_question_and_choices(q)
            opt_text = "\n".join(options)
            for k, v in circled_map.items():
                opt_text = opt_text.replace(k, v)

            # 실제 등장 숫자와 교집합
            cands = sorted(set(re.findall(r"^\s*([1-9][0-9]?)", opt_text, re.M)))
            cands = [x for x in cands if 1 <= int(x) <= args.max_choices]
            nums = cands if cands else allowed

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
    test_sorted = pd.read_csv(args.test_csv).sort_values("ID").reset_index(drop=True)

    if not (sub["ID"].tolist() == test_sorted["ID"].tolist()):
        # 혹시라도 순서가 다르면 merge로 안전하게 매칭
        out_df = sub[["ID"]].merge(
            pd.DataFrame({"ID": test_sorted["ID"].tolist(), "Answer": preds}),
            on="ID",
            how="left"
        )
    else:
        out_df = sub.copy()
        out_df["Answer"] = preds

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved -> {out_csv}")

if __name__ == "__main__":
    main()