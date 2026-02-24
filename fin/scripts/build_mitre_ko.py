#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MITRE ATT&CK Enterprise STIX 2.1 → Korean SFT(JSONL) & RAG(JSONL)

Input :
  data/raw/enterprise-attack.json       # STIX bundle
  data/glossary_en2ko.json              # 용어 매핑(선택)

Output:
  data/cleaned/mitre_sft_ko.jsonl       # QLoRA SFT 학습용
  data/cleaned/mitre_bm25_ko.jsonl      # RAG(BM25) 코퍼스용

설명:
- Technique name(기술명)은 영문 유지 (원문 보존; 필요한 경우 별칭은 RAG 단계에서 다룸)
- description만 한국어 번역 (facebook/nllb-200-distilled-600M)
- 번역 후 글로서리(영→한) 적용으로 용어 일관화
- 긴 본문은 문장 경계 기준으로 나눠 배치 번역
- 번역 실패 시 원문 통과(로버스트)
- 중간 체크포인트(.mitre_ko_progress.json)로 재시작 내구성 확보
"""

import os
import re
import json
import html
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# =========================
# Config
# =========================
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "kor_Hang"

RAW = Path("data/raw/enterprise-attack.json")
GLOSS = Path("data/glossary_en2ko.json")
OUT_SFT = Path("data/cleaned/mitre_sft_ko.jsonl")
OUT_BM25 = Path("data/cleaned/mitre_bm25_ko.jsonl")
CKPT_PATH = Path("data/cleaned/.mitre_ko_progress.json")

# 번역 청크 길이 / 배치 크기
CHUNK_MAX_CHARS = 900
BATCH_SIZE = 16

# =========================
# Utils
# =========================
def clean_markups(s: str) -> str:
    """MITRE 설명 텍스트에서 HTML/마크다운/URL/중복공백 정리 + HTML 엔티티 해제"""
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)                                        # HTML 태그
    s = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", s)            # [txt](url)
    s = re.sub(r"`+", " ", s)                                             # 인라인 코드 표기
    s = re.sub(r"\*\*?|__", " ", s)                                       # 굵게/이탤릭
    s = re.sub(r"https?://\S+", " ", s)                                   # URL 제거
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s).strip()
    s = html.unescape(s)                                                  # &amp; → &
    return s

def normalize_ko(s: str) -> str:
    """번역 후 한국어 텍스트의 개행/공백 정규화"""
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def apply_glossary_ko(s: str, gloss: dict) -> str:
    """번역 후 용어 한글화를 글로서리 기준으로 일괄 치환 (대소문자 무시, 단어 경계 기준)"""
    if not gloss:
        return s
    for en, ko in gloss.items():
        pattern = re.compile(rf"\b{re.escape(en)}\b", flags=re.IGNORECASE)
        s = pattern.sub(ko, s)
    return s

def split_chunks(text: str, max_chars=CHUNK_MAX_CHARS):
    """너무 긴 문단을 문장 경계 기준으로 잘라 리스트로 반환"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[\.\?!])\s+", text)
    chunks, cur = [], ""
    for sent in sentences:
        if not sent:
            continue
        if len(cur) + len(sent) + 1 > max_chars and cur:
            chunks.append(cur.strip())
            cur = sent
        else:
            cur = (cur + " " + sent).strip()
    if cur:
        chunks.append(cur.strip())
    return chunks

def load_done_ids():
    """이미 처리 완료한 STIX object id 집합"""
    if CKPT_PATH.exists():
        try:
            return set(json.loads(CKPT_PATH.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()

def save_done_ids(done_ids):
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.write_text(json.dumps(sorted(list(done_ids)), ensure_ascii=False), encoding="utf-8")

# =========================
# Translation
# =========================
def build_mt_pipeline():
    """NLLB 파이프라인 초기화 (FP16, device_map='auto')"""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    )
    nlp = pipeline(
        "translation",
        model=mt_model,
        tokenizer=tok,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        device_map="auto",
        max_length=1024,
        truncation=True
    )
    return nlp

def translate_en2ko_batch(nlp, texts, batch_size=BATCH_SIZE):
    """여러 문장을 배치로 번역. 실패 시 원문 통과(로버스트)."""
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            res = nlp(batch)
            outs.extend([r["translation_text"].strip() for r in res])
        except Exception:
            # 실패한 청크는 원문 유지 (필요시 여기서 재시도 로직 추가 가능)
            outs.extend(batch)
    return outs

def translate_en2ko(nlp, text: str) -> str:
    """긴 텍스트를 문장 경계 기반 청크로 나눠 번역 후 합침"""
    parts = split_chunks(text, max_chars=CHUNK_MAX_CHARS)
    if not parts:
        return ""
    ko_parts = translate_en2ko_batch(nlp, parts, batch_size=BATCH_SIZE)
    return " ".join(ko_parts).strip()

# =========================
# Main
# =========================
def main():
    assert RAW.exists(), f"STIX file not found: {RAW}"
    os.makedirs(OUT_SFT.parent, exist_ok=True)

    # 글로서리 로드 (선택)
    gloss = {}
    if GLOSS.exists():
        with open(GLOSS, "r", encoding="utf-8") as gf:
            gloss = json.load(gf)

    # MT 파이프라인 로드
    nlp = build_mt_pipeline()

    # 진행 상황 복원
    done = load_done_ids()

    with open(RAW, "r", encoding="utf-8") as f:
        data = json.load(f)

    sft_cnt = 0
    rag_cnt = 0
    processed = 0

    # append 모드로 열어두면 재시작 시 이어쓰기 가능
    with open(OUT_SFT, "a", encoding="utf-8") as f_sft, \
         open(OUT_BM25, "a", encoding="utf-8") as f_bm25:

        for obj in data.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue

            stix_id = obj.get("id")
            if stix_id in done:
                continue

            name = (obj.get("name") or "").strip()
            desc_en = (obj.get("description") or "").strip()
            if not name or not desc_en:
                done.add(stix_id)
                continue

            # tactic 태깅(선택)
            tactics = []
            for kp in obj.get("kill_chain_phases", []) or []:
                if kp.get("kill_chain_name") == "mitre-attack":
                    t = kp.get("phase_name")
                    if t:
                        tactics.append(t)
            tactic = ",".join(sorted(set(tactics)))

            # --- 정제 & 번역 ---
            desc_clean = clean_markups(desc_en)
            desc_ko = translate_en2ko(nlp, desc_clean)
            if gloss:
                desc_ko = apply_glossary_ko(desc_ko, gloss)
            desc_ko = normalize_ko(desc_ko)

            # --- 산출: SFT ---
            sft_item = {
                "instruction": f"다음 공격 기법을 설명하시오: {name}",
                "input": "",
                "output": desc_ko
            }
            f_sft.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            sft_cnt += 1

            # --- 산출: RAG(BM25) ---
            rag_item = {
                "text": f"{name} — {desc_ko}",
                "source": "mitre/enterprise-attack",
                "tactic": tactic
            }
            f_bm25.write(json.dumps(rag_item, ensure_ascii=False) + "\n")
            rag_cnt += 1

            # 진행 체크포인트
            done.add(stix_id)
            processed += 1
            if processed % 50 == 0:
                save_done_ids(done)
                print(f"[Info] processed={processed}, sft={sft_cnt}, rag={rag_cnt}")

    # 마지막 저장
    save_done_ids(done)
    print(f"[OK] SFT KO -> {OUT_SFT} ({sft_cnt} rows)")
    print(f"[OK] RAG KO -> {OUT_BM25} ({rag_cnt} rows)")
    print(f"[OK] Progress saved -> {CKPT_PATH}")

if __name__ == "__main__":
    main()