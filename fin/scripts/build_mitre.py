#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MITRE ATT&CK (Enterprise) STIX 2.1 → SFT(JSONL) & RAG(JSONL)

Input :
  data/raw/enterprise-attack.json     # 방금 받아둔 STIX 번들

Output:
  data/cleaned/mitre_sft.jsonl        # QLoRA SFT 학습용
  data/cleaned/mitre_bm25.jsonl       # RAG(BM25) 코퍼스용

정제 규칙:
- HTML/마크다운/URL 제거, 공백 정리
- 너무 긴 설명은 문장 경계 기준으로 1,400자 근방에서 잘라냄
- STIX objects 중 type == "attack-pattern"만 사용 (Technique/Sub-technique)
"""
import os, re, json
from pathlib import Path

RAW = Path("data/raw/enterprise-attack.json")
OUT_SFT = Path("data/cleaned/mitre_sft.jsonl")
OUT_BM25 = Path("data/cleaned/mitre_bm25.jsonl")

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)  # HTML 태그
    s = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", s)  # [txt](url)
    s = re.sub(r"`+", " ", s)  # 인라인 코드
    s = re.sub(r"\*\*?|__", " ", s)  # 굵게/이탤릭
    s = re.sub(r"https?://\S+", " ", s)  # URL
    s = re.sub(r"[ \t]+", " ", s)  # 중복 공백
    s = re.sub(r"\s*\n\s*", "\n", s).strip()  # 줄바꿈 정리
    return s.strip()

def trim_by_chars(s: str, max_len: int = 1400) -> str:
    if len(s) <= max_len:
        return s
    cut = s[:max_len]
    # 문장 경계(., ?, !)에서 깔끔하게 자르기
    m = re.search(r"[\.?!](?!.*[\.?!]).*$", cut)
    if m:
        end = m.end()
        return cut[:end].strip()
    return cut.strip()

def main():
    os.makedirs(OUT_SFT.parent, exist_ok=True)

    if not RAW.exists():
        raise FileNotFoundError(f"STIX file not found: {RAW}")

    with open(RAW, "r", encoding="utf-8") as f:
        data = json.load(f)

    sft_cnt = 0
    rag_cnt = 0

    with open(OUT_SFT, "w", encoding="utf-8") as f_sft, \
         open(OUT_BM25, "w", encoding="utf-8") as f_bm25:

        for obj in data.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue

            name = (obj.get("name") or "").strip()
            desc = (obj.get("description") or "").strip()
            if not name or not desc:
                continue

            # tactic 태깅(선택)
            tactics = []
            for kp in obj.get("kill_chain_phases", []) or []:
                if kp.get("kill_chain_name") == "mitre-attack":
                    t = kp.get("phase_name")
                    if t:
                        tactics.append(t)
            tactic = ",".join(sorted(set(tactics)))

            cleaned = trim_by_chars(clean_text(desc), max_len=1400)

            # SFT: instruction/output
            sft_item = {
                "instruction": f"다음 공격 기법을 설명하시오: {name}",
                "input": "",
                "output": cleaned
            }
            f_sft.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            sft_cnt += 1

            # RAG(BM25) 코퍼스
            rag_item = {
                "text": f"{name} — {cleaned}",
                "source": "mitre/enterprise-attack",
                "tactic": tactic
            }
            f_bm25.write(json.dumps(rag_item, ensure_ascii=False) + "\n")
            rag_cnt += 1

    print(f"[OK] SFT -> {OUT_SFT} ({sft_cnt} rows)")
    print(f"[OK] RAG -> {OUT_BM25} ({rag_cnt} rows)")

if __name__ == "__main__":
    main()