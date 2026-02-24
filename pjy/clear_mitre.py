#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_non_kr_en.py
- JSONL 파일에서 문자열 필드 전체 정제(중첩 dict/list 포함)
- (1) 괄호쌍 내부에 콜론(반각 ':' 또는 전각 '：')이 있으면, 괄호 포함 통째로 삭제
- (2) 허용 문자만 유지: 한글/영문/숫자/공백/기본 문장부호(반각)
- (3) 전각 괄호쌍도 처리: （）, ［］, ｛｝
"""

import json
import re
from pathlib import Path

# 반각/전각 콜론 둘 다 매치
COLON = r"[:：]"

# 괄호쌍 내부에 콜론이 있으면 그 괄호 블록 전체 제거
# - 중첩 단순 처리: 한 줄에 여러 개 있거나 중첩 느낌이 있어도 반복 적용으로 걷어냄
BRACKET_PATTERNS = [
    r"\([^()\n]*" + COLON + r"[^()\n]*\)",       # ()
    r"\[[^\[\]\n]*" + COLON + r"[^\[\]\n]*\]",   # []
    r"\{[^{}\n]*" + COLON + r"[^{}\n]*\}",       # {}
    r"（[^（）\n]*" + COLON + r"[^（）\n]*）",    # （）
    r"［[^［］\n]*" + COLON + r"[^［］\n]*］",    # ［］
    r"｛[^｛｝\n]*" + COLON + r"[^｛｝\n]*｝",     # ｛｝
]

def remove_colon_brackets(s: str) -> str:
    if not isinstance(s, str):
        return s
    changed = True
    while changed:
        changed = False
        for pat in BRACKET_PATTERNS:
            new_s, n = re.subn(pat, "", s)
            if n > 0:
                changed = True
                s = new_s
    return s

# 허용 문자: 한글/영문/숫자/공백/기본 문장부호(반각) + -, _, /
# 하이픈(-)은 클래스 끝에 두거나 이스케이프해야 함
ALLOWED = re.compile(r"[^가-힣a-zA-Z0-9\s\.\,\;\:\!\?\"\'\(\)\[\]\{\}_/\-]")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    # (1) 콜론 포함 괄호 블록 제거
    s = remove_colon_brackets(s)
    # (2) 허용 집합 외 문자 제거
    s = ALLOWED.sub("", s)
    # (3) 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_value(v):
    """obj의 모든 문자열 값을 재귀적으로 정제"""
    if isinstance(v, str):
        return clean_text(v)
    if isinstance(v, list):
        return [clean_value(x) for x in v]
    if isinstance(v, dict):
        return {k: clean_value(val) for k, val in v.items()}
    return v

def process_jsonl(src_path: str, dst_path: str):
    src = Path(src_path)
    dst = Path(dst_path)

    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            obj = clean_value(obj)

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] cleaned file saved → {dst}")

if __name__ == "__main__":
    # 사용 예시
    process_jsonl(
        "../dataset/all/mitre_bm25_ko_cleaned.jsonl",   # 입력
        "../dataset/all/mitre_bm25_ko_cleaned2.jsonl"    # 출력
    )