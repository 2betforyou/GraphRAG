#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rationaleify_mcq.py
- 기존 SFT(JSONL: {"instruction","input","output"})에서 MCQ를
  "해설 → 마지막 줄 '정답: N'" 포맷으로 변환.
- 선택지 패턴: "선택지:\n1 ...\n2 ...\n..."
- 부정형(옳지 않은/해당하지 않는) 자동 감지.

Usage:
  python scripts/rationaleify_mcq.py \
    --src data/cleaned/tta_sft_plus.jsonl \
    --dst data/cleaned/tta_sft_plus_rationale.jsonl \
    --keep_non_mcq
"""
import re, json, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

MCQ_BLOCK_RE = re.compile(r"(선택지\s*:\s*\n)(.+)$", re.DOTALL)
CHOICE_LINE_RE = re.compile(r"^\s*([1-9][0-9]?)\s+(.+?)\s*$")

NEG_CUES = [
    "옳지 않은", "해당하지 않는", "일치하지 않는", "부적절한", "부합하지 않는"
]

def is_negative_question(text: str) -> bool:
    return any(cue in text for cue in NEG_CUES)

def parse_choices(instruction: str) -> Optional[List[Tuple[str,str]]]:
    """
    instruction에서 '선택지:' 이후 라인을 파싱해 [(번호, 내용), ...] 반환.
    실패 시 None.
    """
    m = MCQ_BLOCK_RE.search(instruction)
    if not m:
        return None
    choices_block = m.group(2).strip()
    lines = [l for l in choices_block.splitlines() if l.strip()]
    out: List[Tuple[str,str]] = []
    for l in lines:
        m2 = CHOICE_LINE_RE.match(l)
        if not m2:
            # 선택지 라인이 아닌 것이 섞여 있으면 실패로 간주
            return None
        out.append((m2.group(1), m2.group(2)))
    return out if out else None

def build_explanation(instr: str, choices: List[Tuple[str,str]], answer_idx_1based: int) -> str:
    is_neg = is_negative_question(instr)
    # 정상/부정 공통: 정답 텍스트 확보
    number_map = {int(n): txt for n, txt in choices}
    ans_txt = number_map.get(answer_idx_1based, "").strip()

    # 간단 해설 템플릿 (정보 누설 없이 형식 학습에 집중)
    if is_neg:
        expl = (
            f"부정형 문항입니다. 제시된 설명(또는 용어)에 부합하지 않는 선택지를 고릅니다. "
            f"{answer_idx_1based}번 선택지는 다른 선택지들과 달리 제시된 조건과 일치하지 않아 정답이 됩니다. "
            f"정답 선택지 요약: {ans_txt}"
        )
    else:
        expl = (
            f"정상형 문항입니다. 제시된 설명(또는 용어)에 가장 부합하는 선택지를 고릅니다. "
            f"{answer_idx_1based}번 선택지는 핵심 정의와 일치하여 정답이 됩니다. "
            f"정답 선택지 요약: {ans_txt}"
        )
    return expl

def transform_record(rec: Dict) -> Optional[Dict]:
    """
    MCQ면 output을 '해설 … \\n정답: N'으로 변환해 반환.
    MCQ가 아니면 None 반환.
    """
    instr = (rec.get("instruction") or "").strip()
    output_raw = (rec.get("output") or "").strip()
    choices = parse_choices(instr)
    if not choices:
        return None  # MCQ 아님

    # 정답 번호 파싱
    m = re.match(r"^\s*([1-9][0-9]?)\s*$", output_raw)
    if not m:
        # 숫자만이 아닌 경우(이미 서술형 등) 변환 스킵
        return None
    ans_idx = int(m.group(1))
    # explain 생성
    explain = build_explanation(instr, choices, ans_idx)
    new_output = f"{explain}\n정답: {ans_idx}"
    out = dict(rec)
    out["output"] = new_output
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--dst", type=str, required=True)
    ap.add_argument("--keep_non_mcq", action="store_true", help="MCQ가 아닌 샘플은 원본 그대로 유지")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    total = 0
    mcq = 0
    kept_non = 0
    converted: List[str] = []

    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            new_rec = transform_record(rec)
            if new_rec is None:
                if args.keep_non_mcq:
                    kept_non += 1
                    converted.append(json.dumps(rec, ensure_ascii=False))
                # keep_non_mcq가 아니면 버림(= MCQ만 남김)
            else:
                mcq += 1
                converted.append(json.dumps(new_rec, ensure_ascii=False))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for s in converted:
            f.write(s + "\n")

    print(f"[OK] wrote -> {dst}")
    print(f"  total read     : {total:,}")
    print(f"  MCQ converted  : {mcq:,}")
    if args.keep_non_mcq:
        print(f"  non-MCQ kept   : {kept_non:,}")
    print(f"  final records  : {len(converted):,}")

if __name__ == "__main__":
    main()