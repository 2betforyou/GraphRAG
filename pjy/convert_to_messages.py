#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_messages.py
- instruction/input/output → messages 형태로 변환
- 이미 messages가 있으면 통과(간단 정규화)
- 기타 필드(id, meta 등)는 보존
"""

import json
import argparse
from pathlib import Path

DEFAULT_SYSTEM = (
    "당신은 금융보안 전문가입니다. 모든 답변은 한국어로 간결하고 정확하게 작성하세요." 
)

def to_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)

def normalize_messages(msgs):
    """role/content 키만 남기고 문자열화, 빈 문자열 정리"""
    out = []
    for m in msgs:
        role = m.get("role", "")
        content = to_str(m.get("content", "")).strip()
        if not role:
            continue
        out.append({"role": role, "content": content})
    return out

def build_messages_from_instruct(obj, system_text: str):
    instr = to_str(obj.get("instruction", "")).strip()
    inp   = to_str(obj.get("input", "")).strip()
    out   = to_str(obj.get("output", "")).strip()

    user_text = instr if inp == "" else (instr + ("\n" + inp if instr else inp))
    msgs = [
        {"role": "system", "content": system_text.strip()},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": out},
    ]
    return normalize_messages(msgs)

def convert_file(src_path: str, dst_path: str, system_text: str = DEFAULT_SYSTEM):
    src = Path(src_path); dst = Path(dst_path)
    n_in = n_out = n_from_instruct = n_pass = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except Exception:
                # 형식 깨진 라인은 스킵
                continue

            new_obj = dict(obj)  # 기타 필드 보존
            if isinstance(obj.get("messages"), list) and obj["messages"]:
                # 이미 messages가 있으면 정규화만
                new_obj["messages"] = normalize_messages(obj["messages"])
                n_pass += 1
            else:
                new_obj["messages"] = build_messages_from_instruct(obj, system_text)
                # instruction/input/output 원본 키를 남기고 싶지 않다면 주석 해제:
                # for k in ("instruction", "input", "output"):
                #     new_obj.pop(k, None)
                n_from_instruct += 1

            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[OK] wrote: {dst}")
    print(f"  lines_in={n_in}, lines_out={n_out},"
          f" built_from_instruction={n_from_instruct}, pass_through_messages={n_pass}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True,
                    help="입력 JSONL (instruction/input/output 또는 messages)")
    ap.add_argument("--dst", type=str, required=True,
                    help="출력 JSONL (messages 스키마)")
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM,
                    help="system 프롬프트 문구")
    args = ap.parse_args()

    convert_file(args.src, args.dst, args.system)