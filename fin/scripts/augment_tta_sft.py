#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_tta_sft_neg.py
- Create augmented SFT data from TTA term-definition pairs.
- Adds NEGATION variants ("옳지 않은/해당하지 않는") for MCQ.
- Generates:
  (A) term -> 정의 선택 MCQ (정상형 + 부정형)
  (B) 정의 -> 용어 선택 MCQ (정상형 + 부정형)
  (C) 추가 정의형 SFT(서술형)

Inputs (default):
  data/cleaned/tta_terms.jsonl    # {"term": "...", "definition": "..."}

Outputs:
  data/cleaned/tta_sft_aug.jsonl   # augmented only
  data/cleaned/tta_sft_plus.jsonl  # base SFT(if exists) + augmented

Usage:
  python scripts/augment_tta_sft_neg.py \
    --src data/cleaned/tta_terms.jsonl \
    --dst_aug data/cleaned/tta_sft_aug.jsonl \
    --dst_plus data/cleaned/tta_sft_plus.jsonl \
    --mcq_per_item 2 \
    --options 4 5 6 \
    --add_neg
"""
import json, random, argparse, re
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_SEED_DEFAULT = 42

# ----- 템플릿 -----
DEF_TEMPLATES = [
    "다음 용어를 간결히 정의하시오: {term}",
    "아래 보안 용어의 의미를 1~2문장으로 설명하시오: {term}",
    "전문가 관점에서 다음 용어의 핵심을 요약하시오: {term}",
    "다음 용어의 특징과 목적을 간략히 말하시오: {term}",
]

# 정의 본문을 입력으로 주고 용어를 답으로 고르는 형태 (정상형)
TERM_Q_TEMPLATES = [
    "다음 설명에 해당하는 용어를 선택하시오.",
    "아래 정의가 가리키는 용어는 무엇인가?",
    "설명과 일치하는 올바른 용어를 고르시오.",
]
# 정의 -> 용어 (부정형)
TERM_Q_NEG_TEMPLATES = [
    "다음 설명에 해당하지 않는 용어를 고르시오.",
    "아래 정의와 일치하지 않는 용어는 무엇인가?",
    "설명과 부합하지 않는 부적절한 용어를 선택하시오.",
]

# 용어를 입력으로 주고 정의를 고르는 형태 (정상형)
DEF_Q_TEMPLATES = [
    "다음 용어에 대한 올바른 정의를 선택하시오: {term}",
    "아래 용어의 의미로 적절한 것을 고르시오: {term}",
    "다음 용어에 대한 설명 중 올바른 것은? {term}",
]
# 용어 -> 정의 (부정형)
DEF_Q_NEG_TEMPLATES = [
    "다음 용어에 대한 설명 중 옳지 않은 것은? {term}",
    "아래 용어의 의미로 부적절한 설명을 고르시오: {term}",
    "다음 용어에 대한 정의 중 해당하지 않는 것은? {term}",
]

# ----- 유틸 -----
def load_terms(src: Path) -> List[Dict[str,str]]:
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            obj = json.loads(line)
            term = (obj.get("term","") or "").strip()
            definition = (obj.get("definition","") or "").strip()
            if term and definition:
                rows.append({"term":term, "definition":definition})
    if not rows:
        raise RuntimeError(f"No rows in {src}")
    return rows

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def shorten(s: str, max_len: int = 140) -> str:
    s = normalize_space(s)
    if len(s) <= max_len:
        return s
    return s[:max_len-1].rstrip() + "…"

def build_tfidf(rows: List[Dict[str,str]]):
    # entry 표현은 "term + definition"
    texts = [normalize_space(f"{r['term']} {r['definition']}") for r in rows]
    tfidf = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)
    return tfidf, X

def top_similar(idx: int, X, topk: int = 50) -> List[int]:
    sims = cosine_similarity(X[idx], X).ravel()
    order = sims.argsort()[::-1]
    return [j for j in order if j != idx][:topk]

# ----- MCQ 빌더 코어 -----
def _build_choices_from_defs(rows, i, X, n_options: int) -> Tuple[str, List[str], int, List[int]]:
    """
    기준: 용어 i
    반환:
      correct_def, choices(list), correct_index(1-based), used_neg_indices(list of row indices used as distractors)
    """
    term = rows[i]["term"]
    correct_def = shorten(rows[i]["definition"])
    neg_idx = top_similar(i, X, topk=50)

    choices = [correct_def]
    used_negs = []
    for j in neg_idx:
        if rows[j]["definition"] == rows[i]["definition"]:
            continue
        cand = shorten(rows[j]["definition"])
        if cand and cand not in choices:
            choices.append(cand)
            used_negs.append(j)
        if len(choices) >= n_options:
            break

    # 부족하면 랜덤 보충
    k = 0
    while len(choices) < n_options and k < len(rows):
        cand = shorten(rows[k]["definition"])
        if cand and cand not in choices:
            choices.append(cand)
            if k not in used_negs and k != i:
                used_negs.append(k)
        k += 1

    random.shuffle(choices)
    answer_idx = choices.index(correct_def)  # 0-based
    return correct_def, choices, (answer_idx + 1), used_negs

def _build_choices_from_terms(rows, i, X, n_options: int) -> Tuple[str, List[str], int, List[int]]:
    """
    기준: 정의 i
    반환:
      correct_term, choices(list), correct_index(1-based), used_neg_indices(list of row indices used as distractors)
    """
    definition = shorten(rows[i]["definition"])
    correct_term = rows[i]["term"]
    neg_idx = top_similar(i, X, topk=50)

    choices = [correct_term]
    used_negs = []
    for j in neg_idx:
        cand = rows[j]["term"]
        if not cand or cand == correct_term:
            continue
        if cand not in choices:
            choices.append(cand)
            used_negs.append(j)
        if len(choices) >= n_options:
            break

    # 부족하면 랜덤 보충
    k = 0
    while len(choices) < n_options and k < len(rows):
        cand = rows[k]["term"]
        if cand and cand not in choices:
            choices.append(cand)
            if k not in used_negs and k != i:
                used_negs.append(k)
        k += 1

    random.shuffle(choices)
    answer_idx = choices.index(correct_term)  # 0-based
    return correct_term, choices, (answer_idx + 1), used_negs

# ----- MCQ 기록 생성 -----
def make_mcq_def_normal(rows, X, i, n_options=4) -> Dict[str,str]:
    """ 용어 -> 올바른 정의 선택 (정상형) """
    term = rows[i]["term"]
    _, choices, ans_1based, _ = _build_choices_from_defs(rows, i, X, n_options)
    inst = random.choice(DEF_Q_TEMPLATES).format(term=term)
    opts = "\n".join(f"{k+1} {c}" for k, c in enumerate(choices))
    instruction = f"{inst}\n선택지:\n{opts}"
    return {"instruction": instruction, "input":"", "output": str(ans_1based)}

def make_mcq_def_neg(rows, X, i, n_options=4) -> Dict[str,str]:
    """ 용어 -> 정의 선택 (부정형: 옳지 않은 것 고르기) """
    term = rows[i]["term"]
    correct_def, choices, _, used_negs = _build_choices_from_defs(rows, i, X, n_options)
    # 부정형 정답: 가장 유사한 하드 네거티브(used_negs의 첫 항목이 choices에 들어가 있음)
    # used_negs[0]의 definition을 찾아 choices에서 인덱스 획득
    neg_target = None
    for j in used_negs:
        neg_def = shorten(rows[j]["definition"])
        if neg_def != correct_def and neg_def in choices:
            neg_target = neg_def
            break
    # 방어적 처리: 없으면 첫 번째 distractor(choices에서 correct가 아닌 첫 항)
    if not neg_target:
        for c in choices:
            if c != correct_def:
                neg_target = c
                break
    ans_idx_1based = choices.index(neg_target) + 1

    inst = random.choice(DEF_Q_NEG_TEMPLATES).format(term=term)
    opts = "\n".join(f"{k+1} {c}" for k, c in enumerate(choices))
    instruction = f"{inst}\n선택지:\n{opts}"
    return {"instruction": instruction, "input":"", "output": str(ans_idx_1based)}

def make_mcq_term_normal(rows, X, i, n_options=4) -> Dict[str,str]:
    """ 정의 -> 올바른 용어 선택 (정상형) """
    definition = shorten(rows[i]["definition"])
    _, choices, ans_1based, _ = _build_choices_from_terms(rows, i, X, n_options)
    inst = random.choice(TERM_Q_TEMPLATES)
    opts = "\n".join(f"{k+1} {c}" for k, c in enumerate(choices))
    instruction = f"{inst}\n설명: {definition}\n선택지:\n{opts}"
    return {"instruction": instruction, "input":"", "output": str(ans_1based)}

def make_mcq_term_neg(rows, X, i, n_options=4) -> Dict[str,str]:
    """ 정의 -> 용어 선택 (부정형: 해당하지 않는 용어 고르기) """
    definition = shorten(rows[i]["definition"])
    correct_term, choices, _, used_negs = _build_choices_from_terms(rows, i, X, n_options)
    # 부정형 정답: 가장 유사한 하드 네거티브 용어
    neg_target = None
    for j in used_negs:
        neg_term = rows[j]["term"]
        if neg_term != correct_term and neg_term in choices:
            neg_target = neg_term
            break
    if not neg_target:
        for c in choices:
            if c != correct_term:
                neg_target = c
                break
    ans_idx_1based = choices.index(neg_target) + 1

    inst = random.choice(TERM_Q_NEG_TEMPLATES)
    opts = "\n".join(f"{k+1} {c}" for k, c in enumerate(choices))
    instruction = f"{inst}\n설명: {definition}\n선택지:\n{opts}"
    return {"instruction": instruction, "input":"", "output": str(ans_idx_1based)}

def make_def_pairs(rows, n_per=2) -> List[Dict[str,str]]:
    """ 추가 정의형(서술형) SFT """
    recs = []
    for r in rows:
        for _ in range(n_per):
            inst = random.choice(DEF_TEMPLATES).format(term=r["term"])
            recs.append({"instruction": inst, "input":"", "output": normalize_space(r["definition"])})
    return recs

# ----- 메인 -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="data/cleaned/tta_terms.jsonl")
    ap.add_argument("--dst_aug", type=str, default="data/cleaned/tta_sft_aug.jsonl")
    ap.add_argument("--dst_plus", type=str, default="data/cleaned/tta_sft_plus.jsonl")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    ap.add_argument("--mcq_per_item", type=int, default=2, help="1=only def->choice, 2=both kinds")
    ap.add_argument("--options", type=int, nargs="+", default=[4], help="ex) --options 4 5 6")
    ap.add_argument("--add_neg", action="store_true", help="부정형 변형 MCQ도 함께 생성")
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.src)
    rows = load_terms(src)

    # TF-IDF for hard negatives
    _, X = build_tfidf(rows)

    aug = []
    # (C) 추가 정의형 SFT
    aug.extend(make_def_pairs(rows, n_per=2))

    # (A)(B) MCQ (정상형 + 선택적으로 부정형)
    for i in range(len(rows)):
        for n_opt in args.options:
            # A) term -> 정의
            rec = make_mcq_def_normal(rows, X, i, n_options=n_opt)
            aug.append(rec)
            if args.add_neg:
                rec_neg = make_mcq_def_neg(rows, X, i, n_options=n_opt)
                aug.append(rec_neg)

            if args.mcq_per_item >= 2:
                # B) 정의 -> 용어
                rec2 = make_mcq_term_normal(rows, X, i, n_options=n_opt)
                aug.append(rec2)
                if args.add_neg:
                    rec2_neg = make_mcq_term_neg(rows, X, i, n_options=n_opt)
                    aug.append(rec2_neg)

    # write augmented
    dst_aug = Path(args.dst_aug)
    dst_aug.parent.mkdir(parents=True, exist_ok=True)
    with dst_aug.open("w", encoding="utf-8") as f:
        for r in aug:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] augmented SFT -> {dst_aug} ({len(aug):,} records)")

    # merge with existing base SFT if present
    dst_plus = Path(args.dst_plus)
    base_sft = Path("data/cleaned/tta_sft.jsonl")
    merged = []
    if base_sft.exists():
        with base_sft.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    merged.append(line.rstrip("\n"))
    with dst_aug.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                merged.append(line.rstrip("\n"))
    with dst_plus.open("w", encoding="utf-8") as f:
        f.write("\n".join(merged) + ("\n" if merged else ""))
    print(f"[OK] merged base+aug -> {dst_plus} ({len(merged):,} records)")

if __name__ == "__main__":
    main()