#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_tta_sft_neg.py (enhanced)
- TTA term-definition pairs → 대량 SFT/MCQ 증강
- 정상형/부정형 MCQ + 서술형 SFT
- 라벨 균등 분포 (--balance_labels)
- 다양성 확장:
  * --times: 각 항목당 반복 생성(샘플 수 폭증)
  * 정의 부분절삭/확장 (--def_crop, --def_expand)
  * 하드/소프트 네거티브 혼합 (--hard_ratio)
  * 템플릿 풀 확대
  * 경미한 표면 변형(noise) 옵션 (--minor_noise)

Inputs (default):
  data/cleaned/tta_terms.jsonl    # {"term": "...", "definition": "..."}

Outputs:
  data/cleaned/tta_sft_aug.jsonl   # augmented only
  data/cleaned/tta_sft_plus.jsonl  # base SFT(if exists) + augmented

Usage (예시):
  python scripts/augment_tta_sft2.py \
    --src data/cleaned/tta_terms.jsonl \
    --dst_aug data/cleaned/tta_sft_aug.jsonl \
    --dst_plus data/cleaned/tta_sft_plus.jsonl \
    --mcq_per_item 2 \
    --options 4 5 6 \
    --add_neg \
    --balance_labels \
    --times1 \
    --def_crop \
    --def_expand \
    --hard_ratio 0.7 \
    --minor_noise
"""
import json, random, argparse, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_SEED_DEFAULT = 42
SENT_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s+")


# ----- 템플릿 (확장) -----
DEF_TEMPLATES = [
    "다음 용어를 간결히 정의하시오: {term}",
    "아래 보안 용어의 의미를 설명하시오: {term}",
    "전문가 관점에서 다음 용어의 핵심을 요약하시오: {term}",
    "다음 용어의 특징과 목적을 간략히 말하시오: {term}",
    "해당 용어의 개념과 활용 맥락을 한 문단으로 정리하시오: {term}",
    "보안관리 맥락에서 다음 용어의 의미를 설명하시오: {term}",
]

TERM_Q_TEMPLATES = [
    "다음 설명에 해당하는 용어를 선택하시오.",
    "아래 정의가 가리키는 용어는 무엇인가?",
    "설명과 일치하는 올바른 용어를 고르시오.",
    "다음 설명이 의미하는 보안 용어를 고르시오.",
    "다음 정의의 대상이 되는 용어는?",
]

TERM_Q_NEG_TEMPLATES = [
    "다음 설명에 해당하지 않는 용어를 고르시오.",
    "아래 정의와 일치하지 않는 용어는 무엇인가?",
    "설명과 부합하지 않는 부적절한 용어를 선택하시오.",
    "다음 설명과 무관한 용어를 고르시오.",
]

DEF_Q_TEMPLATES = [
    "다음 용어에 대한 올바른 정의를 선택하시오: {term}",
    "아래 용어의 의미로 적절한 것을 고르시오: {term}",
    "다음 용어에 대한 설명 중 올바른 것은? {term}",
    "해당 용어에 대한 정의로 맞는 것을 고르시오: {term}",
]

DEF_Q_NEG_TEMPLATES = [
    "다음 용어에 대한 설명 중 옳지 않은 것은? {term}",
    "아래 용어의 의미로 부적절한 설명을 고르시오: {term}",
    "다음 용어에 대한 정의 중 해당하지 않는 것은? {term}",
    "다음 용어에 대한 설명 중 틀린 것을 고르시오: {term}",
]

# ----- 경미한 표면 변형용 토큰 맵 (선택) -----
NOISE_MAP = {
    "및": ["및", "와", "그리고"],
    "등": ["등", "등의", "등을"],
    "또는": ["또는", "혹은"],
    "을": ["을", "를"],
    "를": ["를", "을"],
}

# =========================
# Label Balancer
# =========================
class LabelBalancer:
    def __init__(self):
        self.counters = {}  # n_options -> next_position(1-based)
    def next_pos(self, n_options: int) -> int:
        cur = self.counters.get(n_options, 1)
        out = cur
        nxt = cur + 1
        if nxt > n_options:
            nxt = 1
        self.counters[n_options] = nxt
        return out

def place_target_at(choices: List[str], target_value: str, target_pos_1based: int) -> Tuple[List[str], int]:
    if target_value not in choices:
        ans_idx = 1 + choices.index(choices[0])
        return choices, ans_idx
    cur_idx = choices.index(target_value)
    tgt_idx = target_pos_1based - 1
    if cur_idx != tgt_idx:
        choices[cur_idx], choices[tgt_idx] = choices[tgt_idx], choices[cur_idx]
    return choices, target_pos_1based

# ----- 유틸 -----
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def shorten(s: str, max_len: int = 180) -> str:
    s = normalize_space(s)
    return s if len(s) <= max_len else s[:max_len-1].rstrip() + "…"

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

def build_tfidf(rows: List[Dict[str,str]]):
    texts = [normalize_space(f"{r['term']} {r['definition']}") for r in rows]
    tfidf = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)
    return tfidf, X

def top_similar(idx: int, X, topk: int = 50) -> List[int]:
    sims = cosine_similarity(X[idx], X).ravel()
    order = sims.argsort()[::-1]
    return [j for j in order if j != idx][:topk]

def minor_noise_text(s: str, prob: float = 0.15) -> str:
    """표면형 변형: 조사/접속사 치환(작게)"""
    toks = re.split(r"(\s+)", s)
    for i, t in enumerate(toks):
        if t in NOISE_MAP and random.random() < prob:
            toks[i] = random.choice(NOISE_MAP[t])
    return "".join(toks)

def pick_crop(defn: str) -> str:
    """정의 부분절삭: 문장 단위 앞/중/뒤 일부만"""
    if not isinstance(defn, str):
        defn = "" if defn is None else str(defn)
    text = normalize_space(defn)
    # 고정 길이 look-behind만 사용
    sents = SENT_SPLIT_REGEX.split(text)
    sents = [x for x in sents if x]

    if not sents:
        return shorten(defn)

    mode = random.choice(["head", "mid", "tail"])
    if mode == "head":
        k = max(1, min(len(sents), random.randint(1, 2)))
        return shorten(" ".join(sents[:k]))
    if mode == "tail":
        k = max(1, min(len(sents), random.randint(1, 2)))
        return shorten(" ".join(sents[-k:]))

    # mid
    if len(sents) <= 2:
        return shorten(" ".join(sents))
    start = random.randint(1, max(1, len(sents)-2))
    end = min(len(sents), start + random.randint(1, 2))
    return shorten(" ".join(sents[start:end]))

def pick_expand(defn: str, term: str) -> str:
    """정의 확장: 간단한 접두/접미 문구로 맥락 추가"""
    base = shorten(defn, 220)
    prefix = random.choice([
        f"{term}에 대하여, ",
        f"보안 관점에서 {term}은/는 ",
        f"{term}의 개념은 다음과 같다: ",
    ])
    suffix = random.choice([
        " (주요 목적과 적용 범위를 중심으로 요약).",
        " (핵심 속성과 한계도 함께 고려).",
        " (실무 적용 시 주의사항 포함).",
        "",
    ])
    return shorten(prefix + base + suffix, 240)

# ----- MCQ 코어 -----
def _mix_neg_indices(i: int, X, pool_size: int, hard_ratio: float) -> List[int]:
    """하드/소프트 네거티브 인덱스 혼합"""
    cand = top_similar(i, X, topk=max(50, pool_size))
    if not cand:
        return []
    n_hard = max(1, int(pool_size * hard_ratio))
    hard = cand[:n_hard]
    soft_pool = cand[n_hard: n_hard + (pool_size - n_hard)*3]
    random.shuffle(soft_pool)
    soft = soft_pool[:max(0, pool_size - n_hard)]
    out = hard + soft
    random.shuffle(out)
    return out

def _build_choices_from_defs(rows, i, X, n_options: int, hard_ratio: float = 0.7,
                             use_crop=False, use_expand=False, use_noise=False) -> Tuple[str, List[str], int, List[int]]:
    term = rows[i]["term"]
    base_def = rows[i]["definition"]
    # 본문 변형
    if use_crop and random.random() < 0.5:
        correct_def = pick_crop(base_def)
    elif use_expand and random.random() < 0.5:
        correct_def = pick_expand(base_def, term)
    else:
        correct_def = shorten(base_def)
    if use_noise and random.random() < 0.5:
        correct_def = minor_noise_text(correct_def)

    neg_idx = _mix_neg_indices(i, X, pool_size=max(10, n_options*6), hard_ratio=hard_ratio)

    choices = [correct_def]
    used_negs = []
    for j in neg_idx:
        if rows[j]["definition"] == base_def:
            continue
        cand = rows[j]["definition"]
        # 변형 동일 적용
        if use_crop and random.random() < 0.3:
            cand = pick_crop(cand)
        elif use_expand and random.random() < 0.3:
            cand = pick_expand(cand, rows[j]["term"])
        cand = shorten(cand)
        if use_noise and random.random() < 0.2:
            cand = minor_noise_text(cand)
        if cand and cand not in choices:
            choices.append(cand)
            used_negs.append(j)
        if len(choices) >= n_options:
            break

    # 부족하면 랜덤 보충
    k = 0
    while len(choices) < n_options and k < len(rows):
        cand = shorten(rows[k]["definition"])
        if cand and cand not in choices and k != i:
            choices.append(cand); used_negs.append(k)
        k += 1

    random.shuffle(choices)
    answer_idx = choices.index(correct_def)
    return correct_def, choices, (answer_idx + 1), used_negs

def _build_choices_from_terms(rows, i, X, n_options: int, hard_ratio: float = 0.7,
                              use_crop=False, use_expand=False, use_noise=False) -> Tuple[str, List[str], int, List[int]]:
    definition = rows[i]["definition"]
    if use_crop and random.random() < 0.5:
        definition_v = pick_crop(definition)
    elif use_expand and random.random() < 0.5:
        definition_v = pick_expand(definition, rows[i]["term"])
    else:
        definition_v = shorten(definition)
    if use_noise and random.random() < 0.5:
        definition_v = minor_noise_text(definition_v)

    correct_term = rows[i]["term"]
    neg_idx = _mix_neg_indices(i, X, pool_size=max(10, n_options*6), hard_ratio=hard_ratio)

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
        if cand and cand not in choices and k != i:
            choices.append(cand); used_negs.append(k)
        k += 1

    random.shuffle(choices)
    answer_idx = choices.index(correct_term)
    return correct_term, choices, (answer_idx + 1), used_negs, definition_v

# ----- MCQ 기록 생성 -----
def make_mcq_def_normal(rows, X, i, n_options=4, balancer: Optional[LabelBalancer]=None, balance=False,
                        hard_ratio=0.7, use_crop=False, use_expand=False, use_noise=False) -> Dict[str,str]:
    term = rows[i]["term"]
    correct_def, choices, ans_1based, _ = _build_choices_from_defs(
        rows, i, X, n_options, hard_ratio, use_crop, use_expand, use_noise)
    if balance and balancer:
        target_pos = balancer.next_pos(n_options)
        choices, ans_1based = place_target_at(choices, correct_def, target_pos)
    inst = random.choice(DEF_Q_TEMPLATES).format(term=term)
    opts = "\n".join(f"{k+1} {c}" for k, c in enumerate(choices))
    instruction = f"{inst}\n선택지:\n{opts}"
    return {"instruction": instruction, "input":"", "output": str(ans_1based)}

def make_mcq_def_neg(rows, X, i, n_options=4, balancer: Optional[LabelBalancer]=None, balance=False,
                     hard_ratio=0.7, use_crop=False, use_expand=False, use_noise=False) -> Dict[str,str]:
    term = rows[i]["term"]
    correct_def, choices, _, used_negs = _build_choices_from_defs(
        rows, i, X, n_options, hard_ratio, use_crop, use_expand, use_noise)
    neg_target = None
    for j in used_negs:
        neg_def = shorten(rows[j]["definition"])
        if neg_def != correct_def and neg_def in choices:
            neg_target = neg_def
            break
    if not neg_target:
        for c in choices:
            if c != correct_def:
                neg_target = c; break
    ans_idx_1based = choices.index(neg_target) + 1
    if balance and balancer:
        target_pos = balancer.next_pos(n_options)
        choices, ans_idx_1based = place_target_at(choices, neg_target, target_pos)

    inst = random.choice(DEF_Q_NEG_TEMPLATES).format(term=term)
    opts = "\n".join(f"{k+1} {c}" for k, c in enumerate(choices))
    instruction = f"{inst}\n선택지:\n{opts}"
    return {"instruction": instruction, "input":"", "output": str(ans_idx_1based)}

def make_mcq_term_normal(rows, X, i, n_options=4, balancer: Optional[LabelBalancer]=None, balance=False,
                         hard_ratio=0.7, use_crop=False, use_expand=False, use_noise=False) -> Dict[str,str]:
    correct_term, choices, ans_1based, _, def_v = _build_choices_from_terms(
        rows, i, X, n_options, hard_ratio, use_crop, use_expand, use_noise)
    if balance and balancer:
        target_pos = balancer.next_pos(n_options)
        choices, ans_1based = place_target_at(choices, correct_term, target_pos)
    inst = random.choice(TERM_Q_TEMPLATES)
    opts = "\n".join(f"{k+1} {c}" for k, c in enumerate(choices))
    instruction = f"{inst}\n설명: {def_v}\n선택지:\n{opts}"
    return {"instruction": instruction, "input":"", "output": str(ans_1based)}

def make_mcq_term_neg(rows, X, i, n_options=4, balancer: Optional[LabelBalancer]=None, balance=False,
                      hard_ratio=0.7, use_crop=False, use_expand=False, use_noise=False) -> Dict[str,str]:
    correct_term, choices, _, used_negs, def_v = _build_choices_from_terms(
        rows, i, X, n_options, hard_ratio, use_crop, use_expand, use_noise)
    neg_target = None
    for j in used_negs:
        neg_term = rows[j]["term"]
        if neg_term != correct_term and neg_term in choices:
            neg_target = neg_term; break
    if not neg_target:
        for c in choices:
            if c != correct_term:
                neg_target = c; break
    ans_idx_1based = choices.index(neg_target) + 1
    if balance and balancer:
        target_pos = balancer.next_pos(n_options)
        choices, ans_idx_1based = place_target_at(choices, neg_target, target_pos)

    inst = random.choice(TERM_Q_NEG_TEMPLATES)
    opts = "\n".join(f"{k+1} {c}" for k, c in enumerate(choices))
    instruction = f"{inst}\n설명: {def_v}\n선택지:\n{opts}"
    return {"instruction": instruction, "input":"", "output": str(ans_idx_1based)}

def make_def_pairs(rows, n_per=2, use_crop=False, use_expand=False, use_noise=False) -> List[Dict[str,str]]:
    recs = []
    for r in rows:
        for _ in range(n_per):
            inst = random.choice(DEF_TEMPLATES).format(term=r["term"])
            defn = r["definition"]
            # 변형
            if use_crop and random.random() < 0.5:
                defn_v = pick_crop(defn)
            elif use_expand and random.random() < 0.5:
                defn_v = pick_expand(defn, r["term"])
            else:
                defn_v = normalize_space(defn)
            if use_noise and random.random() < 0.3:
                defn_v = minor_noise_text(defn_v)
            recs.append({"instruction": inst, "input":"", "output": defn_v})
    return recs

# ----- 메인 -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="data/cleaned/tta_terms.jsonl")
    ap.add_argument("--dst_aug", type=str, default="data/cleaned/tta_sft_aug.jsonl")
    ap.add_argument("--dst_plus", type=str, default="data/cleaned/tta_sft_plus.jsonl")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)

    # 기존 옵션
    ap.add_argument("--mcq_per_item", type=int, default=2, help="1=only def->choice, 2=both kinds")
    ap.add_argument("--options", type=int, nargs="+", default=[4], help="ex) --options 4 5 6")
    ap.add_argument("--add_neg", action="store_true", help="부정형 변형 MCQ도 함께 생성")
    ap.add_argument("--balance_labels", action="store_true", help="정답 라벨(1..N) 균등 분포")

    # 새 옵션(양/다양성 확대 핵심)
    ap.add_argument("--times", type=int, default=1, help="각 항목당 반복 생성 횟수(데이터 폭증)")
    ap.add_argument("--hard_ratio", type=float, default=0.7, help="하드 네거티브 비율(0~1)")
    ap.add_argument("--def_crop", action="store_true", help="정의 부분절삭 활성화")
    ap.add_argument("--def_expand", action="store_true", help="정의 확장(접두/접미) 활성화")
    ap.add_argument("--minor_noise", action="store_true", help="경미한 표면 변형 활성화")

    args = ap.parse_args()
    random.seed(args.seed)

    src = Path(args.src)
    rows = load_terms(src)
    _, X = build_tfidf(rows)

    aug = []
    balancer = LabelBalancer() if args.balance_labels else None
    balance = bool(args.balance_labels)

    # (C) 서술형 SFT — times 반영
    aug.extend(
        make_def_pairs(
            rows,
            n_per=max(2, args.times),  # 기본 2, times만큼 늘어남
            use_crop=args.def_crop,
            use_expand=args.def_expand,
            use_noise=args.minor_noise
        )
    )

    # (A)(B) MCQ — times × options 조합으로 대량 생성
    for i in range(len(rows)):
        for _ in range(max(1, args.times)):
            for n_opt in args.options:
                # A) term -> 정의
                rec = make_mcq_def_normal(
                    rows, X, i, n_options=n_opt, balancer=balancer, balance=balance,
                    hard_ratio=args.hard_ratio,
                    use_crop=args.def_crop, use_expand=args.def_expand, use_noise=args.minor_noise
                )
                aug.append(rec)
                if args.add_neg:
                    rec_neg = make_mcq_def_neg(
                        rows, X, i, n_options=n_opt, balancer=balancer, balance=balance,
                        hard_ratio=args.hard_ratio,
                        use_crop=args.def_crop, use_expand=args.def_expand, use_noise=args.minor_noise
                    )
                    aug.append(rec_neg)

                if args.mcq_per_item >= 2:
                    # B) 정의 -> 용어
                    rec2 = make_mcq_term_normal(
                        rows, X, i, n_options=n_opt, balancer=balancer, balance=balance,
                        hard_ratio=args.hard_ratio,
                        use_crop=args.def_crop, use_expand=args.def_expand, use_noise=args.minor_noise
                    )
                    aug.append(rec2)
                    if args.add_neg:
                        rec2_neg = make_mcq_term_neg(
                            rows, X, i, n_options=n_opt, balancer=balancer, balance=balance,
                            hard_ratio=args.hard_ratio,
                            use_crop=args.def_crop, use_expand=args.def_expand, use_noise=args.minor_noise
                        )
                        aug.append(rec2_neg)

    # write augmented
    dst_aug = Path(args.dst_aug); dst_aug.parent.mkdir(parents=True, exist_ok=True)
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