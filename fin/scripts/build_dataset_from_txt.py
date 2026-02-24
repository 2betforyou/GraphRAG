# -*- coding: utf-8 -*-
"""
원본 .txt(법령/시행령, TTA) → 학습용 JSONL 생성
- 주관식(요약/정의) + 객관식(보기 4개 중 1개 고르기) 동시 생성
- 외부 지식 없이 '다른 문서의 항목'을 오답 보기로 써서 객관식 생성
"""
import re, json, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LAWS_DIR = RAW_DIR / "laws"          # *.txt (파일명=법령명)
TTA_TXT  = RAW_DIR / "TTA_cut.txt"   # TTA 용어사전 텍스트

OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUT_DIR / "train.jsonl"
VAL_OUT   = OUT_DIR / "val.jsonl"
ALL_OUT   = OUT_DIR / "all.jsonl"

# ---------------- 공통 유틸 ----------------
BAD_TOKEN_RE = re.compile(r'(?:�|_(?:[A-Z]{1,5})\b|<\|endoftext\|>)')

def write_jsonl(path: Path, rows: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def split_train_val(rows: List[Dict], train_frac=0.9):
    idx = list(range(len(rows)))
    random.shuffle(idx)
    cut = max(1, int(round(len(idx) * train_frac))) if len(idx) > 1 else len(idx)
    train = [rows[i] for i in idx[:cut]]
    val   = [rows[i] for i in idx[cut:]] if len(idx) > 1 else []
    return train, val

def normalize_text(s: str) -> str:
    s = s.replace("\x0c", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------- 법령 파서 ----------------
ART_SPLIT = re.compile(r"(?=제\s*\d+\s*조)")
ART_HEAD  = re.compile(r"^(제\s*\d+\s*조)\s*(.*)", re.S)

ENUM_PATTERNS = [
    r"^\(?\d+\)\s*(.+)$",            # (1) 항목
    r"^\d+\.\s*(.+)$",               # 1. 항목
    r"^[가-하]\)\s*(.+)$",           # 가) 항목
    r"^[가-하]\.\s*(.+)$",           # 가. 항목
    r"^[①-⑳]\s*(.+)$",              # ① 항목
    r"^[-–—•·]\s+(.+)$",            # - 항목
]

def extract_enums(text: str) -> List[str]:
    out = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t: 
            continue
        got = None
        for p in ENUM_PATTERNS:
            m = re.match(p, t)
            if m:
                got = m.group(1).strip()
                break
        if got:
            got = re.sub(r"\s+", " ", got).rstrip(".,;: ")
            if len(got) > 2:
                out.append(got)
    return out

def parse_law_file(path: Path) -> List[Tuple[str, str, str, List[str]]]:
    """
    returns list of (law_name, article_no, body, enums[])
    """
    name = path.stem
    txt = path.read_text(encoding="utf-8", errors="ignore")
    parts = ART_SPLIT.split(txt)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = ART_HEAD.match(p)
        if not m:
            continue
        art_no = re.sub(r"\s+", "", m.group(1))  # '제1조'
        body = m.group(2).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if len(body) < 30:
            continue
        enums = extract_enums(body)
        out.append((name, art_no, body, enums))
    return out

# ---------------- TTA 파서 ----------------
def parse_tta_terms(txt: str) -> List[Tuple[str, str, Optional[str]]]:
    tagged = re.sub(r"\n\s*(\d{1,2}\.\d{1,3})\s*\n", r"\n@@SECTION@@ \1\n", txt)
    secs = [s.strip() for s in tagged.split("@@SECTION@@") if s.strip()]
    out, seen = [], set()
    for sec in secs:
        lines = [l for l in sec.splitlines() if l.strip()]
        if not lines:
            continue
        section_id = None
        if re.fullmatch(r"\d{1,2}\.\d{1,3}", lines[0].strip()):
            section_id = lines[0].strip()
            lines = lines[1:]
        if not lines:
            continue
        term = lines[0].strip()
        junk = [r"^정보통신단체표준.*$", r"^국문표준.*$", r"^\d+$"]
        body_start = 1
        if any(re.match(p, term) for p in junk):
            if len(lines) >= 2:
                term = lines[1].strip()
                body_start = 2
            else:
                continue
        body = " ".join([re.sub(r"\s+", " ", l.strip()) for l in lines[body_start:] if l.strip()])
        body = normalize_text(body)
        # 맨꼬리 숫자/각주 제거
        body = re.sub(r'\s*(?:\(?\d{1,3}\)?\s*)+$', '', body)
        term = re.sub(r"\s*-\s*", "-", term).strip(" -–—:•·")
        if len(body) < 10:
            continue
        key = (term.lower(), section_id or "")
        if key in seen:
            continue
        seen.add(key)
        out.append((term, body, section_id))
    return out

# ---------------- 샘플 생성기 ----------------
def sys_mc():
    return "당신은 금융보안 시험 채점 도우미입니다. 보기 중 정답 번호(1~4)만 출력하세요. 다른 말 금지."

def sys_gen():
    return ("법령/용어를 정확히 설명하는 도우미입니다. "
            "주어진 원문만 근거로 간결하게 답하세요(외부지식 금지).")

def make_msg(system: str, user: str, assistant: str, meta: Dict) -> Dict:
    if BAD_TOKEN_RE.search(user) or BAD_TOKEN_RE.search(assistant):
        return {}
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ],
        "meta": meta
    }

def summarize_law(law: str, art: str, body: str) -> Dict:
    # 간단 요약(첫 2~3문장)
    sents = re.split(r"(?<=다\.)\s+|(?<=[.!?])\s+", normalize_text(body))
    summ = " ".join(sents[:3]) if sents else normalize_text(body)[:300]
    user = f"법령: {law}\n조문: {art}\n\n다음 조문을 3~5문장으로 요약해 주세요.\n\n{body}"
    return make_msg(
        sys_gen(),
        user,
        summ,
        {"task":"gen_summarize_law","law":law,"article":art}
    )

def tta_define(term: str, definition: str, section_id: Optional[str]) -> Dict:
    # 2문장 정의
    sents = re.split(r"(?<=다\.)\s+|(?<=[.!?])\s+", normalize_text(definition))
    ans = " ".join(sents[:2]) if sents else normalize_text(definition)[:200]
    user = f"용어 정의를 2~3문장으로 간결히 설명하세요.\n용어: {term}\n\n원문:\n{definition}"
    return make_msg(
        sys_gen(),
        user,
        ans,
        {"task":"gen_define_tta","term":term,"section":section_id}
    )

def mc_from_law(law: str, art: str, body: str, enums: List[str], pool_neg: List[str]) -> Optional[Dict]:
    # 이 조문에 '포함되지 않는' 항목 고르기
    if len(enums) < 3 or not pool_neg:
        return None
    pos = random.sample(enums, k=min(3, len(enums)))
    neg = random.choice(pool_neg)
    options = pos + [neg]
    random.shuffle(options)
    correct_idx = options.index(neg) + 1  # 1~4
    view = "\n".join([f"{i}) {opt}" for i,opt in enumerate(options,1)])
    snippet = normalize_text(body)[:500]
    user = (f"[객관식] 다음 중 <{law} {art}>에 포함되지 않는 항목을 고르세요.\n"
            f"{view}\n\n(참고: 조문 일부) {snippet}\n정답:")
    return make_msg(sys_mc(), user, str(correct_idx),
                    {"task":"mc_law_not_included","law":law,"article":art})

def mc_from_tta(term: str, definition: str, section_id: Optional[str], pool_defs: List[str]) -> Optional[Dict]:
    # 이 용어의 '정의로 옳은' 보기 고르기
    if len(pool_defs) < 3:
        return None
    negs = random.sample(pool_defs, k=3)
    options = [normalize_text(definition)] + [normalize_text(n) for n in negs]
    random.shuffle(options)
    correct_idx = options.index(normalize_text(definition)) + 1
    view = "\n".join([f"{i}) {opt}" for i,opt in enumerate(options,1)])
    user = (f"[객관식] 다음 중 '{term}'의 정의로 옳은 것을 고르세요.\n"
            f"{view}\n정답:")
    return make_msg(sys_mc(), user, str(correct_idx),
                    {"task":"mc_tta_definition","term":term,"section":section_id})

def main():
    # 1) 법령 파싱
    law_files = sorted(LAWS_DIR.glob("*.txt"))
    all_articles = []
    for p in law_files:
        all_articles.extend(parse_law_file(p))

    # 법령 기반 객관식용 전체 '타 문서 열거항목' 풀
    other_items_pool = []
    for (law, art, body, enums) in all_articles:
        for it in enums:
            other_items_pool.append((law, art, it))
    # 2) TTA 파싱
    assert TTA_TXT.exists(), f"없음: {TTA_TXT}"
    tta_text = TTA_TXT.read_text(encoding="utf-8", errors="ignore")
    tta_pairs = parse_tta_terms(tta_text)
    tta_defs_pool = [d for _, d, _ in tta_pairs]

    rows: List[Dict] = []

    # 3) 주관식(법령 요약)
    for (law, art, body, enums) in all_articles:
        ex = summarize_law(law, art, body)
        if ex: rows.append(ex)

    # 4) 주관식(TTA 정의)
    for (term, definition, sec) in tta_pairs:
        ex = tta_define(term, definition, sec)
        if ex: rows.append(ex)

    # 5) 객관식(법령): not-included
    only_other = [x for x in other_items_pool]
    for (law, art, body, enums) in all_articles:
        # 타 법령 항목만 오답 풀로 사용(동일 조문 항목은 제외)
        pool_neg = [it for (l2,a2,it) in only_other if not (l2==law and a2==art)]
        ex = mc_from_law(law, art, body, enums, pool_neg)
        if ex: rows.append(ex)

    # 6) 객관식(TTA): 정의 맞히기
    for (term, definition, sec) in tta_pairs:
        pool = [d for d in tta_defs_pool if d != definition]
        ex = mc_from_tta(term, definition, sec, pool)
        if ex: rows.append(ex)

    # 7) 필터/셔플/분할
    rows = [r for r in rows if r and isinstance(r, dict)]
    random.shuffle(rows)
    train, val = split_train_val(rows, train_frac=0.9)

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VAL_OUT, val)
    write_jsonl(ALL_OUT, rows)

    print(f"[OK] 생성 완료: total={len(rows)} train={len(train)} val={len(val)}")
    print(f" -> {TRAIN_OUT}\n -> {VAL_OUT}\n -> {ALL_OUT}")

if __name__ == "__main__":
    main()
