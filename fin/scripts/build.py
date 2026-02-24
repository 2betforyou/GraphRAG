# -*- coding: utf-8 -*-
"""
원본 .txt(TTA) → 학습용 JSONL 생성
- 주관식(용어 정의/요약) + 객관식(정의 맞히기: 보기 4개 중 1개 고르기)
- 법령(법/시행령) 기반 생성은 전부 제거 — 법 데이터는 RAG로 처리
"""
import re, json, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
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

# ---------------- TTA 파서 ----------------
def parse_tta_terms(txt: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    TTA 용어사전 텍스트에서 (용어, 정의문, 섹션ID) 튜플 리스트 추출
    """
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
    return ("TTA 용어를 정확히 설명하는 도우미입니다. "
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

def tta_define(term: str, definition: str, section_id: Optional[str]) -> Dict:
    """
    TTA 용어 정의 → 2문장 내 요약(주관식)
    """
    sents = re.split(r"(?<=다\.)\s+|(?<=[.!?])\s+", normalize_text(definition))
    ans = " ".join(sents[:2]) if sents else normalize_text(definition)[:200]
    user = f"용어 정의를 2~3문장으로 간결히 설명하세요.\n용어: {term}\n\n원문:\n{definition}"
    return make_msg(
        sys_gen(),
        user,
        ans,
        {"task":"gen_define_tta","term":term,"section":section_id}
    )

def mc_from_tta(term: str, definition: str, section_id: Optional[str], pool_defs: List[str]) -> Optional[Dict]:
    """
    객관식: 주어진 용어의 '정의로 옳은 보기'를 고르기(4지선다)
    - 정답(해당 정의) + 오답(다른 용어 정의 3개)
    """
    pool_defs = [d for d in pool_defs if d and d.strip()]
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

# ---------------- 메인 ----------------
def main():
    # 1) TTA 파싱 (법령 관련 전부 제거)
    assert TTA_TXT.exists(), f"없음: {TTA_TXT}"
    tta_text = TTA_TXT.read_text(encoding="utf-8", errors="ignore")
    tta_pairs = parse_tta_terms(tta_text)  # [(term, definition, section)]

    # 정의 풀(객관식 오답 생성용)
    tta_defs_pool = [d for _, d, _ in tta_pairs]

    rows: List[Dict] = []

    # 2) 주관식: TTA 정의(2문장 내)
    for (term, definition, sec) in tta_pairs:
        ex = tta_define(term, definition, sec)
        if ex: rows.append(ex)

    # 3) 객관식: TTA 정의 맞히기(4지선다)
    for (term, definition, sec) in tta_pairs:
        pool = [d for d in tta_defs_pool if d != definition]
        ex = mc_from_tta(term, definition, sec, pool)
        if ex: rows.append(ex)

    # 4) 필터/셔플/분할
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