# -*- coding: utf-8 -*-
import json, re, os, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

DATA_DIR = Path("../dataset/")
ALL_DIR  = DATA_DIR / "all"

train_path = ALL_DIR / "train.jsonl"
val_path   = ALL_DIR / "val.jsonl"

tta_txt_path         = DATA_DIR / "TTA_cut.txt"
out_jsonl_path       = DATA_DIR / "TTA_cut.jsonl"
tta_train_jsonl_path = ALL_DIR / "TTA_cut_train.jsonl"
tta_val_jsonl_path   = ALL_DIR / "TTA_cut_val.jsonl"

merged_train_jsonl_path = ALL_DIR / "train_plus_TTA.jsonl"
merged_val_jsonl_path   = ALL_DIR / "val_plus_TTA.jsonl"

# ✅ 학습 최적화: 고정된 출력 포맷 + 외부지식 금지 + 간결 문장
NEW_SYSTEM_PROMPT = (
    "다음 용어의 정의를 설명하세요. 다음 규칙을 엄격히 따르십시오:\n"
    "1) 제공된 원문만 근거로 사실을 기술하고 추측·외부지식·각주 삽입을 금지합니다.\n"
    "2) 출력 형식은 아래와 같습니다(형식을 변경하지 마십시오).\n"
    "[정의]: 2~3문장으로 간결 요약\n"
    "[핵심포인트]: 불릿 3~6개(원문 열거가 있으면 해당 항목을 사용)\n"
    "[예시]: 원문에 예가 있으면 1~2개, 없으면 '해당 없음'\n"
    "[주의/대비개념]: 혼동 가능 개념·주의사항 요약 또는 '해당 없음'\n"
    "[근거]: TTA 표준{섹션번호}\n"
    "3) 한글·존댓말을 사용하고 과도한 수식어를 피합니다."
)
SEED = 42

# ---------------- 공통 유틸 ----------------
BAD_TOKEN_RE = re.compile(r'(?:�|_(?:[A-Z]{1,5})\b|<\|endoftext\|>)')
RE_MULTI_SPACE = re.compile(r'\s{2,}')

def read_first_jsonl_record(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except Exception:
                continue
    raise RuntimeError(f"JSONL에서 유효한 첫 레코드를 찾지 못했습니다: {path}")

def count_nonempty_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except FileNotFoundError:
        return 0

def write_jsonl(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(src_paths: List[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for p in src_paths:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else (line + "\n"))

def detect_schema(example: Dict) -> str:
    if "messages" in example and isinstance(example["messages"], list):
        return "messages"
    if "conversations" in example and isinstance(example["conversations"], list):
        return "conversations"
    if all(k in example for k in ("instruction", "output")):
        return "alpaca"
    if all(k in example for k in ("prompt", "completion")):
        return "prompt_completion"
    return "messages"

# ---------------- TTA 파서(강화) ----------------
def parse_tta_terms(txt: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    TTA_cut.txt → (term, body, section_id) 목록
    - 섹션번호는 라인 단독 '^\s*\d{1,3}\.\d{1,4}\s*$'로 식별
    - 용어는 앞 1~2줄을 결합(짧음/괄호 등 휴리스틱)
    - 본문은 이후 줄을 결합, 꼬리 숫자 제거
    """
    # 1) 개행 보존 정규화(절대 전체 공백 치환 금지)
    t = txt.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")

    # 2) 섹션 헤더(숫자만 있는 줄) 위치 수집
    sec_re = re.compile(r"(?m)^\s*(\d{1,3}\.\d{1,4})\s*$")
    marks = list(sec_re.finditer(t))
    pairs: List[Tuple[str, str, Optional[str]]] = []
    seen = set()

    for i, m in enumerate(marks):
        section_id = m.group(1)
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(t)
        block = t[start:end].strip()
        if not block:
            continue

        # 3) 줄 단위 분해(빈 줄 제거)
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        # 4) 용어 조립(앞 1~2줄 결합 휴리스틱)
        term = lines[0]
        consumed = 1
        if len(lines) >= 2:
            cond_short = (len(term) < 6)           # 너무 짧은 1단어(예: "데이터가")
            cond_openp = term.endswith(("(", "-", "–", "—"))
            cond_no_close = (")" not in term and len(term) < 30 and len(lines[1]) < 60)
            if cond_short or cond_openp or cond_no_close:
                term = f"{term} {lines[1]}".strip()
                consumed = 2

        # 5) 정리(필드별 공백 정규화)
        term = re.sub(r"\s*-\s*", "-", term)                 # 하이픈 간격 정리
        term = re.sub(r"\s+", " ", term).strip(" -–—:•·")

        # 6) 본문 구성(+ 꼬리 페이지/각주 제거)
        body_lines = lines[consumed:]
        if not body_lines:
            continue
        body = " ".join(body_lines)
        body = re.sub(r"\s*(?:\(?\d{1,3}\)?\s*)+$", "", body)  # '... 합니다. 67' / '(67)' 꼬리 제거
        body = re.sub(r"\s*-\s*", "-", body)
        body = re.sub(r"\s+", " ", body).strip()

        if len(body) < 10:
            continue

        key = (term.lower(), section_id or "")
        if key in seen:
            continue
        seen.add(key)
        pairs.append((term, body, section_id))

    return pairs

# ---------------- 텍스트 전처리 & 구성 ----------------
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _to_polite(s: str) -> str:
    s = s.strip()
    # 흔한 비격식/체언종결 보정
    s = re.sub(r'해줌[.]?$', '하도록 합니다.', s)
    s = re.sub(r'함[.]?$', '합니다.', s)
    s = re.sub(r'임[.]?$', '입니다.', s)
    # 마침표 보강
    if not re.search(r'[.!?]$', s):
        s += '.'
    return s

def _make_definition(text: str, max_chars: int = 300) -> str:
    """
    2~3문장 정도로 간결 요약(법령체 문장 경계 우선) + 존댓말 보정
    """
    t = _normalize(text)
    sents = re.split(r"(?<=다\.)\s+|(?<=[.!?])\s+", t)
    sents = [x for x in sents if x]
    if not sents:
        return ''
    sents = [_to_polite(x) for x in sents]
    out = " ".join(sents[:3])
    return out[:max_chars].strip()

def _extract_bullets_min3(text: str, limit: int = 6) -> List[str]:
    """
    열거 패턴 우선 추출 → 없으면 절(clause) 분해로 최소 3개 보장
    """
    bullets: List[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"^(?:[①-⑳]|[가-하]\.|[가-하]\)|\(\d+\)|\d+\.)\s*(.+)$", ln)
        if not m:
            m = re.match(r"^(?:[-–—•·]\s+)(.+)$", ln)
        if m:
            item = _normalize(m.group(1)).rstrip('.,; ')
            bullets.append(item)
        if len(bullets) >= limit:
            break

    if not bullets:
        # 쉼표/접속사 기준 절 분해
        t = _normalize(text)
        parts = re.split(r'(?:,|\s(?:및|그리고|또는|하여|하도록|으로|에\s*따라)\s)', t)
        parts = [p.strip().rstrip('.,; ') for p in parts if len(p.strip()) > 3]
        # 행위/대상/효과 힌트 기반 정렬
        key = {'출처':[], '행위':[], '효과':[], '기타':[]}
        for p in parts:
            if re.search(r'(신고|확인된|근거|원문)', p): key['출처'].append(p)
            elif re.search(r'(제공|연동|전달|등록|관리|저장|암호화|검증)', p): key['행위'].append(p)
            elif re.search(r'(차단|방지|제한|탐지|완화|보호)', p): key['효과'].append(p)
            else: key['기타'].append(p)
        ordered = key['출처'] + key['행위'] + key['효과'] + key['기타']
        bullets = [p for p in ordered if p][:limit]

    # 중복 제거
    dedup, seen = [], set()
    for b in bullets:
        k = b.lower()
        if k in seen: 
            continue
        seen.add(k)
        dedup.append(b)

    while len(dedup) < 3:
        dedup.append(f"핵심 사항 {len(dedup)+1}")
    return dedup[:limit]

def _extract_examples(text: str, limit: int = 2) -> List[str]:
    ex = []
    for m in re.finditer(r"(?:예:|예\)|예시:|예를 들어)\s*([^\.。]+)", text):
        cand = _normalize(m.group(1))
        if cand and len(cand) > 2:
            ex.append(cand)
    if not ex:
        for m in re.finditer(r"\((예:|예시:)\s*([^)]{2,80})\)", text):
            cand = _normalize(m.group(2))
            if cand:
                ex.append(cand)
    return ex[:limit]

def _extract_contrast(text: str) -> Optional[str]:
    if re.search(r"(혼동|구분|대비|다르다|유사하지만)", text):
        t = _normalize(text)
        sents = re.split(r"(?<=다\.)\s+|(?<=[\.\?！!])\s+", t)
        if sents:
            return sents[0]
    return None

# ---------------- 레코드 생성(학습 최적화) ----------------
def build_record(schema: str, term: str, definition: str, section_id: Optional[str]) -> Optional[Dict]:
    # 노이즈/깨짐 검사 — 발견 시 레코드 폐기
    if BAD_TOKEN_RE.search(term) or BAD_TOKEN_RE.search(definition):
        return None

    defn = _make_definition(definition)
    bullets = _extract_bullets_min3(definition)
    examples = _extract_examples(definition)
    contrast = _extract_contrast(definition)

    bullets_str = "\n".join(f"- {b}" for b in bullets)
    examples_str = "\n".join(f"- {e}" for e in examples) if examples else "해당 없음"
    contrast_str = contrast if contrast else "해당 없음"
    ref = f"TTA 표준({section_id})" if section_id else "TTA 표준"

    assistant = (
        f"[정의] {defn}\n\n"
        f"[핵심포인트]\n{bullets_str}\n\n"
        f"[예시] {examples_str}\n"
        f"[주의/대비개념] {contrast_str}\n"
        f"[근거] {ref}"
    )

    if schema == "messages":
        return {
            "messages": [
                {"role": "system", "content": NEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"용어: {term}\n\n원문:\n{definition}"},
                {"role": "assistant", "content": assistant},
            ],
            "meta": {"term": term, "section": section_id}
        }
    if schema == "conversations":
        return {
            "conversations": [
                {"role": "system", "content": NEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"용어: {term}\n\n원문:\n{definition}"},
                {"role": "assistant", "content": assistant},
            ],
            "meta": {"term": term, "section": section_id}
        }
    if schema == "alpaca":
        return {
            "instruction": NEW_SYSTEM_PROMPT,
            "input": f"용어: {term}\n\n원문:\n{definition}",
            "output": assistant,
            "meta": {"term": term, "section": section_id}
        }
    if schema == "prompt_completion":
        return {
            "prompt": f"{NEW_SYSTEM_PROMPT}\n용어: {term}\n원문:\n{definition}\n",
            "completion": assistant,
            "meta": {"term": term, "section": section_id}
        }
    # fallback
    return {
        "messages": [
            {"role": "system", "content": NEW_SYSTEM_PROMPT},
            {"role": "user", "content": f"용어: {term}\n\n원문:\n{definition}"},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {"term": term, "section": section_id}
    }

# ---------------- split ----------------
def split_records(records: List[Dict], train_frac: float, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    rnd = random.Random(seed)
    idx = list(range(len(records)))
    rnd.shuffle(idx)
    cut = max(1, min(len(idx)-1, int(round(len(idx) * train_frac)))) if len(idx) > 1 else len(idx)
    train_idx = set(idx[:cut])
    train_recs = [records[i] for i in range(len(records)) if i in train_idx]
    val_recs   = [records[i] for i in range(len(records)) if i not in train_idx]
    return train_recs, val_recs

# ---------------- 메인 ----------------
def main():
    random.seed(SEED)
    assert train_path.exists(), f"train.jsonl 경로 없음: {train_path}"
    assert tta_txt_path.exists(), f"TTA_cut.txt 경로 없음: {tta_txt_path}"

    # 1) 스키마 감지(원본 train 기준)
    example = read_first_jsonl_record(train_path)
    schema = detect_schema(example)

    # 2) TTA 텍스트 파싱 → (term, definition, section_id) 목록
    tta_text = tta_txt_path.read_text(encoding="utf-8", errors="ignore")
    pairs = parse_tta_terms(tta_text)

    # 3) 레코드 생성 (+ 깨짐 레코드 제거)
    built = [build_record(schema, term, definition, section_id) for term, definition, section_id in pairs]
    records = [r for r in built if r is not None]

    # 4) 전체 TTA JSONL 저장(옵션)
    write_jsonl(out_jsonl_path, records)

    # 5) 기존 train/val 비율로 split 비율 결정 (fallback 0.9)
    n_train = count_nonempty_lines(train_path)
    n_val   = count_nonempty_lines(val_path)
    total   = n_train + n_val
    train_frac = (n_train / total) if total > 0 else 0.9

    # 6) 분할 및 저장
    tta_train, tta_val = split_records(records, train_frac, seed=SEED)
    write_jsonl(tta_train_jsonl_path, tta_train)
    write_jsonl(tta_val_jsonl_path, tta_val)

    # 7) 병합본 생성: (원본 train + TTA_train), (원본 val + TTA_val)
    append_jsonl([train_path, tta_train_jsonl_path], merged_train_jsonl_path)
    append_jsonl([val_path,   tta_val_jsonl_path],   merged_val_jsonl_path)

    print(f"[OK] schema={schema}")
    print(f"TTA pairs: {len(pairs)} → valid records: {len(records)} → split ratio(train): {train_frac:.3f}")
    print(f"저장: {out_jsonl_path}")
    print(f"저장: {tta_train_jsonl_path}, {tta_val_jsonl_path}")
    print(f"저장: {merged_train_jsonl_path}, {merged_val_jsonl_path}")

if __name__ == "__main__":
    main()