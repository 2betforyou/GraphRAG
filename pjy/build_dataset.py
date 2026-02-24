import re, json, random
from pathlib import Path

random.seed(42)

laws = ["개인정보보호법", "금융실명법", "신용정보법", "전자거래기본법", "전자금융거래법", "전자서명법", "정보통신망법", "교육부정보보안기본지침"] 

for law in laws:
    SRC = f"../dataset/{law}.txt" 
    OUT_DIR = Path(f"../dataset/{law}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    txt = Path(SRC).read_text(encoding="utf-8")

    # 잡항목/연락처/시행표기 등 제거(필요시 규칙 추가)
    # txt = re.sub(r"\[시행.*?\]\s*", "", txt)
    # txt = re.sub(r"^\s*전자금융거래법\s*", "", txt)
    # txt = re.sub(r"금융위원회\(.*?\)\s*", "", txt)

    # '제 n 조'로 분할
    parts = re.split(r"(?=제\s*\d+\s*조)", txt)
    articles = []
    for p in parts:
        m = re.match(r"(제\s*\d+\s*조)\s*(.*)", p, re.S)
        if not m: 
            continue
        a_no = re.sub(r"\s+", "", m.group(1))  # '제1조'형태
        body = m.group(2).strip()
        if len(body) < 30: 
            continue
        # 과도한 공백 정리
        body = re.sub(r"\n{3,}", "\n\n", body)
        articles.append((a_no, body))

    def to_example(a_no, body):
        """
        학습 최적화 버전:
        - 시스템 프롬프트를 엄격·일관 포맷으로 지정
        - 조문만 근거로 요약/핵심포인트/예외·벌칙/근거를 구성
        - 규칙 기반으로 열거항 추출(①, 1., (1), 가. 등) → 불릿화
        - 예외/벌칙/의무 여부 간단 태깅
        """
        # ---------- 내부 유틸 ----------
        def normalize(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()

        def make_summary(text: str, max_chars: int = 350) -> str:
            # 문장 단위로 대략 3~5문장 범위 확보 시도 → 부족하면 길이 제한 요약
            t = normalize(text)
            # 간이 문장 분리(법령체 '...한다.' 중심)
            sents = re.split(r"(?<=다\.)\s+|(?<=[\.\?！!])\s+", t)
            sents = [s for s in sents if s]
            if len(sents) >= 3:
                summ = " ".join(sents[:5])
            else:
                summ = t[:max_chars]
            return summ

        def extract_bullets(text: str, limit: int = 6):
            bullets = []
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                # 열거 패턴: ①~⑳, (1), 1., 가./나./다., - 항목 등
                m = re.match(
                    r"^(?:[①-⑳]|[가-하]\.|[가-하]\)|\(\d+\)|\d+\.)\s*(.+)$", ln
                )
                if not m:
                    m = re.match(r"^(?:[-–—•·]\s+)(.+)$", ln)
                if m:
                    item = normalize(m.group(1))
                    if len(item) > 2:
                        bullets.append(item)
                if len(bullets) >= limit:
                    break
            # 열거가 없으면 본문 앞부분을 2~3개로 쪼개 대체
            if not bullets:
                t = normalize(text)
                chunks = re.split(r"(?<=다\.)\s+|(?<=[\.\?！!])\s+", t)
                bullets = [c for c in chunks[:3] if len(c) > 5]
            return bullets[:limit]

        def detect_flags(text: str):
            txt = text
            flags = []
            if re.search(r"(과태료|벌금|징역|형에 처한다|벌칙)", txt):
                flags.append("벌칙/제재 포함")
            if re.search(r"(다만|예외|단,\s|단\s)", txt):
                flags.append("예외 규정 존재")
            if re.search(r"(하여야 한다|해야 한다|금지한다|하지 아니한다)", txt):
                flags.append("의무/금지 규정")
            if re.search(r"(정의|뜻을|의미를)", txt):
                flags.append("정의 규정")
            if re.search(r"(목적)", txt):
                flags.append("목적 규정")
            return flags

        # ---------- 메시지 구성 ----------
        system = (
            "법령에 대해 설명하시오. 다음 원칙을 지키십시오:\n"
            "1) 주어진 조문만 근거로 사실만 기술하고 추측·외부지식은 금지합니다.\n"
            "2) 출력 형식은 아래와 같습니다.\n"
            "   [요약]: 3~5문장으로 핵심을 간결히 정리\n"
            "   [핵심포인트]: 불릿 3~6개(열거항이 있으면 추출, 없으면 핵심 문장)\n"
            "   [예외·벌칙]: 해당 시 간단 정리, 없으면 '해당 없음'\n"
            "   [근거]: <법령명> <조문번호>\n"
            "3) 한글, 존댓말, 간결·정확한 표현을 사용합니다."
        )

        # 사용자 메시지(원문을 그대로 제공)
        user = f"법령: {law}\n조문: {a_no}\n\n{body}"

        # 어시스턴트 레이블(규칙 기반 생성)
        summary  = make_summary(body)
        bullets  = extract_bullets(body)
        flags    = detect_flags(body)

        flags_str = " · ".join(flags) if flags else "해당 없음"
        bullets_str = "\n".join(f"- {b}" for b in bullets)

        assistant = (
            f"[요약] {summary}\n\n"
            f"[핵심포인트]\n{bullets_str}\n\n"
            f"[예외·벌칙] {flags_str}\n"
            f"[근거] {law} {a_no}"
        )

        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ],
            "meta": {"law": law, "article": a_no}
        }

    dataset = [to_example(a,b) for a,b in articles]
    random.shuffle(dataset)
    n_val = max(1, int(len(dataset)*0.1))
    val, train = dataset[:n_val], dataset[n_val:]

    def dump(items, path):
        with open(path, "w", encoding="utf-8") as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False)+"\n")

    dump(train, OUT_DIR/"train.jsonl")
    dump(val,   OUT_DIR/"val.jsonl")
    print(f"articles={len(articles)}, train={len(train)}, val={len(val)}")

