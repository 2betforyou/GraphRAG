# utils.py (revised)
import re
import os
import unicodedata
from typing import List, Dict, Tuple
import re, unicodedata, json
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────────────────────
# 공통 정규화 유틸
#  - NFKC 정규화로 전각/호환문자 통일
#  - 제로폭/비표준 공백 제거
#  - 다중 공백을 단일 공백으로 축소
# ─────────────────────────────────────────────────────────────
_ZWS = "\u200b"
_WS_ALL = r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000]"

def _normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace(_ZWS, "")
    t = re.sub(_WS_ALL, " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ─────────────────────────────────────────────────────────────
# 조항 분할/참조 패턴
#  - 제목 분할: '제5조', '제5조의2'까지 인식
#  - 단일 참조: '제5조', '제5조의2', '제5조 제1항'(항은 무시, 조 단위 엣지)
#  - 범위 참조: '제5조부터 제7조까지' (이상/이하도 지원)
#  - 타법 참조: '개인정보보호법 제28조의2' (현 단계는 법명은 미사용)
# ─────────────────────────────────────────────────────────────
ARTICLE_PATTERN = re.compile(r"(제\s*\d+\s*조(?:\s*의\s*\d+)?)")

_SINGLE_REF = re.compile(
    r"(?:제\s*)?(\d+)\s*조"           # 제 5 조
    r"(?:\s*의\s*(\d+))?"             # 의 2 (옵션)
    r"(?:\s*제\s*\d+\s*항)?"          # 제 1 항 (옵션, 조 엣지에서는 무시)
)

_RANGE_REF = re.compile(
    r"(?:제\s*)?(\d+)\s*조\s*(?:부터|이상)\s*(?:제\s*)?(\d+)\s*조\s*(?:까지|이하)"
)

_OTHER_LAW_REF = re.compile(        # 개인정보보호법 제 28 조 의 2
    r"[가-힣A-Za-z0-9\(\)]+법\s*제\s*(\d+)\s*조(?:\s*의\s*(\d+))?"
)

# ─────────────────────────────────────────────────────────────
# 토크나이저/관계추론: 기존 인터페이스 유지 (동작 동일)
# ─────────────────────────────────────────────────────────────
def simple_tokenize(text: str) -> List[str]:
    """아주 단순한 토크나이저 (공백/기호 분리 + 한글 보존)"""
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []

def infer_relation(sent: str) -> str:
    """조항의 문맥에서 관계 유형 추정"""
    s = sent.strip()
    if any(k in s for k in ["벌칙", "형", "과태료", "처벌", "징역", "벌금"]):
        return "벌칙"
    if any(k in s for k in ["예외로", "다만", "예외", "제외한다"]):
        return "예외"
    if any(k in s for k in ["면제", "면제한다"]):
        return "면제"
    if any(k in s for k in ["적용한다", "적용받", "따른다", "준용"]):
        return "적용"
    return "참조"

# ─────────────────────────────────────────────────────────────
# 조항 분할/ID 생성/참조 추출
# ─────────────────────────────────────────────────────────────
def split_articles(raw_text: str) -> List[Tuple[str, str]]:
    """'제XX조(의n)' 기준으로 문서를 조항 단위로 나눔"""
    raw_text = _normalize(raw_text)
    parts = ARTICLE_PATTERN.split(raw_text)
    articles = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        articles.append((title, body))
    return articles

def _ref_key(num: str, sub: str = None) -> str:
    """노드/참조 키를 '5' 또는 '5의2' 형태로 통일"""
    return f"{num}" + (f"의{sub}" if sub else "")

def extract_refs(text: str) -> List[str]:
    """본문에서 참조된 '제XX조(의n)' 목록 추출 (범위/타법/복수 표기 포함)"""
    text = _normalize(text)
    out = set()

    # 1) 범위: 제5조부터 제7조까지 → 5,6,7
    for m in _RANGE_REF.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b:
            for k in range(a, b + 1):
                out.add(_ref_key(str(k)))
        else:
            for k in range(b, a + 1):
                out.add(_ref_key(str(k)))

    # 2) 단일 참조: 제5조, 제5조의2, (제5조 제1항 → 조 단위로만 취급)
    for m in _SINGLE_REF.finditer(text):
        num, sub = m.group(1), m.group(2)
        out.add(_ref_key(num, sub))

    # 3) 타법 참조: 개인정보보호법 제28조의2 → 조 키만 추출(법명은 현 단계 미사용)
    for m in _OTHER_LAW_REF.finditer(text):
        num, sub = m.group(1), m.group(2)
        out.add(_ref_key(num, sub))

    return sorted(out)

def article_id_from_title(title: str, prefix: str) -> str:
    """조항 제목으로 고유 ID 생성 ('제5조의2' → prefix_5의2)"""
    title = _normalize(title)
    m = re.search(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?", title)
    if not m:
        return f"{prefix}_unknown"
    num, sub = m.group(1), m.group(2)
    return f"{prefix}_{_ref_key(num, sub)}"

# ─────────────────────────────────────────────────────────────
# 파일 로더: 기존 인터페이스 유지
#  - 필요 시 줄바꿈 통일 정도만 수행 (정규화는 split/extract 단계에서)
# ─────────────────────────────────────────────────────────────
def load_txts(data_dir: str) -> Dict[str, str]:
    """지정 폴더에서 모든 .txt 파일 로드"""
    out = {}
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".txt"):
            path = os.path.join(data_dir, fn)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                out[fn] = f.read().replace("\r\n", "\n")
    return out

# 1) 정규화
_ZWS = "\u200b"
_WS_ALL = r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000]"

def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace(_ZWS, "")
    t = re.sub(_WS_ALL, " ", t)
    t = re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE)  # 라인 끝 공백
    return t

# 2) 라벨 정규식
R_CHAPTER = re.compile(r"^\s*(제\s*\d+\s*장)\s*(.+)?\s*$")                  # 제1장 총칙
R_SECTION = re.compile(r"^\s*(제\s*\d+\s*절)\s*(.+)?\s*$")                  # 제1절 …
R_ARTICLE = re.compile(r"^\s*(제\s*\d+\s*조(?:\s*의\s*\d+)?)\s*(\([^)]+\))?\s*(.*)$")
# 예: "제7조의2(보호위원회의 구성 등) ..." 또는 "제7조 ..."

R_ARTICLE_DELETE = re.compile(r"^\s*제\s*\d+\s*조(?:\s*의\s*\d+)?\s*삭제\s*$")

# 항: ①(U+2460)–⑳(U+2473) 등
R_PARAGRAPH_BULLET = re.compile(r"^\s*([\u2460-\u2473]+)\s*(.*)$")          # ① …, ② …

# 호: "1. ...", "2. ..."
R_ITEM_BULLET = re.compile(r"^\s*(\d+)\.\s+(.*)$")

# 목: "가. ...", "나. ..."
R_SUBITEM_BULLET = re.compile(r"^\s*([가-힣])\.\s+(.*)$")

def new_node(label: str, title: Optional[str]=None) -> Dict[str, Any]:
    return {"label": label, "title": title or "", "text": "", "children": []}

def parse_law_text(raw: str) -> Dict[str, Any]:
    text = normalize(raw)
    lines = [ln for ln in text.split("\n") if ln.strip() != ""]
    
    root = {"label": "법률", "title": "", "children": []}
    cur_ch = cur_sec = cur_art = None
    cur_par = cur_item = cur_subitem = None

    def push_text(target, s):
        if target is None:
            return
        if target.get("text"):
            target["text"] += "\n" + s
        else:
            target["text"] = s

    for ln in lines:
        # 0) 삭제 조항 단독 라인
        if R_ARTICLE_DELETE.match(ln):
            m = R_ARTICLE.match(ln.replace(" 삭제", ""))
            if m:
                art_label = m.group(1)
                node = new_node("조", art_label)
                node["deleted"] = True
                # 계층 연결
                if cur_sec is not None:
                    cur_sec["children"].append(node)
                elif cur_ch is not None:
                    cur_ch["children"].append(node)
                else:
                    root["children"].append(node)
                cur_art = node
                cur_par = cur_item = cur_subitem = None
            continue

        # 1) 장
        m = R_CHAPTER.match(ln)
        if m:
            cur_ch = new_node("장", f"{m.group(1)} {m.group(2) or ''}".strip())
            root["children"].append(cur_ch)
            cur_sec = cur_art = cur_par = cur_item = cur_subitem = None
            continue

        # 2) 절
        m = R_SECTION.match(ln)
        if m:
            cur_sec = new_node("절", f"{m.group(1)} {m.group(2) or ''}".strip())
            # 절은 장 아래로, 없으면 root 아래로
            if cur_ch is not None:
                cur_ch["children"].append(cur_sec)
            else:
                root["children"].append(cur_sec)
            cur_art = cur_par = cur_item = cur_subitem = None
            continue

        # 3) 조
        m = R_ARTICLE.match(ln)
        if m:
            art_num = m.group(1)                                  # 제7조의2
            art_title = (m.group(2) or "").strip(" ()")           # (보호위원회의 구성 등)
            remainder = m.group(3).strip()                        # 같은 줄의 본문 잔여

            cur_art = new_node("조", f"{art_num}{(' ' + art_title) if art_title else ''}")
            # 조 위치 연결 (절 > 장 > root)
            if cur_sec is not None:
                cur_sec["children"].append(cur_art)
            elif cur_ch is not None:
                cur_ch["children"].append(cur_art)
            else:
                root["children"].append(cur_art)

            # 하위 포인터 초기화
            cur_par = cur_item = cur_subitem = None

            # 같은 줄에 본문이 이어지면 본문으로
            if remainder:
                push_text(cur_art, remainder)
            continue

        # 4) 항
        m = R_PARAGRAPH_BULLET.match(ln)
        if m and cur_art is not None:
            bullet = m.group(1)    # ①
            content = m.group(2)
            cur_par = new_node("항", bullet)
            cur_art["children"].append(cur_par)
            cur_item = cur_subitem = None
            if content:
                push_text(cur_par, content)
            continue

        # 5) 호
        m = R_ITEM_BULLET.match(ln)
        if m and (cur_par is not None or cur_art is not None):
            num = m.group(1)       # 1
            content = m.group(2)
            cur_item = new_node("호", num)
            parent = cur_par if cur_par is not None else cur_art
            parent["children"].append(cur_item)
            cur_subitem = None
            if content:
                push_text(cur_item, content)
            continue

        # 6) 목
        m = R_SUBITEM_BULLET.match(ln)
        if m and (cur_item is not None or cur_par is not None or cur_art is not None):
            alpha = m.group(1)     # 가
            content = m.group(2)
            cur_subitem = new_node("목", alpha)
            # 목은 보통 호 아래, 호가 없으면 항/조 아래
            if cur_item is not None:
                cur_item["children"].append(cur_subitem)
            elif cur_par is not None:
                cur_par["children"].append(cur_subitem)
            else:
                cur_art["children"].append(cur_subitem)
            if content:
                push_text(cur_subitem, content)
            continue

        # 7) 일반 텍스트: 가장 가까운 하위 노드에 축적
        if cur_subitem is not None:
            push_text(cur_subitem, ln)
        elif cur_item is not None:
            push_text(cur_item, ln)
        elif cur_par is not None:
            push_text(cur_par, ln)
        elif cur_art is not None:
            push_text(cur_art, ln)
        elif cur_sec is not None:
            push_text(cur_sec, ln)
        elif cur_ch is not None:
            push_text(cur_ch, ln)
        else:
            push_text(root, ln)

    return root

# — 실행 예시 —
# with open("개인정보보호법.txt", "r", encoding="utf-8") as f:
#     data = parse_law_text(f.read())
# print(json.dumps(data, ensure_ascii=False, indent=2))

# ─────────────────────────────────────────────────────────────
# (선택) 빠른 자가진단용 스모크 테스트
#  - 필요 시 아래 블록을 임시로 열어 간단 확인 가능
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    제2조 (정의) 이 법에서 사용하는 용어는 다음과 같다.
    … 제5조의2 및 제7조에 따른다. 또한 개인정보보호법 제28조의2에 의한다.
    필요시 제10조부터 제12조까지의 규정을 준용한다.
    """
    print("extract_refs(sample) ->", extract_refs(sample))
    # 기대: ['5의2', '7', '10', '11', '12', '28의2']