#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_corpus.py
- 현재 폴더의 *.txt를 읽어 법령 보일러플레이트/불용어/숫자/기호 등을 정리
- 출력:
  cleaned/<이름>.cleaned.txt         : 공백으로 토큰 나열(학습 전처리용)
  cleaned/<이름>.cleaned_lines.txt   : 40토큰 단위로 줄바꿈(가독성/청크화 용이)
  cleaned/token_freq_top2000.csv     : 상위 토큰 빈도
- 커스터마이즈: stopwords_custom.txt 를 같은 폴더에 두면 합쳐서 사용

python scripts/clean_corpus.py
"""
import re, unicodedata
from pathlib import Path
from collections import Counter

DEFAULT_STOPWORDS = sorted(list(set("""
가령 각각 간의 간에 가장 각종 각호 각항 같은 같은바 같은점 같은중 같은지 같은편 같은해 해당 해당함 해당하는 경우 경우에 경우에는 경우에도 경우이며 경우임 경우임에도 경우임을 경우임이 경우이므로 경우이나 경우이며 경우이나도 경우이나도불구하고
것 것이다 것과 것까지 것들 것만 것보다 것으로 것으로서 것으로써 것과는 그 그간 그및 그밖에 그중 그에 그의 그와 그외 그중에 그리고 그러나 하지만 또는 또한
가 나 다 라 마 바 사 아 자 차 카 타 파 하
및 등 등의 등은 등에서 등이 등으로 등으로서 등으로써 등과 등과의 등과같이 등과같은 등과같으며
이 이가 이나 이다 이며 이고 이를 이와 이외 이외의 이외에는 이외에도 이외임 이중 이중에 이중에서 이중에도 이중이며
이상 이하 초과 미만
때 때에 때에는 때에도 경우 때로 때마다 때와 때에는의
따라 따라서 그러므로 그러한 그러한바 다만 다만, 다만,다만
대한 대하여 대한다 대할 때 대하는 바
되어 되며 되고 되는 되는바 되지 아니한다 아니하며 아니하고 아니할 수 없다 아니한 경우 아니다 아니다.
아닌 아닌바 아닌바, 아닌바이며 아닌지 아닌한 아닌자의
있다 없다 있는 없는 있으며 없으며 있을 수 있다 할 수 있다 하여야 한다 하여야 하며 해서는 아니된다 아니한다 아니하며 아니하고
여부 여부를 여부에 여부에따라 여부와
여러 여러가지 여러 개 다양한 각종
제 조 항 호 목 장 절 관 칙 편 부
해당 해당자 해당기관 해당업무 해당목적
같은 법 이 법 같은 영 같은 규정 이 규정 같은 조 같은 항 같은 호 같은 목 같은 장 같은 절 같은 편 같은 부 이 장 이 조 이 항 이 호 이 목
개정 신설 삭제 준용 전부개정 일부개정
규정 규정은 규정에 규정의 규정하여 규정하고 규정한 규정된 규정됨
정의 정의는 정의에 정의의 정의하고 정의된
목적 목적은 목적에 목적의
원칙 원칙은 원칙에
권리 책무 책임
시행 시행일 시행령 시행규칙
그 밖에 그밖에 그밖에도 그 밖의 그밖의
또는 그리고 그러나 하지만 또한 및 등 등의 등은 등에서 등이 등으로 등에서 등에게 등에도
관한 관하여 관련 관련된 통하여 통한 위하여 위한 같이 같은 경우 때문에 대하여 대한 이러한 그러한 해당
한다 한다. 한다면 한다는 한다는바 하여 하여야 하여야한다 하여서는 아니한다 아니하며 아니하고 아니할 수 수 있는 없는 있으며 없으며 있다 없다 된다 된다. 되어 되고 이다 이다.
""".split())))

RE_HANGUL_OR_ASCII = re.compile(r"[가-힣A-Za-z]")
RE_PUNCT_EDGES = re.compile(r"^[^\w가-힣]+|[^\w가-힣]+$")
RE_ALL_DIGITS = re.compile(r"^[0-9]+$")

PATTERNS = {
    "angle_notes": re.compile(r"<[^>]{1,80}>"),
    "law_markers": re.compile(r"제\s*\d+\s*(장|조|항|호|목)"),
    "bullets": re.compile(r"(^|\n)[\s]*((\(?[0-9]+\)?|[가-힣]\.|\([가-힣0-9]+\)|[•·\-–—○●]))\s*", flags=re.MULTILINE),
    "alias": re.compile(r"\(\s*이하\s*[\"“']?([^\"”')]+)[\"”']?\s*라\s*한다\s*\)"),
    "noise_paren": re.compile(r"\(\s*(?:\d{2,4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}|[0-9,\.\s]+)\s*\)"),
    "corner_quotes": re.compile(r"[「」]"),
}

def normalize(txt: str) -> str:
    txt = unicodedata.normalize("NFKC", txt)
    txt = txt.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    txt = PATTERNS["angle_notes"].sub(" ", txt)
    txt = PATTERNS["alias"].sub(r" \1 ", txt)        # (이하 "별칭"이라 한다) → 별칭만 보존
    txt = PATTERNS["law_markers"].sub(" ", txt)      # 제1조/제2항 등 제거
    txt = PATTERNS["bullets"].sub(r"\1", txt)        # 1. (1) 가. • 등 머리기호 제거
    txt = PATTERNS["noise_paren"].sub(" ", txt)      # 날짜/숫자만 든 괄호 제거
    txt = PATTERNS["corner_quotes"].sub("", txt)     # 「」 기호 제거
    txt = txt.replace("〈", " ").replace("〉", " ").replace("《"," ").replace("》"," ")
    txt = re.sub(r"\b\d{2,4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\b", " ", txt)  # 2024. 7. 31.
    txt = re.sub(r"\b제\s*\d+\b", " ", txt)          # '제 3' 잔재 제거
    return txt

def tokenize_and_filter(txt: str, stopwords: set[str], min_len: int = 2):
    toks = []
    for raw in txt.split():
        t = RE_PUNCT_EDGES.sub("", raw.lower())
        if not t: continue
        if RE_ALL_DIGITS.match(t): continue
        if not RE_HANGUL_OR_ASCII.search(t): continue
        if len(t) < min_len: continue
        if t in stopwords: continue
        toks.append(t)
    return toks

def load_text(path: Path) -> str:
    for enc in ("utf-8","cp949","euc-kr"):
        try:
            return path.read_text(encoding=enc)
        except Exception: pass
    return path.read_bytes().decode("utf-8","ignore")

def main(in_glob: str = "./data/raw/laws/*.txt", out_dir: str = "./data/cleaned"):
    root = Path(".")
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)
    stop_file = Path("stopwords_custom.txt")
    stop = set(DEFAULT_STOPWORDS)
    if stop_file.exists():
        stop |= set(w.strip() for w in stop_file.read_text(encoding="utf-8").splitlines() if w.strip())

    all_tokens = []
    for p in root.glob(in_glob):
        raw = load_text(p)
        toks = tokenize_and_filter(normalize(raw), stop, min_len=2)
        (out / f"{p.stem}.cleaned.txt").write_text(" ".join(toks), encoding="utf-8")
        lines = [" ".join(toks[i:i+40]) for i in range(0, len(toks), 40)]
        (out / f"{p.stem}.cleaned_lines.txt").write_text("\n".join(lines), encoding="utf-8")
        all_tokens.extend(toks)
        print(f"[OK] {p.name} -> {len(toks)} tokens")

    freq = Counter(all_tokens)
    with (out / "token_freq_top2000.csv").open("w", encoding="utf-8") as f:
        f.write("token,count\n")
        for tok, c in freq.most_common(2000):
            f.write(f"{tok},{c}\n")

if __name__ == "__main__":
    main()
