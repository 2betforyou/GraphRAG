import re
from pathlib import Path

# 입력 폴더 (원본 .txt)
src_dir = Path("../dataset/laws")
src_files = list(src_dir.glob("*.txt"))

# 출력 폴더
out_dir = Path("../dataset/laws/cleaned")
out_dir.mkdir(parents=True, exist_ok=True)

# 동그라미 숫자 매핑 (①~⑳)
circled_map = {
    "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5",
    "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9", "⑩": "10",
    "⑪": "11", "⑫": "12", "⑬": "13", "⑭": "14", "⑮": "15",
    "⑯": "16", "⑰": "17", "⑱": "18", "⑲": "19", "⑳": "20",
}

def clean_text(text: str) -> str:
    # <...> 꺾쇠 표기 삭제 (예: <개정 2020.2.4>)
    text = re.sub(r"<[^>]*>", "", text)
    # [...] 대괄호 표기 삭제 (예: [본조신설 2020.2.4.])
    text = re.sub(r"\[[^\]]*\]", "", text)
    # 동그라미 숫자 치환
    for k, v in circled_map.items():
        text = text.replace(k, v)
    # 공백/개행 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

for path in src_files:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(raw)

    out_path = out_dir / f"{path.stem}_cleaned.txt"
    out_path.write_text(cleaned, encoding="utf-8")
    print(f"[OK] {path.name} → {out_path}")