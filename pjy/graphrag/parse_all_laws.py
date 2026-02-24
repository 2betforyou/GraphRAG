import os, glob, json
from tqdm import tqdm
from utils import parse_law_text  # 방금 만든 utils.py에 포함되어 있어야 함

IN_DIR = "../../dataset/laws/clean_laws"
OUT_DIR = "../../dataset/laws/parsed_laws"

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    files = glob.glob(os.path.join(IN_DIR, "*.txt"))
    print(f"[+] Found {len(files)} law files.")
    for path in tqdm(files, desc="parsing laws"):
        fname = os.path.basename(path)
        out_path = os.path.join(OUT_DIR, fname.replace(".txt", ".json"))

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        data = parse_law_text(raw)   # 앞서 만든 함수 사용

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] Parsed laws saved under {OUT_DIR}/")

if __name__ == "__main__":
    main()