# -*- coding: utf-8 -*-
from pathlib import Path
import re, json
import pandas as pd

def main(src_path="./data/raw/TTA_cut.txt"):
    SRC = Path(src_path)
    RAW = SRC.read_text(encoding="utf-8", errors="ignore")

    txt = RAW.replace("\ufeff","").replace("\x0c","\n")
    txt = re.sub(r"^\s*\d+\s*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^\s*정보통신단체표준\(국문표준\)\s*$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n\n", txt).strip()
    entry_pat = re.compile(r"(?:^|\n)\s*(\d{1,2}\.\d{1,3})\s*\n+([^\n]+?)\s*(?:\n+|$)", flags=re.MULTILINE)

    starts = []
    for m in entry_pat.finditer(txt):
        starts.append((m.start(), m.end(), m.group(1), m.group(2).strip()))

    entries = []
    for idx, (s_begin, s_end, sec_no, term_line) in enumerate(starts):
        def_start = s_end
        def_end = starts[idx+1][0] if idx + 1 < len(starts) else len(txt)
        definition = txt[def_start:def_end].strip()
        definition = re.sub(r"\n{2,}", "\n", definition).strip()
        definition = re.sub(r"^\s*\d{1,2}\.\d{1,3}\s*$", "", definition, flags=re.MULTILINE).strip()

        term = term_line.strip(" :-\u200b").replace("  ", " ").strip()

        if len(term) < 2 or len(definition) < 3:
            continue

        definition = re.sub(r"[ \t]+", " ", definition)
        # entries.append({"section": sec_no, "term": term, "definition": definition})
        entries.append({"term": term, "definition": definition})

    out_csv = Path("./data/cleaned/tta_terms.csv")
    out_jsonl = Path("./data/cleaned/tta_terms.jsonl")
    out_txt = Path("./data/cleaned/tta_terms.txt")

    # df = pd.DataFrame(entries, columns=["section","term","definition"])
    df = pd.DataFrame(entries, columns=["term","definition"])
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in entries:
            f.write(json.dumps({"term": row["term"], "definition": row["definition"]}, ensure_ascii=False) + "\n")

    with out_txt.open("w", encoding="utf-8") as f:
        for row in entries:
            f.write(f'{row["term"]} / {row["definition"]}\n')

if __name__ == "__main__":
    main()
