import csv, json, sys

SYS = {"role":"system","content":"당신은 금융보안 전문 AI 어시스턴트입니다. 질문에 대해 명확하고 간결하게 답변하세요."}

def row_to_example(r):
    t = r["type"].strip().upper()
    if t == "MCQ":
        q = f'{r["question"].strip()}\n{r["options"].strip()}'
        a = r["answer"].strip()  # 숫자만
    else:
        q = r["question"].strip()
        a = r["answer"].strip()
    return {"messages":[SYS, {"role":"user","content":q}, {"role":"assistant","content":a}]}

def main(csv_path, out_jsonl):
    with open(csv_path, newline="", encoding="utf-8") as fh, open(out_jsonl,"w",encoding="utf-8") as out:
        for r in csv.DictReader(fh):
            ex = row_to_example(r)
            out.write(json.dumps(ex, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])