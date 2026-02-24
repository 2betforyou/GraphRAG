from pathlib import Path
import json, random
random.seed(42)

roots = [
  "../dataset/개인정보보호법",
  "../dataset/금융실명법",
  "../dataset/신용정보법",
  "../dataset/전자거래기본법",
  "../dataset/전자금융거래법",
  "../dataset/전자서명법",
  "../dataset/전자통신망법", 
  "../dataset/교육부정보보한기본지침", 
]

def load_jsonl(p):
    p = Path(p)
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

train, val = [], []
for r in roots:
    train += load_jsonl(f"{r}/train.jsonl")
    val   += load_jsonl(f"{r}/val.jsonl")

random.shuffle(train); random.shuffle(val)
Path("../dataset/all").mkdir(parents=True, exist_ok=True)
Path("../dataset/all/train.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in train), encoding="utf-8")
Path("../dataset/all/val.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in val), encoding="utf-8")
print(len(train), len(val))
