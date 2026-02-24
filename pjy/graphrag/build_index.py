# build_index.py
import argparse, json, pickle
from rank_bm25 import BM25Okapi
from utils import simple_tokenize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_json", required=True)
    ap.add_argument("--bm25_path", required=True)
    args = ap.parse_args()

    graph = json.load(open(args.graph_json, "r", encoding="utf-8"))
    docs = []
    ids = []
    for n in graph["nodes"]:
        text = f"{n.get('title','')}\n{n.get('text','')}"
        docs.append(simple_tokenize(text))
        ids.append(n["id"])
    bm25 = BM25Okapi(docs)
    with open(args.bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "graph": graph}, f)
    print(f"Indexed {len(ids)} nodes. Saved -> {args.bm25_path}")

if __name__ == "__main__":
    main()