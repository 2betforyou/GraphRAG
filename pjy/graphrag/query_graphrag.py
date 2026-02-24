# query_graphrag.py
import argparse, json, pickle, networkx as nx
from utils import simple_tokenize

def load_graph(graph_json):
    data = json.load(open(graph_json, "r", encoding="utf-8"))
    G = nx.DiGraph()
    for n in data["nodes"]:
        G.add_node(n["id"], **n)
    for e in data["edges"]:
        G.add_edge(e["src"], e["dst"], relation=e.get("relation","참조"))
    return G

def expand_neighbors(G, seeds, hop=1):
    seen = set(seeds)
    frontier = list(seeds)
    for _ in range(hop):
        nxt = []
        for s in frontier:
            for _, v in G.out_edges(s):
                if v not in seen:
                    seen.add(v); nxt.append(v)
            for u, _ in G.in_edges(s):
                if u not in seen:
                    seen.add(u); nxt.append(u)
        frontier = nxt
    return list(seeds) + [n for n in seen if n not in seeds]

def summarize_answer(G, nodes):
    lines = ["■ 근거 조항"]
    for nid in nodes[:5]:
        node = G.nodes[nid]
        lines.append(f"- [{node.get('law')}] {node.get('title')} : {node.get('text','')[:120].strip()}...")
    lines.append("\n※ GraphRAG 기반 요약 (BM25+graph hop 확장)")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_json", required=True)
    ap.add_argument("--bm25_path", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--hop", type=int, default=1)
    args = ap.parse_args()

    with open(args.bm25_path, "rb") as f:
        pack = pickle.load(f)
    bm25, ids = pack["bm25"], pack["ids"]
    G = load_graph(args.graph_json)

    q_tokens = simple_tokenize(args.query)
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    seeds = [rid for rid, _ in ranked[:args.topk]]
    expanded = expand_neighbors(G, seeds, hop=args.hop)

    summary = summarize_answer(G, expanded)
    print(summary)

if __name__ == "__main__":
    main()