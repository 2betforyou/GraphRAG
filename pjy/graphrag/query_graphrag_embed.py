# query_graphrag_embed.py
import argparse, json, faiss, numpy as np, networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_graph(graph_json):
    data = json.load(open(graph_json, "r", encoding="utf-8"))
    G = nx.DiGraph()
    for n in data["nodes"]:
        G.add_node(n["id"], **n)
    for e in data["edges"]:
        G.add_edge(e["src"], e["dst"], relation=e.get("relation", "참조"))
    return G, data

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
    return list(seen)

def summarize_answer(G, nodes):
    lines = ["■ 근거 조항"]
    for nid in nodes[:5]:
        node = G.nodes[nid]
        lines.append(f"- [{node.get('law')}] {node.get('title')} : {node.get('text','')[:120].strip()}...")
    lines.append("\n※ GraphRAG (E5-ko 임베딩 + 그래프 hop 확장)")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_json", required=True)
    ap.add_argument("--faiss_index", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--hop", type=int, default=1)
    ap.add_argument("--model_name", default="intfloat/multilingual-e5-base")
    args = ap.parse_args()

    # 그래프 + FAISS 로드
    G, _ = load_graph(args.graph_json)
    index = faiss.read_index(args.faiss_index)
    meta = json.load(open(args.faiss_index + ".meta.json", "r", encoding="utf-8"))
    ids = meta["ids"]

    model = SentenceTransformer(args.model_name)
    q_emb = model.encode([args.query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")

    # FAISS 검색
    D, I = index.search(q_emb, args.topk)
    seeds = [ids[i] for i in I[0]]

    expanded = expand_neighbors(G, seeds, hop=args.hop)
    summary = summarize_answer(G, expanded)
    print(summary)

if __name__ == "__main__":
    main()