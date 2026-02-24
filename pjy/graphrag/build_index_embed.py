# build_index_embed.py
import argparse, json, faiss, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_json", required=True)
    ap.add_argument("--faiss_index", required=True)
    ap.add_argument("--emb_path", default="../../dataset/out/embeddings.npy")
    ap.add_argument("--model_name", default="intfloat/multilingual-e5-base")
    args = ap.parse_args()

    print(f"모델 로드 중: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    # 그래프 로드
    graph = json.load(open(args.graph_json, "r", encoding="utf-8"))
    texts = [f"{n.get('title','')}\n{n.get('text','')}" for n in graph["nodes"]]
    ids = [n["id"] for n in graph["nodes"]]

    # 문장 임베딩
    print(f"문장 임베딩 생성 중 ({len(texts)} 조항)...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS 인덱스 구축
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity용
    index.add(embeddings)

    np.save(args.emb_path, embeddings)
    faiss.write_index(index, args.faiss_index)

    meta = {
        "ids": ids,
        "model": args.model_name,
        "dim": embeddings.shape[1],
        "emb_path": args.emb_path,
        "faiss_index": args.faiss_index
    }
    json.dump(meta, open(args.faiss_index + ".meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"저장 완료: {args.faiss_index}")

if __name__ == "__main__":
    main() 