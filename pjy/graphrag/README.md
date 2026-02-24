
# build_graph.py
python build_graph.py \
  --json_dir ../../dataset/laws/parsed_laws \
  --out_json ../../dataset/laws/graphrag/law_graph.json \
  --out_html ../../dataset/laws/graphrag/law_graph.html



# graph_build_crosslaw.py - 교차 법령 참조 버전 
python graph_build_crosslaw.py \
  --json_dir ../../dataset/laws/parsed_laws \
  --out_json ../../dataset/laws/graphrag/law_graph.json \
  --out_html ../../dataset/laws/graphrag/law_graph.html

# build_index_embed.py
python build_index_embed.py \
  --graph_json ../../dataset/laws/graphrag/law_graph.json \
  --faiss_index ../../dataset/laws/graphrag/law_graph.index \
  --emb_path ../../dataset/laws/graphrag/embeddings.npy \
  --model_name intfloat/multilingual-e5-base

# inference_graphrag.py
python inference_graphrag.py \
  --adapter out-exaone-law-qlora/20250828_040425/adapter \
  --rag on \
  --rag_mode graph \
  --index_dir ../dataset/laws/graphrag \
  --graph_json_path ../dataset/laws/graphrag/law_graph.json \
  --rag_topk 3 \
  --rag_max_chars 600 \
  --rag_injection user \
  --bf16
