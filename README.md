# GraphRAG 기반 금융보안 규제 해석 프레임워크 
## 한국전자거래학회 2025 추계학술대회
## 주성용 박준영 오병훈 이재우


## RAG for EXAONE-3.5-7.8B (Korean Laws/Guidelines)

이 패키지는 로컬 텍스트/JSONL 자료(법령, 시행령, 지침, MITRE 등)를 기반으로 **EXAONE-3.5-7.8B-Instruct**에 연결하는 RAG 파이프라인입니다.

## 요구 패키지
```
pip install "transformers>=4.41.0" "accelerate>=0.31.0" "sentence-transformers>=3.0.1"             "faiss-cpu>=1.8.0" rank_bm25 "bitsandbytes>=0.43.1"             "peft>=0.11.1"
# (CUDA가 있다면 faiss-gpu를 권장합니다. 환경에 맞는 whl 설치)
```

## 1) 인덱스 빌드
```
python build_index.py   --sources "/mnt/data/*.txt" "/mnt/data/*.jsonl"   --index-dir "./data/index/law_rag"   --embedding-model "intfloat/multilingual-e5-base"   --chunk-chars 900 --chunk-overlap 200   --bm25
```

## 2) 질의/생성
```
python ask_rag.py   --index-dir "./data/index/law_rag"   --base-model "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"   --topk 6 --alpha 0.2   --max-new-tokens 320 --temperature 0.2
```

**alpha**: BM25와 벡터 스코어 결합 가중치 (0=벡터만, 1=BM25만).

---


### build_graph.py
```
python build_graph.py \
  --json_dir ../../dataset/laws/parsed_laws \
  --out_json ../../dataset/laws/graphrag/law_graph.json \
  --out_html ../../dataset/laws/graphrag/law_graph.html
```


### graph_build_crosslaw.py - 교차 법령 참조 버전 
```
python graph_build_crosslaw.py \
  --json_dir ../../dataset/laws/parsed_laws \
  --out_json ../../dataset/laws/graphrag/law_graph.json \
  --out_html ../../dataset/laws/graphrag/law_graph.html
```


### build_index_embed.py
```
python build_index_embed.py \
  --graph_json ../../dataset/laws/graphrag/law_graph.json \
  --faiss_index ../../dataset/laws/graphrag/law_graph.index \
  --emb_path ../../dataset/laws/graphrag/embeddings.npy \
  --model_name intfloat/multilingual-e5-base
```


### inference_graphrag.py
```
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
```


