# RAG for EXAONE-3.5-7.8B (Korean Laws/Guidelines)

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