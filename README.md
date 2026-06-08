# GraphRAG 기반 금융보안 규제 해석 프레임워크

## 한국전자거래학회 2025 추계학술대회

주성용 · 박준영 · 오병훈 · 이재우

금융보안 법령 해석을 위한 **Graph-augmented Hybrid RAG** 프레임워크입니다. 이 프로젝트는 법령, 시행령, 지침, MITRE ATT&CK 자료를 EXAONE-3.5-7.8B-Instruct 기반 질의응답 파이프라인에 연결하고, 법령 조항 간 참조 그래프와 dense/sparse retrieval을 결합해 규제 질의의 근거 문맥 검색 품질을 높이는 것을 목표로 합니다.

단순히 문서를 벡터 DB에 넣고 검색하는 방식이 아니라, 한국 법령 문서의 특징인 **조항 간 참조**, **교차 법령 참조**, **시행령/지침 연결**, **법령명 기반 라우팅**을 retrieval 단계에 반영합니다.

## 핵심 아이디어

금융보안 규제 질의는 특정 조항 하나만으로 답하기 어려운 경우가 많습니다. 예를 들어 "개인정보 처리 제한", "전자금융거래 사고 책임", "정보통신망 보호조치" 같은 질문은 원 조항, 예외 조항, 시행령, 관련 지침이 함께 검색되어야 합니다.

이 프로젝트는 다음 방식으로 RAG의 검색 병목을 줄입니다.

| 문제 | 접근 방식 |
| --- | --- |
| 벡터 검색만으로 정확한 조항을 놓침 | E5 dense retrieval과 BM25 sparse retrieval 결합 |
| 법령 문맥이 조항 단위로 분산됨 | 법령 조항을 노드로 만들고 참조 관계를 엣지로 구성 |
| "제n조" 참조가 여러 법령에 중복됨 | 법령명 + 조항 패턴을 우선 연결하고, 단순 조항 참조는 후보 수에 따라 fallback |
| 중복 문맥이 많이 들어감 | optional reranker와 MMR로 재정렬 및 다양화 |
| 관련성이 낮은 검색 결과가 주입됨 | low-confidence gating으로 RAG 적용 여부 제어 |

## Research Framing

### 1. Goal: what problem are we tackling?

금융보안 규제 질의응답에서는 모델이 최신 법령, 시행령, 고시, 지침, MITRE 자료를 정확히 참조해야 합니다. 하지만 일반 LLM은 학습 시점 이후의 규제 변화나 세부 조항 관계를 안정적으로 기억하지 못하고, naive RAG는 의미적으로 비슷한 chunk를 검색하더라도 실제 정답에 필요한 **관련 조항**, **예외 조항**, **준용 조항**, **시행령 연결**을 놓칠 수 있습니다.

따라서 이 프로젝트의 목표는 다음과 같습니다.

> 금융보안 법령 문서의 구조적 참조 관계를 활용해, 규제 질의응답에서 필요한 근거 문맥을 더 정확하게 검색하고 생성 답변의 근거성을 높이는 RAG 시스템을 구축한다.

### 2. Existing works: how do recent prior works tackle this problem?

최근 RAG 연구는 단순 dense retrieval을 넘어 문서 간 관계, 그래프 구조, retrieval 품질 평가를 함께 다루는 방향으로 발전하고 있습니다.

| Prior work / trend | Main idea | Limitation for Korean financial regulation QA |
| --- | --- | --- |
| Naive RAG | 문서를 chunking한 뒤 vector search로 관련 문맥 검색 | 조항 간 참조, 시행령 연결, 예외 조항처럼 구조적 관계를 놓치기 쉬움 |
| Hybrid RAG | dense retrieval과 BM25를 결합해 semantic match와 lexical match를 함께 활용 | 검색 recall은 개선되지만 법령 간 참조 구조 자체를 모델링하지는 않음 |
| Microsoft GraphRAG | entity graph와 community summary를 만들어 corpus-level global question answering에 활용 ([arXiv:2404.16130](https://arxiv.org/abs/2404.16130)) | global summarization에는 강하지만, 한국 법령의 조항 번호/교차 법령 참조를 직접 해석하는 도메인 규칙은 별도로 필요 |
| LightRAG | graph structure를 indexing/retrieval에 통합하고 dual-level retrieval로 효율과 문맥성을 개선 ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779)) | 범용 graph retrieval 구조이며, 법령명 + 제n조 같은 규제 문서 고유 패턴은 도메인 특화 처리가 필요 |
| RAG evaluation frameworks | context precision, context recall, faithfulness, answer relevancy처럼 RAG를 retrieval/generation 단위로 분해 평가 ([arXiv:2309.15217](https://arxiv.org/abs/2309.15217)) | 평가 프레임은 제공하지만, 법령 참조 그래프를 어떻게 retrieval에 결합할지는 시스템 설계 문제로 남음 |

### 3. Main challenge: what do recent works fail to solve?

최근 GraphRAG와 Hybrid RAG는 문서 관계를 검색에 활용한다는 큰 방향을 제시하지만, 금융보안 법령 질의에는 다음과 같은 도메인 특화 문제가 남아 있습니다.

1. **법령 조항 참조의 모호성**
   - `"제21조"` 같은 표현은 여러 법령에 동시에 존재합니다.
   - 법령명을 함께 보지 않으면 잘못된 조항으로 연결될 수 있습니다.

2. **교차 법령 참조**
   - 한 법령의 조항이 다른 법령, 시행령, 시행규칙, 고시를 참조하는 경우가 많습니다.
   - flat chunk retrieval은 이런 연결을 검색 시점에 안정적으로 따라가기 어렵습니다.

3. **정답 근거의 분산**
   - 하나의 질문에 답하기 위해 원칙 조항, 예외 조항, 벌칙 조항, 정의 조항이 함께 필요할 수 있습니다.
   - top-k vector search만으로는 관련 조항 묶음이 빠질 수 있습니다.

4. **한국어 법령 표현의 lexical dependency**
   - 법령명, 조항 번호, 약칭, 고시명 등은 semantic similarity보다 정확한 문자열 매칭이 중요한 경우가 많습니다.
   - dense retrieval만으로는 조항 번호 기반 질의에서 recall이 흔들릴 수 있습니다.

### 4. Our method: how does it solve the challenge?

본 프로젝트는 범용 GraphRAG를 그대로 재현하기보다, 한국 금융보안 법령 문서의 구조를 직접 활용하는 **domain-specific lightweight GraphRAG**를 제안합니다.

핵심 방법은 다음과 같습니다.

| Challenge | Our method |
| --- | --- |
| 조항 참조 모호성 | `(law, article)` 역색인을 만들고 `"법령명 + 제n조"` 패턴을 우선 연결 |
| 교차 법령 참조 | `graph_build_crosslaw.py`에서 법령 간 참조 edge 생성 |
| 분산된 근거 조항 | FAISS 검색으로 seed 조항을 찾은 뒤 graph hop으로 인접 조항 확장 |
| lexical dependency | BM25와 dense retrieval을 결합해 조항 번호/법령명 match 보강 |
| irrelevant context | reranker, MMR, low-confidence gating으로 context 품질 제어 |

즉, 이 방법은 검색을 단순한 chunk ranking 문제가 아니라 **법령 구조를 따라 관련 근거를 확장하는 문제**로 재정의합니다.

### 5. Experimental results: how do we show that the method solves the challenge?

현재 코드는 answer-only PPL 평가와 RAG 실행 로그를 제공하며, 최종적으로는 다음 실험으로 효과를 검증할 수 있습니다.

| Experiment | Comparison | Metric |
| --- | --- | --- |
| Retrieval ablation | dense only vs BM25 only vs hybrid vs hybrid + graph | Recall@k, MRR, nDCG |
| Graph contribution | hybrid RAG vs hybrid RAG + graph hop | gold article hit rate, cross-law reference hit rate |
| Context quality | retrieved context before/after graph expansion | Context Precision, Context Recall |
| Generation quality | no RAG vs FAISS RAG vs GraphRAG | accuracy, faithfulness, answer relevancy |
| Robustness | law-name queries, article-number queries, cross-law queries | per-type accuracy and retrieval recall |
| Efficiency | FAISS RAG vs GraphRAG | latency, context length, tokens/sec |

권장 evaluation set은 다음 query type을 포함해야 합니다.

- 특정 법령명과 조항 번호가 명시된 질문
- 조항 번호만 주어진 질문
- 시행령/시행규칙까지 함께 봐야 하는 질문
- 정의 조항과 벌칙 조항이 함께 필요한 질문
- MITRE ATT&CK와 법령 근거를 함께 요구하는 질문

성공 기준은 GraphRAG가 단순 FAISS RAG 대비 특히 **cross-law reference hit rate**, **gold article Recall@k**, **faithfulness**에서 개선되는지 확인하는 것입니다.

## 주요 기능

- **Domain-specific RAG**
  - 금융보안 관련 법령, 시행령, 지침, MITRE 자료를 로컬 텍스트/JSONL 기반으로 색인
  - EXAONE-3.5-7.8B-Instruct 및 LoRA adapter 기반 추론 지원

- **Hybrid Search**
  - `intfloat/multilingual-e5-base` 임베딩 기반 FAISS dense retrieval
  - 선택적 BM25 sparse retrieval
  - `rag_alpha`로 dense/BM25 결합 비율 조정

- **Lightweight GraphRAG**
  - 법령 조항을 그래프 노드로 구성
  - 조항 참조, 교차 법령 참조, 역색인을 활용해 엣지 생성
  - FAISS로 seed 조항을 찾은 뒤 graph hop으로 관련 조항 확장

- **Retrieval Control**
  - `rag_mode=faiss|graph`로 검색 모드 전환
  - `rag_topk`, `rag_max_chars`, `rag_injection`으로 검색량과 context injection 제어
  - `rag_log_top1`로 검색 상위 결과 디버깅
  - 객관식 문제는 질문과 보기를 함께 사용해 검색 질의 강화

## 아키텍처

```text
Question
  -> law/MITRE trigger and routing
  -> E5 dense retrieval + optional BM25 retrieval
  -> score fusion
  -> optional reranker / MMR
  -> optional graph hop expansion
  -> context injection into system or user message
  -> EXAONE-3.5-7.8B-Instruct + LoRA generation
  -> answer post-processing
```

## Repository Structure

```text
.
├── README.md
├── dataset/
├── fin/
└── pjy/
    ├── build_index.py              # 텍스트/JSONL 기반 FAISS/BM25 인덱스 생성
    ├── rag_searcher.py             # hybrid retrieval, routing, reranker, MMR, gating
    ├── inference_graphrag.py       # FAISS/GraphRAG 추론 파이프라인
    ├── evals.py                    # answer-only PPL 평가
    └── graphrag/
        ├── graph_build_crosslaw.py # 교차 법령 참조 그래프 생성
        ├── build_index_embed.py    # 그래프 노드 임베딩 및 FAISS 인덱스 생성
        └── query_graphrag_embed.py # 그래프 기반 검색 테스트
```

## Requirements

```bash
pip install "transformers>=4.41.0" \
  "accelerate>=0.31.0" \
  "sentence-transformers>=3.0.1" \
  "faiss-cpu>=1.8.0" \
  rank_bm25 \
  "bitsandbytes>=0.43.1" \
  "peft>=0.11.1"
```

CUDA 환경에서는 환경에 맞는 `faiss-gpu` wheel 사용을 권장합니다.

## 1. Hybrid RAG Index Build

로컬 텍스트/JSONL 자료를 chunking한 뒤 E5 임베딩, FAISS 인덱스, 선택적 BM25 인덱스를 생성합니다.

```bash
cd pjy

python build_index.py \
  --sources "/mnt/data/*.txt" "/mnt/data/*.jsonl" \
  --index-dir "./data/index/law_rag" \
  --embedding-model "intfloat/multilingual-e5-base" \
  --chunk-chars 900 \
  --chunk-overlap 200 \
  --bm25
```

## 2. Basic RAG QA

```bash
cd pjy

python ask_rag.py \
  --index-dir "./data/index/law_rag" \
  --base-model "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" \
  --topk 6 \
  --alpha 0.2 \
  --max-new-tokens 320 \
  --temperature 0.2
```

`alpha`는 BM25와 벡터 스코어 결합 가중치입니다. `0`은 벡터 검색만, `1`은 BM25 검색만 사용합니다.

## 3. Law Graph Build

법령 구조 JSON에서 조항 단위 노드를 만들고, 조항 참조와 교차 법령 참조를 엣지로 연결합니다.

```bash
cd pjy/graphrag

python graph_build_crosslaw.py \
  --json_dir ../../dataset/laws/parsed_laws \
  --out_json ../../dataset/laws/graphrag/law_graph.json \
  --out_html ../../dataset/laws/graphrag/law_graph.html
```

`graph_build_crosslaw.py`는 다음 전략을 사용합니다.

- `"법령명 + 제n조"` 패턴은 명시적 교차 법령 참조로 우선 연결
- `"이 법"`, `"같은 법"`은 현재 법령으로 해석
- `"제n조"`만 있는 경우 전체 법령에서 후보가 하나면 연결
- 후보가 여러 개면 현재 법령의 해당 조항으로 fallback
- 중복 조항은 병합하고 `(law, article)` 역색인으로 탐색 효율 개선

## 4. GraphRAG Index Build

그래프 노드 텍스트를 임베딩하고 FAISS 인덱스를 생성합니다.

```bash
cd pjy/graphrag

python build_index_embed.py \
  --graph_json ../../dataset/laws/graphrag/law_graph.json \
  --faiss_index ../../dataset/laws/graphrag/law_graph.index \
  --emb_path ../../dataset/laws/graphrag/embeddings.npy \
  --model_name intfloat/multilingual-e5-base
```

## 5. Inference with FAISS or GraphRAG

`inference_graphrag.py`는 FAISS 기반 RAG와 그래프 기반 RAG를 모두 지원합니다.

```bash
cd pjy

python inference_graphrag.py \
  --adapter out-exaone-law-qlora/20250828_040425/adapter \
  --rag on \
  --rag_mode graph \
  --index_dir ../dataset/laws/graphrag \
  --graph_json_path ../dataset/laws/graphrag/law_graph.json \
  --rag_topk 3 \
  --rag_max_chars 600 \
  --rag_injection user \
  --rag_log_top1 \
  --bf16
```

주요 옵션:

| 옵션 | 설명 |
| --- | --- |
| `--rag_mode faiss` | E5 + FAISS + optional BM25 기반 검색 |
| `--rag_mode graph` | 그래프 seed 검색 후 인접 조항 확장 |
| `--rag_alpha` | dense/BM25 결합 비율 |
| `--rag_topk` | 검색 상위 문맥 수 |
| `--rag_max_chars` | 문맥당 최대 문자 수 |
| `--rag_injection system|user` | RAG 문맥 주입 위치 |
| `--rag_log_top1` | 검색 상위 1개 결과 로그 출력 |

## Evaluation

현재 `pjy/evals.py`는 answer-only perplexity(PPL) 중심 평가를 제공합니다.

```bash
cd pjy

python evals.py \
  --base LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --adapter out-exaone-law-qlora/20250816_014322/adapter \
  --val_jsonl ../dataset/all/val.jsonl \
  --per_device_eval_batch_size 2
```

향후 RAG 품질 평가는 다음 지표로 확장할 수 있습니다.

| 평가 축 | 권장 지표 |
| --- | --- |
| Retrieval quality | Recall@k, MRR, nDCG |
| Context quality | Context Precision, Context Recall |
| Generation quality | Answer Relevancy, Faithfulness |
| System efficiency | latency, tokens/sec, context length |
| Error analysis | retrieval miss, wrong graph expansion, irrelevant context, hallucination |

## Positioning

이 레포는 Microsoft GraphRAG의 full pipeline, 즉 entity extraction, community detection, community summary 기반 global QA 시스템을 그대로 재현하는 프로젝트는 아닙니다. 대신 한국 법령 도메인의 구조적 특징인 **조항 참조 관계**를 활용하는 **domain-specific lightweight GraphRAG**에 가깝습니다.

따라서 이 프로젝트의 핵심 가치는 "GraphRAG를 사용했다"보다 다음 문장에 더 가깝습니다.

> 법령 조항 간 참조 그래프와 hybrid retrieval을 결합해 금융보안 규제 질의응답의 검색 병목을 줄이는 RAG 시스템

## Current Limitations and Next Steps

- 정량 RAG 평가는 아직 answer-only PPL 중심이며, retrieval metric과 faithfulness metric 보강이 필요합니다.
- 질의별 retriever 선택, top-k 점수, graph hop, context length, latency, generation token 수를 저장하는 구조화 tracing이 아직 부족합니다.
- LLM이 검색 전략을 동적으로 계획하고 수정하는 agentic RAG 구조는 아직 포함되어 있지 않습니다.
- 향후 ablation study로 `dense only`, `BM25 only`, `hybrid`, `hybrid + graph`, `hybrid + graph + rerank/MMR`를 비교하면 프로젝트의 기여가 더 명확해집니다.

## Keywords

`GraphRAG` · `Hybrid Search` · `Dense Retrieval` · `BM25` · `FAISS` · `Korean Legal RAG` · `Financial Security Regulation` · `EXAONE` · `LoRA` · `MITRE ATT&CK`
