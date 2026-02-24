# RAG (BM25) + TTA-only Training Pipeline

## Layout
```
data/
  cleaned/
    tta_terms.jsonl           # (already parsed) from TTA_cut.txt
    *.cleaned_lines.txt       # law corpus (for RAG only)
scripts/
  make_tta_sft.py
  train_tta_qlora.py
  build_bm25.py
  retriever_bm25.py
  infer_bm25.py
  load.py                     # (provided by you)
out-tta-sft-qlora/            # produced by training
data/index/
  bm25_laws.npz               # produced by build_bm25.py
  bm25_laws_meta.jsonl
```

## 1) Create SFT data from TTA terms
```bash
python scripts/make_tta_sft.py
# writes data/cleaned/tta_sft.jsonl
```

## 2) Train QLoRA on TTA-only
```bash
python scripts/train_tta_qlora.py \
  --base taetae030/fin-term-model \
  --data data/cleaned/tta_sft.jsonl \
  --out out-tta-sft-qlora
```

## 3) Build BM25 index for law RAG
```bash
pip install rank-bm25
python scripts/build_bm25.py
# outputs data/index/bm25_laws.npz and bm25_laws_meta.jsonl
```

## 4) Inference with optional RAG (BM25) + your load.make_prompt_auto
```bash
# If you have a test.csv with column "Question"
python scripts/infer_bm25.py \
  --base taetae030/fin-term-model \
  --adapter out-tta-sft-qlora/adapter \
  --rag --topk 5 --max_new 128 --test_csv data/test.csv

# Or run interactive shell without test.csv
python scripts/infer_bm25.py \
  --base taetae030/fin-term-model \
  --adapter out-tta-sft-qlora/adapter \
  --rag --topk 5 --max_new 128
```

### Notes
- **Training uses TTA only.** No law texts are used for SFT.  
- **RAG uses only law corpus** at inference (BM25).  
- For 객관식, generation first token is constrained to digits 1–9.  
- If memory is tight, reduce `--seq_len` or increase `--grad_accum`.

Happy hacking!

---
# doc > docx 
libreoffice --headless --convert-to docx s.docx

# 0) (선택) 새 가상환경/캐시 정리
# rm -rf out-tta-sft-qlora  # 타임스탬프 저장 쓰면 생략 가능

# 1) TTA 용어/정의 → SFT 기본셋
python scripts/make_tta_sft.py \
  # in: data/cleaned/tta_terms.jsonl
  # out: data/cleaned/tta_sft.jsonl

# 2) TTA SFT 증강 (정의형 + 객관식 자동 생성)
# 4지선다
python scripts/augment_tta_sft.py \
  --src data/cleaned/tta_terms.jsonl \
  --dst_aug data/cleaned/tta_sft_aug_4.jsonl \
  --mcq_per_item 2 --options 4

# 5지선다
python scripts/augment_tta_sft.py \
  --src data/cleaned/tta_terms.jsonl \
  --dst_aug data/cleaned/tta_sft_aug_5.jsonl \
  --mcq_per_item 2 --options 5

# 6지선다
python scripts/augment_tta_sft.py \
  --src data/cleaned/tta_terms.jsonl \
  --dst_aug data/cleaned/tta_sft_aug_6.jsonl \
  --mcq_per_item 2 --options 6

# 합치기
cat data/cleaned/tta_sft.jsonl \
    data/cleaned/tta_sft_aug_4.jsonl \
    data/cleaned/tta_sft_aug_5.jsonl \
    data/cleaned/tta_sft_aug_6.jsonl \
    > data/cleaned/tta_sft_plus.jsonl

# or 
python scripts/augment_tta_sft.py \
  --src data/cleaned/tta_terms.jsonl \
  --dst_aug data/cleaned/tta_sft_aug.jsonl \
  --dst_plus data/cleaned/tta_sft_plus.jsonl \
  --mcq_per_item 2 --options 4 5 6

# or 
python scripts/augment_tta_sft.py \
  --src data/cleaned/tta_terms.jsonl \
  --dst_aug data/cleaned/tta_sft_aug.jsonl \
  --dst_plus data/cleaned/tta_sft_plus.jsonl \
  --mcq_per_item 2 \
  --options 4 5 \
  --add_neg \
  --seed 42

# 3) MITRE ATT&CK 데이터 추가, 한글로 번역
python scripts/build_mitre.py

python scripts/build_mitre_ko.py

<!-- # 4) 병합 (정의/단문 중심의 MITRE만 포함 권장)
cat data/cleaned/tta_sft_plus.jsonl data/cleaned/mitre_sft_ko.jsonl \
  > data/cleaned/tta_mitre_sft_plus_ko.jsonl -->

# 5) 학습 (처음부터)
python scripts/train_tta_qlora.py \
  --base LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --data data/cleaned/tta_sft_plus.jsonl \
  --out out-tta-sft-qlora \
  --epochs 2 --seq_len 1536 --lr 8e-5 --grad_accum 16

python scripts/train_tta_qlora.py \
  --base LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --data data/cleaned/tta_mitre_sft_plus_ko.jsonl \
  --out out-tta-sft-qlora \
  --epochs 2 --seq_len 1536 --lr 8e-5 --grad_accum 16

# 6) RAG 인덱스 재생성
cat data/cleaned/bm25_laws.jsonl data/cleaned/mitre_bm25_ko.jsonl \
  > data/cleaned/bm25_all_ko.jsonl
or 
cat data/cleaned/dapt_laws.jsonl data/cleaned/mitre_bm25_ko.jsonl \
  > data/cleaned/bm25_all_ko.jsonl

cat data/cleaned/dapt_laws.jsonl data/cleaned/mitre_bm25_ko_cleaned.jsonl \
  > data/cleaned/bm25_all_ko.jsonl

python scripts/build_bm25_all.py

# 7) 추론
python scripts/infer_bm25.py \
  --base LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --adapter out-tta-sft-qlora/20250827_003755/adapter \
  --rag --topk 7 --max_new 2048 --test_csv data/test.csv

python scripts/infer_bm25.py \
  --base mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter out-tta-sft-qlora/<타임스탬프>/adapter \
  --rag --topk 7 --max_new 2048 --test_csv data/test.csv


python scripts/infer_bm25.py \
  --base taetae030/fin-term-model \
  --adapter out-tta-sft-qlora/20250826_033325/adapter \
  --rag --topk 5 --max_new 1024 \
  --mcq_explain --mcq_explain_tokens 512 \
  --test_csv data/test.csv

python scripts/infer_bm25.py \
  --base LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --adapter out-tta-sft-qlora/20250827_011246/adapter \
  --rag --topk 7 --max_new 1024 \
  --mcq_explain --mcq_explain_tokens 512 \
  --test_csv data/test.csv
