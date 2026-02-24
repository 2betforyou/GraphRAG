0. 가상환경
> pip install -r requirements.txt

1. 데이터 준비
> python scripts/build_dataset_from_txt.py \
  --laws_txt data/raw/. \
  --tta_txt  data/raw/TTA_cut.txt \
  --out_train data/processed/train_plus_TTA.jsonl \
  --out_val   data/processed/val_plus_TTA.jsonl

2. 학습(QLoRA)
> python scripts/train_causallm.py \
  --base taetae030/fin-term-model \
  --train_jsonl data/processed/train.jsonl \
  --val_jsonl   data/processed/val.jsonl \
  --epochs 12 --lr 2e-4 --max_len 4096 \
  --per_device_bs 4 --grad_accum 32

3. 추론
> python scripts/infer.py \
  --test_csv data/test.csv \
  --sample_csv data/sample_submission.csv \
  --out_csv submission.csv \
  --print_each
  --use_rag


python inference5.py \
  --adapter out-causallm-qlora/20250825_200843/adapter \
  --rag_on \
  --rag_index_dir data/index/laws_e5 \
  --rag_query_encoder intfloat/multilingual-e5-base \
  --rag_topk 4 \
  --rag_max_ctx_tokens 800




