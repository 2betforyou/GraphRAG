# -*- coding: utf-8 -*-
import json, re, os, random, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    EarlyStoppingCallback, BitsAndBytesConfig, set_seed
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import Any

# ===================== 기본 설정 =====================
SEED = 42
set_seed(SEED)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

base = "taetae030/fin-term-model"

# ===================== 토크나이저 =====================
tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ===================== 4-bit 로드 =====================
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.pad_token_id = tok.pad_token_id
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# ===================== QLoRA =====================
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # 모델마다 모듈명이 다를 수 있으므로 넓게 커버
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# ===================== 데이터 로드 =====================
# 주: messages 스키마이며, 마지막이 assistant여야 함
raw = load_dataset("json", data_files={
    "train":"../dataset/all/train_plus_TTA.jsonl",
    "eval":"../dataset/all/val_plus_TTA.jsonl"
})

def build_supervised(ex):
    msgs = ex["messages"]
    assert isinstance(msgs, list) and msgs[-1]["role"] == "assistant"

    # 1) 프롬프트(assistant 직전까지) → 토크나이즈
    prompt_out = tok.apply_chat_template(
        msgs[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    p_ids = prompt_out["input_ids"].squeeze(0).tolist() if isinstance(prompt_out, dict) else \
            (prompt_out.squeeze(0).tolist() if prompt_out.dim()==2 else prompt_out.tolist())

    # 2) 정답(assistant) → 토크나이즈 (EOS 포함)
    answer = msgs[-1]["content"] + tok.eos_token
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]

    # 3) 레이블 마스킹(프롬프트는 -100)
    input_ids = p_ids + a_ids
    labels    = [-100] * len(p_ids) + a_ids
    attn_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

ds = raw.map(build_supervised, remove_columns=raw["train"].column_names)

# ===================== Constant-length Packing (라벨 보존) =====================
MAX_LEN = 2048

def pack_with_labels(block_size):
    """
    - examples["input_ids"]와 examples["labels"]를 '같이' 이어붙여 고정 길이로 슬라이싱
    - 프롬프트 마스크(-100) 보존
    - 경계 EOS는 이미 정답 끝에 있으므로 중복 삽입 금지
    """
    def _fn(examples):
        stream_ids, stream_labels = [], []
        for ids, labs in zip(examples["input_ids"], examples["labels"]):
            # 샘플 경계 EOS 점검(이미 정답 끝에 EOS가 있으므로 보통 불필요)
            if ids and labs and labs[-1] != tok.eos_token_id:
                ids  = ids  + [tok.eos_token_id]
                labs = labs + [tok.eos_token_id]
            stream_ids.extend(ids)
            stream_labels.extend(labs)

        # block 단위 슬라이스(라벨 동일 오프셋 유지)
        chunks_ids   = [stream_ids[i:i+block_size]   for i in range(0, len(stream_ids),   block_size)]
        chunks_label = [stream_labels[i:i+block_size]for i in range(0, len(stream_labels),block_size)]

        # 너무 짧은 꼬리 chunk 제거(패딩 낭비 방지)
        if chunks_ids and len(chunks_ids[-1]) < int(block_size * 0.5):
            chunks_ids.pop(); chunks_label.pop()

        attn = [[1]*len(c) for c in chunks_ids]
        return {"input_ids": chunks_ids, "attention_mask": attn, "labels": chunks_label}
    return _fn

tok_ds = ds.map(pack_with_labels(MAX_LEN), batched=True, batch_size=1000)
train_ds, eval_ds = tok_ds["train"], tok_ds["eval"]

# ===================== Collator =====================
@dataclass
class DataCollatorCausalWithLabels:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ids  = [torch.tensor(f["input_ids"]) for f in features]
        attn = [torch.tensor(f["attention_mask"]) for f in features]
        labs = [torch.tensor(f["labels"]) for f in features]
        batch = self.tokenizer.pad(
            {"input_ids": ids, "attention_mask": attn},
            padding=True, return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        maxlen = batch["input_ids"].size(1)
        lab_pad = torch.full((len(labs), maxlen), -100, dtype=torch.long)
        for i, l in enumerate(labs):
            lab_pad[i, :l.size(0)] = l
        batch["labels"] = lab_pad
        return batch

collator = DataCollatorCausalWithLabels(tok)

# ===================== 런/경로 =====================
RUN_ID = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"out-exaone-law-qlora/{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== 학습 하이퍼파라미터 =====================
PER_DEVICE_BS = 4
GAS = 32
GLOBAL_BS = PER_DEVICE_BS * GAS
STEPS_PER_EPOCH = math.ceil(len(train_ds) / GLOBAL_BS)
# 소형/패킹일수록 steps 기반 평가가 유리
EVAL_STEPS = max(50, min(200, STEPS_PER_EPOCH))  # 데이터 크기에 맞춰 자동 조정

args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=12,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    learning_rate=2e-4,

    seed=SEED, data_seed=SEED,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GAS,
    per_device_eval_batch_size=4,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    bf16=True,
    tf32=True,
    optim="paged_adamw_8bit",
    report_to="none",

    # 안정화
    label_smoothing_factor=0.0,
    weight_decay=0.01,
    max_grad_norm=1.0,

    remove_unused_columns=False, 
    logging_steps=20,
    logging_first_step=True,
    prediction_loss_only=True,
    eval_accumulation_steps=8,
    save_safetensors=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.005, 
    )],
)

trainer.train()

# ===================== 저장 =====================
trainer.save_model(f"{OUT_DIR}/adapter")
tok.save_pretrained(f"{OUT_DIR}/tokenizer")
print(f"Saved!{OUT_DIR}")

# ===================== 평가 =====================
metrics = trainer.evaluate(eval_dataset=eval_ds)
print(metrics)

# ===================== Tail fine-tuning (옵션) =====================
"""
추가 1~2 epoch 저LR로 이어 달리기.
주의: 이어 달리기는 '최선 체크포인트'에서 시작하는 게 일반적으로 안전함.
"""
TAIL_EPOCHS = 2
NEW_LR = 1e-4

new_num_epochs = trainer.state.epoch + TAIL_EPOCHS
trainer.args.num_train_epochs = new_num_epochs

# 옵티마이저 LR만 하향
if trainer.optimizer is None:
    trainer.create_optimizer()
for g in trainer.optimizer.param_groups:
    g["lr"] = NEW_LR

# 새 총 스텝으로 스케줄러 갱신
base_steps = STEPS_PER_EPOCH
new_total_steps = int(base_steps * new_num_epochs)
trainer.create_scheduler(num_training_steps=new_total_steps)

# 이어서 학습
trainer.train(resume_from_checkpoint=True)
m_tail = trainer.evaluate(eval_dataset=eval_ds)
print({"tail_eval_loss": m_tail.get("eval_loss")})
trainer.save_model(f"out-exaone-law-qlora-tail/{RUN_ID}/adapter")
tok.save_pretrained(f"out-exaone-law-qlora-tail/{RUN_ID}/tokenizer")


