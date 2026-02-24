# -*- coding: utf-8 -*-
"""
단일 CausalLM(taetae030/fin-term-model)로
객관식(숫자 1~4) + 주관식(짧은 서술문) 동시 학습 (QLoRA)
"""
import os, math, random, numpy as np, torch
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          EarlyStoppingCallback, BitsAndBytesConfig, set_seed)
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from peft import LoraConfig, get_peft_model
from datetime import datetime
from zoneinfo import ZoneInfo

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TRAIN = DATA_DIR / "train.jsonl"
VAL   = DATA_DIR / "val.jsonl"

BASE = "taetae030/fin-term-model"   # 공개 허깅페이스 모델

# 1) 토크나이저
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 2) 4-bit 로드 (QLoRA)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.pad_token_id = tok.pad_token_id
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# 3) LoRA
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    task_type="CAUSAL_LM", bias="none",
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# 4) 데이터 로드 (messages 스키마, 마지막 assistant 필수)
raw = load_dataset("json", data_files={"train": str(TRAIN), "eval": str(VAL)})

def build_supervised(ex):
    msgs = ex["messages"]
    assert isinstance(msgs, list) and msgs[-1]["role"] == "assistant"

    # assistant 직전까지 프롬프트
    prompt_ids = tok.apply_chat_template(
        msgs[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(prompt_ids, dict):
        p = prompt_ids["input_ids"]
    else:
        p = prompt_ids
    if p.dim() == 2:
        p_ids = p.squeeze(0).tolist()
    else:
        p_ids = p.tolist()

    # 정답
    answer = msgs[-1]["content"] + tok.eos_token
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]

    input_ids = p_ids + a_ids
    labels    = [-100]*len(p_ids) + a_ids
    attn_mask = [1]*len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

ds = raw.map(build_supervised, remove_columns=raw["train"].column_names)

# 5) 길이 패킹 (라벨 보존)
MAX_LEN = 2048
def pack_with_labels(block_size):
    def _fn(examples):
        stream_ids, stream_labels = [], []
        for ids, labs in zip(examples["input_ids"], examples["labels"]):
            if ids and labs and labs[-1] != tok.eos_token_id:
                ids  = ids  + [tok.eos_token_id]
                labs = labs + [tok.eos_token_id]
            stream_ids.extend(ids)
            stream_labels.extend(labs)
        chunks_ids   = [stream_ids[i:i+block_size]    for i in range(0, len(stream_ids),    block_size)]
        chunks_label = [stream_labels[i:i+block_size] for i in range(0, len(stream_labels), block_size)]
        if chunks_ids and len(chunks_ids[-1]) < int(block_size*0.5):
            chunks_ids.pop(); chunks_label.pop()
        attn = [[1]*len(c) for c in chunks_ids]
        return {"input_ids": chunks_ids, "attention_mask": attn, "labels": chunks_label}
    return _fn

tok_ds = ds.map(pack_with_labels(MAX_LEN), batched=True, batch_size=800)
train_ds, eval_ds = tok_ds["train"], tok_ds["eval"]

# 6) Collator
@dataclass
class DataCollatorCausalWithLabels:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ids  = [torch.tensor(f["input_ids"]) for f in features]
        attn = [torch.tensor(f["attention_mask"]) for f in features]
        labs = [torch.tensor(f["labels"]) for f in features]
        batch = self.tokenizer.pad({"input_ids": ids, "attention_mask": attn},
                                   padding=True, return_tensors="pt",
                                   pad_to_multiple_of=self.pad_to_multiple_of)
        maxlen = batch["input_ids"].size(1)
        lab_pad = torch.full((len(labs), maxlen), -100, dtype=torch.long)
        for i, l in enumerate(labs):
            lab_pad[i, :l.size(0)] = l
        batch["labels"] = lab_pad
        return batch

collator = DataCollatorCausalWithLabels(tok)

# 7) 학습 설정
RUN_ID = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
OUT_DIR = ROOT / "out-causallm-qlora" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_DEVICE_BS = 4
GAS = 32
GLOBAL_BS = PER_DEVICE_BS * GAS
steps_per_epoch = math.ceil(len(train_ds) / GLOBAL_BS)
eval_steps = max(50, min(200, steps_per_epoch))

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=10,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    learning_rate=2e-4,

    seed=SEED, data_seed=SEED,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    per_device_train_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GAS,
    per_device_eval_batch_size=4,

    eval_strategy="steps", eval_steps=eval_steps,
    save_strategy="steps", save_steps=eval_steps,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    bf16=True, tf32=True,
    optim="paged_adamw_8bit",
    report_to="none",

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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.003)],
)

trainer.train()

# 8) 저장
trainer.save_model(str(OUT_DIR / "adapter"))
tok.save_pretrained(str(OUT_DIR / "tokenizer"))
print(f"[SAVED] {OUT_DIR}")

# 9) 평가
metrics = trainer.evaluate(eval_dataset=eval_ds)
print(metrics)