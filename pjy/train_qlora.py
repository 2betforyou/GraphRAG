from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch, math
import random, numpy as np, torch
from transformers import set_seed
from datetime import datetime
from zoneinfo import ZoneInfo
import os 

SEED = 42
set_seed(SEED)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


base = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"


# 1) 토크나이저
tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# 2) 4-bit 로드
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
model.config.use_cache = False                           # 학습 시 권장
model.gradient_checkpointing_enable(                     # v4.53 권장 방식
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# QLoRA 준비
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)
model.enable_input_require_grads()


from datasets import load_dataset, DatasetDict

# 단일 jsonl 로드
raw_all = load_dataset(
    "json",
    data_files={
        "data": "../dataset/all/tta_mitre_sft_plus_ko1.jsonl"
    }
)["data"]

# 90/10 split
split = raw_all.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split["train"], split["test"]

# ✅ dict가 아니라 DatasetDict 로!
raw = DatasetDict({"train": train_ds, "eval": val_ds})

print(train_ds, val_ds)

def build_supervised(ex):
    # 데이터에 messages가 없으면 instruction/input/output로 messages를 만들어줌
    if "messages" in ex and ex["messages"]:
        msgs = ex["messages"]
    else:
        user = ex.get("instruction", "") or ""
        if ex.get("input"):
            user += "\n" + (ex.get("input") or "")
        system = "당신은 금융보안 전문가입니다. 모든 답변은 한국어로 간결하고 정확하게 작성하세요."
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": ex.get("output","")},
        ]

    # assistant 답변만 loss에 반영
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    answer = (msgs[-1]["content"] or "") + tok.eos_token

    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]

    return {
        "input_ids": p_ids + a_ids,
        "labels":    [-100]*len(p_ids) + a_ids,
        "attention_mask": [1]*(len(p_ids)+len(a_ids)),
    }

ds = raw.map(build_supervised, remove_columns=raw["train"].column_names)

# (선택) 너무 긴 샘플을 잘라 OOM 예방
# MAX_LEN = 4096
# def truncate(ex):
#     if len(ex["input_ids"]) > MAX_LEN:
#         ex["input_ids"]      = ex["input_ids"][:MAX_LEN]
#         ex["attention_mask"] = ex["attention_mask"][:MAX_LEN]
#         ex["labels"]         = ex["labels"][:MAX_LEN]
#     return ex
# ds = ds.map(truncate)

train_ds, eval_ds = ds["train"], ds["eval"]


# # 4) 토크나이즈 (주의: add_special_tokens=False, 마스크는 만들지 않음)
# MAX_LEN = 4096  # 24GB 안전권은 4096~6144부터 권장, 여유되면 8192 시도
# def tokenize(batch):
#     out = tok(batch["text"], add_special_tokens=False)     # padding/truncation 없음
#     return {"input_ids": out["input_ids"]}                 # attention_mask는 여기서 만들지 않음

# tok_ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

# # 패킹: input_ids를 이어붙여 block_size로 자르고,
# # 같은 길이의 attention_mask(전부 1)를 함께 생성
# def pack(block_size):
#     def _fn(examples):
#         stream = []
#         for ids in examples["input_ids"]:
#             # 각 샘플 경계에 EOS 삽입(문맥 분리)
#             stream.extend(ids + [tok.eos_token_id])

#         chunks = [stream[i:i+block_size] for i in range(0, len(stream), block_size)]
#         # 마지막 청크가 너무 짧으면 버려 패딩 낭비 방지
#         if chunks and len(chunks[-1]) < int(block_size * 0.5):
#             chunks = chunks[:-1]

#         attn = [[1] * len(c) for c in chunks]             # attention_mask 동시 생성
#         return {"input_ids": chunks, "attention_mask": attn}
#     return _fn

# tok_ds = tok_ds.map(pack(MAX_LEN), batched=True, batch_size=1000)

# train_ds, eval_ds = tok_ds["train"], tok_ds["eval"]


# 5) Collator (Causal LM) - collator가 길이 맞춰 패딩해줌
from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class DataCollatorCausalWithLabels:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8
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


RUN_ID = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"out-exaone-law-qlora/{RUN_ID}"  # ← 런별 폴더
os.makedirs(OUT_DIR, exist_ok=True)


# 6) TrainingArguments (v4.53.3)
args = TrainingArguments(
    output_dir=OUT_DIR, 
    num_train_epochs=18, 
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    
    seed=SEED, 
    data_seed=SEED, 
    dataloader_num_workers=0, 

    per_device_train_batch_size=4, 
    gradient_accumulation_steps=32,
    
    learning_rate=2e-4,
    logging_steps=20, 
    save_steps=500, 
    eval_steps=500, 
    
    bf16=True,
    tf32=True, 
    
    optim="paged_adamw_8bit",
    report_to="none",
    
    weight_decay=0.01,
    max_grad_norm=1.0,
    group_by_length=True,
    per_device_eval_batch_size=1, 
    prediction_loss_only=True,   
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds, 
    eval_dataset=eval_ds, 
    data_collator=collator, 
)
trainer.train()


# 7) 저장 (PEFT 어댑터)
trainer.save_model(f"{OUT_DIR}/adapter")
tok.save_pretrained(f"{OUT_DIR}/tokenizer")


# 8) 학습 후 평가 
metrics = trainer.evaluate(eval_dataset=eval_ds)
print(metrics)




"""
# ========= Tail fine-tuning in the SAME session =========
# 추가로 1~2 에폭, 낮은 LR로 이어서 학습

import math

TAIL_EPOCHS = 2          # 더 돌릴 에폭 수 (원하면 1로 줄여도 됨)
NEW_LR = 1e-4            # 낮춘 러닝레이트

# 1) 총 스텝 계산 함수 (단일 GPU 기준; DDP면 world_size 곱 고려)
def steps_per_epoch(n_samples, bs, gas):
    return math.ceil(n_samples / (bs * gas))

base_steps = steps_per_epoch(len(train_ds),
                             args.per_device_train_batch_size,
                             args.gradient_accumulation_steps)

# 2) epoch 확장
new_num_epochs = trainer.state.epoch + TAIL_EPOCHS
trainer.args.num_train_epochs = new_num_epochs

# 3) 옵티마이저/스케줄러 재생성 (새 총 스텝 반영)
if trainer.optimizer is None:
    trainer.create_optimizer()
else:
    # 옵티마이저는 그대로 두고 LR만 업데이트
    for g in trainer.optimizer.param_groups:
        g["lr"] = NEW_LR

# 새 총 스텝(전체 학습 기준) 계산
new_total_steps = int(base_steps * new_num_epochs)

# 스케줄러를 새 총 스텝으로 재생성
trainer.create_scheduler(num_training_steps=new_total_steps)

# 4) 이어서 학습 (메모리 상 상태를 그대로 이어감)
trainer.train(resume_from_checkpoint=True)

# 5) 평가 및 저장(새 폴더 권장)
m_tail = trainer.evaluate(eval_dataset=eval_ds)
print({"tail_eval_loss": m_tail.get("eval_loss")})
trainer.save_model("out-exaone-law-qlora-tail/adapter")
tok.save_pretrained("out-exaone-law-qlora-tail/tokenizer")
# ========================================================

"""