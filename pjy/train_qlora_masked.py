from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import List, Dict, Any
import torch, math

base = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # 메모리 절약
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

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)
model.enable_input_require_grads()

# ---------- 데이터: messages -> (prompt_ids, answer_ids) ----------
raw = load_dataset("json", data_files={"train": "../dataset/all/train.jsonl", "eval": "../dataset/all/val.jsonl"})

def build_supervised(ex):
    msgs = ex["messages"]
    assert msgs[-1]["role"] == "assistant", "마지막 메시지는 assistant여야 합니다."

    # 1) 프롬프트: 마지막 assistant 제외 + generation 프롬프트 추가
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    # 2) 정답: 마지막 assistant content
    answer = msgs[-1]["content"] + tok.eos_token

    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]

    input_ids = p_ids + a_ids
    labels    = [-100]*len(p_ids) + a_ids
    attn      = [1]*len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

ds = raw.map(build_supervised, remove_columns=raw["train"].column_names)

# 길이 기준 버킷팅(패딩 낭비↓)
def get_len(ex): return {"len": len(ex["input_ids"])}
ds = ds.map(get_len)
train_ds, eval_ds = ds["train"], ds["eval"]

# ---------- 커스텀 collator: labels까지 패딩 ----------
@dataclass
class DataCollatorCausalWithLabels:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn      = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels    = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attn},
            padding=True, return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        maxlen = batch["input_ids"].size(1)
        lab_pad = torch.full((len(labels), maxlen), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab_pad[i, :l.size(0)] = l
        batch["labels"] = lab_pad
        return batch

collator = DataCollatorCausalWithLabels(tok)

args = TrainingArguments(
    output_dir="out-exaone-law-qlora-masked",
    num_train_epochs=3,                         # 2→3 (언더핏이면)
    per_device_train_batch_size=2,              # 24GB 안전 시작
    gradient_accumulation_steps=32,             # 유효배치 유지
    learning_rate=2e-4,
    warmup_ratio=0.05, 
    lr_scheduler_type="cosine",

    logging_steps=50,
    save_steps=500,

    bf16=True,
    tf32=True,
    optim="paged_adamw_8bit",
    group_by_length=True,                       # 길이 버킷팅
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)

trainer.train()
trainer.save_model("out-exaone-law-qlora-masked/adapter")
tok.save_pretrained("out-exaone-law-qlora-masked/tokenizer")