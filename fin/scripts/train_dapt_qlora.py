#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dapt_qlora.py
- DAPT(비지도, 다음 토큰 예측)로 법령 도메인 적응
- Base: --base (예: taetae030/fin-term-model)
- Data: --data_jsonl (예: /mnt/data/dapt_laws.jsonl)
- 출력: out-dapt-qlora/adapter, tokenizer
"""
import os, math, argparse
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    BitsAndBytesConfig, DataCollatorForLanguageModeling, set_seed
)
from peft import LoraConfig, get_peft_model

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="taetae030/fin-term-model")
    ap.add_argument("--data_jsonl", type=str, default="./data/dapt_laws.jsonl")
    ap.add_argument("--out_dir", type=str, default="./out-dapt-qlora")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)      # per-device 
    ap.add_argument("--grad_accum", type=int, default=16)     # effective 16 
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    return ap.parse_args()

def main():
    args = parse_args(); set_seed(args.seed)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files={"train": args.data_jsonl})
    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=args.seq_len)
    ds = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    steps_per_epoch = math.ceil(len(ds["train"]) / (args.batch_size * args.grad_accum))
    train_args = TrainingArguments(
        output_dir=args.out_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        logging_steps=max(1, steps_per_epoch // 50), save_steps=max(1, steps_per_epoch // 5),
        learning_rate=args.lr, warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine",
        weight_decay=0.0, bf16=True, fp16=False, optim="paged_adamw_32bit",
        gradient_checkpointing=True, torch_compile=False, report_to="none",
        save_total_limit=2, max_grad_norm=1.0,
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=ds["train"],
                      tokenizer=tok, data_collator=collator)
    trainer.train()
    model.save_pretrained(os.path.join(args.out_dir, "adapter"))
    tok.save_pretrained(os.path.join(args.out_dir, "tokenizer"))

if __name__ == "__main__":
    main()