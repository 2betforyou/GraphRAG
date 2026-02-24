#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA training for EXAONE (robust across architectures)
- 4bit QLoRA + auto target_modules detection (multi-stage, with fallback)
- Safe chat-template rendering (fallback if missing)
- Sanity check that some params are trainable (else widen targets)
"""
import os, re, sys, datetime, argparse
from typing import List, Tuple
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
    set_seed, EarlyStoppingCallback
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    ap.add_argument("--data", type=str, default="data/cleaned/tta_sft_plus.jsonl")
    ap.add_argument("--out", type=str, default="out-tta-sft-qlora")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--seq_len", type=int, default=1536)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    return ap.parse_args()

def build_messages(ex):
    user = ex.get("instruction", "")
    if ex.get("input"):
        user += "\n" + ex["input"]
    system = "당신은 금융보안 전문가입니다. 모든 답변은 한국어로 간결하고 정확하게 작성하세요."
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex.get("output","")},
    ]
    ex["messages"] = msgs
    return ex

def render_with_fallback(tok, msgs):
    """
    Prefer tokenizer.chat_template; if missing/broken, fall back to a plain format.
    """
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        sys_txt = ""
        user_txt = ""
        asst_txt = ""
        for m in msgs:
            if m["role"] == "system":
                sys_txt = m["content"]
            elif m["role"] == "user":
                user_txt = m["content"]
            elif m["role"] == "assistant":
                asst_txt = m["content"]
        # very simple fallback format
        return (
            f"<|system|>\n{sys_txt}\n</|system|>\n"
            f"<|user|>\n{user_txt}\n</|user|>\n"
            f"<|assistant|>\n{asst_txt}\n</|assistant|>\n"
        )

def _present_suffixes(model, suffixes) -> List[str]:
    names = [n for n, _ in model.named_modules()]
    found = set()
    for n in names:
        last = n.split(".")[-1]
        if last in suffixes:
            found.add(last)
    return sorted(found)

def guess_target_modules_stage(model, stage: int) -> List[str]:
    """
    Stage-wise widening:
      1) llama/mistral style
      2) falcon/qwen style
      3) generic proj names
      4) as a last resort: ALL Linear layer suffixes
    Returns exact module name suffixes.
    """
    if stage == 1:
        cand = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","wi","wo","wq","wk","wv"]
        got = _present_suffixes(model, cand)
        if got: return got
    elif stage == 2:
        cand = ["query_key_value","dense","dense_h_to_4h","dense_4h_to_h","c_attn","c_proj","proj","out_proj"]
        got = _present_suffixes(model, cand)
        if got: return got
    elif stage == 3:
        cand = ["in_proj","attn_proj","ffn_up","ffn_down","ffn_gate","Wqkv","W_pack"]
        got = _present_suffixes(model, cand)
        if got: return got
    elif stage == 4:
        # last resort: every Linear's suffix
        linear_suffixes = set()
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                linear_suffixes.add(n.split(".")[-1])
        if linear_suffixes:
            return sorted(list(linear_suffixes))
    return []

def apply_lora_with_fallback(model, args) -> Tuple[torch.nn.Module, List[str]]:
    """
    Detect target suffixes; wrap once with PEFT.
    """
    tried = []
    for stage in (1,2,3,4):
        tmods = guess_target_modules_stage(model, stage)
        if not tmods:
            tried.append((stage, []))
            continue
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            task_type="CAUSAL_LM", target_modules=tmods
        )
        wrapped = get_peft_model(model, lcfg)
        t, a = count_trainable(wrapped)
        if t > 0:
            print(f"[Info] LoRA target_modules (stage {stage}): {tmods}")
            print(f"[Info] Trainable params: {t:,} / {a:,} ({100.0*t/a:.4f}%)")
            return wrapped, tmods
        tried.append((stage, tmods))
    raise RuntimeError(f"LoRA target_modules detection failed. Tried: {tried}")

def count_trainable(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return t, a

def main():
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{args.out}/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 4bit QLoRA quant
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # (이제 생성된 model을 참조) Flash-Attn2 가능하면 사용
    try:
        model.config.attn_implementation = "flash_attention_2"
        print("[Info] attn_implementation = flash_attention_2")
    except Exception as e:
        print(f"[Warn] flash_attention_2 not set ({e}); using default attention")

    model.config.use_cache = False
    # VRAM 여유가 있다면 끄는 것도 속도에 유리함. 기본은 켜둠.
    model.gradient_checkpointing_enable()

    # k-bit prep (필수)
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA (auto-detect with fallbacks)
    model, used_tmods = apply_lora_with_fallback(model, args)

    # ===== Dataset =====
    raw = load_dataset("json", data_files={"train": args.data})["train"]
    split = raw.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, val_ds = split["train"], split["test"]

    train_ds = train_ds.map(build_messages, remove_columns=[c for c in train_ds.column_names if c != "messages"])
    val_ds   = val_ds.map(build_messages,   remove_columns=[c for c in val_ds.column_names   if c != "messages"])

    def render(batch):
        texts = []
        for msgs in batch["messages"]:
            txt = render_with_fallback(tok, msgs)
            texts.append(txt)
        return {"text": texts}

    train_ds = train_ds.map(render, batched=True, remove_columns=["messages"])
    val_ds   = val_ds.map(render,   batched=True, remove_columns=["messages"])

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.seq_len)

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tok_fn,   batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))

    train_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=max(1, steps_per_epoch // 50),
        save_steps=max(1, steps_per_epoch // 5),
        eval_strategy="steps",
        eval_steps=max(1, steps_per_epoch // 5),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        weight_decay=0.0,
        bf16=True, fp16=False,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        torch_compile=False,
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    trainer.train()
    model.save_pretrained(os.path.join(out_dir, "adapter"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    print(f"[OK] saved adapter -> {out_dir}/adapter")

if __name__ == "__main__":
    main()