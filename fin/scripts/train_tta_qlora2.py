#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA training for EXAONE (robust + upgraded)
- 4bit QLoRA + multi-stage LoRA target_modules auto-detection (with fallback)
- Optional manual target override (--tmods)
- Safe chat-template rendering (fallback if missing)
- Flash-Attention 2 selectable (--attn flash2|eager|auto)
- BF16/FP16 auto fallback, gradient checkpointing toggle
- Resume support, safetensors saving, training args dump
"""
import os, re, sys, json, datetime, argparse
from typing import List, Tuple, Optional
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
    set_seed, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    ap.add_argument("--data", type=str, default="data/cleaned/tta_sft_plus.jsonl")
    ap.add_argument("--out", type=str, default="out-tta-sft-qlora")
    ap.add_argument("--seed", type=int, default=42)

    # Train schedule
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--seq_len", type=int, default=1536)

    # Loader
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=32)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--tmods", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="Comma-separated target module suffixes to override auto (e.g. q_proj,k_proj,v_proj,gate_proj,up_proj,down_proj). Empty = auto-detect.")

    # System
    ap.add_argument("--attn", type=str, default="auto", choices=["auto","flash2","eager"], help="Attention impl preference")
    ap.add_argument("--no_gc", action="store_true", help="Disable gradient checkpointing")
    ap.add_argument("--resume", type=str, default="", help="Path to resume from checkpoint (Trainer)")
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--eval_steps", type=int, default=0, help="Override eval/save/logging steps. 0 = auto")
    return ap.parse_args()


# ---------------------------
# Data utils
# ---------------------------
def build_messages(ex):
    user = ex.get("instruction", "")
    if ex.get("input"):
        user += "\n" + ex["input"]
    system = "당신은 금융보안 전문가입니다. 모든 답변은 한국어로 간결하고 정확하게 작성하세요."
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex.get("output", "")},
    ]
    ex["messages"] = msgs
    return ex


def render_with_fallback(tok, msgs):
    """Prefer tokenizer.chat_template; if missing/broken, fall back to a plain format."""
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
        return (
            f"<|system|>\n{sys_txt}\n</|system|>\n"
            f"<|user|>\n{user_txt}\n</|user|>\n"
            f"<|assistant|>\n{asst_txt}\n</|assistant|>\n"
        )


# ---------------------------
# LoRA target detection
# ---------------------------
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
      1) llama/mistral style (attn + mlp common)
      2) falcon/qwen style
      3) generic proj names
      4) last resort: ALL Linear layer suffixes
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
        linear_suffixes = set()
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                linear_suffixes.add(n.split(".")[-1])
        if linear_suffixes:
            return sorted(list(linear_suffixes))
    return []


def count_trainable(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return t, a


def apply_lora(model, args, override: Optional[List[str]] = None) -> Tuple[torch.nn.Module, List[str], int]:
    """Apply LoRA; if override is provided, use it; else auto-detect with fallback stages."""
    if override and len(override) > 0:
        tmods = override
        stage = 0
    else:
        tmods, stage = [], -1
        for s in (1, 2, 3, 4):
            cand = guess_target_modules_stage(model, s)
            if cand:
                tmods, stage = cand, s
                break
        if not tmods:
            raise RuntimeError("LoRA target_modules detection failed (no candidates found).")

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=tmods
    )
    wrapped = get_peft_model(model, lcfg)
    t, a = count_trainable(wrapped)
    if t == 0:
        raise RuntimeError(f"LoRA wrapped but 0 trainable params. target_modules={tmods}")
    print(f"[Info] LoRA target_modules (stage {stage}): {tmods}")
    print(f"[Info] Trainable params: {t:,} / {a:,} ({100.0*t/a:.4f}%)")
    return wrapped, tmods, stage


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # Dump args for reproducibility
    with open(os.path.join(out_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Quant (4-bit QLoRA)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Attention impl
    try:
        if args.attn == "flash2":
            model.config.attn_implementation = "flash_attention_2"
        elif args.attn == "eager":
            model.config.attn_implementation = "eager"
        else:  # auto: try flash2 then fallback silently
            try:
                model.config.attn_implementation = "flash_attention_2"
            except Exception:
                model.config.attn_implementation = "eager"
        print(f"[Info] attn_implementation = {model.config.attn_implementation}")
    except Exception as e:
        print(f"[Warn] could not set attn_implementation ({e})")

    model.config.use_cache = False

    # Gradient checkpointing
    if not args.no_gc:
        model.gradient_checkpointing_enable()
        gc_flag = True
    else:
        gc_flag = False
    print(f"[Info] gradient_checkpointing = {gc_flag}")

    # Prepare for k-bit training (important)
    model = prepare_model_for_kbit_training(model)

    # LoRA override (if provided)
    override = [s.strip() for s in args.tmods.split(",") if s.strip()] if args.tmods else []
    model, used_tmods, stage = apply_lora(model, args, override if override else None)

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

    # Step geometry
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    if args.eval_steps > 0:
        eval_steps = args.eval_steps
        save_steps = args.eval_steps
        log_steps  = max(1, args.eval_steps // 2)
    else:
        eval_steps = max(1, steps_per_epoch // 5)
        save_steps = eval_steps
        log_steps  = max(1, steps_per_epoch // 50)

    # TrainingArguments
    use_bf16 = (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
    train_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=log_steps,
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        weight_decay=0.0,
        bf16=use_bf16, fp16=not use_bf16,
        optim="paged_adamw_32bit",
        gradient_checkpointing=(not args.no_gc),
        torch_compile=False,
        report_to="none",
        max_grad_norm=1.0,
        remove_unused_columns=False,  # important for custom dicts
    )

    # Trainer (try new processing_class to avoid deprecation; fallback to tokenizer)
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer = None
    try:
        trainer = Trainer(processing_class=tok, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tok, **trainer_kwargs)

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    # Train (resume if provided)
    resume_ckpt = args.resume if args.resume else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save adapter + tokenizer
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    # Save a small metadata file
    meta = {
        "base": args.base,
        "used_target_modules": used_tmods,
        "stage": stage,
        "attn_impl": getattr(model.config, "attn_implementation", "unknown"),
        "gradient_checkpointing": not args.no_gc,
        "seq_len": args.seq_len,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    with open(os.path.join(adapter_dir, "adapter_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved adapter -> {adapter_dir}")
    print(f"[OK] target_modules used -> {used_tmods}")


if __name__ == "__main__":
    main()