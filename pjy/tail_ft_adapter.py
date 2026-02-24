#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tail_ft_adapter.py
- 기존 LoRA 어댑터(에폭 12 등)를 불러와 '테일 파인튜닝'만 수행
- Phase-1: +E1 epochs @ LR1 (기본 2ep @ 5e-5)
- Phase-2: +E2 epochs @ LR2 (기본 2ep @ 2e-5)
- EarlyStopping + load_best_model_at_end 지원
"""

import os, math, argparse, random, numpy as np, torch
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    BitsAndBytesConfig, EarlyStoppingCallback, set_seed
)
from peft import PeftModel
from peft import LoraConfig  # (미사용 지만 import 에러 방지)
# ↑ adapter는 로드만 하므로 새 LoRA 구성 생성은 하지 않습니다.

# -------------------------
# Collator (labels 포함 패딩)
# -------------------------
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

# -------------------------
# 데이터 전처리(SFT: assistant만 loss)
# -------------------------
def build_supervised(tok, ex):
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
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    answer = (msgs[-1]["content"] or "") + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": p_ids + a_ids,
        "labels":    [-100]*len(p_ids) + a_ids,
        "attention_mask": [1]*(len(p_ids)+len(a_ids)),
    }

def steps_per_epoch(n_samples, bs, gas):
    return math.ceil(n_samples / (bs * gas))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    ap.add_argument("--adapter_in", type=str, required=True, help="이어학습할 기존 LoRA 어댑터 경로 (예: out-exaone-law-qlora/20250827_xxx/adapter)")
    ap.add_argument("--data", type=str, default="../dataset/all/tta_mitre_sft_plus_ko.jsonl")
    ap.add_argument("--seed", type=int, default=42)

    # Phase 설정
    ap.add_argument("--epochs1", type=float, default=2.0, help="Phase-1 추가 epoch")
    ap.add_argument("--lr1", type=float, default=5e-5, help="Phase-1 learning rate")
    ap.add_argument("--epochs2", type=float, default=2.0, help="Phase-2 추가 epoch")
    ap.add_argument("--lr2", type=float, default=2e-5, help="Phase-2 learning rate")

    # 훈련 하이퍼파라미터
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--eval_steps", type=int, default=250)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--early_patience", type=int, default=1)
    ap.add_argument("--early_threshold", type=float, default=0.01)

    ap.add_argument("--out_dir", type=str, default=None)

    args = ap.parse_args()

    # 고정 시드
    set_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # 출력 폴더
    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    OUT_DIR = args.out_dir or f"out-exaone-law-qlora-tail/{run_id}"
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------------
    # 토크나이저 & 모델 로드(4bit)
    # -------------------------
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.pad_token_id = tok.pad_token_id
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 🔑 기존 어댑터 로드
    model = PeftModel.from_pretrained(base_model, args.adapter_in)
    model.enable_input_require_grads()

    # -------------------------
    # 데이터셋 준비
    # -------------------------
    raw_all = load_dataset("json", data_files={"data": args.data})["data"]
    split = raw_all.train_test_split(test_size=0.1, seed=args.seed)
    raw = DatasetDict({"train": split["train"], "eval": split["test"]})

    def _fmt(ex): return build_supervised(tok, ex)
    ds = raw.map(_fmt, remove_columns=raw["train"].column_names)
    train_ds, eval_ds = ds["train"], ds["eval"]

    collator = DataCollatorCausalWithLabels(tok)

    # -------------------------
    # 공통 TrainingArguments (단, epoch/LR은 Phase마다 조절)
    # -------------------------
    common_kwargs = dict(
        output_dir=OUT_DIR,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        logging_steps=20,

        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,

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

    # -------------------------
    # Phase-1: +epochs1 @ lr1
    # -------------------------
    args1 = TrainingArguments(
        **common_kwargs,
        num_train_epochs=args.epochs1,
        learning_rate=args.lr1,
    )
    trainer = Trainer(
        model=model,
        args=args1,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_patience,
            early_stopping_threshold=args.early_threshold,
        )],
    )
    print(f"[Tail-1] epochs={args.epochs1}, lr={args.lr1}")
    trainer.train()
    m1 = trainer.evaluate(eval_dataset=eval_ds)
    print({"tail1_eval_loss": m1.get("eval_loss")})

    # 체크포인트 저장
    adapter_dir_1 = os.path.join(OUT_DIR, "adapter_tail_phase1")
    trainer.save_model(adapter_dir_1)
    tok.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))

    # -------------------------
    # Phase-2: +epochs2 @ lr2
    #  - 동일 Trainer 인스턴스를 사용해 '이어달리기'
    #  - 전체 num_train_epochs를 현재 완료 에폭에서 추가
    # -------------------------
    if args.epochs2 > 0:
        # 총 스텝 재계산을 위해 현재 스텝 기준 전체 에폭 갱신
        # (아래는 Trainer 내부 스케줄러 재생성을 위해 수동 설정)
        # 1) 총 에폭 확장
        base_steps = steps_per_epoch(len(train_ds), args.batch_size, args.grad_accum)
        new_total_epochs = args.epochs1 + args.epochs2
        trainer.args.num_train_epochs = new_total_epochs

        # 2) LR 하향
        if trainer.optimizer is None:
            trainer.create_optimizer()
        for g in trainer.optimizer.param_groups:
            g["lr"] = args.lr2

        # 3) 스케줄러 재생성
        new_total_steps = int(base_steps * new_total_epochs)
        trainer.create_scheduler(num_training_steps=new_total_steps)

        print(f"[Tail-2] epochs(+{args.epochs2}) → total={new_total_epochs}, lr={args.lr2}")
        trainer.train(resume_from_checkpoint=True)
        m2 = trainer.evaluate(eval_dataset=eval_ds)
        print({"tail2_eval_loss": m2.get("eval_loss")})

        adapter_dir_2 = os.path.join(OUT_DIR, "adapter_tail_phase2")
        trainer.save_model(adapter_dir_2)

    print(f"[DONE] Tail fine-tuning finished. OUT={OUT_DIR}")

if __name__ == "__main__":
    main()