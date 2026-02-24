# evals.py  — answer-only PPL (학습 목적과 동일하게 평가)
import argparse, time, math, os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 경고 억제(선호사항)

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, BitsAndBytesConfig
)
from peft import PeftModel
from dataclasses import dataclass
from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence


def build_answer_only_eval(val_jsonl, tokenizer, max_len=4096):
    """
    messages -> (prompt_ids, answer_ids) -> labels는 '정답 토큰만'
    """
    raw = load_dataset("json", data_files={"eval": val_jsonl})

    def to_io(ex):
        msgs = ex["messages"]
        assert msgs[-1]["role"] == "assistant"
        prompt = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        answer = msgs[-1]["content"] + tokenizer.eos_token

        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = (p_ids + a_ids)[:max_len]
        labels    = ([-100]*len(p_ids) + a_ids)[:max_len]
        attn      = [1]*len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    ds = raw.map(to_io, remove_columns=raw["eval"].column_names)
    return ds["eval"]


@dataclass
class DataCollatorCausalWithLabels_NoTokenizerPad:
    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ids  = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labs = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids      = pad_sequence(ids,  batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attn, batch_first=True, padding_value=0)
        labels         = pad_sequence(labs, batch_first=True, padding_value=self.label_pad_token_id)

        # 패딩 위치 레이블은 반드시 무시
        labels[attention_mask == 0] = self.label_pad_token_id

        # (선택) 8의 배수로 패딩 → Tensor Core 효율
        if self.pad_to_multiple_of is not None:
            def _pad_to_multiple(x, pad_val):
                L = x.size(1)
                m = self.pad_to_multiple_of
                if L % m != 0:
                    pad_len = m - (L % m)
                    pad = torch.full((x.size(0), pad_len), pad_val, dtype=x.dtype, device=x.device)
                    x = torch.cat([x, pad], dim=1)
                return x
            input_ids      = _pad_to_multiple(input_ids, self.pad_token_id)
            attention_mask = _pad_to_multiple(attention_mask, 0)
            labels         = _pad_to_multiple(labels, self.label_pad_token_id)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    ap.add_argument("--adapter", default="out-exaone-law-qlora/adapter")
    ap.add_argument("--val_jsonl", default="../dataset/all/val.jsonl")
    ap.add_argument("--block_size", type=int, default=4096)   # 호환 유지용(미사용)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--tf32", action="store_true", default=True)
    ap.add_argument("--out_dir", default="out-exaone-eval")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    eval_ds = build_answer_only_eval(args.val_jsonl, tok, max_len=4096)

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
    base_model.config.use_cache = True
    base_model.config.ignore_index = -100  # 안전장치
    try:
        base_model.gradient_checkpointing_disable()
    except Exception:
        pass

    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    collator = DataCollatorCausalWithLabels_NoTokenizerPad(
        pad_token_id=tok.pad_token_id, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=args.bf16,
        tf32=args.tf32,
        report_to="none",
        prediction_loss_only=True,
        dataloader_num_workers=2,
        eval_accumulation_steps=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # 토큰 수 집계(속도 지표)
    total_tokens = sum(len(x["input_ids"]) for x in eval_ds)

    t0 = time.time()
    metrics = trainer.evaluate()
    t1 = time.time()

    ppl = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics else float("nan")
    tok_per_sec = total_tokens / max(t1 - t0, 1e-6)

    print({
        "eval_loss": metrics.get("eval_loss"),
        "eval_ppl": ppl,
        "eval_runtime": metrics.get("eval_runtime"),
        "eval_samples_per_second": metrics.get("eval_samples_per_second"),
        "approx_tokens_per_second": tok_per_sec,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
    })



"""
python evals.py \
  --base LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --adapter out-exaone-law-qlora/20250816_014322/adapter \
  --val_jsonl ../dataset/all/val.jsonl \
  --per_device_eval_batch_size 2
"""



if __name__ == "__main__":
    main()