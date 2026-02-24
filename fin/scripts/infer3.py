#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from load import is_multiple_choice, make_prompt_auto, extract_question_and_choices
from allowed_tokens import get_digit_processor
from tqdm.auto import tqdm 

def load_model():
    base_model = "taetae030/fin-term-model"
    adapter_path = "./out-causallm-qlora/20250824_035137/adapter"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "./out-causallm-qlora/20250824_035137/tokenizer",
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def main():
    model, tokenizer = load_model()
    df = pd.read_csv("./data/test.csv")
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference", unit="Question"):
        qid, question = row["ID"], row["Question"]

        prompt_data = make_prompt_auto(question)
        if isinstance(prompt_data, list):
            prompt_text = prompt_data[0]["content"] + prompt_data[1]["content"]
        else:
            prompt_text = prompt_data

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(model.device)

        if is_multiple_choice(question):
            _, options = extract_question_and_choices(question)
            digits = "".join(str(i+1) for i in range(len(options)))
            if len(options) >= 10:
                digits = "0123456789"
            processor = get_digit_processor(tokenizer, digits=digits)
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=1, logits_processor=[processor])
        else:
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=2048, early_stopping=True) 

        new_tokens = outputs[0][inputs["input_ids"].size(1):]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append((qid, answer))

        print(f"\n[ID={qid}]")
        print(f"Q: {question}")
        print(f"A: {answer}")

    out_df = pd.DataFrame(results, columns=["ID", "Answer"])
    out_df.to_csv("submission2.csv", index=False)
    print("\n[완료] submission2.csv")

if __name__ == "__main__":
    main()