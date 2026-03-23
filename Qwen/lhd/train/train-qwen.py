#!/usr/bin/env python3
# coding: utf-8

import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# =========================
# 路径 & 列名配置（按需改）
# =========================
MODEL_PATH = "/file/songnn/Models/Qwen/Qwen2.5-7B-Instruct"
DATA_PATH  = "/home/ubuntu/songnn/hata experiment/experiment/qwen/filter/sbic/sbic_train.csv"

TEXT_COLUMN  = "text"  # 用插入emoji的文本训练
LABEL_COLUMN = "label"                  # 监督标签（0/1）

OUTPUT_DIR = "/home/ubuntu/songnn/hata experiment/experiment/train/sbic"

# 训练超参（你可以先跑小一点验证流程）
SEED = 42
VAL_RATIO = 0.03
NUM_EPOCHS = 1
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACC = 16
MAX_LENGTH = 512

# LoRA 配置（Qwen 常用 target_modules）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# =========================
# 固定随机种子
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 读数据
# =========================
df = pd.read_csv(DATA_PATH, encoding="utf-8")
if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    raise ValueError(f"缺列：需要 {TEXT_COLUMN} 和 {LABEL_COLUMN}，但现在是 {list(df.columns)}")

# label 转 int（处理 1.0/0.0）
df = df[df[LABEL_COLUMN].notna()].copy()
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

# 划分 train/val
val_size = max(1, int(len(df) * VAL_RATIO))
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df_val = df.iloc[:val_size].reset_index(drop=True)
df_tr  = df.iloc[val_size:].reset_index(drop=True)

train_ds = Dataset.from_pandas(df_tr)
val_ds   = Dataset.from_pandas(df_val)

print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

# =========================
# 加载 tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    use_fast=True,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# Prompt 构造：让模型学会严格输出 0/1
# =========================
SYSTEM_PROMPT = (
    "你是一个文本二分类模型，请判断下面的内容是否属于 'hata'（仇恨）。\n\n"
    "输出要求：\n"
    "- 如果是仇恨内容，输出 1\n"
    "- 如果不是仇恨内容，输出 0\n\n"
    "只输出一个数字 0 或 1，不要输出其他任何内容。"
)

def build_messages(text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"文本内容：\n{text}"},
    ]

def preprocess_text(text: str) -> str:
    # 和你推理脚本一致：简单清洗 @ 和 URL
    if not isinstance(text, str):
        text = str(text)
    parts = []
    for t in text.split(" "):
        if t.startswith("@") and len(t) > 1:
            t = "@user"
        elif t.startswith("http"):
            t = "http"
        parts.append(t)
    return " ".join(parts)

def tokenize_example(example):
    text = preprocess_text(example[TEXT_COLUMN])
    label = int(example[LABEL_COLUMN])

    # add_generation_prompt=True 会在末尾加上 assistant 的生成起始标记
    prompt = tokenizer.apply_chat_template(
        build_messages(text),
        tokenize=False,
        add_generation_prompt=True
    )

    # 监督目标：只训练生成的 0/1
    full_text = prompt + str(label)

    full = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH)
    prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"]

    prompt_len = len(prompt_ids)
    input_ids = full["input_ids"]
    attn = full["attention_mask"]

    labels = [-100] * len(input_ids)
    # 仅让模型在“答案字符”上算 loss
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }

train_ds = train_ds.map(tokenize_example, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(tokenize_example, remove_columns=val_ds.column_names)

# =========================
# QLoRA 4bit 加载模型
# =========================
use_cuda = torch.cuda.is_available()
compute_dtype = torch.bfloat16 if use_cuda else torch.float32

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 数据 collator：pad input_ids & labels
# =========================
def collate_fn(features):
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attn, labels = [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attn.append(f["attention_mask"] + [0] * pad_len)
        labels.append(f["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# =========================
# 训练参数
# =========================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    bf16=use_cuda,              # A100/H100 等支持 bf16 更好
    fp16=(use_cuda and not torch.cuda.is_bf16_supported()),
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
)

trainer.train()

# 保存 LoRA adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ LoRA adapter saved to:", OUTPUT_DIR)
