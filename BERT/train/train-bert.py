import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# ======================
# 配置
# ======================
MODEL_NAME = "/home/ubuntu/Models/bert-base-uncased"
DATA_PATH  = "/home/ubuntu/songnn/hata experiment/experiment/BERT/train/bert/date-use-train/toxigen_utf8bom.csv"
TEXT_COL   = "text"
LABEL_COL  = "label"
OUTPUT_DIR = "/home/ubuntu/songnn/hata experiment/experiment/BERT/train/bert/toxigen"

SEED = 42
MAX_LEN = 256
VAL_RATIO = 0.1

LR = 2e-5
EPOCHS = 3
TRAIN_BS = 16
EVAL_BS = 32
EVAL_STEPS = 200
SAVE_STEPS = 200
LOG_STEPS = 50

# ======================
# 环境 & 随机种子
# ======================
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(SEED)

use_cuda = torch.cuda.is_available()
print(f"CUDA available: {use_cuda}")

# （可选）彻底离线，防止 transformers/hf 去联网探测
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ======================
# 读数据
# ======================
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
# 防 BOM 导致列名异常
df.columns = [c.replace("\ufeff", "") for c in df.columns]

if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
    raise ValueError(f"缺列：需要 {TEXT_COL}/{LABEL_COL}，但实际列为：{list(df.columns)}")

df = df[[TEXT_COL, LABEL_COL]].dropna().copy()
df[LABEL_COL] = df[LABEL_COL].astype(int)

print("Dataset size:", len(df))
print("Label distribution:", df[LABEL_COL].value_counts().to_dict())

# 分层划分 train/val
train_df, val_df = train_test_split(
    df,
    test_size=VAL_RATIO,
    random_state=SEED,
    shuffle=True,
    stratify=df[LABEL_COL],
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Train size:", len(train_df), "Val size:", len(val_df))
print("Train label ratio:", train_df[LABEL_COL].value_counts(normalize=True).to_dict())
print("Val   label ratio:", val_df[LABEL_COL].value_counts(normalize=True).to_dict())

train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)

# ======================
# Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

def tokenize_fn(batch):
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,  # 交给 DataCollatorWithPadding 动态 pad
    )

train_ds = train_ds.map(tokenize_fn, batched=True, desc="Tokenizing train")
val_ds   = val_ds.map(tokenize_fn, batched=True, desc="Tokenizing val")

# Trainer 期望 label 列名为 labels
train_ds = train_ds.rename_column(LABEL_COL, "labels")
val_ds   = val_ds.rename_column(LABEL_COL, "labels")

# 只保留训练需要的列，避免把 text 等无关列带进 dataloader
keep_cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format(type="torch", columns=keep_cols)
val_ds.set_format(type="torch", columns=keep_cols)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ======================
# Model
# ======================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    local_files_only=True
)

# ======================
# Metrics
# ======================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ======================
# TrainingArguments
# ======================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,

    logging_steps=LOG_STEPS,
    learning_rate=LR,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    num_train_epochs=EPOCHS,

    weight_decay=0.01,
    warmup_ratio=0.06,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    fp16=use_cuda,          # 有 GPU 才开 fp16
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ======================
# Train
# ======================
trainer.train()

# 保存最好模型（Trainer 已自动回滚到 best）
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Saved to:", OUTPUT_DIR)
