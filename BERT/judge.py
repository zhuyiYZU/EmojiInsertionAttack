import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================
# 用户配置区（只改这里）
# ======================
MODEL_DIR   = r"D:\python-wenjian\python wenjian\emoji\experiment\BERT\train\bert\toxigen"

INPUT_FILE  = r"D:\python-wenjian\python wenjian\emoji\experiment\BERT\toxigen\attack-my\1.csv"
TEXT_COLUMN = "text_with_emoji"

# 如果你的文件里有真实标签，用它来算初始准确率
# 没有就保持 None（仅输出预测统计）
GT_LABEL_COLUMN = "label"   # 没有标签就改成 None

# 输出
OUTPUT_DIR = r"D:\python-wenjian\python wenjian\emoji\experiment\BERT\toxigen\attack-my\out"
OUTPUT_FILE_NAME = "1.csv"

# 预测标签列名（对应 judge-3.py 的 attack-after-label）
PRED_LABEL_COLUMN = "attack-after-label"

# 推理参数
MAX_LEN = 256
BATCH_SIZE = 64

# ======================
# 基础设置
# ======================
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)

# （可选）避免 tokenizers fork 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================
# 读取数据
# ======================
# 兜底编码：优先 utf-8-sig（BOM），不行再 gb18030
try:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(INPUT_FILE, encoding="gb18030")

df.columns = [c.replace("\ufeff", "") for c in df.columns]

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"找不到文本列 '{TEXT_COLUMN}'，当前列为：{list(df.columns)}")

# 如果预测列不存在先创建
if PRED_LABEL_COLUMN not in df.columns:
    df[PRED_LABEL_COLUMN] = None

# 是否有真实标签
has_gt = (GT_LABEL_COLUMN is not None) and (GT_LABEL_COLUMN in df.columns) and df[GT_LABEL_COLUMN].notna().any()
if has_gt:
    df = df[df[GT_LABEL_COLUMN].notna()].copy()
    df[GT_LABEL_COLUMN] = df[GT_LABEL_COLUMN].astype(int)

texts = df[TEXT_COLUMN].astype(str).tolist()

# ======================
# 加载模型（离线）
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"开始分类，总样本数：{len(df)}")
print(f"使用列：{TEXT_COLUMN}，预测结果写入列：{PRED_LABEL_COLUMN}")
print(f"Using device: {device}")
print("====================================")

# ======================
# 批量预测
# ======================
all_preds = []
all_probs = []

with torch.no_grad():
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits

        probs_1 = torch.softmax(logits, dim=-1)[:, 1]  # 预测为1的概率
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs_1.cpu().numpy().tolist())

# 写回 DF
df[PRED_LABEL_COLUMN] = all_preds
df["prob_1"] = all_probs  # 你不想要可以删掉这一行

# ======================
# 统计信息（judge-3.py 同款风格）
# ======================
total = len(df)
valid_df = df[df[PRED_LABEL_COLUMN].notna()]  # 这里理论上全都有结果
valid_total = len(valid_df)
invalid_total = total - valid_total

pred_1_num = (valid_df[PRED_LABEL_COLUMN] == 1).sum()
pred_0_num = (valid_df[PRED_LABEL_COLUMN] == 0).sum()

pred_1_ratio = pred_1_num / valid_total if valid_total > 0 else 0.0
pred_0_ratio = pred_0_num / valid_total if valid_total > 0 else 0.0

print("\n==================== 统计结果 ====================")
print(f"样本总数量: {total}")
print(f"有预测结果的样本数: {valid_total}")
print(f"无结果/解析失败的样本数: {invalid_total}")
print(f"{PRED_LABEL_COLUMN}=1 的个数(在有结果样本里): {pred_1_num}")
print(f"{PRED_LABEL_COLUMN}=1 的占比(在有结果样本里): {pred_1_ratio:.4f}")
print(f"{PRED_LABEL_COLUMN}=0 的个数(在有结果样本里): {pred_0_num}")
print(f"{PRED_LABEL_COLUMN}=0 的占比(在有结果样本里): {pred_0_ratio:.4f}")

# ======================
# 如果有 GT label，输出 label-based stats + Metrics
# ======================
if has_gt:
    y_true = valid_df[GT_LABEL_COLUMN].astype(int).tolist()
    y_pred = valid_df[PRED_LABEL_COLUMN].astype(int).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # [[TN, FP],[FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)

    print("\n[Label-based stats]")
    print(f"有效有label的样本数: {len(y_true)}")
    print(f"label=1 的总数: {sum(1 for x in y_true if x == 1)}")
    print(f"label=1 且 {PRED_LABEL_COLUMN}=1 的个数(TP): {tp}")
    print(f"在 label=1 中 {PRED_LABEL_COLUMN}=1 的占比(Recall): {r:.4f}")

    print("\n[Metrics]")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall   : {r:.4f}")
    print(f"F1       : {f1:.4f}")
    print("=================================================\n")
else:
    print("\n（该文件没有可用的真实标签列，因此不计算 Accuracy/Precision/Recall/F1）")
    print("=================================================\n")

# ======================
# 保存结果
# ======================
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"已生成输出文件:\n{OUTPUT_FILE}")
