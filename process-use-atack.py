import pandas as pd

# ====== 改成你的文件路径 ======
INPUT_PATH  = r"D:\python-wenjian\python wenjian\emoji\experiment\RoBert\toxigen\pre-attack-before\toxigen.csv"
OUTPUT_PATH = r"D:\python-wenjian\python wenjian\emoji\experiment\RoBert\toxigen\date-use-attack\toxigen.csv"

# 读文件（兼容 UTF-8 BOM）
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
df.columns = [c.replace("\ufeff", "") for c in df.columns]

# 检查必要列
need_cols = ["attack-after-label", "label", "text"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺少列: {missing}，当前列为：{list(df.columns)}")

# 筛选 attack-after-label == 1
df["attack-after-label"] = pd.to_numeric(df["attack-after-label"], errors="coerce")
filtered = df[df["attack-after-label"] == 1][["label", "text"]].copy()

# 保存（UTF-8 BOM）
filtered.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print("✅ Saved:", OUTPUT_PATH)
print("筛选后样本数:", len(filtered))
print("label 分布:", filtered["label"].value_counts().to_dict())
