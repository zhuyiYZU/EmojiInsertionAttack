import os
import sys
import time
import re
import pandas as pd
from openai import OpenAI

# ======================
# 用户配置区（只改这里）
# ======================

# 输入文件路径
INPUT_FILE = r"D:\python-wenjian\python wenjian\emoji\experiment\gpt-4o\lhd-all\TH\lhdresults_final.csv"

# 文本所在列名（例如："text"、"text-insert-strong-k5-2emoji-bestpos" 等）
TEXT_COLUMN = "adv_text"

# 输出目录（会自动创建）
OUTPUT_DIR = r"D:\python-wenjian\python wenjian\emoji\experiment\gpt-4o\lhd-all\TH\out"

# 输出文件名（如果为 None，则默认使用输入文件名）
OUTPUT_FILE_NAME = "lhd.csv"   # 或者 None

# 预测标签列名（可以是 "model_pred"、"attack-after-label" 等）
LABEL_COLUMN = "attack-after-label"

# 使用的模型名称
MODEL_NAME = "gpt-4o"

# 最大重试次数
MAX_RETRIES = 5

# 每次请求间隔（秒）
SLEEP_BETWEEN_CALLS = 0.2

# ======================
# OpenAI 客户端配置（按你原来的写法写死 key）
# ======================

client = OpenAI(
    base_url="https://chat.xiaohuapi.site/v1",
    api_key="sk-xzkf5d6Wm0ZlWIQxXSCpbkOVqqLQM8XIKA5Fh4Gxah8zf5fX",  # ★ 你自己的 key（注意安全）
    timeout=120
)

# ======================
# 基础设置
# ======================

# 尝试让控制台支持 utf-8（旧版本没有 reconfigure 就忽略）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.makedirs(OUTPUT_DIR, exist_ok=True)

if OUTPUT_FILE_NAME is None:
    # 默认用原文件名
    base_name = os.path.basename(INPUT_FILE)
    OUTPUT_FILE_NAME = base_name

OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)

# ======================
# 读取数据（带编码兜底）
# ======================

try:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(INPUT_FILE, encoding="gb18030")

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"找不到文本列 '{TEXT_COLUMN}'，当前列为：{list(df.columns)}")

# 如果标签列不存在，先创建为 None，方便后面写入
if LABEL_COLUMN not in df.columns:
    df[LABEL_COLUMN] = None

# ======================
# 分类函数（通用 + 更鲁棒的输出解析）
# ======================

def classify(text, max_retries=MAX_RETRIES):
    """
    对单条文本进行 hate / non-hate 二分类：
    - 返回 1 表示 hate
    - 返回 0 表示 非 hate
    - 返回 None 表示解析失败或被内容过滤 / 多次重试失败
    """

    # 文本清洗，避免奇怪编码导致请求报错
    if isinstance(text, str):
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
    else:
        # 不是字符串（例如 NaN），直接返回 None
        return None

    # 提示词
    prompt = f"""
你是一个文本二分类模型，请判断下面的内容是否属于 "hate"（仇恨）。

输出要求：
- 如果是仇恨内容，只输出 1
- 如果不是仇恨内容，只输出 0
- 不要输出任何其他字符或解释

文本内容：
{text}
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    # 如果你想更严格，可以打开这一行：
                    # {"role": "system", "content": "你是一个严格的文本分类器，只能输出 0 或 1"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            output = response.choices[0].message.content.strip()

            # ==================== 关键改动在这里 ====================
            # 不再用 output == "1" 的粗暴判断，改为用正则提取第一个独立的 0/1
            # 例如： "1（仇恨）"、"标签：1"、"The label is 1"
            # 都能正确解析为 1
            m = re.search(r"\b[01]\b", output)
            if m:
                return int(m.group())
            else:
                print("[解析失败] 模型输出内容为：", repr(output))
                return None
            # ======================================================

        except Exception as e:
            msg = str(e)

            # 处理限流 / 上游繁忙
            if "429" in msg or "负载已饱和" in msg or "upstream_error" in msg:
                print(f"[重试 {attempt}/{max_retries}] 请求被限流或上游繁忙：{msg}")
                time.sleep(1 * attempt)  # 等待时间逐渐增加
                continue

            # 被内容审核/过滤
            if "content_filter" in msg:
                print("[content_filter] 该样本被内容审核拦截，标记为 None：", msg)
                return None

            # 其他错误：直接打印并返回 None
            print("[其他错误]：", msg)
            return None

    # 多次重试依然失败
    print(f"[错误] 多次重试({max_retries})后仍失败，返回 None")
    return None

# ======================
# 主循环：逐行调用模型
# ======================

preds = []

total_rows = len(df)
print(f"开始分类，总样本数：{total_rows}")
print(f"使用列：{TEXT_COLUMN}，预测结果写入列：{LABEL_COLUMN}")
print("====================================")

for i, row in df.iterrows():
    text = row[TEXT_COLUMN]

    pred = classify(text)
    preds.append(pred)

    print(f"{i + 1}/{total_rows}  →  {LABEL_COLUMN} = {pred}")
    time.sleep(SLEEP_BETWEEN_CALLS)  # 简单限流

# 写回到 DataFrame
df[LABEL_COLUMN] = preds

# ======================
# 统计信息（通用版）
# ======================

total = len(df)
valid_df = df[df[LABEL_COLUMN].notna()]   # 去掉 None / NaN
valid_total = len(valid_df)

hate_num = (valid_df[LABEL_COLUMN] == 1).sum()
non_hate_num = (valid_df[LABEL_COLUMN] == 0).sum()
filtered_num = total - valid_total

hate_ratio = hate_num / valid_total if valid_total > 0 else 0.0
non_hate_ratio = non_hate_num / valid_total if valid_total > 0 else 0.0

print("\n==================== 统计结果 ====================")
print(f"样本总数量: {total}")
print(f"有预测结果的样本数: {valid_total}")
print(f"被过滤/无结果的样本数: {filtered_num}")
print(f"预测为 hate 的数量: {hate_num}")
print(f"预测为 非 hate 的数量: {non_hate_num}")
print(f"在有预测结果的样本中，hate 比例: {hate_ratio:.4f}")
print(f"在有预测结果的样本中，非 hate 比例: {non_hate_ratio:.4f}")
print("（如果这是攻击后的样本，非 hate 比例可以理解为“攻击成功率”）")
print("=================================================\n")

# ======================
# 保存结果
# ======================

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"已生成输出文件:\n{OUTPUT_FILE}")
