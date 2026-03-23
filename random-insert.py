import math
import random
import pandas as pd

# ==== 配置 ====
INPUT_FILE = r"D:\python-wenjian\python wenjian\emoji\experiment\RoBert\lhd\date-use-attack\lhd.csv"      # 换成你自己的 sbic.csv 路径
OUTPUT_FILE = r"D:\python-wenjian\python wenjian\emoji\experiment\RoBert\lhd\random\lhd.csv"

TEXT_COLUMN = "text"              # 原始文本列
NEW_TEXT_COLUMN = "text_with_emoji"

# 表情集合
EMOJIS = ["😻","😂","😹","🤣","😸","😊","😆","😃","😺","😇","😄","😁"]

# 固定随机种子，方便复现实验（想要每次不同可以去掉这一行）
random.seed(20260202)

def insert_emojis_inside_simple(sentence: str) -> str:
    if sentence is None:
        return sentence
    s = str(sentence)
    words = s.split()
    n = len(words)
    if n == 0:
        return s

    # k = max(1, ceil(len(words)/5))
    k = max(1, math.ceil(n / 5))
    k = min(k, n)

    # 随机选 k 个词的位置
    positions = random.sample(range(n), k)

    for idx in positions:
        w = words[idx]
        # 在单词内部（字母部分居中）插入 1 个 emoji
        letters = [i for i, ch in enumerate(w) if ch.isalpha()]
        if letters:
            start = letters[0]
            end = letters[-1]
            mid = (start + end + 1) // 2
        else:
            mid = len(w) // 2

        emo = random.choice(EMOJIS)
        words[idx] = w[:mid] + emo + w[mid:]

    return " ".join(words)

def main():
    # 读数据
    try:
        df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding="gb18030")

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"找不到文本列 '{TEXT_COLUMN}'，当前列为：{list(df.columns)}")

    # 插表情
    df[NEW_TEXT_COLUMN] = df[TEXT_COLUMN].apply(insert_emojis_inside_simple)

    # 保存
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("已保存到：", OUTPUT_FILE)

if __name__ == "__main__":
    main()
