import pandas as pd
import re

# 输入文件路径
file1 = r"D:\python-wenjian\python wenjian\emoji\experiment\RoBert\toxigen\duibi-attack\hy-3.txt"

# ✅ 输出文件路径（改成你要求的位置）
output_file = r"D:\python-wenjian\python wenjian\emoji\experiment\RoBert\toxigen\duibi-attack\hy3.csv"

def parse_file(filepath):
    """
    读取文件，提取 adv sent 后的文本
    返回 DataFrame，列为 orig_label(1) 和 new_text
    """
    new_texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("adv sent"):
            # 使用正则匹配冒号后的文本
            match = re.match(r"adv sent\s*\(\d+\):\s*(.*)", line)
            if match:
                new_texts.append(match.group(1))
    df = pd.DataFrame({
        "orig_label": 1,
        "new_text": new_texts
    })
    return df

# 解析三个文件
df1 = parse_file(file1)


# 合并
df_merged = pd.concat([df1], ignore_index=True)

# 保存 CSV
df_merged.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ 合并完成，新文件已保存为：{output_file}")


