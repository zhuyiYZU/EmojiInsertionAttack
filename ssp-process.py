import json
import pandas as pd

def extract_fields_from_json(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        inside_obj = False
        for line in f:
            line_strip = line.strip().rstrip(",")
            if line_strip.endswith("{") and not inside_obj:
                inside_obj = True
                buffer = "{"
            elif inside_obj:
                buffer += line
                if line_strip in ["}", "},"]:
                    try:
                        obj = json.loads(buffer.rstrip(",\n "))
                        if isinstance(obj, dict):
                            new_text = obj.get("new_text")
                            # ✅ 跳过 new_text 为 None、空字符串 或 "null" 的情况
                            if new_text is None or str(new_text).strip().lower() in ["", "null"]:
                                pass
                            else:
                                rows.append({
                                    "orig_label": obj.get("orig_label"),
                                    "new_text": new_text,
                                })
                    except Exception:
                        pass
                    inside_obj = False
                    buffer = ""
    return pd.DataFrame(rows)

# 解析三个文件
df1 = extract_fields_from_json(r"D:\python-wenjian\python wenjian\emoji\experiment\RoBert\toxigen\duibi-attack\ssp-roberta-toxige4401after\ssp_retry_summary.json")



# 合并
merged_df = pd.concat([df1], ignore_index=True)

# 保存到 CSV
merged_df.to_csv("ssp3.csv", index=False, encoding="utf-8-sig")

print("✅ 合并完成，输出文件：ssp.csv-all.csv-toxigen-attacked-out-judue1.csv")
