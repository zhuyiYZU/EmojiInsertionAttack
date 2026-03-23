import pandas as pd
from pathlib import Path

in_path = Path(r"D:\python-wenjian\python wenjian\emoji\experiment\RoBert\toxigen\ce-attack\toxigen\stream_EN9999_MTllama3_TAlhd_PTstep2_k_pred_avg_PSTzs_STUSE_NT1.csv")
out_path = in_path.with_name(in_path.stem + "_keep2cols.csv")

df = pd.read_csv(in_path)

# 只保留两列
keep_cols = ["original_text", "perturbed_text"]
out_df = df.loc[:, keep_cols]

out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Saved to:", out_path)