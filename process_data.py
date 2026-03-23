# import csv
# import os
# import sys
#
# # Use the existing clean_str implementation from the project to ensure consistency
# try:
#     from dataloader import clean_str  # type: ignore
# except Exception:
#     # Fallback: replicate the same logic to avoid runtime import issues
#     import re
#
#     def clean_str(string, TREC=False):
#         """Tokenization/string cleaning for all datasets except for SST.
#         Every dataset is lower cased except for TREC
#         """
#         string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#         string = re.sub(r"\'s", " \'s", string)
#         string = re.sub(r"\'ve", " \'ve", string)
#         string = re.sub(r"n\'t", " n\'t", string)
#         string = re.sub(r"\'re", " \'re", string)
#         string = re.sub(r"\'d", " \'d", string)
#         string = re.sub(r"\'ll", " \'ll", string)
#         string = re.sub(r",", " , ", string)
#         string = re.sub(r"!", " ! ", string)
#         string = re.sub(r"\(", " \\( ", string)
#         string = re.sub(r"\)", " \\) ", string)
#         string = re.sub(r"\?", " \\? ", string)
#         string = re.sub(r"\s{2,}", " ", string)
#         return string.strip() if TREC else string.strip().lower()
#
#
# # Hardcode your absolute input/output paths here before running
# INPUT_CSV_ABS_PATH = "/absolute/path/to/your_input.csv"  # e.g., "/home/ubuntu/.../data/yahoo.csv"
# OUTPUT_FILE_ABS_PATH = "/absolute/path/to/output_single_file.txt"  # e.g., "/home/ubuntu/.../data/yahoo_imdb_format"
#
#
# def process_csv_to_imdb_single_file(input_csv_path, output_file_path):
#     """Read a two-column CSV (label,text) and write a single file with lines: "label text".
#
#     - Assumes no header in the CSV.
#     - Applies the same cleaning and lowercasing as the project (clean_str).
#     - Writes UTF-8 encoded output.
#     """
#     num_written = 0
#     with open(input_csv_path, "r", encoding="utf-8") as fin, open(
#         output_file_path, "w", encoding="utf-8"
#     ) as fout:
#         reader = csv.reader(fin, delimiter=",")
#         for row in reader:
#             if not row:
#                 continue
#             # Expecting [label, text]
#             try:
#                 raw_label = row[0].strip()
#                 label_int = int(raw_label)
#                 raw_text = row[1]
#             except Exception:
#                 # If the CSV is unexpectedly [text, label], try to recover gracefully
#                 try:
#                     raw_label = row[1].strip()
#                     label_int = int(raw_label)
#                     raw_text = row[0]
#                 except Exception:
#                     # Skip malformed lines
#                     continue
#
#             text_clean = clean_str(raw_text.strip(), TREC=False)
#             # Compose line as: label + space + text
#             fout.write(f"{label_int} {text_clean}\n")
#             num_written += 1
#
#     return num_written
#
#
# def main():
#     # Allow overriding via CLI for convenience, but default to hardcoded paths
#     # Usage: python data/process_data.py [input_csv_abs_path] [output_file_abs_path]
#     input_path = "/home/ubuntu/songnn/PaperCode/hard/hard-label-attack/data/data-qwen/judge-1-qwen/Qwen-pre-before-judge1.csv"
#     output_path = "/home/ubuntu/songnn/PaperCode/hard/hard-label-attack/data/data-qwen/judge-1-qwen/Qwen-pre-before-judge1"
#
#     if len(sys.argv) >= 2:
#         input_path = sys.argv[1]
#     if len(sys.argv) >= 3:
#         output_path = sys.argv[2]
#
#     if not os.path.isabs(input_path) or not os.path.isabs(output_path):
#         print("Please provide absolute paths for both input and output.")
#         sys.exit(1)
#
#     if not os.path.exists(os.path.dirname(output_path)):
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     count = process_csv_to_imdb_single_file(input_path, output_path)
#     print(f"Wrote {count} lines to {output_path}")
#
#
# if __name__ == "__main__":
#     main()
#
#
import csv
import os
import sys

# Use the existing clean_str implementation from the project to ensure consistency
try:
    from dataloader import clean_str  # type: ignore
except Exception:
    # Fallback: replicate the same logic to avoid runtime import issues
    import re

    def clean_str(string, TREC=False):
        """Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \\( ", string)
        string = re.sub(r"\)", " \\) ", string)
        string = re.sub(r"\?", " \\? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip() if TREC else string.strip().lower()


def process_csv_to_imdb_single_file(input_csv_path, output_file_path):
    """Read a two-column CSV and write a single file with lines: "label text".

    - 支持有表头（如: text,attack-before-label）
    - 标签可以是 0 / 1 或 0.0 / 1.0
    - 会用 clean_str 做清洗和小写化
    """
    num_written = 0
    with open(input_csv_path, "r", encoding="utf-8") as fin, open(
        output_file_path, "w", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin, delimiter=",")

        # 跳过表头（如果没有表头也不会出错）
        header = next(reader, None)

        for row in reader:
            if not row:
                continue

            # 尝试两种列顺序：[label, text] 或 [text, label]
            label_int = None
            raw_text = None

            # 先假设第 0 列是 label
            try:
                raw_label = row[0].strip()
                # 支持 "1" / "0" / "1.0" / "0.0"
                label_int = int(float(raw_label))
                raw_text = row[1]
            except Exception:
                # 再尝试第 1 列是 label
                try:
                    raw_label = row[1].strip()
                    label_int = int(float(raw_label))
                    raw_text = row[0]
                except Exception:
                    # 这一行既不是 [label,text] 也不是 [text,label]，跳过
                    continue

            text_clean = clean_str(raw_text.strip(), TREC=False)
            fout.write(f"{label_int} {text_clean}\n")
            num_written += 1

    return num_written


def main():
    # 默认用你现在这两个路径
    input_path = r"D:\python-wenjian\python wenjian\emoji\experiment\TextCNN\lhd\date-use-attack\lhd.csv"
    output_path = r"D:\python-wenjian\python wenjian\emoji\experiment\TextCNN\lhd\date-use-attack\lhd"

    # 可以用命令行覆盖：python script.py input.csv output.txt
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    if not os.path.isabs(input_path) or not os.path.isabs(output_path):
        print("Please provide absolute paths for both input and output.")
        sys.exit(1)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = process_csv_to_imdb_single_file(input_path, output_path)
    print(f"Wrote {count} lines to {output_path}")


if __name__ == "__main__":
    main()


