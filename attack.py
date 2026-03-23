import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

import pandas as pd

EMOJIS = ["😻", "😂", "😹", "🤣", "😸", "😊", "😆", "😃", "😺", "😇", "😄", "😁"]
PREFIXES = ["un", "in", "im", "ir", "il", "non", "dis", "anti"]
SUFFIXES = [
    "able", "ible", "al", "ive", "less", "ous", "ly", "tion", "sion",
    "ment", "ness", "ity", "er", "ist", "ism", "ize", "ise", "ify"
]

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
}

WORD_RE = re.compile(r"\b\w+\b")
TOKEN_RE = re.compile(r"\S+|\s+")
LETTER_ONLY_RE = re.compile(r"^[A-Za-z]+$")
LEADING_PUNCT_RE = re.compile(r"^[^A-Za-z]*")
TRAILING_PUNCT_RE = re.compile(r"[^A-Za-z]*$")
VOWELS = set("aeiou")


@dataclass
class TransformResult:
    text: str
    inserted_any: bool


def normalize_word(word: str) -> str:
    return re.sub(r"[^A-Za-z]", "", word).lower()


def build_hate_keywords(texts: Sequence[str], top_k: int = 200) -> List[str]:
    counter = Counter()
    for text in texts:
        for token in WORD_RE.findall(str(text).lower()):
            if len(token) <= 2:
                continue
            if not LETTER_ONLY_RE.fullmatch(token):
                continue
            if token in STOPWORDS:
                continue
            counter[token] += 1
    return [w for w, _ in counter.most_common(top_k)]


def best_split_index(alpha: str) -> int:
    """
    在词内部选择一个较自然的位置，优先靠近中间，避免切在首尾。
    规则偏启发式，目的是尽量接近你示例里的“内部插入”效果。
    """
    n = len(alpha)
    if n <= 3:
        return 1

    candidates = list(range(1, n))
    center = (n - 1) / 2

    def score(i: int) -> Tuple[float, int, int]:
        left = alpha[i - 1].lower()
        right = alpha[i].lower()
        # 先靠近中间
        distance = abs(i - center)
        # 优先元音/辅音边界，再其次双辅音边界，尽量少切双元音
        if (left in VOWELS) != (right in VOWELS):
            boundary = 0
        elif (left not in VOWELS) and (right not in VOWELS):
            boundary = 1
        else:
            boundary = 2
        # 尽量别贴边
        edge_penalty = 0 if (1 < i < n - 1) else 1
        return (distance, boundary, edge_penalty)

    return min(candidates, key=score)


def inject_into_token(token: str, emoji: str, repeat: int = 1) -> str:
    leading = LEADING_PUNCT_RE.match(token).group(0)
    trailing = TRAILING_PUNCT_RE.search(token).group(0)
    core = token[len(leading): len(token) - len(trailing) if trailing else len(token)]
    if not core:
        return token

    alpha = re.sub(r"[^A-Za-z]", "", core)
    if len(alpha) < 2:
        return token

    target_idx = best_split_index(alpha)
    count = 0
    out = []
    for ch in core:
        out.append(ch)
        if ch.isalpha():
            count += 1
            if count == target_idx:
                out.append(emoji * repeat)
    return leading + "".join(out) + trailing


def local_transform_sentence(text: str, hate_keywords: Set[str], rng: random.Random) -> TransformResult:
    parts = TOKEN_RE.findall(str(text))
    if not parts:
        return TransformResult(text=str(text), inserted_any=False)

    transformed = parts[:]
    inserted_positions = set()
    inserted_any = False

    # Step1: 前缀 / 后缀命中 -> 插 1 个 emoji
    for idx, token in enumerate(parts):
        if token.isspace():
            continue
        base = normalize_word(token)
        if len(base) < 3 or base in hate_keywords:
            continue
        has_prefix = any(base.startswith(p) and len(base) > len(p) + 1 for p in PREFIXES)
        has_suffix = any(base.endswith(s) and len(base) > len(s) + 1 for s in SUFFIXES)
        if has_prefix or has_suffix:
            transformed[idx] = inject_into_token(token, rng.choice(EMOJIS), repeat=1)
            inserted_positions.add(idx)
            inserted_any = True

    # Step3: hate_keywords 命中的词，随机至多 k 个位置，每个位置插 2 个 emoji
    word_indexes = [i for i, x in enumerate(parts) if not x.isspace()]
    candidate_indexes = [
        i for i in word_indexes
        if normalize_word(parts[i]) in hate_keywords and i not in inserted_positions
    ]
    if candidate_indexes:
        k = max(1, math.ceil(len(word_indexes) / 5))
        chosen = rng.sample(candidate_indexes, k=min(k, len(candidate_indexes)))
        for i in chosen:
            emoji = rng.choice(EMOJIS)
            transformed[i] = inject_into_token(transformed[i], emoji, repeat=2)
            inserted_any = True

    # fallback: 整句没有任何 emoji -> 最长词插 2 个相同 emoji
    if not inserted_any:
        valid_words = [i for i in word_indexes if len(normalize_word(parts[i])) >= 2]
        if valid_words:
            longest_idx = max(valid_words, key=lambda i: len(normalize_word(parts[i])))
            emoji = rng.choice(EMOJIS)
            transformed[longest_idx] = inject_into_token(parts[longest_idx], emoji, repeat=2)
            inserted_any = True

    return TransformResult(text="".join(transformed), inserted_any=inserted_any)


SYSTEM_PROMPT = """You are a text transformation engine.
You must transform exactly one input sentence and return JSON only.
Do not explain anything. Do not add extra keys.
Preserve the original text as much as possible except inserting emojis inside words.
"""


def build_user_prompt(text: str, hate_keywords: Sequence[str], seed: int) -> str:
    hate_preview = ", ".join(hate_keywords)
    return f"""
Transform the sentence according to these rules.

Emoji pool:
😻 😂 😹 🤣 😸 😊 😆 😃 😺 😇 😄 😁
Use randomness controlled by this seed: {seed}

Rules:
1. Scan words using English word logic. Base word = lowercase letters only, stripping punctuation.
2. Step1: for words matching prefix or suffix patterns, and base word NOT in hate_keywords, insert exactly 1 emoji inside that word.
Prefixes: un, in, im, ir, il, non, dis, anti
Suffixes: able, ible, al, ive, less, ous, ly, tion, sion, ment, ness, ity, er, ist, ism, ize, ise, ify
3. Step3: for words whose base word IS in hate_keywords, treat them as candidates. Let k = max(1, ceil(number_of_space_split_words / 5)). Randomly choose up to k candidate positions. Insert 2 identical emojis inside each chosen word.
4. If no emoji was inserted anywhere in the whole sentence, find the longest base word in the sentence and insert 2 identical emojis inside it.
5. Emoji must be inserted INSIDE a word, not before or after the word.
6. Keep punctuation, spacing, and all non-target text unchanged as much as possible.
7. Return strict JSON: {{"text_insert": "..."}}

hate_keywords:
[{hate_preview}]

Input sentence:
{text}
""".strip()


def gpt_transform_sentence(client, model: str, text: str, hate_keywords: Sequence[str], seed: int) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, hate_keywords, seed)},
        ],
        text={"format": {"type": "json_object"}},
    )
    raw = response.output_text
    try:
        data = json.loads(raw)
        return str(data["text_insert"])
    except Exception as e:
        raise ValueError(f"Failed to parse model JSON. Raw output: {raw}") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Emoji insertion experiment with local rules + GPT-5.1 API")
    parser.add_argument("--input", default="/mnt/data/lhd-all.csv")
    parser.add_argument("--output", default="/mnt/data/lhd-all_emoji_gpt51_output.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--sample", type=int, default=None, help="Only process the first N rows for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["gpt", "local"], default="gpt")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between API calls, useful for rate limits")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column {args.text_col!r} not found. Columns: {list(df.columns)}")

    if args.sample:
        df = df.head(args.sample).copy()

    texts = df[args.text_col].fillna("").astype(str).tolist()
    hate_keywords = build_hate_keywords(texts, top_k=args.top_k)
    hate_set = set(hate_keywords)

    print(f"Loaded rows: {len(df)}")
    print(f"Top {len(hate_keywords)} hate_keywords built.")
    print("Preview:", hate_keywords[:30])

    rng = random.Random(args.seed)
    outputs: List[str] = []

    client = None
    if args.mode == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    for i, text in enumerate(texts, start=1):
        row_seed = args.seed + i
        if args.mode == "local":
            transformed = local_transform_sentence(text, hate_set, random.Random(row_seed)).text
        else:
            transformed = gpt_transform_sentence(
                client=client,
                model=args.model,
                text=text,
                hate_keywords=hate_keywords,
                seed=row_seed,
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

        outputs.append(transformed)
        if i <= 5 or i % 50 == 0:
            print(f"[{i}/{len(texts)}] done")
            print("SRC:", text)
            print("OUT:", transformed)
            print("-" * 60)

    df["text-insert"] = outputs
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
