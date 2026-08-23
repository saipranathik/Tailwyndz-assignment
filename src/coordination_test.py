import pandas as pd
import json

from analyze_data import parse_timestamps


POSTS = "data/final/posts.csv"
TRUTH = "data/ground_truth/final_ground_truth.json"


print("Loading posts...")

p = pd.read_csv(
    POSTS,
    usecols=[
        "post_id",
        "author_id",
        "timestamp",
        "text",
    ],
)

with open(TRUTH, encoding="utf-8") as f:
    truth = json.load(f)

truth_ids = set(
    map(str, truth["posts"]["automated_post_ids"])
)

# ------------------------------------------------------------
# Correct timestamp parsing
# ------------------------------------------------------------

p["timestamp"] = parse_timestamps(
    p["timestamp"]
)

print(
    "Unparsed timestamps:",
    int(p["timestamp"].isna().sum())
)

p = p.dropna(
    subset=["timestamp"]
)

# ------------------------------------------------------------
# Text normalization
# ------------------------------------------------------------

p["text_norm"] = (
    p["text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

p["truth_bot"] = (
    p["post_id"]
    .astype(str)
    .isin(truth_ids)
)

print(
    "Rows after timestamp parsing:",
    len(p)
)

print(
    "True automated posts present:",
    int(p["truth_bot"].sum())
)


def evaluate(name, mask):

    mask = mask.fillna(False)

    predicted = int(mask.sum())

    tp = int(
        (mask & p["truth_bot"]).sum()
    )

    fp = int(
        (mask & ~p["truth_bot"]).sum()
    )

    fn = int(
        (~mask & p["truth_bot"]).sum()
    )

    precision = (
        tp / predicted
        if predicted else 0
    )

    recall = (
        tp / p["truth_bot"].sum()
        if p["truth_bot"].sum()
        else 0
    )

    error = (
        fp + fn
    ) / len(p)

    print(
        f"{name:<35}"
        f"{predicted:>10,}"
        f"{tp:>10,}"
        f"{precision:>10.3f}"
        f"{recall:>10.3f}"
        f"{error:>10.3f}"
    )


print()
print("=" * 90)
print("COORDINATED REPEATED-TEXT DETECTOR")
print("=" * 90)

print(
    f"{'Rule':<35}"
    f"{'Pred':>10}"
    f"{'TP':>10}"
    f"{'Prec':>10}"
    f"{'Recall':>10}"
    f"{'Error':>10}"
)

print("-" * 90)

# ------------------------------------------------------------
# Test coordination windows
# ------------------------------------------------------------

for seconds in [5, 10, 20, 30, 60]:

    bucket = (
        p["timestamp"]
        .dt.floor(f"{seconds}s")
    )

    grouped = (
        p.assign(bucket=bucket)
        .groupby(
            ["bucket", "text_norm"],
            dropna=False,
        )
        .agg(
            posts=("post_id", "size"),
            authors=("author_id", "nunique"),
        )
        .reset_index()
    )

    for author_threshold in [3, 5, 8, 10]:

        suspicious = grouped[
            grouped["authors"] >= author_threshold
        ]

        keys = pd.MultiIndex.from_frame(
            suspicious[
                ["bucket", "text_norm"]
            ]
        )

        post_keys = pd.MultiIndex.from_frame(
            pd.DataFrame({
                "bucket": bucket,
                "text_norm": p["text_norm"],
            })
        )

        mask = post_keys.isin(keys)

        evaluate(
            f"{seconds}s + {author_threshold} authors",
            pd.Series(mask, index=p.index),
        )


print("=" * 90)
print(
    f"Ground-truth automated posts: "
    f"{len(truth_ids):,}"
)