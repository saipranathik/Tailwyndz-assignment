import pandas as pd
import json
import numpy as np

POSTS = "data/final/posts.csv"
AUTHORS = "data/final/authors_and_reference.xlsx"
TRUTH = "data/ground_truth/final_ground_truth.json"


print("Loading...")

p = pd.read_csv(
    POSTS,
    usecols=[
        "post_id",
        "author_id",
        "text",
        "follower_count",
    ],
)

a = pd.read_excel(
    AUTHORS,
    sheet_name="authors",
)

with open(TRUTH, encoding="utf-8") as f:
    truth = json.load(f)

truth_ids = set(
    map(str, truth["posts"]["automated_post_ids"])
)

p["author_id_key"] = pd.to_numeric(
    p["author_id"],
    errors="coerce",
)

a["author_id_key"] = pd.to_numeric(
    a["author_id"],
    errors="coerce",
)

p = p.merge(
    a[
        [
            "author_id_key",
            "posting_frequency",
            "has_picture",
            "has_bio",
            "following_count",
        ]
    ],
    on="author_id_key",
    how="left",
)

p["truth_bot"] = (
    p["post_id"].astype(str).isin(truth_ids)
)

p["text_norm"] = (
    p["text"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

p["text_len"] = p["text_norm"].str.len()

author_stats = (
    p.groupby("author_id_key")
    .agg(
        author_posts=("post_id", "size"),
        unique_texts=("text_norm", "nunique"),
    )
    .reset_index()
)

author_stats["repeat_rate"] = (
    1
    - author_stats["unique_texts"]
    / author_stats["author_posts"]
)

author_stats["text_diversity"] = (
    author_stats["unique_texts"]
    / author_stats["author_posts"]
)

p = p.merge(
    author_stats,
    on="author_id_key",
    how="left",
)

truth_count = int(p["truth_bot"].sum())


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
        tp / truth_count
        if truth_count else 0
    )

    error = (
        fp + fn
    ) / len(p)

    print(
        f"{name:<30}"
        f"{predicted:>10,}"
        f"{tp:>10,}"
        f"{precision:>10.3f}"
        f"{recall:>10.3f}"
        f"{error:>10.3f}"
    )


print()
print("=" * 85)
print("POST-LEVEL BOT SIGNAL TEST")
print("=" * 85)

print(
    f"{'Rule':<30}"
    f"{'Pred':>10}"
    f"{'TP':>10}"
    f"{'Prec':>10}"
    f"{'Recall':>10}"
    f"{'Error':>10}"
)

print("-" * 85)

# ------------------------------------------------------------
# Repeat-rate thresholds
# ------------------------------------------------------------

for threshold in [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
]:

    evaluate(
        f"repeat >= {threshold:.0%}",
        p["repeat_rate"] >= threshold,
    )


# ------------------------------------------------------------
# Author volume + repeat
# ------------------------------------------------------------

for threshold in [
    0.20,
    0.30,
    0.40,
]:

    evaluate(
        f"posts>=10 + repeat>={threshold:.0%}",
        (
            (p["author_posts"] >= 10)
            & (p["repeat_rate"] >= threshold)
        ),
    )


# ------------------------------------------------------------
# Posting frequency
# ------------------------------------------------------------

for threshold in [
    5,
    10,
    15,
    20,
    30,
]:

    evaluate(
        f"posting_frequency >= {threshold}",
        p["posting_frequency"] >= threshold,
    )


# ------------------------------------------------------------
# Profile metadata
# ------------------------------------------------------------

evaluate(
    "no picture",
    p["has_picture"].eq(False),
)

evaluate(
    "no bio",
    p["has_bio"].eq(False),
)


print("=" * 85)
print(
    f"True automated posts: {truth_count:,}"
)