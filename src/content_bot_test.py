import pandas as pd
import json
import re


POSTS = "data/final/posts.csv"
TRUTH = "data/ground_truth/final_ground_truth.json"


p = pd.read_csv(
    POSTS,
    usecols=["post_id", "text"]
)

with open(TRUTH, encoding="utf-8") as f:
    truth = json.load(f)

truth_ids = set(
    map(str, truth["posts"]["automated_post_ids"])
)

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


def evaluate(name, mask):

    mask = mask.fillna(False)

    pred = int(mask.sum())
    tp = int((mask & p["truth_bot"]).sum())
    fp = int((mask & ~p["truth_bot"]).sum())
    fn = int((~mask & p["truth_bot"]).sum())

    precision = tp / pred if pred else 0
    recall = tp / p["truth_bot"].sum()

    error = (fp + fn) / len(p)

    print(
        f"{name:<35}"
        f"{pred:>10,}"
        f"{tp:>10,}"
        f"{precision:>10.3f}"
        f"{recall:>10.3f}"
        f"{error:>10.3f}"
    )


# ------------------------------------------------------------
# Signal definitions
# ------------------------------------------------------------

spam_words = [
    "win free",
    "giveaway",
    "claim now",
    "click here",
    "click link",
    "free drinks",
]

amplification_words = [
    "disaster",
    "complete failure",
    "ruining",
    "avoid kestrel",
    "nobody should attend",
]

monitoring_words = [
    "monitoring summary",
    "automated summary",
    "brand monitoring",
    "mentions increased",
    "volume increased",
    "activity is elevated",
]


def contains_any(words):
    pattern = "|".join(
        re.escape(x)
        for x in words
    )

    return p["text_norm"].str.contains(
        pattern,
        regex=True,
        na=False,
    )


spam = contains_any(spam_words)
amplification = contains_any(amplification_words)
monitoring = contains_any(monitoring_words)

combined = (
    spam
    | amplification
    | monitoring
)


print()
print("=" * 95)
print("CONTENT-BASED BOT SIGNAL TEST")
print("=" * 95)

print(
    f"{'Rule':<35}"
    f"{'Pred':>10}"
    f"{'TP':>10}"
    f"{'Prec':>10}"
    f"{'Recall':>10}"
    f"{'Error':>10}"
)

print("-" * 95)

evaluate("Promotional/spam", spam)
evaluate("Negative amplification", amplification)
evaluate("Monitoring language", monitoring)
evaluate("Any content signal", combined)

print("=" * 95)

print()
print("Overlap:")
print("Spam + amplification:", int((spam & amplification).sum()))
print("Spam + monitoring:", int((spam & monitoring).sum()))
print("Amplification + monitoring:", int((amplification & monitoring).sum()))
print("All three:", int((spam & amplification & monitoring).sum()))