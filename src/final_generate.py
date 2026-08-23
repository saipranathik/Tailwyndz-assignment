"""
Tailwyndz Assessment 4 - final synthetic dataset generator.

Creates the deliberately messy synthetic social-listening dataset required by
the assessment brief. This script GENERATES mess; it does not clean it.

Outputs:
  data/final/posts.csv
  data/final/posts.jsonl
  data/final/authors_and_reference.xlsx
  data/final/event_timeline.csv
  data/final/brand_terms.csv
  data/ground_truth/final_ground_truth.json
  data/ground_truth/generation_report.json
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import random
import re

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FINAL_DIR = DATA_DIR / "final"
GT_DIR = DATA_DIR / "ground_truth"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)

AUTHOR_COUNT = 75_000
BASELINE_DAYS = 14
EVENT_HOURS = 14
BASELINE_POSTS = 55_000
EVENT_NORMAL_POSTS = 961_000

EVENT_START = datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)
EVENT_END = EVENT_START + timedelta(hours=EVENT_HOURS)
BASELINE_START = EVENT_START - timedelta(days=BASELINE_DAYS)
BASELINE_END = EVENT_START

BRAND = "Kestrel"

PLATFORMS = ["platform_a", "platform_b", "platform_c", "platform_d"]
PLATFORM_P = [0.40, 0.25, 0.20, 0.15]

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata"]
CITY_VARIANTS = [
    "Mumbai", "mumbai", "MUMBAI",
    "Bengaluru", "Bangalore", "BENGALURU",
    "Delhi", "New Delhi", "DELHI",
    "Chennai", "CHENNAI", "Madras",
    "Hyderabad", "HYDERABAD", "Hyd",
]
COUNTRY_VARIANTS = ["IN", "India", "INDIA", "DE", "Germany", "PT", "Portugal"]
CHANNEL_VARIANTS = ["social", "Social", "SOCIAL", "social media", "SM"]
DEVICE_VARIANTS = ["mobile", "Mobile", "MOBILE", "android", "iOS", "web"]

LANGUAGES = ["en", "hi", "hinglish", "pt", "de"]
LANG_P = [0.55, 0.12, 0.18, 0.08, 0.07]

NORMAL_TEXT = [
    "The festival atmosphere is amazing today.",
    "Waiting for the next set to start.",
    "The music here is so good.",
    "Just arrived at the venue.",
    "Having a great time with friends.",
    "The weather is perfect for the festival.",
    "That performance was incredible.",
    "This place is packed but the vibe is great.",
    "Not bad at all actually, having a good time.",
    "This is sick 🔥",
    "great, another hour in this queue 🙃",
    "आज का माहौल बहुत अच्छा है।",
    "Aaj festival ka vibe bahut mast hai yaar.",
    "O ambiente do festival está incrível.",
    "Die Stimmung beim Festival ist unglaublich.",
]

IRRELEVANT_KESTREL = [
    "Saw a kestrel hovering over the field this morning.",
    "The kestrel is back near the old bridge.",
    "Looking for a used Kestrel car this weekend.",
    "Kestrel was the surname on the team sheet.",
    "That Kestrel documentary was brilliant.",
    "Anyone know where to photograph a kestrel?",
]

RELEVANT_KESTREL = [
    "Kestrel festival is unreal tonight.",
    "Kestrel stage is absolutely packed.",
    "Kestrel drinks everywhere at the venue.",
    "The Kestrel event team needs to fix this.",
    "Having a great time at Kestrel.",
    "Kestrel crowd is getting huge near gate three.",
]

NEGATIVE_GENERIC = [
    "This is awful.",
    "What a mess.",
    "Why is nobody fixing this?",
    "This is ridiculous.",
    "People are getting frustrated.",
    "Not happy about this at all.",
]

POSITIVE_GENERIC = [
    "Amazing performance!",
    "Best set of the night!",
    "This is incredible!",
    "What a surprise!",
    "Absolutely loving this!",
    "Best festival moment so far!",
]

EVENTS = [
    {
        "incident_id": "INC_QUEUE_CRUSH",
        "incident_type": "queue_crush",
        "name": "Queue crush at gate three",
        "start": datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 4, 18, 12, 40, tzinfo=timezone.utc),
        "sentiment": "negative",
        "volume": 1200,
    },
    {
        "incident_id": "INC_SOUND_FAILURE",
        "incident_type": "sound_failure",
        "name": "Main stage sound failure",
        "start": datetime(2026, 4, 18, 15, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 4, 18, 15, 12, tzinfo=timezone.utc),
        "sentiment": "negative",
        "volume": 9000,
    },
    {
        "incident_id": "INC_SPONSOR_ANNOUNCEMENT",
        "incident_type": "sponsor_announcement",
        "name": "Sponsor announcement reaction",
        "start": datetime(2026, 4, 18, 17, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 4, 18, 18, 30, tzinfo=timezone.utc),
        "sentiment": "negative",
        "volume": 18000,
    },
    {
        "incident_id": "INC_PAYMENT_OUTAGE",
        "incident_type": "payment_outage",
        "name": "Payment outage at bars",
        "start": datetime(2026, 4, 18, 19, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 4, 18, 19, 30, tzinfo=timezone.utc),
        "sentiment": "neutral",
        "volume": 3500,
    },
    {
        "incident_id": "INC_POSITIVE_DECOY_1",
        "incident_type": "positive_decoy",
        "name": "Surprise guest appearance",
        "start": datetime(2026, 4, 18, 13, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 4, 18, 13, 30, tzinfo=timezone.utc),
        "sentiment": "positive",
        "volume": 10000,
    },
    {
        "incident_id": "INC_POSITIVE_DECOY_2",
        "incident_type": "positive_decoy",
        "name": "Positive crowd reaction",
        "start": datetime(2026, 4, 18, 18, 0, tzinfo=timezone.utc),
        "end": datetime(2026, 4, 18, 18, 40, tzinfo=timezone.utc),
        "sentiment": "positive",
        "volume": 8000,
    },
]

def choose_ts(start, end, n):
    """Natural daily rhythm; vectorised sampling."""
    seconds = int((end - start).total_seconds())
    # approximate daytime/evening activity weighting
    hours = np.arange(seconds) / 3600.0
    # sampling minutes is enough for 1.1M rows; seconds added afterward
    minutes = max(1, seconds // 60)
    m = np.arange(minutes)
    h = ((start.hour * 60 + m) / 60.0) % 24
    w = np.where((h >= 0) & (h < 6), .25,
        np.where((h < 9), .6,
        np.where((h < 12), 1.0,
        np.where((h < 17), 1.25,
        np.where((h < 22), 1.5, .75)))))
    w = w / w.sum()
    chosen = rng.choice(m, size=n, p=w)
    secs = rng.integers(0, 60, size=n)
    return [start + timedelta(minutes=int(x), seconds=int(s))
            for x, s in zip(chosen, secs)]

def make_authors():
    ids = np.arange(1, AUTHOR_COUNT + 1)
    ages = rng.integers(30, 8 * 365, AUTHOR_COUNT)
    followers = np.maximum(1, rng.lognormal(4.5, 2.0, AUTHOR_COUNT).astype(int))
    following = np.maximum(1, rng.lognormal(4.0, 1.5, AUTHOR_COUNT).astype(int))
    freq = np.round(rng.lognormal(.2, 1.0, AUTHOR_COUNT), 2)
    creation = [EVENT_START - timedelta(days=int(x)) for x in ages]

    df = pd.DataFrame({
        "author_id": ids,
        "account_creation_date": creation,
        "follower_count": followers,
        "following_count": following,
        "posting_frequency": freq,
        "has_picture": rng.random(AUTHOR_COUNT) < .87,
        "has_bio": rng.random(AUTHOR_COUNT) < .78,
        "country": rng.choice(COUNTRY_VARIANTS, AUTHOR_COUNT),
        "city": rng.choice(CITY_VARIANTS, AUTHOR_COUNT),
        "device_type": rng.choice(DEVICE_VARIANTS, AUTHOR_COUNT),
        "channel": rng.choice(CHANNEL_VARIANTS, AUTHOR_COUNT),
        "email": [f"user{i}@example.com" for i in ids],
    })

    # ~0.5% internal QA/test accounts
    qa_n = int(round(AUTHOR_COUNT * .005))
    qa_idx = rng.choice(AUTHOR_COUNT, qa_n, replace=False)
    qa_ids = df.iloc[qa_idx]["author_id"].tolist()
    for j, idx in enumerate(qa_idx):
        df.loc[idx, "email"] = f"qa+{j+1:03d}@tailwyndz.test"
        df.loc[idx, "posting_frequency"] = 999.0
        df.loc[idx, "follower_count"] = 1_000_000
    return df, qa_ids

def base_posts(authors, n, start, end, seed_offset=0):
    r = np.random.default_rng(SEED + seed_offset)
    idx = r.integers(0, len(authors), n)
    a = authors.iloc[idx].reset_index(drop=True)
    platforms = r.choice(PLATFORMS, n, p=PLATFORM_P)
    languages = r.choice(LANGUAGES, n, p=LANG_P)
    ts = choose_ts(start, end, n)

    text = []
    relevant = np.zeros(n, dtype=bool)
    for i in range(n):
        if r.random() < .30 and "Kestrel" in (t := str(r.choice(RELEVANT_KESTREL))):
            # Roughly 30% of Kestrel-word posts will later be irrelevant.
            pass
        if r.random() < .11:
            t = str(r.choice(IRRELEVANT_KESTREL))
            relevant[i] = False
        elif r.random() < .13:
            t = str(r.choice(RELEVANT_KESTREL))
            relevant[i] = True
        else:
            t = str(r.choice(NORMAL_TEXT))
        if r.random() < .08:
            t += " " + str(r.choice(["😂", "😭", "🔥", "🙃", "😡", "❤️"]))
        text.append(t)

    followers = a["follower_count"].to_numpy()
    likes = r.poisson(np.maximum(1, followers / 100))
    reshares = r.poisson(np.maximum(.2, followers / 500))
    replies = r.poisson(np.maximum(.2, followers / 700))

    geo = [
        str(r.choice(CITIES)) if p != "platform_b" else None
        for p in platforms
    ]

    return pd.DataFrame({
        "post_id": [f"P{i:09d}" for i in range(n)],
        "platform": platforms,
        "author_id": a["author_id"].to_numpy(),
        "timestamp": ts,
        "text": text,
        "language": languages,
        "geo": geo,
        "follower_count": followers,
        "likes": likes,
        "reshares": reshares,
        "replies": replies,
        "_truth_source": "normal",
    }), relevant

def incident_posts(authors, incident, r):
    n = incident["volume"]
    idx = r.integers(0, len(authors), n)
    a = authors.iloc[idx].reset_index(drop=True)
    ts = choose_ts(incident["start"], incident["end"], n)
    platforms = r.choice(PLATFORMS, n, p=PLATFORM_P)
    texts = []

    for _ in range(n):
        kind = incident["incident_type"]
        if kind == "queue_crush":
            t = str(r.choice([
                "Gate three is completely jammed.",
                "Stuck in the queue at gate three.",
                "People are getting crushed at gate three.",
                "great, another hour in this queue 🙃",
            ]))
        elif kind == "sound_failure":
            t = str(r.choice([
                "The main stage sound has completely failed.",
                "No sound at the main stage!",
                "What happened to the audio?",
                "We can't hear anything.",
            ]))
        elif kind == "sponsor_announcement":
            t = str(r.choice([
                "Why did Kestrel make this sponsor announcement?",
                "Not happy about the sponsor decision.",
                "This sponsor announcement is disappointing.",
                "Kestrel really thought this was a good idea?",
            ]))
        elif kind == "payment_outage":
            t = str(r.choice([
                "Card payment isn't working at the bar.",
                "Cash only right now, payment terminals are down.",
                "The card machine is not accepting payments.",
                "Can't pay for drinks by card.",
            ]))
        else:
            t = str(r.choice(POSITIVE_GENERIC))
        texts.append(t)

    # Queue posts deliberately skew toward low-follower authors.
    if incident["incident_type"] == "queue_crush":
        low = authors.sort_values("follower_count").head(max(1000, len(authors)//8))
        a = low.iloc[r.integers(0, len(low), n)].reset_index(drop=True)

    followers = a["follower_count"].to_numpy()
    return pd.DataFrame({
        "post_id": [f"I_{incident['incident_id']}_{i:07d}" for i in range(n)],
        "platform": platforms,
        "author_id": a["author_id"].to_numpy(),
        "timestamp": ts,
        "text": texts,
        "language": r.choice(["en", "hinglish", "hi"], n, p=[.6,.3,.1]),
        "geo": [str(r.choice(CITIES)) if p != "platform_b" else None for p in platforms],
        "follower_count": followers,
        "likes": r.poisson(np.maximum(1, followers / 100)),
        "reshares": r.poisson(np.maximum(.2, followers / 500)),
        "replies": r.poisson(np.maximum(.2, followers / 700)),
        "_truth_source": incident["incident_id"],
    })

def automated_posts(authors, r):
    """Three automated types; ground truth kept outside raw data."""
    # Bot volume is added on top of organic traffic; target 7.5% of the
    # eventual combined stream. This keeps the final share inside the
    # assessment's 6-9% requirement.
    organic_n = BASELINE_POSTS + EVENT_NORMAL_POSTS + sum(e["volume"] for e in EVENTS) + 18_000
    bot_n = int(round((0.075 / (1 - 0.075)) * organic_n))
    bot_ids = r.choice(authors["author_id"].to_numpy(), bot_n, replace=True)
    types = r.choice(["spam", "amplifier", "monitor"], bot_n, p=[.35, .45, .20])
    ts = choose_ts(EVENT_START, EVENT_END, bot_n)

    rows = []
    for i, (aid, typ) in enumerate(zip(bot_ids, types)):
        if typ == "spam":
            text = str(r.choice([
                "WIN FREE KESTREL NOW!!! CLICK HERE!!!",
                "Kestrel giveaway!!! Claim now!!!",
                "FREE DRINKS CLICK LINK NOW!!!",
            ]))
        elif typ == "amplifier":
            text = str(r.choice([
                "Kestrel festival is a disaster and nobody should attend.",
                "Avoid Kestrel, the festival is a complete failure.",
                "Kestrel is ruining the festival experience.",
            ]))
        else:
            text = str(r.choice([
                "Monitoring summary: Kestrel mentions increased.",
                "Automated summary: Kestrel activity is elevated.",
                "Brand monitoring update: Kestrel volume increased.",
            ]))
        rows.append({
            "post_id": f"B_{i:09d}",
            "platform": str(r.choice(PLATFORMS)),
            "author_id": int(aid),
            "timestamp": ts[i],
            "text": text,
            "language": "en",
            "geo": str(r.choice(CITIES)),
            "follower_count": int(authors.loc[authors["author_id"].eq(aid), "follower_count"].iloc[0]),
            "likes": int(r.poisson(2)),
            "reshares": int(r.poisson(1)),
            "replies": int(r.poisson(1)),
            "_bot_type": typ,
            "_truth_source": "automated",
        })
    return pd.DataFrame(rows)

def event_timeline():
    rows = [
        ["event", "Kestrel Festival", EVENT_START, EVENT_END],
        ["gate_opening", "Gate three opens", EVENT_START, EVENT_START + timedelta(minutes=30)],
        ["set", "Main stage set", EVENT_START + timedelta(hours=2), EVENT_START + timedelta(hours=4)],
        ["sponsor", "Sponsor announcement", datetime(2026,4,18,17,tzinfo=timezone.utc), datetime(2026,4,18,18,30,tzinfo=timezone.utc)],
    ]
    rows += [[e["incident_id"], e["name"], e["start"], e["end"]] for e in EVENTS]
    return pd.DataFrame(rows, columns=["event_id","description","start","end"])

def inject_mess(posts, authors, r):
    """Inject requested data-quality problems into the raw outputs."""
    posts = posts.copy()
    authors = authors.copy()
    report = {}

    n = len(posts)

    # Exact duplicates ~2%
    dup_n = int(round(n * .02))
    exact = posts.sample(dup_n, random_state=SEED).copy()
    exact["post_id"] = exact["post_id"].astype(str) + "_DUP"
    posts = pd.concat([posts, exact], ignore_index=True)
    report["exact_duplicate_rows"] = dup_n

    # Near duplicates ~1%: same event a few seconds later, one field changed.
    near_n = int(round(n * .01))
    near = posts.sample(near_n, random_state=SEED + 1).copy()
    near["post_id"] = near["post_id"].astype(str) + "_NEAR"
    near["timestamp"] = pd.to_datetime(near["timestamp"]) + pd.to_timedelta(
        r.integers(1, 6, near_n), unit="s"
    )
    near["text"] = near["text"].astype(str) + " "
    posts = pd.concat([posts, near], ignore_index=True)
    report["near_duplicate_rows"] = near_n

    # -----------------------------------------------------------------------
    # Missingness
    # -----------------------------------------------------------------------
    # The assessment requires approximately 4-9% missingness in at least
    # three columns, while the missingness must NOT be random.
    #
    # We therefore create a target overall rate for each field and make the
    # probability higher for a particular platform/source.
    # -----------------------------------------------------------------------

    missingness_config = {
        "text": {
            "overall_rate": 0.05,
            "preferred_platform": "platform_c",
            "preferred_share": 0.70,
        },
        "language": {
            "overall_rate": 0.06,
            "preferred_platform": "platform_d",
            "preferred_share": 0.70,
        },
        "follower_count": {
            "overall_rate": 0.045,
            "preferred_platform": "platform_a",
            "preferred_share": 0.70,
        },
        "geo": {
            "overall_rate": 0.07,
            "preferred_platform": "platform_b",
            "preferred_share": 0.70,
        },
    }

    for col, config in missingness_config.items():

        target_n = int(round(
            len(posts) * config["overall_rate"]
        ))

        preferred_mask = (
            posts["platform"]
            .eq(config["preferred_platform"])
            & posts[col].notna()
        )

        preferred_idx = posts.index[preferred_mask].to_numpy()

        preferred_n = min(
            int(round(target_n * config["preferred_share"])),
            len(preferred_idx),
        )

        selected_preferred = (
            r.choice(
                preferred_idx,
                size=preferred_n,
                replace=False,
            )
            if preferred_n > 0
            else np.array([], dtype=int)
        )

        remaining_n = target_n - preferred_n

        other_idx = posts.index[
            (~posts.index.isin(selected_preferred))
            & posts[col].notna()
        ].to_numpy()

        selected_other = (
            r.choice(
                other_idx,
                size=remaining_n,
                replace=False,
            )
            if remaining_n > 0
            else np.array([], dtype=int)
        )

        selected_idx = np.concatenate([
            selected_preferred,
            selected_other,
        ])

        posts.loc[selected_idx, col] = np.nan

        report[f"missing_{col}_rows"] = int(
            len(selected_idx)
        )

        report[f"missing_{col}_rate"] = float(
            len(selected_idx) / len(posts)
        )

    # Category spelling variation.
    posts["geo"] = posts["geo"].map(
        lambda x: x if pd.isna(x) else str(r.choice(CITY_VARIANTS))
    )

    # Unmatched event author IDs 3-6%.
    unmatched_n = int(round(n * .04))
    unmatched_idx = r.choice(posts.index.to_numpy(), unmatched_n, replace=False)
    fake_ids = np.arange(9000001, 9000001 + unmatched_n)
    posts.loc[unmatched_idx, "author_id"] = fake_ids
    report["unmatched_event_author_ids"] = unmatched_n

    # Impossible values ~0.3%.
    bad_n = int(round(n * .003))
    bad_idx = r.choice(posts.index.to_numpy(), bad_n, replace=False)
    posts.loc[bad_idx[:bad_n//3], "likes"] = -5
    posts.loc[bad_idx[bad_n//3:2*bad_n//3], "replies"] = 10_000_000
    posts.loc[bad_idx[2*bad_n//3:], "follower_count"] = -20
    report["impossible_value_rows"] = bad_n

    # Mixed timestamp representations.
    ts = pd.to_datetime(posts["timestamp"], utc=True)
    formats = r.choice(["dmy","ymd","mdy","iso","epoch"], len(posts),
                        p=[.18,.18,.18,.28,.18])
    out = []
    for t, f in zip(ts, formats):
        if f == "dmy":
            out.append(t.strftime("%d-%m-%Y %H:%M:%S"))
        elif f == "ymd":
            out.append(t.strftime("%Y/%m/%d %H:%M:%S"))
        elif f == "mdy":
            out.append(t.strftime("%m-%d-%y %H:%M:%S"))
        elif f == "iso":
            out.append(t.isoformat())
        else:
            out.append(str(int(t.timestamp() * 1000)))
    posts["timestamp"] = out

    # Late-arriving events: keep event timestamp in a separate arrival_time.
    late_n = int(round(len(posts) * .03))
    late_idx = r.choice(posts.index.to_numpy(), late_n, replace=False)
    true_ts = pd.to_datetime(posts.loc[late_idx, "timestamp"], errors="coerce", utc=True)
    # Where mixed strings cannot parse directly, use a deterministic recent timestamp.
    arrival = pd.Timestamp(EVENT_END) + pd.to_timedelta(
        r.integers(9*24*3600 + 1, 14*24*3600, late_n), unit="s"
    )
    posts["arrival_time"] = pd.NaT
    posts.loc[late_idx, "arrival_time"] = arrival
    report["late_arriving_rows"] = late_n

    # Remove generator-only truth labels before analytical outputs.
    if "_truth_source" in posts.columns:
        posts = posts.drop(columns=["_truth_source"])
    if "_bot_type" in posts.columns:
        posts = posts.drop(columns=["_bot_type"])

    # Shuffle rows to ensure files are out of order.
    posts = posts.sample(frac=1, random_state=SEED + 22).reset_index(drop=True)

    # Internal QA authors are already injected.
    report["qa_accounts"] = int(round(len(authors) * .005))

    return posts, authors, report

def main():
    print("=" * 72)
    print("TAILWYNDZ ASSESSMENT 4 - FINAL SYNTHETIC DATA GENERATOR")
    print("=" * 72)

    authors, qa_ids = make_authors()
    print(f"Authors: {len(authors):,}")

    baseline, baseline_rel = base_posts(
        authors, BASELINE_POSTS, BASELINE_START, BASELINE_END, 10
    )
    event_normal, event_rel = base_posts(
        authors, EVENT_NORMAL_POSTS, EVENT_START, EVENT_END, 20
    )

    all_incidents = []
    incident_truth = []
    for i, e in enumerate(EVENTS):
        p = incident_posts(authors, e, np.random.default_rng(SEED + 100 + i))
        all_incidents.append(p)
        incident_truth.append({
            "incident_id": e["incident_id"],
            "incident_type": e["incident_type"],
            "name": e["name"],
            "start": e["start"].isoformat(),
            "end": e["end"].isoformat(),
            "true_sentiment": e["sentiment"],
            "post_count": len(p),
        })
    incidents = pd.concat(all_incidents, ignore_index=True)

    bots = automated_posts(authors, np.random.default_rng(SEED + 900))
    bot_truth_ids = bots["post_id"].tolist()
    bot_author_ids = bots["author_id"].astype(int).unique().tolist()

    # Add event-specific irrelevant football spike.
    football_n = 18_000
    football_ts = choose_ts(datetime(2026,4,18,18,tzinfo=timezone.utc),
                            datetime(2026,4,18,20,tzinfo=timezone.utc), football_n)
    football = event_normal.sample(football_n, replace=True,
                                   random_state=SEED + 33).copy()
    football["post_id"] = [f"F_{i:08d}" for i in range(football_n)]
    football["timestamp"] = football_ts
    football["text"] = rng.choice([
        "Kestrel scored again, what a terrible result.",
        "That Kestrel loss is painful.",
        "Kestrel bottled it again. Awful football.",
        "Can't believe the Kestrel result tonight.",
    ], football_n)
    football["_truth_source"] = "football_decoy"
    football_rel = np.zeros(football_n, dtype=bool)

    combined = pd.concat([baseline, event_normal, incidents, bots.drop(columns=["_bot_type"]), football],
                         ignore_index=True)

    # Assign globally unique IDs before quality injection.
    # Internal source labels are retained only long enough to build private
    # ground truth and are removed before writing analytical outputs.
    combined["post_id"] = [f"P{i:010d}" for i in range(1, len(combined)+1)]

    # Build final private truth from final IDs.
    automated_post_ids = combined.loc[
        combined["_truth_source"].eq("automated"), "post_id"
    ].astype(str).tolist()

    incident_post_ids = combined.loc[
        combined["_truth_source"].str.startswith("INC_", na=False), "post_id"
    ].astype(str).tolist()

    irrelevant_post_ids = combined.loc[
        combined["_truth_source"].eq("football_decoy"), "post_id"
    ].astype(str).tolist()

    combined, authors, dq_report = inject_mess(
        combined, authors, np.random.default_rng(SEED + 1200)
    )

    # Brand reference.
    brand_terms = pd.DataFrame({
        "term_type": ["brand", "product", "hashtag", "misspelling", "misspelling"],
        "term": ["Kestrel", "Kestrel Energy", "#KestrelFest", "KestrelBev", "Kestral"],
    })

    timeline = event_timeline()

    # Save CSV.
    combined.to_csv(FINAL_DIR / "posts.csv", index=False)

    # Save JSONL.
    with (FINAL_DIR / "posts.jsonl").open("w", encoding="utf-8") as f:
        for row in combined.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

   # Excel cannot store timezone-aware datetimes.
    authors_excel = authors.copy()

    if "account_creation_date" in authors_excel.columns:
        authors_excel["account_creation_date"] = (
            pd.to_datetime(
                authors_excel["account_creation_date"],
                utc=True,
            )
            .dt.tz_localize(None)
        )

    timeline_excel = timeline.copy()

    for column in ["start", "end"]:
        if column in timeline_excel.columns:
            timeline_excel[column] = (
                pd.to_datetime(
                    timeline_excel[column],
                    utc=True,
                )
                .dt.tz_localize(None)
            )

    with pd.ExcelWriter(
        FINAL_DIR / "authors_and_reference.xlsx",
        engine="openpyxl",
    ) as writer:

        authors_excel.to_excel(
            writer,
            sheet_name="authors",
            index=False,
        )

        brand_terms.to_excel(
            writer,
            sheet_name="brand_terms",
            index=False,
        )

        timeline_excel.to_excel(
            writer,
            sheet_name="event_timeline",
            index=False,
            startrow=3,
        )

    # Private ground truth.
    relevant_ids = incident_post_ids

    ground_truth = {
        "seed": SEED,
        "event": {
            "name": "Kestrel Festival",
            "start": EVENT_START.isoformat(),
            "end": EVENT_END.isoformat(),
            "baseline_start": BASELINE_START.isoformat(),
            "baseline_end": BASELINE_END.isoformat(),
        },
        "authors": {
            "total": len(authors),
            "automated_author_ids": bot_author_ids,
            "qa_author_ids": qa_ids,
        },
        "posts": {
            "total_generated_before_quality_injection": int(len(combined)),
            "relevant_post_ids": relevant_ids,
            "irrelevant_post_ids": irrelevant_post_ids,
            "automated_post_ids": automated_post_ids,
        },
        "incidents": incident_truth,
        "injected_data_quality_issues": dq_report,
        "notes": {
            "bot_truth_is_private": True,
            "raw_dataset_contains_no_bot_flag": True,
            "raw_dataset_contains_no_incident_label": True,
        },
    }

    with (GT_DIR / "final_ground_truth.json").open("w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    report = {
        "rows": int(len(combined)),
        "authors": int(len(authors)),
        "automated_post_count": int(len(automated_post_ids)),
        "automated_share": float(len(automated_post_ids) / len(combined)),
        "irrelevant_football_decoy_rows": int(len(irrelevant_post_ids)),
        "injected_data_quality": dq_report,
        "incidents": incident_truth,
        "outputs": [
            "posts.csv",
            "posts.jsonl",
            "authors_and_reference.xlsx",
            "event_timeline.csv",
            "brand_terms.csv",
        ],
    }

    timeline.to_csv(FINAL_DIR / "event_timeline.csv", index=False)
    brand_terms.to_csv(FINAL_DIR / "brand_terms.csv", index=False)

    with (GT_DIR / "generation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Final rows: {len(combined):,}")
    print(f"Automated rows: {len(bots):,}")
    bot_share = len(automated_post_ids) / len(combined)
    print(f"Automated share: {bot_share:.2%}")
    if not 0.06 <= bot_share <= 0.09:
        raise RuntimeError(
            f"Automated share {bot_share:.2%} is outside the required 6-9% range."
        )
    print(f"Exact duplicate injections: {dq_report['exact_duplicate_rows']:,}")
    print(f"Near duplicate injections: {dq_report['near_duplicate_rows']:,}")
    print(f"Late arrivals: {dq_report['late_arriving_rows']:,}")
    print(f"Saved final outputs under {FINAL_DIR}")
    print(f"Saved private ground truth under {GT_DIR}")

if __name__ == "__main__":
    main()