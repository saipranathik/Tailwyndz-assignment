"""
Tailwyndz Assessment 4 - Analysis Pipeline

Reads the deliberately messy final dataset without modifying the raw files.

Pipeline:
1. Parse mixed timestamps.
2. Audit data quality.
3. Join posts to authors.
4. Detect likely automated posts with transparent heuristics.
5. Classify Kestrel/event relevance with an auditable text rule set.
6. Score sentiment with a transparent multilingual lexicon + negation/sarcasm handling.
7. Compute VSS exactly from human + relevant posts in 15-minute windows.
8. Detect sustained <= -14 VSS breaches.
9. Evaluate bot/relevance/incident results against PRIVATE ground truth only after
   predictions are complete.

This is an assessment analysis script, not a cleaning script. Raw data is never
overwritten.
"""

from __future__ import annotations

from pathlib import Path
from datetime import timezone
import json
import re
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = BASE_DIR / "data" / "final"
GT_DIR = BASE_DIR / "data" / "ground_truth"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSTS_PATH = FINAL_DIR / "posts.csv"
AUTHORS_PATH = FINAL_DIR / "authors_and_reference.xlsx"
GT_PATH = GT_DIR / "final_ground_truth.json"

EVENT_START = pd.Timestamp("2026-04-18 10:00:00", tz="UTC")
EVENT_END = pd.Timestamp("2026-04-19 00:00:00", tz="UTC")
BASELINE_START = pd.Timestamp("2026-04-04 10:00:00", tz="UTC")
BASELINE_END = EVENT_START

VSS_THRESHOLD = -14.0
MIN_QUALIFYING_POSTS = 220
WINDOW_MINUTES = 15


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def parse_timestamps(series: pd.Series) -> pd.Series:
    """
    Parse the mixed timestamp formats present in the final dataset.

    Epoch-millisecond values are handled separately. All remaining
    timestamp strings are parsed using pandas' mixed-format parser.
    """

    s = series.astype("string").str.strip()

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns, UTC]",
    )

    # ---------------------------------------------------------------
    # Epoch milliseconds
    # ---------------------------------------------------------------

    epoch_mask = s.str.fullmatch(
        r"\d{12,14}",
        na=False,
    )

    if epoch_mask.any():
        result.loc[epoch_mask] = pd.to_datetime(
            pd.to_numeric(s.loc[epoch_mask], errors="coerce"),
            #format="mixed",
            unit="ms",
            errors="coerce",
            utc=True,
        )

    # ---------------------------------------------------------------
    # Everything else
    # ---------------------------------------------------------------

    # ISO timestamps
    remaining = result.isna()

    if remaining.any():
        mask = (
            remaining
            & s.str.match(
                r"^\d{4}-\d{2}-\d{2}",
                na=False,
            )
        )

        if mask.any():
            result.loc[mask] = pd.to_datetime(
                s.loc[mask],
                format="ISO8601",
                errors="coerce",
                utc=True,
            )

    # YYYY/MM/DD
    remaining = result.isna()

    if remaining.any():
        mask = (
            remaining
            & s.str.fullmatch(
                r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}",
                na=False,
            )
        )

        if mask.any():
            result.loc[mask] = pd.to_datetime(
                s.loc[mask],
                format="%Y/%m/%d %H:%M:%S",
                errors="coerce",
                utc=True,
            )

    # DD-MM-YYYY
    remaining = result.isna()

    if remaining.any():
        mask = (
            remaining
            & s.str.fullmatch(
                r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}",
                na=False,
            )
        )

        if mask.any():
            result.loc[mask] = pd.to_datetime(
                s.loc[mask],
                format="%d-%m-%Y %H:%M:%S",
                errors="coerce",
                utc=True,
            )

    # MM-DD-YY
    remaining = result.isna()

    if remaining.any():
        mask = (
            remaining
            & s.str.fullmatch(
                r"\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
                na=False,
            )
        )

        if mask.any():
            result.loc[mask] = pd.to_datetime(
                s.loc[mask],
                format="%m-%d-%y %H:%M:%S",
                errors="coerce",
                utc=True,
            )

    return result

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

CITY_MAP = {
    "mumbai": "Mumbai",
    "mumbai ": "Mumbai",
    "bombay": "Mumbai",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "hyderabad": "Hyderabad",
    "hyd": "Hyderabad",
    "chennai": "Chennai",
    "madras": "Chennai",
    "pune": "Pune",
    "kolkata": "Kolkata",
}

def normalize_city(value):
    if pd.isna(value):
        return pd.NA
    key = re.sub(r"\s+", " ", str(value).strip().lower())
    return CITY_MAP.get(key, str(value).strip().title())


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------

IRRELEVANT_PATTERNS = [
    r"\bbird\b",
    r"\bbirdwatch",
    r"\bphotograph",
    r"\bphotography\b",
    r"\bcar\b",
    r"\bused\b",
    r"\bsecondhand\b",
    r"\bfootball\b",
    r"\bscored\b",
    r"\bscore\b",
    r"\bloss\b",
    r"\bbottled\b",
    r"\bdocumentary\b",
    r"\btv\b",
    r"\btelevision\b",
    r"\bsurname\b",
    r"\bplayer\b",
    r"\bteam sheet\b",
]

RELEVANT_PATTERNS = [
    r"\bfestival\b",
    r"\bvenue\b",
    r"\bstage\b",
    r"\bgate\b",
    r"\bqueue\b",
    r"\bcrowd\b",
    r"\bconcert\b",
    r"\bset\b",
    r"\bsponsor\b",
    r"\bbar\b",
    r"\bpayment\b",
    r"\bdrink\b",
    r"\bmain stage\b",
    r"\bKestrelFest\b",
    r"#kestrelfest",
    r"\bkestrel energy\b",
]

def classify_relevance(text: str) -> str:
    """
    Three-way relevance:
      relevant      = clear Kestrel/event context
      irrelevant    = clear alternative meaning
      ambiguous     = brand word present but context insufficient

    Posts without Kestrel/event terms are not relevant to VSS.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"

    t = text.lower()

    has_kestrel = bool(re.search(r"\bkestrel\b", t))
    has_relevant_context = any(re.search(p, t) for p in RELEVANT_PATTERNS)
    has_irrelevant_context = any(re.search(p, t) for p in IRRELEVANT_PATTERNS)

    if has_kestrel and has_irrelevant_context and not has_relevant_context:
        return "irrelevant"

    if has_kestrel and has_relevant_context:
        return "relevant"

    # Strong event terms can establish relevance even when the brand is absent.
    event_terms = [
        "gate three", 
        "main stage", 
        "sound", 
        "payment terminal",
        "festival",
        "sponsor announcement", 
        "sponsor decision",
        "queue"
    ]
    if any(term in t for term in event_terms):
        return "relevant"

    if has_kestrel:
        return "ambiguous"

    return "not_target"


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

POSITIVE = {
    "amazing", "incredible", "great", "good", "love", "loving", "best",
    "brilliant", "awesome", "perfect", "happy", "excellent", "sick",
    "mast", "accha", "अच्छा", "अद्भुत", "बहुत अच्छा",
    "incrível", "excelente", "großartig", "wunderbar",
}

NEGATIVE = {
    "awful", "terrible", "bad", "horrible", "worst", "disaster",
    "failure", "failed", "broken", "crush", "crushed", "jammed",
    "stuck", "frustrated", "ridiculous", "angry", "disappointing",
    "disappointed", "painful", "loss", "lost", "ruining", "mess",
    "not", "down", "outage", "can't", "cannot",
    "बुरा", "खराब", "परेशान", "नहीं",
    "terrível", "falha", "ruim",
    "schlecht", "schlimm", "fehler",
}

EMOJI_POSITIVE = set("❤️❤🔥😍🥰😊😁🎉")
EMOJI_NEGATIVE = set("😡😠😭💔😞😤🙃")

NEGATION = {
    "not", "never", "no", "neither", "nor", "can't", "cannot",
    "dont", "don't", "isn't", "wasn't", "didn't", "hardly",
    "नहीं", "मत", "न", "não", "nicht", "kein",
}

SARCASM_PATTERNS = [
    r"great, another",
    r"great another",
    r"yeah right",
    r"thanks for nothing",
    r"what a .* mess",
    r"really thought .* good idea",
    r"what were .* thinking",
]

def sentiment_score(text: str) -> int:
    """Return -1, 0, +1 using transparent lexical rules."""
    if not isinstance(text, str) or not text.strip():
        return 0

    t = text.lower()
    tokens = re.findall(r"[a-zA-ZÀ-ÿ\u0900-\u097F']+", t)

    score = 0

    for i, token in enumerate(tokens):
        if token in POSITIVE:
            multiplier = -1 if i > 0 and tokens[i - 1] in NEGATION else 1
            score += multiplier
        elif token in NEGATIVE:
            multiplier = -1 if i > 0 and tokens[i - 1] in NEGATION else 1
            score -= multiplier

    score += sum(1 for ch in text if ch in EMOJI_POSITIVE)
    score -= sum(1 for ch in text if ch in EMOJI_NEGATIVE)

    if any(re.search(p, t) for p in SARCASM_PATTERNS):
        # Explicitly handle the planted sarcastic queue construction.
        score = min(score, -1)

    if "not bad at all" in t:
        score = max(score, 1)

    if "this is sick" in t:
        score = max(score, 1)

    return int(np.sign(score))


def sentiment_label(score: int) -> str:
    return {-1: "negative", 0: "neutral", 1: "positive"}[score]


# ---------------------------------------------------------------------------
# Bot detection
# ---------------------------------------------------------------------------

def build_bot_features(
    posts: pd.DataFrame,
    authors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Post-level automated-traffic detector.

    The detector uses observable content patterns rather than private
    automated-author/post labels.

    Three transparent signal families are used:

      1. Promotional / spam language
      2. Negative amplification language
      3. Monitoring / automated-summary language

    These correspond to the observable forms of automated traffic
    described in the assessment.

    Ground-truth automated labels are NOT used here. They are used
    only later for evaluation.
    """

    p = posts.copy()

    # ------------------------------------------------------------
    # Normalize text
    # ------------------------------------------------------------

    p["text_norm"] = (
        p["text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # ------------------------------------------------------------
    # Promotional / spam signal
    # ------------------------------------------------------------

    spam_patterns = [
        r"\bwin free\b",
        r"\bgiveaway\b",
        r"\bclaim now\b",
        r"\bclick here\b",
        r"\bclick link\b",
        r"\bfree drinks\b",
    ]

    spam_pattern = "|".join(spam_patterns)

    p["spam_signal"] = (
        p["text_norm"]
        .str.contains(
            spam_pattern,
            regex=True,
            na=False,
        )
    )

    # ------------------------------------------------------------
    # Negative amplification signal
    # ------------------------------------------------------------

    amplification_patterns = [
        r"\bdisaster\b",
        r"\bcomplete failure\b",
        r"\bruining\b",
        r"\bavoid kestrel\b",
        r"\bnobody should attend\b",
    ]

    amplification_pattern = "|".join(
        amplification_patterns
    )

    p["amplification_signal"] = (
        p["text_norm"]
        .str.contains(
            amplification_pattern,
            regex=True,
            na=False,
        )
    )

    # ------------------------------------------------------------
    # Monitoring / automated-summary signal
    # ------------------------------------------------------------

    monitoring_patterns = [
        r"\bmonitoring summary\b",
        r"\bautomated summary\b",
        r"\bbrand monitoring\b",
        r"\bmentions increased\b",
        r"\bvolume increased\b",
        r"\bactivity is elevated\b",
    ]

    monitoring_pattern = "|".join(
        monitoring_patterns
    )

    p["monitoring_signal"] = (
        p["text_norm"]
        .str.contains(
            monitoring_pattern,
            regex=True,
            na=False,
        )
    )

    # ------------------------------------------------------------
    # Final post-level prediction
    # ------------------------------------------------------------

    p["predicted_automated"] = (
        p["spam_signal"]
        | p["amplification_signal"]
        | p["monitoring_signal"]
    )

    # ------------------------------------------------------------
    # Transparent score
    #
    # 1 = one automated-content family detected
    # 2/3 = multiple families detected
    #
    # Multiple signals are not expected to overlap in this dataset,
    # but retaining the score makes the detector auditable.
    # ------------------------------------------------------------

    p["automation_score"] = (
        p["spam_signal"].astype(int)
        + p["amplification_signal"].astype(int)
        + p["monitoring_signal"].astype(int)
    )

    return p

# ---------------------------------------------------------------------------
# VSS
# ---------------------------------------------------------------------------

def make_vss(posts: pd.DataFrame) -> pd.DataFrame:
    p = posts.copy()

    p["timestamp"] = parse_timestamps(p["timestamp"])
    p = p.dropna(subset=["timestamp"])

    p["market"] = p["geo"].map(normalize_city)
    p["sentiment_score"] = p["text"].map(sentiment_score)

    # Human + relevant only.
    qualifying = p[
        (~p["predicted_automated"])
        & (p["relevance"] == "relevant")
        & (p["market"].notna())
    ].copy()

    # Baseline: same market, 14 days before event.
    baseline = qualifying[
        (qualifying["timestamp"] >= BASELINE_START)
        & (qualifying["timestamp"] < BASELINE_END)
    ].copy()

    event = qualifying[
        (qualifying["timestamp"] >= EVENT_START)
        & (qualifying["timestamp"] < EVENT_END)
    ].copy()

    baseline_group = (
        baseline.groupby("market", dropna=False)["sentiment_score"]
        .agg(
            baseline_n="size",
            baseline_net=lambda x: 100 * (
                (x > 0).mean() - (x < 0).mean()
            ),
        )
        .reset_index()
    )

    # 15-minute event windows.
    event["window"] = event["timestamp"].dt.floor("15min")

    windows = (
        event.groupby(["window", "market"], dropna=False)["sentiment_score"]
        .agg(
            qualifying_posts="size",
            positive=lambda x: int((x > 0).sum()),
            negative=lambda x: int((x < 0).sum()),
            neutral=lambda x: int((x == 0).sum()),
        )
        .reset_index()
    )

    windows["current_net"] = 100 * (
        windows["positive"] / windows["qualifying_posts"]
        - windows["negative"] / windows["qualifying_posts"]
    )

    windows = windows.merge(
        baseline_group,
        on="market",
        how="left",
    )

    windows["vss"] = windows["current_net"] - windows["baseline_net"]

    windows["signal"] = np.where(
        windows["qualifying_posts"] < MIN_QUALIFYING_POSTS,
        "NO SIGNAL",
        np.where(
            windows["vss"] <= VSS_THRESHOLD,
            "BREACH",
            "NORMAL",
        ),
    )

    windows = windows.sort_values(["market", "window"]).reset_index(drop=True)

    # Consecutive breach windows.
    windows["breach_run"] = 0
    for market, idx in windows.groupby("market").groups.items():
        run = 0
        for i in idx:
            if windows.loc[i, "signal"] == "BREACH":
                run += 1
            else:
                run = 0
            windows.loc[i, "breach_run"] = run

    windows["escalation"] = windows["breach_run"] >= 3

    return windows


# ---------------------------------------------------------------------------
# Incident evaluation
# ---------------------------------------------------------------------------

def evaluate_incidents(vss: pd.DataFrame, ground_truth: dict) -> pd.DataFrame:
    rows = []

    for incident in ground_truth["incidents"]:
        start = pd.Timestamp(incident["start"])
        end = pd.Timestamp(incident["end"])

        # ---------------------------------------------------------------
        # Detection attribution
        # ---------------------------------------------------------------
        #
        # Incident attribution is evaluated separately from the VSS rules.
        # An incident is considered detected when a NEW escalation begins
        # during the incident window.
        #
        # breach_run == 3 identifies the first escalation window.
        # This prevents an escalation that began before an incident from
        # being incorrectly attributed to that later incident.
        # ---------------------------------------------------------------

        attributed = vss[
            (vss["window"] >= start)
            & (vss["window"] < end)
            & (vss["escalation"])
            & (vss["breach_run"]==3)
        ]

        prior_escalation = vss[
            (vss["window"] < start)
            & (vss["escalation"])
        ]

        if attributed.empty:
            detected = False
            first_detection = pd.NaT
            delay_minutes = np.nan
        else:
            detected = True
            first_detection = attributed["window"].min()
            delay_minutes = (
                first_detection - start
            ).total_seconds() / 60

        rows.append({
            "incident_id": incident["incident_id"],
            "incident_type": incident["incident_type"],
            "start": start,
            "end": end,
            "truth_post_count": incident["post_count"],
            "detected": detected,
            "first_detection": first_detection,
            "detection_delay_minutes": delay_minutes,
            "prior_escalation_at_start": not prior_escalation.empty,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TAILWYNDZ ASSESSMENT 4 - ANALYSIS PIPELINE")
    print("=" * 72)

    print("Loading posts...")
    posts = pd.read_csv(POSTS_PATH)

    print(f"Raw rows: {len(posts):,}")

    # Data-quality audit before analytical transformations.
    quality = {
        "rows": int(len(posts)),
        "columns": posts.columns.tolist(),
        "missing_percent": (
            posts.isna().mean() * 100
        ).round(3).to_dict(),
        "unique_post_ids": int(posts["post_id"].nunique()),
        "duplicate_full_rows": int(posts.duplicated().sum()),
        "platform_counts": posts["platform"].value_counts().to_dict(),
    }

    # Timestamp audit.
    parsed_ts = parse_timestamps(posts["timestamp"])
    quality["unparsed_timestamps"] = int(parsed_ts.isna().sum())
    quality["parsed_timestamp_min"] = (
        parsed_ts.min().isoformat() if parsed_ts.notna().any() else None
    )
    quality["parsed_timestamp_max"] = (
        parsed_ts.max().isoformat() if parsed_ts.notna().any() else None
    )

    posts["parsed_timestamp"] = parsed_ts

    # -------------------------------------------------------------------
    # Extended data-quality audit
    # -------------------------------------------------------------------

    # Exact duplicate events.
    # The generator changes post_id to *_DUP, so post_id is excluded.
    exact_duplicate_cols = [
        "platform",
        "author_id",
        "timestamp",
        "text",
        "language",
        "geo",
        "follower_count",
        "likes",
        "reshares",
        "replies",
        "arrival_time",
    ]

    quality["exact_duplicate_event_rows"] = int(
        posts.duplicated(
            subset=exact_duplicate_cols,
            keep=False,
        ).sum()
    )

    # Impossible values injected by the generator.
    quality["impossible_likes"] = int(
        pd.to_numeric(posts["likes"], errors="coerce").lt(0).sum()
    )

    quality["extreme_replies"] = int(
        pd.to_numeric(posts["replies"], errors="coerce").gt(1_000_000).sum()
    )

    quality["impossible_followers"] = int(
        pd.to_numeric(
            posts["follower_count"],
            errors="coerce",
        ).lt(0).sum()
    )

    # Late-arriving events.
    # arrival_time is populated only for deliberately late rows.
    arrival_ts = pd.to_datetime(
        posts["arrival_time"],
        errors="coerce",
        utc=True,
    )

    quality["late_arriving_rows"] = int(
        (
            arrival_ts.notna()
            & posts["parsed_timestamp"].notna()
            & (arrival_ts > posts["parsed_timestamp"])
        ).sum()
    )

    # Timestamp representation audit.
    raw_ts = (
        posts["timestamp"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    quality["timestamp_format_counts"] = {
        "epoch_ms": int(
            raw_ts.str.fullmatch(
                r"\d{12,14}",
                na=False,
            ).sum()
        ),
        "iso": int(
            raw_ts.str.match(
                r"^\d{4}-\d{2}-\d{2}",
                na=False,
            ).sum()
        ),
        "ymd_slash": int(
            raw_ts.str.fullmatch(
                r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}",
                na=False,
            ).sum()
        ),
        "dmy_dash": int(
            raw_ts.str.fullmatch(
                r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}",
                na=False,
            ).sum()
        ),
        "mdy_dash": int(
            raw_ts.str.fullmatch(
                r"\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
                na=False,
            ).sum()
        ),
    }

    # Relevance.
    print("Classifying relevance...")
    posts["relevance"] = posts["text"].map(classify_relevance)

    relevance_counts = posts["relevance"].value_counts(dropna=False)
    quality["relevance_counts"] = relevance_counts.to_dict()

    kestrel_mask = posts["text"].fillna("").str.contains(
        r"\bkestrel\b", case=False, regex=True
    )
    quality["kestrel_mentions"] = int(kestrel_mask.sum())
    quality["kestrel_relevance_breakdown"] = (
        posts.loc[kestrel_mask, "relevance"]
        .value_counts(dropna=False)
        .to_dict()
    )

    quality["geo_raw_variants"] = (
        posts["geo"]
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
    )

    # Load authors from the workbook.

    # Load authors from the workbook.
    print("Loading authors...")
    authors = pd.read_excel(
        AUTHORS_PATH,
        sheet_name="authors",
    )

    # Unmatched author IDs.
    post_author_ids = pd.to_numeric(
        posts["author_id"],
        errors="coerce",
    )

    quality["author_category_variants"] = {
        "device_type": (
            authors["device_type"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        ),
        "channel": (
            authors["channel"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        ),
        "country": (
            authors["country"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        ),
        "city": (
            authors["city"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        ),
    }

    reference_author_ids = set(
        pd.to_numeric(
            authors["author_id"],
            errors="coerce",
        ).dropna()
    )

    quality["unmatched_author_id_rows"] = int(
        (~post_author_ids.isin(reference_author_ids)).sum()
    )

    # Bot detection.
    print("Detecting automated traffic...")
    enriched = build_bot_features(posts, authors)

    # Sentiment.
    print("Scoring sentiment...")
    enriched["sentiment_score"] = enriched["text"].map(sentiment_score)
    enriched["sentiment"] = enriched["sentiment_score"].map(sentiment_label)

    # Bot evaluation is deliberately performed only after predictions exist.
    with GT_PATH.open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    truth_bot_ids = set(map(str, ground_truth["posts"]["automated_post_ids"]))
    pred_bot_ids = set(
        enriched.loc[enriched["predicted_automated"], "post_id"].astype(str)
    )

    all_ids = set(enriched["post_id"].astype(str))

    tp = len(pred_bot_ids & truth_bot_ids)
    fp = len(pred_bot_ids - truth_bot_ids)
    fn = len(truth_bot_ids - pred_bot_ids)
    tn = len(all_ids - pred_bot_ids - truth_bot_ids)

    bot_eval = {
        "true_bot_count": len(truth_bot_ids),
        "predicted_bot_count": len(pred_bot_ids),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0,
        "error_rate": (fp + fn) / len(all_ids) if all_ids else 0,
    }

    # VSS.
    print("Computing VSS...")
    vss = make_vss(enriched)

    # Incident evaluation.
    incident_eval = evaluate_incidents(vss, ground_truth)
    

    # Save compact analytical outputs.
    quality_path = OUTPUT_DIR / "data_quality_audit.json"
    bot_path = OUTPUT_DIR / "bot_evaluation.json"
    relevance_path = OUTPUT_DIR / "relevance_summary.json"
    sentiment_path = OUTPUT_DIR / "sentiment_summary.json"

    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2, default=str)

    with bot_path.open("w", encoding="utf-8") as f:
        json.dump(bot_eval, f, indent=2)

    with relevance_path.open("w", encoding="utf-8") as f:
        json.dump(quality["kestrel_relevance_breakdown"], f, indent=2)

    sentiment_summary = (
        enriched[
            (~enriched["predicted_automated"])
            & (enriched["relevance"] == "relevant")
        ]["sentiment"]
        .value_counts()
        .to_dict()
    )

    with sentiment_path.open("w", encoding="utf-8") as f:
        json.dump(sentiment_summary, f, indent=2)

    vss.to_csv(OUTPUT_DIR / "vss_15min.csv", index=False)
    incident_eval.to_csv(OUTPUT_DIR / "incident_evaluation.csv", index=False)

    # Save a manageable scored sample, not another 1.2M-row duplicate.
    enriched.sample(
        min(50_000, len(enriched)),
        random_state=42,
    ).to_csv(
        OUTPUT_DIR / "scored_sample.csv",
        index=False,
    )

    # Console summary.
    print()
    print("DATA QUALITY")
    print(f"Rows: {len(posts):,}")
    print(f"Unparsed timestamps: {quality['unparsed_timestamps']:,}")
    print(f"Kestrel mentions: {quality['kestrel_mentions']:,}")

    print()
    print("RELEVANCE")
    print(relevance_counts)

    print()
    print("BOT DETECTION")
    print(f"Predicted automated: {len(pred_bot_ids):,}")
    print(f"True automated:      {len(truth_bot_ids):,}")
    print(f"Precision:           {bot_eval['precision']:.3f}")
    print(f"Recall:              {bot_eval['recall']:.3f}")
    print(f"Error rate:          {bot_eval['error_rate']:.3%}")

    print()
    print("SENTIMENT")
    print(sentiment_summary)

    print()
    print("VSS")
    print(f"15-minute windows: {len(vss):,}")
    print(f"NO SIGNAL windows: {(vss['signal'] == 'NO SIGNAL').sum():,}")
    print(f"BREACH windows:    {(vss['signal'] == 'BREACH').sum():,}")
    print(f"Escalation windows:{vss['escalation'].sum():,}")

    print()
    print("INCIDENT EVALUATION")
    print(incident_eval[
        ["incident_id", "incident_type", "detected", "detection_delay_minutes"]
    ].to_string(index=False))

    print()
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()