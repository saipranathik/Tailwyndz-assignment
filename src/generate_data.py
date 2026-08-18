"""
Tailwyndz Propel Lateral Drive 2026
Assessment No. 4 - Problem 4
"Forty Minutes Into The Set"

Synthetic social-listening dataset generator.

This module currently establishes:
- reproducible configuration
- event timing
- author generation
- internal ground-truth tracking

The generated dataset is intentionally messy. Cleaning and analysis
will happen in later stages of the assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

AUTHOR_COUNT = 75_000

EVENT_DATE = datetime(2026, 4, 18, tzinfo=timezone.utc)
EVENT_START = EVENT_DATE.replace(hour=10, minute=0)
EVENT_END = EVENT_START + timedelta(hours=14)

BASELINE_START = EVENT_START - timedelta(days=14)
BASELINE_END = EVENT_START

BRAND_NAME = "Kestrel"

PLATFORMS = [
    "platform_a",
    "platform_b",
    "platform_c",
    "platform_d",
]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def initialise_randomness() -> None:
    """Initialise all random number generators with the same seed."""
    random.seed(SEED)
    np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def create_directories() -> None:
    """Create directories required by the generator."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Author generation
# ---------------------------------------------------------------------------

def generate_authors(n: int = AUTHOR_COUNT) -> pd.DataFrame:
    """
    Generate the author/account population.

    Important:
    The final authors table intentionally contains no explicit bot flag.
    Automated-account ground truth will be tracked separately.
    """

    rng = np.random.default_rng(SEED)

    author_ids = np.arange(1, n + 1)

    # Account ages vary from a few weeks to several years.
    account_age_days = rng.integers(
        low=30,
        high=365 * 8,
        size=n,
    )

    account_creation_dates = [
        EVENT_DATE - timedelta(days=int(days))
        for days in account_age_days
    ]

    followers = np.maximum(
        1,
        rng.lognormal(
            mean=4.5,
            sigma=2.0,
            size=n,
        ).astype(int),
    )

    following = np.maximum(
        1,
        rng.lognormal(
            mean=4.0,
            sigma=1.5,
            size=n,
        ).astype(int),
    )

    posting_frequency = np.round(
        rng.lognormal(
            mean=0.2,
            sigma=1.0,
            size=n,
        ),
        2,
    )

    has_picture = rng.random(n) < 0.87
    has_bio = rng.random(n) < 0.78

    authors = pd.DataFrame(
        {
            "author_id": author_ids,
            "account_creation_date": account_creation_dates,
            "follower_count": followers,
            "following_count": following,
            "posting_frequency": posting_frequency,
            "has_picture": has_picture,
            "has_bio": has_bio,
        }
    )

    return authors


# ---------------------------------------------------------------------------
# Ground-truth bookkeeping
# ---------------------------------------------------------------------------

def initialise_ground_truth() -> dict:
    """
    Create a private structure containing information that should not appear
    in the analytical dataset.
    """

    return {
        "seed": SEED,
        "event": {
            "name": "Kestrel Festival",
            "start": EVENT_START.isoformat(),
            "end": EVENT_END.isoformat(),
            "baseline_start": BASELINE_START.isoformat(),
            "baseline_end": BASELINE_END.isoformat(),
        },
        "authors": {
            "total": AUTHOR_COUNT,
            "automated_author_ids": [],
        },
        "posts": {
            "total_generated": 0,
            "relevant_post_ids": [],
            "irrelevant_post_ids": [],
            "automated_post_ids": [],
        },
        "incidents": [],
        "injected_data_quality_issues": {},
    }


# ---------------------------------------------------------------------------
# Save intermediate output
# ---------------------------------------------------------------------------

def save_author_preview(authors: pd.DataFrame) -> None:
    """
    Save an intermediate author file.

    This is only a development checkpoint. The final generator will write
    the required multi-format outputs later.
    """

    output_path = RAW_DIR / "authors_preview.csv"
    authors.to_csv(output_path, index=False)

    print(f"Saved {len(authors):,} authors to {output_path}")


def save_ground_truth(ground_truth: dict) -> None:
    """Save private generator metadata outside the analytical dataset."""

    output_path = GROUND_TRUTH_DIR / "generation_state.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            ground_truth,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved private ground truth to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    initialise_randomness()
    create_directories()

    print("=" * 70)
    print("TAILWYNDZ ASSESSMENT 4 - SYNTHETIC DATA GENERATOR")
    print("=" * 70)

    print(f"Random seed       : {SEED}")
    print(f"Authors           : {AUTHOR_COUNT:,}")
    print(f"Baseline start    : {BASELINE_START}")
    print(f"Baseline end      : {BASELINE_END}")
    print(f"Event start       : {EVENT_START}")
    print(f"Event end         : {EVENT_END}")

    ground_truth = initialise_ground_truth()

    authors = generate_authors()

    save_author_preview(authors)
    save_ground_truth(ground_truth)

    print("\nStage 1 complete.")
    print("Author generation is working.")
    print("Post generation will be added in the next stage.")


if __name__ == "__main__":
    main()