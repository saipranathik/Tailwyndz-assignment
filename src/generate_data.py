"""
Tailwyndz Propel Lateral Drive 2026
Assessment No. 4 - Problem 4
"Forty Minutes Into The Set"

Synthetic social-listening dataset generator.

Current development stage:
- reproducible configuration
- event timing
- author generation
- normal baseline post generation
- internal ground-truth tracking

The generated dataset will intentionally become messy in later stages.
Cleaning and analysis will happen after the synthetic data is created.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd


# ===========================================================================
# Configuration
# ===========================================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

AUTHOR_COUNT = 75_000

# ---------------------------------------------------------------------------
# Event timing
# ---------------------------------------------------------------------------

EVENT_DATE = datetime(
    2026,
    4,
    18,
    tzinfo=timezone.utc,
)

EVENT_START = EVENT_DATE.replace(
    hour=10,
    minute=0,
)

EVENT_END = EVENT_START + timedelta(
    hours=14
)

# 14-day period immediately before the event.
BASELINE_START = EVENT_START - timedelta(
    days=14
)

BASELINE_END = EVENT_START

BRAND_NAME = "Kestrel"

PLATFORMS = [
    "platform_a",
    "platform_b",
    "platform_c",
    "platform_d",
]


# ===========================================================================
# Platform configuration
# ===========================================================================

PLATFORM_CONFIG = {
    "platform_a": {
        "text_limit": 280,
        "geo_available": True,
    },
    "platform_b": {
        "text_limit": 500,
        "geo_available": False,
    },
    "platform_c": {
        "text_limit": 2200,
        "geo_available": True,
    },
    "platform_d": {
        "text_limit": 300,
        "geo_available": True,
    },
}


# ===========================================================================
# Reference values for normal posts
# ===========================================================================

LANGUAGES = [
    "en",
    "hi",
    "hinglish",
    "pt",
    "de",
]

CITIES = [
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
]


# ===========================================================================
# Normal text templates
# ===========================================================================

NORMAL_TEXT_TEMPLATES = {
    "en": [
        "The festival atmosphere is amazing today.",
        "Waiting for the next set to start.",
        "Crowd is getting bigger near the main stage.",
        "The music here is so good.",
        "Just arrived at the venue.",
        "Having a great time with friends.",
        "The weather is perfect for the festival.",
        "That performance was incredible.",
    ],

    "hi": [
        "आज का माहौल बहुत अच्छा है।",
        "अगला सेट शुरू होने का इंतज़ार है।",
        "यहाँ का म्यूजिक बहुत बढ़िया है।",
        "दोस्तों के साथ बहुत मज़ा आ रहा है।",
        "अभी अभी वेन्यू पहुँचे हैं।",
    ],

    "hinglish": [
        "Aaj festival ka vibe bahut mast hai.",
        "Next set ka wait kar raha hoon.",
        "Music kaafi accha hai yaar.",
        "Friends ke saath full enjoy kar rahe hain.",
        "Venue ka atmosphere kaafi energetic hai.",
    ],

    "pt": [
        "O ambiente do festival está incrível.",
        "Esperando o próximo show começar.",
        "A música está muito boa.",
        "Cheguei agora ao festival.",
    ],

    "de": [
        "Die Stimmung beim Festival ist unglaublich.",
        "Ich warte auf den nächsten Auftritt.",
        "Die Musik ist wirklich gut.",
        "Gerade beim Festival angekommen.",
    ],
}


# ===========================================================================
# Reproducibility
# ===========================================================================

def initialise_randomness() -> None:
    """
    Initialise Python and NumPy random number generators with the same seed.
    """

    random.seed(SEED)
    np.random.seed(SEED)


# ===========================================================================
# Directory setup
# ===========================================================================

def create_directories() -> None:
    """
    Create directories required by the generator.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    GROUND_TRUTH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ===========================================================================
# Author generation
# ===========================================================================

def generate_authors(
    n: int = AUTHOR_COUNT,
) -> pd.DataFrame:
    """
    Generate the author/account population.

    Important:
    The final authors table intentionally contains no explicit bot flag.

    Automated-account ground truth will be tracked separately.
    """

    rng = np.random.default_rng(SEED)

    author_ids = np.arange(
        1,
        n + 1,
    )

    # Account ages vary from a few weeks to several years.
    account_age_days = rng.integers(
        low=30,
        high=365 * 8,
        size=n,
    )

    account_creation_dates = [
        EVENT_DATE - timedelta(
            days=int(days)
        )
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


# ===========================================================================
# Language selection
# ===========================================================================

def choose_language(
    rng: np.random.Generator,
) -> str:
    """
    Choose a language using a non-uniform distribution.

    English is dominant, with Hindi, Hinglish, Portuguese and German
    represented as smaller groups.
    """

    languages = [
        "en",
        "hi",
        "hinglish",
        "pt",
        "de",
    ]

    probabilities = [
        0.55,
        0.12,
        0.18,
        0.08,
        0.07,
    ]

    return str(
        rng.choice(
            languages,
            p=probabilities,
        )
    )


# ===========================================================================
# Text generation
# ===========================================================================

def generate_post_text(
    language: str,
    platform: str,
    rng: np.random.Generator,
) -> str:
    """
    Generate ordinary non-incident social text.

    Later stages will add:
    - sarcasm
    - negation
    - emojis
    - irrelevant brand mentions
    - planted incidents
    - bot-generated text
    """

    templates = NORMAL_TEXT_TEMPLATES[language]

    text = str(
        rng.choice(templates)
    )

    # Add occasional emoji usage.
    if rng.random() < 0.25:

        text += str(
            rng.choice(
                [
                    " 🔥",
                    " 🎵",
                    " ❤️",
                    " 😍",
                    " 🙌",
                    " 🎉",
                ]
            )
        )

    # Respect the platform-specific text limit.
    limit = PLATFORM_CONFIG[platform]["text_limit"]

    return text[:limit]


# ===========================================================================
# Timestamp generation
# ===========================================================================

def generate_timestamps(
    start: datetime,
    end: datetime,
    n_posts: int,
    rng: np.random.Generator,
) -> list[datetime]:
    """
    Generate multiple timestamps using a realistic daily activity rhythm.

    The probability distribution is calculated ONCE and then sampled for
    all requested posts.

    This is deliberately written this way because the generator will
    eventually need to handle around 1.2 million posts.
    """

    total_minutes = int(
        (end - start).total_seconds() / 60
    )

    candidate_minutes = np.arange(
        total_minutes
    )

    weights = np.zeros(
        total_minutes,
        dtype=float,
    )

    # Assign activity weights to each minute.
    for minute_offset in candidate_minutes:

        timestamp = start + timedelta(
            minutes=int(minute_offset)
        )

        hour = timestamp.hour

        if 0 <= hour < 6:
            # Very low overnight activity.
            weight = 0.25

        elif 6 <= hour < 9:
            # Morning activity starts increasing.
            weight = 0.60

        elif 9 <= hour < 12:
            weight = 1.00

        elif 12 <= hour < 17:
            # Afternoon activity.
            weight = 1.25

        elif 17 <= hour < 22:
            # Evening is the busiest baseline period.
            weight = 1.50

        else:
            weight = 0.75

        weights[minute_offset] = weight

    # Convert weights into probabilities.
    weights /= weights.sum()

    # Sample all required minutes at once.
    selected_minutes = rng.choice(
        candidate_minutes,
        size=n_posts,
        p=weights,
    )

    # Add random seconds within each selected minute.
    seconds = rng.integers(
        low=0,
        high=60,
        size=n_posts,
    )

    timestamps = [
        start
        + timedelta(
            minutes=int(minute),
            seconds=int(second),
        )
        for minute, second in zip(
            selected_minutes,
            seconds,
        )
    ]

    return timestamps


# ===========================================================================
# Normal post generation
# ===========================================================================

def generate_normal_posts(
    authors: pd.DataFrame,
    n_posts: int,
    start: datetime,
    end: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate ordinary social posts for a specified time period.

    This is still the clean internal representation.

    Data-quality problems will be introduced later.
    """

    # Randomly select authors for each post.
    author_indices = rng.integers(
        low=0,
        high=len(authors),
        size=n_posts,
    )

    selected_authors = authors.iloc[
        author_indices
    ].reset_index(drop=True)

    # Assign platforms.
    platforms = rng.choice(
        PLATFORMS,
        size=n_posts,
        p=[
            0.40,
            0.25,
            0.20,
            0.15,
        ],
    )

    # Assign languages.
    languages = [
        choose_language(rng)
        for _ in range(n_posts)
    ]

    # Generate timestamps efficiently.
    timestamps = generate_timestamps(
        start=start,
        end=end,
        n_posts=n_posts,
        rng=rng,
    )

    # Generate text.
    texts = [
        generate_post_text(
            language,
            platform,
            rng,
        )
        for language, platform in zip(
            languages,
            platforms,
        )
    ]

    # Pull follower counts from selected authors.
    follower_counts = (
        selected_authors[
            "follower_count"
        ].to_numpy()
    )

    # Generate likes loosely correlated with follower count.
    likes = np.maximum(
        0,
        rng.poisson(
            lam=np.maximum(
                1,
                follower_counts / 100,
            ),
        ),
    )

    # Generate reshares.
    reshares = np.maximum(
        0,
        rng.poisson(
            lam=np.maximum(
                0.2,
                follower_counts / 500,
            ),
        ),
    )

    # Generate replies.
    replies = np.maximum(
        0,
        rng.poisson(
            lam=np.maximum(
                0.2,
                follower_counts / 700,
            ),
        ),
    )

    # Generate geographic information.
    geo = []

    for platform in platforms:

        if PLATFORM_CONFIG[
            platform
        ]["geo_available"]:

            geo.append(
                str(
                    rng.choice(
                        CITIES
                    )
                )
            )

        else:
            geo.append(None)

    # Build the internal posts table.
    posts = pd.DataFrame(
        {
            "post_id": [
                f"P{i:09d}"
                for i in range(
                    1,
                    n_posts + 1,
                )
            ],

            "platform": platforms,

            "author_id": selected_authors[
                "author_id"
            ],

            "timestamp": timestamps,

            "text": texts,

            "language": languages,

            "geo": geo,

            "follower_count": follower_counts,

            "likes": likes,

            "reshares": reshares,

            "replies": replies,
        }
    )

    return posts


# ===========================================================================
# Ground-truth bookkeeping
# ===========================================================================

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


# ===========================================================================
# Save intermediate output
# ===========================================================================

def save_author_preview(
    authors: pd.DataFrame,
) -> None:
    """
    Save an intermediate author file.

    This is only a development checkpoint.

    The final generator will write the required multi-format outputs later.
    """

    output_path = (
        RAW_DIR / "authors_preview.csv"
    )

    authors.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(authors):,} authors "
        f"to {output_path}"
    )


def save_ground_truth(
    ground_truth: dict,
) -> None:
    """
    Save private generator metadata outside the analytical dataset.
    """

    output_path = (
        GROUND_TRUTH_DIR
        / "generation_state.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            ground_truth,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved private ground truth "
        f"to {output_path}"
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:

    # Initialise reproducible randomness.
    initialise_randomness()

    # Create required directories.
    create_directories()

    print("=" * 70)
    print(
        "TAILWYNDZ ASSESSMENT 4 - "
        "SYNTHETIC DATA GENERATOR"
    )
    print("=" * 70)

    print(
        f"Random seed       : {SEED}"
    )

    print(
        f"Authors           : "
        f"{AUTHOR_COUNT:,}"
    )

    print(
        f"Baseline start    : "
        f"{BASELINE_START}"
    )

    print(
        f"Baseline end      : "
        f"{BASELINE_END}"
    )

    print(
        f"Event start       : "
        f"{EVENT_START}"
    )

    print(
        f"Event end         : "
        f"{EVENT_END}"
    )

    # Initialise private ground truth.
    ground_truth = (
        initialise_ground_truth()
    )

    # Generate authors.
    authors = generate_authors()

    save_author_preview(
        authors
    )

    # -----------------------------------------------------------------------
    # Development sample
    # -----------------------------------------------------------------------
    #
    # We intentionally start small while developing the generator.
    #
    # The final assessment dataset will be much larger.
    # -----------------------------------------------------------------------

    DEVELOPMENT_POST_COUNT = 10_000

    posts = generate_normal_posts(
        authors=authors,
        n_posts=DEVELOPMENT_POST_COUNT,
        start=BASELINE_START,
        end=BASELINE_END,
        rng=np.random.default_rng(SEED),
    )

    # Save development sample.
    posts.to_csv(
        RAW_DIR
        / "posts_baseline_preview.csv",
        index=False,
    )

    # Update private ground truth.
    ground_truth[
        "posts"
    ][
        "total_generated"
    ] = len(posts)

    save_ground_truth(
        ground_truth
    )

    print(
        f"\nGenerated "
        f"{len(posts):,} baseline posts."
    )

    print(
        "Saved to "
        f"{RAW_DIR / 'posts_baseline_preview.csv'}"
    )

    print(
        "\nStage 2 complete."
    )

    print(
        "Normal baseline post "
        "generation is working."
    )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    main()