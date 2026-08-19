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
# Planted incident definitions
# ===========================================================================

INCIDENT_DEFINITIONS = [
    {
        "incident_id": "INC_QUEUE_CRUSH",
        "incident_type": "queue_crush",
        "name": "Queue crush",
        "start": EVENT_START + timedelta(hours=2),
        "duration_minutes": 40,
        "severity": "high",
        "sentiment": "negative",
        "expected_volume_multiplier": 3.0,
    },
    {
        "incident_id": "INC_SOUND_FAILURE",
        "incident_type": "sound_failure",
        "name": "Sound failure",
        "start": EVENT_START + timedelta(hours=5),
        "duration_minutes": 12,
        "severity": "high",
        "sentiment": "negative",
        "expected_volume_multiplier": 4.0,
    },
    {
        "incident_id": "INC_SPONSOR_ANNOUNCEMENT",
        "incident_type": "sponsor_announcement",
        "name": "Sponsor announcement",
        "start": EVENT_START + timedelta(hours=7),
        "duration_minutes": 90,
        "severity": "medium",
        "sentiment": "negative",
        "expected_volume_multiplier": 1.8,
    },
    {
        "incident_id": "INC_PAYMENT_OUTAGE",
        "incident_type": "payment_outage",
        "name": "Payment outage",
        "start": EVENT_START + timedelta(hours=9),
        "duration_minutes": 30,
        "severity": "medium",
        "sentiment": "neutral",
        "expected_volume_multiplier": 2.0,
    },
    {
        "incident_id": "INC_POSITIVE_DECOY_1",
        "incident_type": "positive_decoy",
        "name": "Positive performance reaction",
        "start": EVENT_START + timedelta(hours=3),
        "duration_minutes": 25,
        "severity": "low",
        "sentiment": "positive",
        "expected_volume_multiplier": 2.5,
    },
    {
        "incident_id": "INC_POSITIVE_DECOY_2",
        "incident_type": "positive_decoy",
        "name": "Positive crowd reaction",
        "start": EVENT_START + timedelta(hours=8),
        "duration_minutes": 35,
        "severity": "low",
        "sentiment": "positive",
        "expected_volume_multiplier": 2.2,
    },
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
# Incident-specific text
# ===========================================================================

INCIDENT_TEXT_TEMPLATES = {
    "queue_crush": {
        "en": [
            "Queue near the entrance is getting really bad.",
            "People are being pushed around near the gate.",
            "This queue is completely out of control.",
            "Too many people packed into this entrance.",
            "Security needs to manage the crowd here.",
        ],
        "hi": [
            "गेट के पास बहुत ज्यादा भीड़ हो गई है।",
            "यहाँ लाइन बहुत खराब हो गई है।",
            "गेट पर लोग बहुत ज्यादा धक्का दे रहे हैं।",
        ],
        "hinglish": [
            "Gate ke paas bahut zyada bheed ho gayi.",
            "Queue bilkul out of control ho gayi hai.",
            "Log entrance pe push kar rahe hain.",
        ],
    },

    "sound_failure": {
        "en": [
            "The sound has completely stopped.",
            "There is no audio on the main stage.",
            "Sound system just went down.",
            "We cannot hear anything from the stage.",
        ],
        "hi": [
            "मेन स्टेज की आवाज़ बंद हो गई है।",
            "साउंड सिस्टम काम नहीं कर रहा है।",
        ],
        "hinglish": [
            "Main stage ka sound band ho gaya.",
            "Sound system kaam nahi kar raha.",
            "Stage se kuch sunai nahi de raha.",
        ],
    },

    "sponsor_announcement": {
        "en": [
            "Why are they pushing this sponsor announcement again?",
            "This sponsor announcement is taking forever.",
            "Nobody came here for another sponsor speech.",
            "The sponsor promotion is getting annoying.",
        ],
        "hi": [
            "ये स्पॉन्सर अनाउंसमेंट फिर से क्यों हो रहा है?",
            "स्पॉन्सर का ये प्रमोशन बहुत लंबा हो गया।",
        ],
        "hinglish": [
            "Ye sponsor announcement phir se kyun ho raha hai?",
            "Sponsor promotion kaafi annoying ho gaya.",
            "Itna lamba sponsor segment kyun?",
        ],
    },

    "payment_outage": {
        "en": [
            "Card payments aren't going through at the food stalls.",
            "Payment terminals seem to be down.",
            "Having trouble paying for food right now.",
            "The card machine isn't responding.",
        ],
        "hi": [
            "फूड स्टॉल पर कार्ड पेमेंट नहीं हो रहा है।",
            "पेमेंट मशीन काम नहीं कर रही है।",
        ],
        "hinglish": [
            "Food stalls pe card payment nahi ho raha.",
            "Payment terminal down lag raha hai.",
            "Card machine respond nahi kar rahi.",
        ],
    },

    "positive_decoy": {
        "en": [
            "That performance was absolutely incredible!",
            "Best set of the festival so far!",
            "Everyone is singing along and it is amazing.",
            "This crowd has such great energy tonight!",
        ],
        "hi": [
            "ये परफॉर्मेंस कमाल की थी!",
            "आज का सेट सबसे अच्छा था!",
        ],
        "hinglish": [
            "Ye performance ekdum amazing thi!",
            "Aaj ka set sabse best tha!",
            "Crowd ki energy next level hai!",
        ],
    },
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
# Event-day timestamp generation
# ===========================================================================

def generate_event_timestamps(
    start: datetime,
    end: datetime,
    n_posts: int,
    rng: np.random.Generator,
) -> list[datetime]:
    """
    Generate timestamps for the 14-hour event period.

    Event activity follows a natural festival rhythm:
    - moderate activity when the event begins
    - increasing activity through the afternoon
    - strong evening activity
    - peak activity around the later performances
    - decline toward the end of the event
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

    for minute_offset in candidate_minutes:

        timestamp = start + timedelta(
            minutes=int(minute_offset)
        )

        elapsed_hours = (
            timestamp - start
        ).total_seconds() / 3600

        # ---------------------------------------------------------------
        # Event-day activity curve
        # ---------------------------------------------------------------

        if elapsed_hours < 2:
            # Opening period.
            weight = 1.0

        elif elapsed_hours < 5:
            # Afternoon build-up.
            weight = 1.5

        elif elapsed_hours < 8:
            # Stronger activity as performances progress.
            weight = 2.2

        elif elapsed_hours < 11:
            # Evening / major performances.
            weight = 3.0

        else:
            # Final part of the event.
            weight = 2.0

        weights[minute_offset] = weight

    weights /= weights.sum()

    selected_minutes = rng.choice(
        candidate_minutes,
        size=n_posts,
        p=weights,
    )

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
# Event-day post generation
# ===========================================================================

def generate_event_posts(
    authors: pd.DataFrame,
    n_posts: int,
    start: datetime,
    end: datetime,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate ordinary event-day social posts.

    At this stage these are still clean, non-incident posts.
    Incidents and automated activity will be added later.
    """

    author_indices = rng.integers(
        low=0,
        high=len(authors),
        size=n_posts,
    )

    selected_authors = authors.iloc[
        author_indices
    ].reset_index(drop=True)

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

    languages = [
        choose_language(rng)
        for _ in range(n_posts)
    ]

    timestamps = generate_event_timestamps(
        start=start,
        end=end,
        n_posts=n_posts,
        rng=rng,
    )

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

    follower_counts = (
        selected_authors[
            "follower_count"
        ].to_numpy()
    )

    # Event-day engagement is somewhat higher than baseline.
    likes = np.maximum(
        0,
        rng.poisson(
            lam=np.maximum(
                1,
                follower_counts / 70,
            ),
        ),
    )

    reshares = np.maximum(
        0,
        rng.poisson(
            lam=np.maximum(
                0.3,
                follower_counts / 350,
            ),
        ),
    )

    replies = np.maximum(
        0,
        rng.poisson(
            lam=np.maximum(
                0.3,
                follower_counts / 500,
            ),
        ),
    )

    geo = []

    for platform in platforms:

        if PLATFORM_CONFIG[
            platform
        ]["geo_available"]:

            geo.append(
                str(
                    rng.choice(CITIES)
                )
            )

        else:
            geo.append(None)

    posts = pd.DataFrame(
        {
            "post_id": [
                f"E{i:09d}"
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

def generate_clean_post_ids(
    n_posts: int,
    prefix: str,
) -> list[str]:
    """
    Generate ordinary-looking post IDs.

    The prefix is only used internally to avoid collisions while
    developing. Incident information is never encoded in the ID.
    """

    return [
        f"{prefix}{i:09d}"
        for i in range(1, n_posts + 1)
    ]

# ===========================================================================
# Incident post generation
# ===========================================================================

def generate_incident_posts(
    authors: pd.DataFrame,
    incident: dict,
    n_posts: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate posts associated with one planted incident.

    The incident identity is stored temporarily in an internal column.
    This column is removed before the raw event stream is saved.
    """

    incident_start = incident["start"]

    incident_end = (
        incident_start
        + timedelta(
            minutes=incident["duration_minutes"]
        )
    )

    author_indices = rng.integers(
        low=0,
        high=len(authors),
        size=n_posts,
    )

    selected_authors = authors.iloc[
        author_indices
    ].reset_index(drop=True)

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

    languages = [
        choose_language(rng)
        for _ in range(n_posts)
    ]

    timestamps = generate_timestamps(
        start=incident_start,
        end=incident_end,
        n_posts=n_posts,
        rng=rng,
    )

    incident_type = incident[
        "incident_type"
    ]

    texts = []

    for language in languages:

        language_templates = (
            INCIDENT_TEXT_TEMPLATES[
                incident_type
            ].get(
                language,
                INCIDENT_TEXT_TEMPLATES[
                    incident_type
                ]["en"],
            )
        )

        text = str(
            rng.choice(
                language_templates
            )
        )

        if rng.random() < 0.30:

            text += str(
                rng.choice(
                    [
                        " 😭",
                        " 😡",
                        " 😐",
                        " 🙄",
                        " 😍",
                        " 🔥",
                    ]
                )
            )

        texts.append(text)

    follower_counts = (
        selected_authors[
            "follower_count"
        ].to_numpy()
    )

    multiplier = incident[
        "expected_volume_multiplier"
    ]

    likes = np.maximum(
        0,
        rng.poisson(
            lam=np.maximum(
                1,
                follower_counts
                / 100
                * multiplier,
            ),
        ),
    )

    reshares = np.maximum(
        0,
        rng.poisson(
            lam=np.maximum(
                0.2,
                follower_counts
                / 500
                * multiplier,
            ),
        ),
    )

    replies = np.maximum(
        0,
        rng.poisson(
            lam=np.maximum(
                0.2,
                follower_counts
                / 700
                * multiplier,
            ),
        ),
    )

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

    posts = pd.DataFrame(
        {
            "post_id": [
                f"TEMP_{i:09d}"
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

            # ---------------------------------------------------------------
            # PRIVATE DEVELOPMENT LABEL
            # ---------------------------------------------------------------
            #
            # This is used ONLY while constructing ground truth.
            # It will be removed before the raw event stream is saved.
            #
            "_incident_id": incident[
                "incident_id"
            ],
        }
    )

    return posts

def combine_event_stream(
    normal_posts: pd.DataFrame,
    incident_posts: pd.DataFrame,
    incident_labels: list[dict],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Combine normal event activity and planted incident posts.

    The private incident label is used to create ground truth and is then
    removed before the event stream is saved.
    """

    normal_posts = normal_posts.copy()

    incident_posts = incident_posts.copy()

    # Normal posts are not associated with an incident.
    normal_posts["_incident_id"] = None

    # Combine both streams.
    combined = pd.concat(
        [
            normal_posts,
            incident_posts,
        ],
        ignore_index=True,
    )

    # Shuffle the combined event stream.
    combined = combined.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    # Generate final ordinary-looking IDs.
    combined["post_id"] = [
        f"P{i:09d}"
        for i in range(
            1,
            len(combined) + 1,
        )
    ]

    # -----------------------------------------------------------------------
    # Build private ground-truth mapping.
    # -----------------------------------------------------------------------

    updated_labels = []

    for _, row in combined.iterrows():

        incident_id = row[
            "_incident_id"
        ]

        if pd.notna(incident_id):

            incident_info = next(
                (
                    item
                    for item in incident_labels
                    if item[
                        "incident_id"
                    ]
                    == incident_id
                ),
                None,
            )

            if incident_info is not None:

                updated_labels.append(
                    {
                        "post_id": row[
                            "post_id"
                        ],

                        "incident_id":
                            incident_info[
                                "incident_id"
                            ],

                        "incident_type":
                            incident_info[
                                "incident_type"
                            ],
                    }
                )

    # Remove the private label before saving raw data.
    combined = combined.drop(
        columns=[
            "_incident_id"
        ]
    )

    return (
        combined,
        updated_labels,
    )

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

    # -----------------------------------------------------------------------
    # Initialise private ground truth
    # -----------------------------------------------------------------------

    ground_truth = initialise_ground_truth()

    # -----------------------------------------------------------------------
    # Generate authors
    # -----------------------------------------------------------------------

    authors = generate_authors()

    save_author_preview(
        authors
    )

    # -----------------------------------------------------------------------
    # Generate baseline development sample
    # -----------------------------------------------------------------------

    DEVELOPMENT_POST_COUNT = 10_000

    baseline_posts = generate_normal_posts(
        authors=authors,
        n_posts=DEVELOPMENT_POST_COUNT,
        start=BASELINE_START,
        end=BASELINE_END,
        rng=np.random.default_rng(SEED),
    )

    baseline_posts.to_csv(
        RAW_DIR
        / "posts_baseline_preview.csv",
        index=False,
    )

    print(
        f"\nGenerated "
        f"{len(baseline_posts):,} baseline posts."
    )

    print(
        "Saved to "
        f"{RAW_DIR / 'posts_baseline_preview.csv'}"
    )

    # -----------------------------------------------------------------------
    # Generate event-day development sample
    # -----------------------------------------------------------------------

    DEVELOPMENT_EVENT_POST_COUNT = 20_000

    event_posts = generate_event_posts(
        authors=authors,
        n_posts=DEVELOPMENT_EVENT_POST_COUNT,
        start=EVENT_START,
        end=EVENT_END,
        rng=np.random.default_rng(
            SEED + 1
        ),
    )

    print(
        f"Generated "
        f"{len(event_posts):,} normal event-day posts."
    )

    # -----------------------------------------------------------------------
    # Generate planted incidents
    # -----------------------------------------------------------------------

    incident_posts = []

    incident_labels = []

    for incident_index, incident in enumerate(
        INCIDENT_DEFINITIONS
    ):

        # Incident volume depends on duration and intensity.
        incident_post_count = int(
            250
            * (
                incident[
                    "duration_minutes"
                ]
                / 30
            )
            * incident[
                "expected_volume_multiplier"
            ]
        )

        generated_incident_posts = (
            generate_incident_posts(
                authors=authors,
                incident=incident,
                n_posts=incident_post_count,
                rng=np.random.default_rng(
                    SEED
                    + 100
                    + incident_index
                ),
            )
        )

        incident_posts.append(
            generated_incident_posts
        )

        incident_labels.append(
            {
                "incident_id": incident[
                    "incident_id"
                ],

                "incident_type": incident[
                    "incident_type"
                ],
            }
        )

    incident_posts = pd.concat(
        incident_posts,
        ignore_index=True,
    )

    print(
        f"Generated "
        f"{len(incident_posts):,} incident posts."
    )

    # -----------------------------------------------------------------------
    # Combine normal event activity + incidents
    # -----------------------------------------------------------------------

    event_stream, incident_post_mapping = (
        combine_event_stream(
            normal_posts=event_posts,
            incident_posts=incident_posts,
            incident_labels=incident_labels,
            rng=np.random.default_rng(
                SEED + 500
            ),
        )
    )

    # Save the combined event stream.
    event_stream.to_csv(
        RAW_DIR
        / "posts_event_combined_preview.csv",
        index=False,
    )

    print(
        f"Generated combined event stream "
        f"with {len(event_stream):,} posts."
    )

    print(
        "Saved to "
        f"{RAW_DIR / 'posts_event_combined_preview.csv'}"
    )

    # -----------------------------------------------------------------------
    # Update private ground truth
    # -----------------------------------------------------------------------

    ground_truth["posts"]["total_generated"] = (
        len(baseline_posts)
        + len(event_stream)
    )

    ground_truth["posts"]["relevant_post_ids"] = [
        item["post_id"]
        for item in incident_post_mapping
    ]

    ground_truth[
        "incidents"
    ] = []

    for incident in INCIDENT_DEFINITIONS:

        incident_mapping = [
            item
            for item in incident_post_mapping
            if item[
                "incident_id"
            ]
            == incident[
                "incident_id"
            ]
        ]

        incident_end = (
            incident["start"]
            + timedelta(
                minutes=incident[
                    "duration_minutes"
                ]
            )
        )

        ground_truth[
            "incidents"
        ].append(
            {
                "incident_id": incident[
                    "incident_id"
                ],

                "incident_type": incident[
                    "incident_type"
                ],

                "name": incident[
                    "name"
                ],

                "start": incident[
                    "start"
                ].isoformat(),

                "end": incident_end.isoformat(),

                "true_sentiment": incident[
                    "sentiment"
                ],

                "post_count": len(
                    incident_mapping
                ),
            }
        )

    # -----------------------------------------------------------------------
    # Save private ground truth
    # -----------------------------------------------------------------------

    save_ground_truth(
        ground_truth
    )

    # -----------------------------------------------------------------------
    # Stage completion message
    # -----------------------------------------------------------------------

    print(
        "\nStage 4A complete."
    )

    print(
        "Baseline activity, event-day activity, "
        "and planted incidents have been combined "
        "into one event stream."
    )

    print(
        "Incident labels remain in private "
        "ground truth only."
    )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    main()