import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Kestrel Command Centre",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
VSS_FILE = BASE_DIR / "outputs" / "vss_15min.csv"

VSS_THRESHOLD = -14
MIN_POSTS = 220


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_vss():
    df = pd.read_csv(VSS_FILE)

    df["window"] = pd.to_datetime(df["window"], utc=True)

    numeric_cols = [
        "qualifying_posts",
        "positive",
        "negative",
        "neutral",
        "current_net",
        "baseline_net",
        "vss",
        "breach_run",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["escalation"] = (
        df["escalation"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return df.sort_values(
        ["window", "market"]
    ).reset_index(drop=True)


vss = load_vss()

markets = sorted(vss["market"].dropna().unique())
windows = sorted(vss["window"].dropna().unique())


# ============================================================
# STATUS
# ============================================================

def get_status(row):

    if row["qualifying_posts"] < MIN_POSTS:
        return "NO SIGNAL"

    if row["escalation"]:
        return "ESCALATE"

    if row["signal"] == "BREACH":
        return "BREACH"

    return "NORMAL"


def status_color(status):

    if status == "ESCALATE":
        return "#dc2626"

    if status == "BREACH":
        return "#f59e0b"

    if status == "NORMAL":
        return "#16a34a"

    return "#64748b"


def status_icon(status):

    return {
        "ESCALATE": "🔴",
        "BREACH": "🟠",
        "NORMAL": "🟢",
        "NO SIGNAL": "⚪",
    }.get(status, "⚪")


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>

/* Overall page */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1600px;
}

/* Main title */
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 2px;
    line-height: 1.35;
    padding: 6px 0 6px 0;
    overflow: visible;
}

/* Subtitle */
.subtitle {
    font-size: 15px;
    color: #94a3b8;
    margin-bottom: 18px;
}

/* Section headings */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 12px;
    margin-bottom: 10px;
}

/* Simulation time */
.time-label {
    font-size: 14px;
    font-weight: 600;
}

/* Market cards */
.market-card {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 18px;
    height: 180px;
    box-sizing: border-box;
}

.market-name {
    font-size: 17px;
    font-weight: 750;
    color: #111827;
    margin-bottom: 14px;
}

.vss-label {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 2px;
}

.vss-number {
    font-size: 32px;
    font-weight: 800;
    line-height: 1.1;
}

.market-status {
    font-size: 14px;
    font-weight: 750;
    margin-top: 8px;
}

.market-info {
    font-size: 12px;
    color: #64748b;
    margin-top: 9px;
}

/* Investigation box */
.detail-box {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 20px;
}

.detail-label {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 3px;
}

.detail-value {
    font-size: 21px;
    font-weight: 700;
    color: #111827;
}

/* Footer */
.boundary {
    background: #111827;
    border-radius: 8px;
    padding: 10px 14px;
    color: #cbd5e1;
    font-size: 12px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">KESTREL FESTIVAL</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'SOCIAL LISTENING COMMAND CENTRE &nbsp;•&nbsp; '
    'Historical event simulation'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIMULATION TIME
# ============================================================

st.markdown(
    '<div class="time-label">Simulation time</div>',
    unsafe_allow_html=True,
)

selected_window = st.select_slider(
    "Simulation time",
    options=windows,
    value=windows[-1],
    format_func=lambda x: x.strftime("%H:%M"),
    label_visibility="collapsed",
)

current = vss[
    vss["window"] == selected_window
].copy()

current["status"] = current.apply(
    get_status,
    axis=1,
)


# ============================================================
# SUMMARY
# ============================================================

escalations = int(
    (current["status"] == "ESCALATE").sum()
)

breaches = int(
    (current["status"] == "BREACH").sum()
)

no_signal = int(
    (current["status"] == "NO SIGNAL").sum()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Markets monitored", len(markets))
c2.metric("Escalations", escalations)
c3.metric("Breaches", breaches)
c4.metric("No signal", no_signal)


# ============================================================
# MARKET STATUS
# ============================================================

st.markdown(
    '<div class="section-title">Market status</div>',
    unsafe_allow_html=True,
)

cols = st.columns(5)

for col, market in zip(cols, markets):

    row = current[
        current["market"] == market
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    status = get_status(row)
    color = status_color(status)
    icon = status_icon(status)

    html = (
        '<div class="market-card">'

        f'<div class="market-name">{market}</div>'

        '<div class="vss-label">Verified Sentiment Shift</div>'

        f'<div class="vss-number" style="color:{color};">'
        f'{row["vss"]:.2f}'
        '</div>'

        f'<div class="market-status" style="color:{color};">'
        f'{icon} {status}'
        '</div>'

        '<div class="market-info">'
        f'{int(row["qualifying_posts"]):,} qualifying posts'
        f' &nbsp;•&nbsp; Run {int(row["breach_run"])}'
        '</div>'

        '</div>'
    )

    col.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# MARKET INVESTIGATION
# ============================================================

st.markdown(
    '<div class="section-title">Market investigation</div>',
    unsafe_allow_html=True,
)

selected_market = st.selectbox(
    "Select market",
    markets,
)

market_data = vss[
    (vss["market"] == selected_market)
    & (vss["window"] <= selected_window)
].copy()

selected_row = market_data[
    market_data["window"] == selected_window
]

if not selected_row.empty:

    selected_row = selected_row.iloc[0]

    status = get_status(selected_row)
    color = status_color(status)
    icon = status_icon(status)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            '<div class="detail-box">'
            '<div class="detail-label">STATUS</div>'
            f'<div class="detail-value" '
            f'style="color:{color};">'
            f'{icon} {status}'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="detail-box">'
            '<div class="detail-label">VSS</div>'
            f'<div class="detail-value">'
            f'{selected_row["vss"]:.2f}'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="detail-box">'
            '<div class="detail-label">QUALIFYING POSTS</div>'
            f'<div class="detail-value">'
            f'{int(selected_row["qualifying_posts"]):,}'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            '<div class="detail-box">'
            '<div class="detail-label">CONSECUTIVE BREACHES</div>'
            f'<div class="detail-value">'
            f'{int(selected_row["breach_run"])}'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# SYSTEM BOUNDARY
# ============================================================

st.markdown(
    """
<div class="boundary">
⚠ <b>System boundary:</b>
Social listening is an early-warning signal,
not a standalone incident detector.
</div>
""",
    unsafe_allow_html=True,
)