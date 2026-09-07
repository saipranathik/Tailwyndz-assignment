import pandas as pd

from src.analyze_data import make_vss


def test_vss_returns_no_signal_below_minimum_posts():
    timestamps = pd.date_range(
        start="2026-04-18 10:00:00+00:00",
        periods=100,
        freq="min",
    )

    posts = pd.DataFrame({
        "timestamp": timestamps,
        "geo": ["Hyderabad"] * 100,
        "text": ["bad terrible failure"] * 100,
        "predicted_automated": [False] * 100,
        "relevance": ["relevant"] * 100,
    })

    result = make_vss(posts)

    assert not result.empty
    assert (result["qualifying_posts"] < 220).all()
    assert (result["signal"] == "NO SIGNAL").all()


if __name__ == "__main__":
    test_vss_returns_no_signal_below_minimum_posts()
    print("PASS: VSS NO SIGNAL test")