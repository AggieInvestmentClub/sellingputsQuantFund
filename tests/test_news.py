"""
Tests for src/data/news.py (DAT-08, DAT-09).

Run from the repo root:  python -m pytest tests/test_news.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.news import before_decision, first_versions  # noqa: E402


def test_first_versions_keeps_earliest_update():
    news = pd.DataFrame({
        "storyId": ["urn:newsml:reuters.com:20250514:nA1:1", "urn:newsml:reuters.com:20250514:nA1:0",
                    "urn:newsml:reuters.com:20250514:nB2:0"],
        "event_time": pd.to_datetime(["2025-05-14 15:10", "2025-05-14 15:00", "2025-05-14 16:00"], utc=True),
        "headline": ["update", "first", "other story"]})
    out = first_versions(news)
    assert out["headline"].tolist() == ["first", "other story"]


def test_before_decision_uses_new_york_close_with_daylight_saving():
    summer = pd.DataFrame({"event_time": pd.to_datetime(["2025-05-14 19:59", "2025-05-14 20:01"], utc=True),
                           "headline": ["a", "b"]})                  # close 16:00 EDT = 20:00 UTC
    winter = pd.DataFrame({"event_time": pd.to_datetime(["2025-01-14 20:30", "2025-01-14 21:01"], utc=True),
                           "headline": ["c", "d"]})                  # close 16:00 EST = 21:00 UTC
    assert before_decision(summer, "2025-05-14")["headline"].tolist() == ["a"]
    assert before_decision(winter, "2025-01-14")["headline"].tolist() == ["c"]


def test_empty_inputs_pass_through():
    empty = pd.DataFrame(columns=["storyId", "event_time"])
    assert first_versions(empty).empty and before_decision(empty, "2025-05-14").empty


def test_on_shock_day_runs_from_previous_close():
    from src.data.news import on_shock_day
    news = pd.DataFrame({"event_time": pd.to_datetime(
        ["2025-05-09 19:59", "2025-05-09 20:01", "2025-05-12 19:59", "2025-05-12 20:01"], utc=True),
        "headline": ["fri before close", "fri after close", "mon before close", "mon after close"]})
    out = on_shock_day(news, "2025-05-12")                       # Monday: window starts Friday's close
    assert out["headline"].tolist() == ["fri after close", "mon before close"]
