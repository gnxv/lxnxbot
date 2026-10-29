"""Tests for time parsing and basic pure functions from manager."""
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from app.graphs.manager import parse_datetime, parse_times, extract_topic

MSK = ZoneInfo("Europe/Moscow")

def test_parse_times():
    assert parse_times("в 10:30 и 18:00") == [time(10, 30), time(18, 0)]
    assert parse_times("без времени") == []

def test_parse_datetime_now():
    now = datetime(2026, 10, 29, 14, 0, tzinfo=MSK)
    assert parse_datetime("сейчас", now) == now

def test_parse_datetime_time_today():
    now = datetime(2026, 10, 29, 10, 0, tzinfo=MSK)
    result = parse_datetime("в 18:30", now)
    assert result.hour == 18 and result.minute == 30
    assert result.date() == now.date()

def test_parse_datetime_time_tomorrow_if_passed():
    now = datetime(2026, 10, 29, 20, 0, tzinfo=MSK)
    result = parse_datetime("в 10:00", now)
    assert result.date() == now.date() + timedelta(days=1)
    assert result.hour == 10

def test_extract_topic():
    topic = extract_topic("опубликуй пост про провисшую петлю")
    assert topic and "петл" in topic.lower()
    assert extract_topic("напиши про плёнка или эмаль в 19:00") is not None
