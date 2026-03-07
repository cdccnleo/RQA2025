from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.infrastructure.resource.core.event_storage import EventStorage


def test_store_and_retrieve_events():
    logger = MagicMock()
    storage = EventStorage(max_events=3, retention_hours=1, logger=logger)

    storage.store_event("alpha", {"value": 1})
    storage.store_event("beta", {"value": 2})

    all_events = storage.get_events()
    assert len(all_events) == 2
    assert all_events[0]["type"] == "alpha"
    assert all_events[1]["data"]["value"] == 2
    logger.log_debug.assert_called_with("已存储事件: beta")


def test_get_events_filter_and_limit():
    storage = EventStorage(max_events=5, retention_hours=1, logger=MagicMock())
    now = datetime.now()
    storage.store_event("alpha", 1, timestamp=now - timedelta(minutes=30))
    storage.store_event("alpha", 2, timestamp=now)
    storage.store_event("beta", 3, timestamp=now)

    alpha_events = storage.get_events(event_type="alpha")
    assert len(alpha_events) == 2

    recent_alpha = storage.get_events(event_type="alpha", since=now - timedelta(minutes=1))
    assert len(recent_alpha) == 1
    assert recent_alpha[0]["data"] == 2

    limited = storage.get_events(limit=2)
    assert len(limited) == 2
    assert limited[0]["data"] == 2  # most recent due to slicing


def test_cleanup_expired_events():
    logger = MagicMock()
    storage = EventStorage(max_events=5, retention_hours=1, logger=logger)
    now = datetime.now()

    storage.store_event("old", {}, timestamp=now - timedelta(hours=2))
    storage.store_event("fresh", {}, timestamp=now)

    storage.get_events()
    assert storage.get_event_count() == 1
    recent = storage.get_events()
    assert len(recent) == 1 and recent[0]["type"] == "fresh"
    logger.log_debug.assert_any_call("已清理过期事件: 1个")


def test_get_event_count_and_capacity_trim():
    storage = EventStorage(max_events=2, retention_hours=1, logger=MagicMock())
    storage.store_event("alpha", 1)
    storage.store_event("alpha", 2)
    storage.store_event("alpha", 3)  # deque maxlen trims oldest

    assert storage.get_event_count() == 2
    assert storage.get_event_count("alpha") == 2


def test_clear_events_specific_and_all():
    logger = MagicMock()
    storage = EventStorage(max_events=5, retention_hours=1, logger=logger)
    storage.store_event("alpha", 1)
    storage.store_event("beta", 2)

    storage.clear_events("alpha")
    assert storage.get_event_count() == 1
    logger.log_info.assert_called_with("已清除事件: alpha")

    storage.clear_events()
    assert storage.get_event_count() == 0
    logger.log_info.assert_called_with("已清除所有事件记录")


def test_get_storage_stats():
    storage = EventStorage(max_events=4, retention_hours=2, logger=MagicMock())
    now = datetime.now()
    storage.store_event("alpha", {}, timestamp=now - timedelta(minutes=30))
    storage.store_event("beta", {}, timestamp=now)

    stats = storage.get_storage_stats()

    assert stats["total_events"] == 2
    assert stats["max_capacity"] == 4
    assert stats["retention_hours"] == 2
    assert stats["utilization_percent"] == pytest.approx(50.0)
    assert stats["oldest_event_age_hours"] == pytest.approx(0.5, rel=0.05)

