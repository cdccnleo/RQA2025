#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充 ConnectionHealthChecker 的异常队列与队列跟踪分支覆盖。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List
from unittest.mock import Mock, patch

from src.infrastructure.utils.components.connection_health_checker import (
    ConnectionHealthChecker,
    PoolState,
)


class _BrokenQueue:
    """模拟 qsize 抛异常但保留 queue 属性的队列."""

    def __init__(self, items: Iterable) -> None:
        self.queue = list(items)

    def qsize(self) -> int:  # pragma: no cover - 预期引发异常路径
        raise RuntimeError("qsize boom")


class _TrackableQueue:
    """可精确控制长度的轻量队列."""

    def __init__(self, items: Iterable) -> None:
        self.queue: List = list(items)

    def qsize(self) -> int:
        return len(self.queue)


class _ConnectionInfo:
    """用于测试 is_connection_valid 的轻量连接包装."""

    def __init__(
        self,
        connection: object = None,
        last_used: datetime | None = None,
        created_at: datetime | None = None,
        connection_id: str = "conn-1",
    ) -> None:
        now = datetime.now()
        self.connection = connection
        self.last_used = last_used or now
        self.created_at = created_at or now
        self.connection_id = connection_id
        self.error_count = 0
        self.use_count = 0


def test_assess_pool_health_handles_qsize_error() -> None:
    checker = ConnectionHealthChecker()
    # available 队列 qsize 抛错 -> 回退到 queue 属性
    broken_available = _BrokenQueue([])
    broken_active = _BrokenQueue(["act-1", "act-2"])
    connections = ["act-1", "act-2"]

    result = checker._assess_pool_health(
        connections, broken_available, broken_active, max_size=10
    )

    assert result["raw_available"] == 0
    assert result["raw_active"] == 2
    # 没有可用连接但存在活动连接 -> CRITICAL
    assert result["state_enum"] is PoolState.CRITICAL


def test_assess_pool_health_tracks_queue_delta() -> None:
    checker = ConnectionHealthChecker()
    available = _TrackableQueue(["conn-1", "conn-2", "conn-3"])
    active_dict = {"active-1": object()}
    connections = ["conn-1", "conn-2", "conn-3", "active-1"]

    # 首次评估初始化跟踪信息
    first = checker._assess_pool_health(connections, available, active_dict, max_size=10)
    assert first["available"] == 3
    assert first["active"] == 1

    # 模拟连接被消费导致可用数量下降
    available.queue.pop()
    available.queue.pop()  # 剩余 1 个

    second = checker._assess_pool_health(connections, available, active_dict, max_size=10)

    # display_available 会扣除 queue_delta，降为 0
    assert second["available"] == 0
    # display_active 至少为活跃连接数量，且投影激活后提升到 3
    assert second["active"] >= 3
    assert second["raw_available"] == 1
    assert second["raw_active"] == 1


def test_is_connection_valid_passes_when_within_thresholds() -> None:
    now = datetime.now()
    validator = Mock(return_value=True)
    checker = ConnectionHealthChecker(connection_validator=validator)
    info = _ConnectionInfo(
        connection=object(),
        last_used=now - timedelta(seconds=10),
        created_at=now - timedelta(seconds=30),
    )

    assert checker.is_connection_valid(info, idle_timeout=60, max_lifetime=120)
    validator.assert_called_once_with(info.connection)


def test_is_connection_valid_returns_false_when_connection_missing() -> None:
    checker = ConnectionHealthChecker(connection_validator=Mock())
    info = _ConnectionInfo(connection=None)

    assert not checker.is_connection_valid(info, idle_timeout=60, max_lifetime=120)


def test_is_connection_valid_fails_on_idle_timeout() -> None:
    now = datetime.now()
    checker = ConnectionHealthChecker()
    info = _ConnectionInfo(
        connection=object(),
        last_used=now - timedelta(seconds=200),
        created_at=now - timedelta(seconds=50),
    )

    assert not checker.is_connection_valid(info, idle_timeout=30, max_lifetime=500)


def test_is_connection_valid_fails_on_max_lifetime() -> None:
    now = datetime.now()
    checker = ConnectionHealthChecker()
    info = _ConnectionInfo(
        connection=object(),
        last_used=now - timedelta(seconds=10),
        created_at=now - timedelta(seconds=1_000),
    )

    assert not checker.is_connection_valid(info, idle_timeout=120, max_lifetime=100)


def test_is_connection_valid_respects_validator_result() -> None:
    validator = Mock(return_value=False)
    checker = ConnectionHealthChecker(connection_validator=validator)
    info = _ConnectionInfo(connection=object())

    assert not checker.is_connection_valid(info, idle_timeout=60, max_lifetime=120)
    validator.assert_called_once()


def test_is_connection_valid_logs_on_validator_exception() -> None:
    validator = Mock(side_effect=RuntimeError("boom"))
    info = _ConnectionInfo(connection=object())

    with patch(
        "src.infrastructure.utils.components.connection_health_checker.logger"
    ) as mock_logger:
        checker = ConnectionHealthChecker(connection_validator=validator)
        assert not checker.is_connection_valid(info, idle_timeout=60, max_lifetime=120)
        mock_logger.error.assert_called_once()

