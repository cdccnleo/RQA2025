#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_components 额外单测，覆盖去重、熔断、日志分片及工厂创建相关逻辑。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import pytest

from src.infrastructure.utils.components.optimized_components import (
    MarketDataDeduplicator,
    TradingHoursAwareCircuitBreaker,
    LogShardManager,
    OptimizedComponentFactory,
    OptimizedComponent,
)


def test_market_data_deduplicator_detects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    deduplicator = MarketDataDeduplicator(window_size=10)
    tick: Dict = {
        "symbol": "TEST",
        "price": 100.0,
        "volume": 50,
        "bid": [1, 2],
        "ask": [3, 4],
    }

    assert deduplicator.is_duplicate(tick) is False
    assert deduplicator.is_duplicate(dict(tick)) is True  # 相同数据视为重复

    tick["price"] = 101.0
    assert deduplicator.is_duplicate(tick) is False  # 数据变更不再重复


def test_trading_hours_aware_circuit_breaker_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = {
        "all_day": {"start": "00:00", "end": "23:59", "threshold": 0.7},
    }
    breaker = TradingHoursAwareCircuitBreaker(schedule)

    breaker.update_load(0.6)
    assert breaker.should_trigger() is False

    breaker.update_load(0.75)
    # 保证当前时间在时段内
    monkeypatch.setattr(
        "src.infrastructure.utils.components.optimized_components.datetime",
        datetime,
    )
    assert breaker.should_trigger() is True


def test_log_shard_manager_path_and_rotation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shard_path = tmp_path / "logs"
    manager = LogShardManager(
        base_path=str(shard_path),
        sharding_rules={"by_symbol": True, "by_date": {"format": "%Y%m%d"}},
    )

    entry = {"symbol": "ABC"}
    path = manager.get_shard_path(entry)
    assert path.parent.parent == shard_path
    assert path.parts[-2] == "ABC"

    # 不存在的文件不轮转
    assert manager._should_rotate(path) is False

    # 创建文件并模拟超大大小
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    monkeypatch.setattr(os.path, "getsize", lambda _: 200 * 1024 * 1024)
    assert manager._should_rotate(path) is True


def test_optimized_component_factory_creation() -> None:
    factory = OptimizedComponentFactory()
    component = factory.create_component(1)
    assert isinstance(component, OptimizedComponent)
    assert component.get_component_id() == 1

    info = factory.get_factory_info()
    assert info["factory_name"] == "OptimizedComponentFactory"
    assert set(info["supported_ids"]) == set(factory.SUPPORTED_COMPONENT_IDS)

    all_components = factory.create_all_components()
    assert len(all_components) == len(factory.SUPPORTED_COMPONENT_IDS)

    with pytest.raises(ValueError):
        factory.create_component(999)

