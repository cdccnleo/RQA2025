import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import time
import threading

import pytest

from src.data.preload.preloader import Preloader


def test_register_run_once_and_stats(monkeypatch):
    pl = Preloader()
    ran = {"ok": 0}

    def job():
        ran["ok"] += 1

    assert pl.register_task("t1", job, interval_seconds=1, enabled=True) is True
    assert pl.register_task("t1", job) is False  # 重复注册失败

    # 第一次应当执行
    res = pl.run_once()
    assert res.get("t1") is True
    assert ran["ok"] == 1

    # 立刻再次 run_once 因 interval 未到不应再执行
    res2 = pl.run_once()
    assert "t1" not in res2 or res2["t1"] is False

    stats = pl.get_stats()
    assert stats["total_tasks"] == 1
    assert stats["tasks"]["t1"]["success_count"] == 1


def test_enable_disable_and_scheduler_fast(monkeypatch):
    pl = Preloader()
    executed = {"count": 0}

    def job():
        executed["count"] += 1

    assert pl.register_task("t", job, interval_seconds=1, enabled=False)
    assert pl.run_once() == {}  # 未启用

    assert pl.enable_task("t") is True
    res = pl.run_once()
    assert res.get("t") is True

    # 加速调度循环，避免真实 sleep
    monkeypatch.setattr("src.data.preload.preloader.time.sleep", lambda *_: None)
    pl.start(poll_interval=0)  # 立即轮询（内部用 wait 控制）
    # 等待调度线程触发至少一次 run_once
    time.sleep(0.01)
    pl.stop()

    assert executed["count"] >= 1

import time
from src.data.preload.preloader import Preloader


def test_register_run_once_and_stats():
    pre = Preloader()
    hit = {"count": 0}

    def task_ok():
        hit["count"] += 1

    assert pre.register_task("t1", task_ok, interval_seconds=1, enabled=True) is True
    # 首次应执行
    res1 = pre.run_once()
    assert res1.get("t1") is True
    # 立即再次执行，interval 未到应不执行
    res2 = pre.run_once()
    assert "t1" not in res2
    # 等待超过间隔
    time.sleep(1.1)
    res3 = pre.run_once()
    assert res3.get("t1") is True
    st = pre.get_stats()
    assert st["total_tasks"] == 1
    assert st["tasks"]["t1"]["success_count"] >= 2


def test_disable_enable_unregister_and_scheduler():
    pre = Preloader()
    pre.register_task("t", lambda: None, interval_seconds=1, enabled=True)
    assert pre.disable_task("t") is True
    # disabled 时不执行
    assert pre.run_once() == {}
    assert pre.enable_task("t") is True
    assert "t" in pre.list_tasks()
    # 启动调度器并快速停止
    pre.start(poll_interval=1)
    time.sleep(0.2)
    pre.stop()
    assert pre.unregister_task("t") is True


