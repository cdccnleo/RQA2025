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
import pytest
from src.data.preload.preloader import Preloader


def test_register_invalid_task_name_or_func():
    pre = Preloader()
    # 空名称
    with pytest.raises(ValueError):
        pre.register_task("", lambda: None)
    # 非可调用
    with pytest.raises(ValueError):
        pre.register_task("bad", None)  # type: ignore[arg-type]


def test_task_failure_updates_stats_and_returns_false():
    pre = Preloader()
    def task_fail():
        raise RuntimeError("boom")
    assert pre.register_task("t_fail", task_fail, interval_seconds=1, enabled=True) is True
    # 首次应执行且失败
    res = pre.run_once()
    assert res.get("t_fail") is False
    st = pre.get_stats()
    tstat = st["tasks"]["t_fail"]
    assert tstat["last_status"] == "error"
    assert tstat["fail_count"] == 1
    assert isinstance(tstat["last_run_at"], float)
    # 立刻再次 run_once，不应重复执行（interval 未到）
    res2 = pre.run_once()
    assert "t_fail" not in res2
    # 等待到期后再次执行，失败计数增加
    time.sleep(1.1)
    res3 = pre.run_once()
    assert res3.get("t_fail") is False
    st2 = pre.get_stats()
    assert st2["tasks"]["t_fail"]["fail_count"] == 2


def test_disable_prevents_execution_even_if_interval_due():
    pre = Preloader()
    hit = {"n": 0}
    def task_ok():
        hit["n"] += 1
    assert pre.register_task("t", task_ok, interval_seconds=1, enabled=False) is True
    # disabled 时从未执行
    assert pre.run_once() == {}
    time.sleep(1.1)
    assert pre.run_once() == {}
    assert hit["n"] == 0


