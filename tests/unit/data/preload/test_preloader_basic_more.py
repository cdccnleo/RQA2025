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
from src.data.preload.preloader import Preloader


def test_interval_normalized_and_stats_reflect_interval():
    pre = Preloader()
    pre.register_task("t0", lambda: None, interval_seconds=0, enabled=True)
    st = pre.get_stats()
    assert st["tasks"]["t0"]["interval_seconds"] == 1


def test_should_run_with_now_parameter_via_internal_access():
    pre = Preloader()
    hit = {"n": 0}
    def task_ok():
        hit["n"] += 1
    pre.register_task("t", task_ok, interval_seconds=5, enabled=True)
    # 首次 run_once 执行
    assert pre.run_once().get("t") is True
    # 直接访问内部任务以验证 should_run(now)
    task = pre._tasks["t"]  # type: ignore[attr-defined]
    now = task.last_run_at or time.time()
    # 提前 1 秒，不应执行
    assert task.should_run(now=now + 1) is False
    # 超过 interval，应执行
    assert task.should_run(now=now + 6) is True


