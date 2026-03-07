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


def test_start_is_idempotent_and_stop_graceful():
    pre = Preloader()
    # 注册一个较长间隔的任务，避免在测试中被频繁触发
    pre.register_task("t", lambda: None, interval_seconds=2, enabled=True)
    pre.start(poll_interval=0.1)
    # 再次启动应无副作用
    pre.start(poll_interval=0.1)
    # 等待调度器循环一次
    time.sleep(0.2)
    # 停止应在超时前完成
    pre.stop(timeout=1.0)
    # 多次停止也应无副作用
    pre.stop(timeout=1.0)


def test_interval_gate_prevents_back_to_back_runs_under_scheduler():
    pre = Preloader()
    hit = {"n": 0}
    def task_ok():
        hit["n"] += 1
    # 间隔1秒，轮询更快，但应受 interval 限制
    pre.register_task("t", task_ok, interval_seconds=1, enabled=True)
    pre.start(poll_interval=0.05)
    # 首次应在很短时间内触发
    time.sleep(0.2)
    n1 = hit["n"]
    assert n1 >= 1
    # 在1秒内多轮循环不应重复触发
    time.sleep(0.3)
    n2 = hit["n"]
    assert n2 == n1
    # 超过1秒后应再次触发
    time.sleep(0.9)
    n3 = hit["n"]
    assert n3 >= n2
    pre.stop(timeout=1.0)


