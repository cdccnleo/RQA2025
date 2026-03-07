"""
边界测试：preloader.py
测试边界情况和异常场景
"""
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


import pytest
import time
import threading
from unittest.mock import Mock, MagicMock
from src.data.preload.preloader import PreloadTask, Preloader


def test_preload_task_init():
    """测试 PreloadTask（初始化）"""
    task = PreloadTask("test_task", lambda: None, interval_seconds=60)
    
    assert task.name == "test_task"
    assert task.interval_seconds == 60
    assert task.enabled is True
    assert task.last_run_at is None
    assert task.last_status == "never"
    assert task.last_error is None
    assert task.success_count == 0
    assert task.fail_count == 0


def test_preload_task_init_invalid_name():
    """测试 PreloadTask（初始化，无效名称）"""
    with pytest.raises(ValueError, match="invalid preload task"):
        PreloadTask("", lambda: None)


def test_preload_task_init_invalid_func():
    """测试 PreloadTask（初始化，无效函数）"""
    with pytest.raises(ValueError, match="invalid preload task"):
        PreloadTask("test", None)


def test_preload_task_init_negative_interval():
    """测试 PreloadTask（初始化，负间隔）"""
    task = PreloadTask("test_task", lambda: None, interval_seconds=-10)
    
    # 间隔会被修正为1
    assert task.interval_seconds == 1


def test_preload_task_init_zero_interval():
    """测试 PreloadTask（初始化，零间隔）"""
    task = PreloadTask("test_task", lambda: None, interval_seconds=0)
    
    # 间隔会被修正为1
    assert task.interval_seconds == 1


def test_preload_task_should_run_enabled():
    """测试 PreloadTask（应该运行，已启用）"""
    task = PreloadTask("test_task", lambda: None, interval_seconds=60, enabled=True)
    
    # 首次运行应该返回True
    assert task.should_run() is True


def test_preload_task_should_run_disabled():
    """测试 PreloadTask（应该运行，已禁用）"""
    task = PreloadTask("test_task", lambda: None, interval_seconds=60, enabled=False)
    
    assert task.should_run() is False


def test_preload_task_should_run_interval_not_met():
    """测试 PreloadTask（应该运行，间隔未到）"""
    task = PreloadTask("test_task", lambda: None, interval_seconds=60)
    task.last_run_at = time.time()
    
    # 刚刚运行过，不应该再次运行
    assert task.should_run() is False


def test_preload_task_should_run_interval_met():
    """测试 PreloadTask（应该运行，间隔已到）"""
    task = PreloadTask("test_task", lambda: None, interval_seconds=1)
    task.last_run_at = time.time() - 2  # 2秒前运行过
    
    # 间隔已到，应该运行
    assert task.should_run() is True


def test_preload_task_should_run_custom_now():
    """测试 PreloadTask（应该运行，自定义时间）"""
    task = PreloadTask("test_task", lambda: None, interval_seconds=60)
    now = time.time()
    task.last_run_at = now - 30
    
    # 30秒前运行，间隔60秒，不应该运行
    assert task.should_run(now=now) is False
    
    # 70秒前运行，间隔60秒，应该运行
    task.last_run_at = now - 70
    assert task.should_run(now=now) is True


def test_preload_task_run_success():
    """测试 PreloadTask（运行，成功）"""
    mock_func = Mock(return_value="success")
    task = PreloadTask("test_task", mock_func, interval_seconds=60)
    
    result = task.run()
    
    assert result is True
    assert task.last_status == "success"
    assert task.last_error is None
    assert task.success_count == 1
    assert task.fail_count == 0
    assert task.last_run_at is not None
    mock_func.assert_called_once()


def test_preload_task_run_failure():
    """测试 PreloadTask（运行，失败）"""
    mock_func = Mock(side_effect=ValueError("Test error"))
    task = PreloadTask("test_task", mock_func, interval_seconds=60)
    
    result = task.run()
    
    assert result is False
    assert task.last_status == "error"
    assert task.last_error == "Test error"
    assert task.success_count == 0
    assert task.fail_count == 1
    assert task.last_run_at is not None


def test_preload_task_run_multiple_success():
    """测试 PreloadTask（运行，多次成功）"""
    task = PreloadTask("test_task", lambda: None, interval_seconds=60)
    
    task.run()
    task.run()
    task.run()
    
    assert task.success_count == 3
    assert task.fail_count == 0


def test_preloader_init():
    """测试 Preloader（初始化）"""
    preloader = Preloader()
    
    assert preloader._tasks == {}
    assert preloader._thread is None


def test_preloader_register_task():
    """测试 Preloader（注册任务）"""
    preloader = Preloader()
    mock_func = Mock()
    
    result = preloader.register_task("test_task", mock_func, interval_seconds=60)
    
    assert result is True
    assert "test_task" in preloader._tasks
    assert preloader._tasks["test_task"].name == "test_task"


def test_preloader_register_task_duplicate():
    """测试 Preloader（注册任务，重复）"""
    preloader = Preloader()
    mock_func = Mock()
    
    preloader.register_task("test_task", mock_func)
    result = preloader.register_task("test_task", mock_func)
    
    assert result is False


def test_preloader_unregister_task():
    """测试 Preloader（注销任务）"""
    preloader = Preloader()
    mock_func = Mock()
    
    preloader.register_task("test_task", mock_func)
    result = preloader.unregister_task("test_task")
    
    assert result is True
    assert "test_task" not in preloader._tasks


def test_preloader_unregister_task_nonexistent():
    """测试 Preloader（注销任务，不存在）"""
    preloader = Preloader()
    
    result = preloader.unregister_task("nonexistent")
    
    assert result is False


def test_preloader_enable_task():
    """测试 Preloader（启用任务）"""
    preloader = Preloader()
    mock_func = Mock()
    
    preloader.register_task("test_task", mock_func, enabled=False)
    result = preloader.enable_task("test_task")
    
    assert result is True
    assert preloader._tasks["test_task"].enabled is True


def test_preloader_enable_task_nonexistent():
    """测试 Preloader（启用任务，不存在）"""
    preloader = Preloader()
    
    result = preloader.enable_task("nonexistent")
    
    assert result is False


def test_preloader_disable_task():
    """测试 Preloader（禁用任务）"""
    preloader = Preloader()
    mock_func = Mock()
    
    preloader.register_task("test_task", mock_func, enabled=True)
    result = preloader.disable_task("test_task")
    
    assert result is True
    assert preloader._tasks["test_task"].enabled is False


def test_preloader_disable_task_nonexistent():
    """测试 Preloader（禁用任务，不存在）"""
    preloader = Preloader()
    
    result = preloader.disable_task("nonexistent")
    
    assert result is False


def test_preloader_run_once():
    """测试 Preloader（运行一次）"""
    preloader = Preloader()
    mock_func1 = Mock()
    mock_func2 = Mock()
    
    preloader.register_task("task1", mock_func1, interval_seconds=60)
    preloader.register_task("task2", mock_func2, interval_seconds=60)
    
    results = preloader.run_once()
    
    assert len(results) == 2
    assert results["task1"] is True
    assert results["task2"] is True
    mock_func1.assert_called_once()
    mock_func2.assert_called_once()


def test_preloader_run_once_disabled_task():
    """测试 Preloader（运行一次，禁用任务）"""
    preloader = Preloader()
    mock_func = Mock()
    
    preloader.register_task("task1", mock_func, interval_seconds=60, enabled=False)
    
    results = preloader.run_once()
    
    assert len(results) == 0
    mock_func.assert_not_called()


def test_preloader_run_once_interval_not_met():
    """测试 Preloader（运行一次，间隔未到）"""
    preloader = Preloader()
    mock_func = Mock()
    
    preloader.register_task("task1", mock_func, interval_seconds=60)
    preloader.run_once()  # 第一次运行
    
    results = preloader.run_once()  # 立即再次运行
    
    assert len(results) == 0
    assert mock_func.call_count == 1  # 只调用了一次


def test_preloader_run_once_task_failure():
    """测试 Preloader（运行一次，任务失败）"""
    preloader = Preloader()
    mock_func = Mock(side_effect=ValueError("Test error"))
    
    preloader.register_task("task1", mock_func, interval_seconds=60)
    
    results = preloader.run_once()
    
    assert len(results) == 1
    assert results["task1"] is False
    assert preloader._tasks["task1"].fail_count == 1


def test_preloader_start():
    """测试 Preloader（启动）"""
    preloader = Preloader()
    
    preloader.start()
    
    assert preloader._thread is not None
    assert preloader._thread.is_alive()
    
    # 清理
    preloader.stop()


def test_preloader_start_already_running():
    """测试 Preloader（启动，已在运行）"""
    preloader = Preloader()
    
    preloader.start()
    thread1 = preloader._thread
    
    # 再次启动应该不创建新线程
    preloader.start()
    thread2 = preloader._thread
    
    assert thread1 == thread2
    
    # 清理
    preloader.stop()


def test_preloader_stop():
    """测试 Preloader（停止）"""
    preloader = Preloader()
    
    preloader.start()
    assert preloader._thread.is_alive()
    
    preloader.stop()
    
    # 等待线程结束
    preloader._thread.join(timeout=2)
    assert not preloader._thread.is_alive()


def test_preloader_stop_not_started():
    """测试 Preloader（停止，未启动）"""
    preloader = Preloader()
    
    # 应该不抛出异常
    preloader.stop()


def test_preloader_get_stats():
    """测试 Preloader（获取统计）"""
    preloader = Preloader()
    mock_func = Mock()
    
    preloader.register_task("task1", mock_func, interval_seconds=60)
    preloader.run_once()
    
    stats = preloader.get_stats()
    
    assert "tasks" in stats
    assert "total_tasks" in stats
    assert "task1" in stats["tasks"]
    assert stats["tasks"]["task1"]["success_count"] == 1
    assert stats["tasks"]["task1"]["fail_count"] == 0
    assert stats["tasks"]["task1"]["last_status"] == "success"


def test_preloader_get_stats_empty():
    """测试 Preloader（获取统计，空）"""
    preloader = Preloader()
    
    stats = preloader.get_stats()
    
    assert stats["total_tasks"] == 0
    assert stats["tasks"] == {}


def test_preloader_get_stats_multiple_tasks():
    """测试 Preloader（获取统计，多个任务）"""
    preloader = Preloader()
    mock_func1 = Mock()
    mock_func2 = Mock(side_effect=ValueError("Error"))
    
    preloader.register_task("task1", mock_func1, interval_seconds=60)
    preloader.register_task("task2", mock_func2, interval_seconds=60)
    
    preloader.run_once()
    
    stats = preloader.get_stats()
    
    assert stats["total_tasks"] == 2
    assert len(stats["tasks"]) == 2
    assert stats["tasks"]["task1"]["success_count"] == 1
    assert stats["tasks"]["task2"]["fail_count"] == 1


def test_preloader_list_tasks():
    """测试 Preloader（列出任务）"""
    preloader = Preloader()
    mock_func = Mock()
    
    preloader.register_task("task1", mock_func, interval_seconds=60)
    preloader.register_task("task2", mock_func, interval_seconds=60)
    
    tasks = preloader.list_tasks()
    
    assert len(tasks) == 2
    assert "task1" in tasks
    assert "task2" in tasks


def test_preloader_list_tasks_empty():
    """测试 Preloader（列出任务，空）"""
    preloader = Preloader()
    
    tasks = preloader.list_tasks()
    
    assert tasks == []


def test_preloader_start_exception_handling():
    """测试 Preloader（启动时异常处理）"""
    preloader = Preloader()
    
    # 创建一个会抛出异常的任务
    def failing_task():
        raise RuntimeError("Task failed")
    
    preloader.register_task("failing_task", failing_task, interval_seconds=1)
    
    # 启动预加载器
    preloader.start(poll_interval=0.1)
    
    # 等待一段时间让异常处理代码执行
    time.sleep(0.3)
    
    # 验证预加载器仍在运行（异常被捕获）
    assert preloader._thread is not None
    assert preloader._thread.is_alive()
    
    # 清理
    preloader.stop()


def test_preloader_start_exception_in_run_once():
    """测试 Preloader（启动时 run_once 异常处理）"""
    preloader = Preloader()
    
    # 通过 mock 让 run_once 抛出异常
    original_run_once = preloader.run_once
    
    def failing_run_once():
        raise RuntimeError("run_once failed")
    
    preloader.run_once = failing_run_once
    
    # 启动预加载器
    preloader.start(poll_interval=0.1)
    
    # 等待一段时间让异常处理代码执行
    time.sleep(0.3)
    
    # 验证预加载器仍在运行（异常被捕获）
    assert preloader._thread is not None
    assert preloader._thread.is_alive()
    
    # 恢复原始方法
    preloader.run_once = original_run_once
    
    # 清理
    preloader.stop()
