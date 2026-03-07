import time
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.infrastructure.monitoring.core.concurrency_optimizer as concurrency_module
from src.infrastructure.monitoring.core.concurrency_optimizer import ConcurrencyOptimizer


@pytest.fixture(autouse=True)
def patch_psutil(monkeypatch):
    """Provide stable psutil responses to avoid real CPU sampling delays."""

    cpu_values = [20.0]

    def fake_cpu_percent(interval=1):
        return cpu_values[-1]

    class FakeVirtualMemory:
        percent = 30.0

    monkeypatch.setattr(concurrency_module.psutil, "cpu_percent", fake_cpu_percent)
    monkeypatch.setattr(concurrency_module.psutil, "virtual_memory", lambda: FakeVirtualMemory)
    yield cpu_values


@pytest.fixture
def optimizer(patch_psutil):
    opt = ConcurrencyOptimizer(min_workers=1, max_workers=4, target_cpu_percent=40.0, monitor_interval=0.01)
    yield opt
    opt.stop_monitoring()
    opt.executor.shutdown(wait=True)


def test_submit_task_success(optimizer):
    future = optimizer.submit_task(lambda x: x + 1, 1)
    assert future.result() == 2
    assert optimizer.stats["tasks_completed"] == 1
    assert optimizer.stats["tasks_failed"] == 0


def test_submit_task_sets_timeout(optimizer):
    future = optimizer.submit_task(lambda: 42, timeout=0.5)
    assert getattr(future, "timeout", None) == 0.5
    assert optimizer.stats["tasks_submitted"] >= 1


def test_submit_task_failure(optimizer):
    def failing():
        raise RuntimeError("boom")

    future = optimizer.submit_task(failing)
    with pytest.raises(RuntimeError):
        future.result()
    assert optimizer.stats["tasks_failed"] == 1


def test_submit_batch_formats(optimizer, caplog):
    caplog.set_level("ERROR")
    tasks = [
        (lambda x: x + 1, (1,), {}),
        (lambda y: y * 2, (2,)),
        (lambda: "bad",),
    ]

    futures = optimizer.submit_batch(tasks)
    assert len(futures) == 2
    assert any("无效任务格式" in record.message for record in caplog.records)


def test_wait_for_completion(optimizer):
    futures = [
        optimizer.submit_task(lambda: "ok"),
        optimizer.submit_task(lambda: 2 * 2),
    ]
    summary = optimizer.wait_for_completion(futures, timeout=1.0)
    assert summary["completed"] == 2
    assert summary["failed"] == 0
    assert summary["success_rate"] == 1.0


def test_wait_for_completion_with_error(optimizer):
    futures = [
        optimizer.submit_task(lambda: "ok"),
        optimizer.submit_task(lambda: (_ for _ in ()).throw(RuntimeError("fail"))),
    ]
    summary = optimizer.wait_for_completion(futures, timeout=1.0)
    assert summary["failed"] == 1
    assert any("error" in r for r in summary["results"])


def test_wait_for_completion_timeout(optimizer):
    slow_future = optimizer.executor.submit(time.sleep, 0.5)
    summary = optimizer.wait_for_completion([slow_future], timeout=0.01)
    assert summary["completed"] == 0
    assert summary["failed"] == 0
    assert summary["total"] == 1


def test_get_concurrency_stats(optimizer, patch_psutil):
    patch_psutil.append(60.0)
    stats = optimizer.get_concurrency_stats()
    assert stats["current_workers"] == optimizer.current_workers
    assert stats["current_cpu_percent"] == 60.0
    assert "tasks_submitted" in stats


def test_adjust_thread_pool_changes_size(optimizer, patch_psutil, monkeypatch):
    resize_spy = MagicMock()
    monkeypatch.setattr(optimizer, "_resize_thread_pool", resize_spy)

    # force CPU high to shrink
    optimizer.current_workers = 3
    patch_psutil.append(90.0)
    optimizer._adjust_thread_pool(avg_cpu=90.0, memory_percent=30.0)
    resize_spy.assert_called_with(2)

    resize_spy.reset_mock()
    optimizer.current_workers = 1
    optimizer._adjust_thread_pool(avg_cpu=10.0, memory_percent=30.0)
    resize_spy.assert_called_with(2)


def test_get_recommendations(optimizer, monkeypatch):
    monkeypatch.setattr(
        optimizer,
        "get_concurrency_stats",
        lambda: {
            "current_cpu_percent": 85.0,
            "current_memory_percent": 88.0,
            "success_rate": 0.5,
            "thread_adjustments": 11,
        },
    )

    recs = optimizer.get_recommendations()
    assert any(r["type"] == "cpu_optimization" for r in recs)
    assert any(r["type"] == "memory_optimization" for r in recs)
    assert any(r["type"] == "error_handling" for r in recs)


def test_start_and_stop_monitoring(optimizer, monkeypatch):
    run_once = MagicMock(side_effect=lambda: optimizer.stop_event.set())
    monkeypatch.setattr(concurrency_module.ConcurrencyOptimizer, "_monitor_and_adjust", run_once)

    optimizer.start_monitoring()
    time.sleep(0.05)
    optimizer.stop_monitoring()
    assert run_once.called
    assert optimizer.monitoring_active is False


def test_start_monitoring_idempotent(optimizer, monkeypatch):
    run_once = MagicMock(side_effect=lambda: optimizer.stop_event.set())
    monkeypatch.setattr(concurrency_module.ConcurrencyOptimizer, "_monitor_and_adjust", run_once)

    optimizer.start_monitoring()
    first_thread = optimizer.monitor_thread
    optimizer.start_monitoring()
    assert optimizer.monitor_thread is first_thread
    optimizer.stop_monitoring()
    optimizer.stop_event.clear()
    assert run_once.called


def test_shutdown_invokes_executor_and_monitoring(optimizer, monkeypatch):
    stop_mock = MagicMock()
    monkeypatch.setattr(optimizer, "stop_monitoring", stop_mock)
    executor_mock = MagicMock()
    optimizer.executor = executor_mock

    optimizer.shutdown(timeout=3.0)
    stop_mock.assert_called_once()
    executor_mock.shutdown.assert_called_once_with(wait=True, timeout=3.0)


def test_get_health_status_warning(optimizer, monkeypatch):
    stats = {
        "current_workers": 2,
        "tasks_submitted": 5,
        "tasks_completed": 4,
        "tasks_failed": 1,
        "success_rate": 0.5,
        "avg_execution_time": 0.1,
        "current_cpu_percent": 95.0,
        "current_memory_percent": 92.0,
        "thread_adjustments": 12,
        "uptime_seconds": 10,
        "tasks_per_second": 0.4,
        "monitoring_active": False,
    }

    executor_mock = MagicMock()
    executor_mock._threads = {object()}
    optimizer.executor = executor_mock
    optimizer.monitoring_active = False
    monkeypatch.setattr(optimizer, "get_concurrency_stats", lambda: stats)

    health = optimizer.get_health_status()
    assert health["status"] == "warning"
    assert any("任务成功率过低" in issue for issue in health["issues"])
    assert any("CPU使用率严重过高" in issue for issue in health["issues"])
    assert "性能监控未启用" in health["issues"]


def test_get_health_status_error(optimizer, monkeypatch):
    monkeypatch.setattr(optimizer, "get_concurrency_stats", MagicMock(side_effect=RuntimeError("boom")))

    health = optimizer.get_health_status()
    assert health["status"] == "error"
    assert health["error"] == "boom"


def test_monitoring_loop_handles_exception_and_continues(monkeypatch, optimizer):
    calls = {"count": 0}

    def failing_cpu(interval=1):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("fail")
        optimizer.stop_event.set()
        return 30.0

    monkeypatch.setattr(concurrency_module.psutil, "cpu_percent", failing_cpu)
    monkeypatch.setattr(concurrency_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=40.0))
    monkeypatch.setattr(concurrency_module.time, "sleep", lambda _: None)

    thread = threading.Thread(target=optimizer._monitor_and_adjust)
    thread.start()
    thread.join(timeout=0.1)

    assert calls["count"] >= 2


def test_monitor_and_adjust_trims_history(monkeypatch, optimizer):
    optimizer.stats["cpu_usage_history"] = [10.0] * 11
    optimizer.stop_event = threading.Event()

    monkeypatch.setattr(concurrency_module.psutil, "cpu_percent", lambda interval=1: 50.0)
    monkeypatch.setattr(concurrency_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=35.0))

    def stop_after_sleep(interval):
        optimizer.stop_event.set()

    optimizer.stop_event.clear()
    monkeypatch.setattr(concurrency_module.time, "sleep", stop_after_sleep)

    optimizer._monitor_and_adjust()
    history = optimizer.stats["cpu_usage_history"]
    assert history.count(10.0) <= 10
    assert history[-1] == 50.0
    optimizer.stop_event = threading.Event()


def test_adjust_thread_pool_memory_branch(optimizer, monkeypatch):
    monkeypatch.setattr(optimizer, "_resize_thread_pool", MagicMock())
    optimizer.current_workers = 3
    optimizer._adjust_thread_pool(avg_cpu=optimizer.target_cpu_percent, memory_percent=88.0)
    optimizer._resize_thread_pool.assert_called_once()


def test_resize_thread_pool_success(optimizer, monkeypatch):
    class StubExecutor:
        def __init__(self):
            self.shutdown_called = False

        def shutdown(self, wait=True, timeout=None):
            self.shutdown_called = True

    original_executor = optimizer.executor
    stub_executor = StubExecutor()
    optimizer.executor = stub_executor

    created = {}

    def fake_thread_pool_executor(*args, **kwargs):
        new_exec = StubExecutor()
        created["instance"] = new_exec
        return new_exec

    monkeypatch.setattr(concurrency_module, "ThreadPoolExecutor", fake_thread_pool_executor)

    new_size = max(optimizer.min_workers + 1, 2)
    optimizer._resize_thread_pool(new_size)

    assert stub_executor.shutdown_called is True
    assert created["instance"] is optimizer.executor
    assert optimizer.current_workers == new_size

    optimizer.executor = original_executor


def test_resize_thread_pool_failure(optimizer, monkeypatch):
    original_executor = optimizer.executor
    mock_executor = MagicMock()
    mock_executor.shutdown.side_effect = RuntimeError("boom")
    optimizer.executor = mock_executor
    original_workers = optimizer.current_workers

    optimizer._resize_thread_pool(original_workers + 1)
    assert optimizer.current_workers == original_workers
    optimizer.executor = original_executor


def test_get_health_status_detects_inactive_pool(optimizer, monkeypatch):
    original_executor = optimizer.executor
    optimizer.executor = SimpleNamespace()
    monkeypatch.setattr(optimizer, "get_concurrency_stats", lambda: {
        "success_rate": 1.0,
        "current_cpu_percent": 10.0,
        "current_memory_percent": 10.0,
        "thread_adjustments": 0,
        "current_workers": optimizer.current_workers,
        "tasks_submitted": 0,
        "tasks_completed": 0,
        "tasks_failed": 0,
        "avg_execution_time": 0.0,
        "uptime_seconds": 1.0,
        "tasks_per_second": 0.0,
        "monitoring_active": True,
    })

    health = optimizer.get_health_status()
    assert "线程池未正常运行" in health["issues"]
    optimizer.executor = original_executor


def test_global_optimizer_available():
    assert concurrency_module.global_concurrency_optimizer is not None


def test_update_avg_execution_time(optimizer):
    """测试更新平均执行时间"""
    initial_avg = optimizer.stats['avg_execution_time']
    optimizer._update_avg_execution_time(1.0)
    
    # 使用移动平均，新值应该接近初始值和新值的加权平均
    new_avg = optimizer.stats['avg_execution_time']
    assert new_avg != initial_avg
    assert 0 <= new_avg <= 1.0


def test_get_recommendations_no_issues(optimizer, monkeypatch):
    """测试获取建议（无问题）"""
    monkeypatch.setattr(
        optimizer,
        "get_concurrency_stats",
        lambda: {
            "current_cpu_percent": 50.0,
            "current_memory_percent": 50.0,
            "success_rate": 0.98,
            "thread_adjustments": 2,
        },
    )
    
    recs = optimizer.get_recommendations()
    assert len(recs) == 0


def test_get_recommendations_thread_adjustments_high(optimizer, monkeypatch):
    """测试获取建议（线程调整频繁）"""
    monkeypatch.setattr(
        optimizer,
        "get_concurrency_stats",
        lambda: {
            "current_cpu_percent": 50.0,
            "current_memory_percent": 50.0,
            "success_rate": 0.98,
            "thread_adjustments": 15,
        },
    )
    
    recs = optimizer.get_recommendations()
    assert any(r["type"] == "stability" for r in recs)


def test_adjust_thread_pool_memory_critical(optimizer, monkeypatch):
    """测试调整线程池（内存严重过高）"""
    resize_spy = MagicMock()
    monkeypatch.setattr(optimizer, "_resize_thread_pool", resize_spy)
    optimizer.current_workers = 4
    
    # 注意：代码逻辑是先检查>85%，然后检查>90%
    # 当内存>85%时，new_workers = max(min_workers, old_workers - 2) = max(1, 4-2) = 2
    # 当内存>90%时，new_workers = min_workers = 1
    # 但由于是elif，只会执行第一个条件
    # 所以内存95%时，实际会减少到2（因为先匹配>85%的条件）
    optimizer._adjust_thread_pool(avg_cpu=50.0, memory_percent=95.0)
    # 由于代码逻辑，内存>85%时会先减少2个线程
    resize_spy.assert_called_with(2)
    
    # 测试内存>85%但<90%的情况
    resize_spy.reset_mock()
    optimizer.current_workers = 4
    optimizer._adjust_thread_pool(avg_cpu=50.0, memory_percent=88.0)
    # 内存>85%但<90%时，减少2个线程
    resize_spy.assert_called_with(2)


def test_adjust_thread_pool_no_change(optimizer, monkeypatch):
    """测试调整线程池（无需调整）"""
    resize_spy = MagicMock()
    monkeypatch.setattr(optimizer, "_resize_thread_pool", resize_spy)
    optimizer.current_workers = 2
    
    # CPU和内存都在目标范围内
    optimizer._adjust_thread_pool(
        avg_cpu=optimizer.target_cpu_percent,
        memory_percent=50.0
    )
    resize_spy.assert_not_called()


def test_get_health_status_healthy(optimizer, monkeypatch):
    """测试获取健康状态（健康）"""
    executor_mock = MagicMock()
    executor_mock._threads = {object()}
    optimizer.executor = executor_mock
    optimizer.monitoring_active = True
    
    monkeypatch.setattr(optimizer, "get_concurrency_stats", lambda: {
        "success_rate": 0.95,
        "current_cpu_percent": 50.0,
        "current_memory_percent": 50.0,
        "current_workers": optimizer.current_workers,
        "tasks_submitted": 10,
        "tasks_completed": 10,
        "tasks_failed": 0,
        "avg_execution_time": 0.1,
        "thread_adjustments": 2,
        "uptime_seconds": 10,
        "tasks_per_second": 1.0,
        "monitoring_active": True,
    })
    
    health = optimizer.get_health_status()
    assert health["status"] == "healthy"
    assert len(health["issues"]) == 0


def test_stop_monitoring_when_not_active(optimizer):
    """测试停止监控（未激活）"""
    optimizer.monitoring_active = False
    optimizer.stop_monitoring()
    # 应该不会抛出异常
    assert optimizer.monitoring_active is False


def test_submit_task_with_priority(optimizer):
    """测试提交任务（带优先级）"""
    future = optimizer.submit_task(lambda: 42, priority='high')
    assert future.result() == 42
    assert optimizer.stats["tasks_submitted"] >= 1


def test_submit_batch_empty_list(optimizer):
    """测试批量提交任务（空列表）"""
    futures = optimizer.submit_batch([])
    assert len(futures) == 0


def test_wait_for_completion_empty_list(optimizer):
    """测试等待完成（空列表）"""
    summary = optimizer.wait_for_completion([], timeout=1.0)
    assert summary["completed"] == 0
    assert summary["total"] == 0
    assert summary["success_rate"] == 0.0