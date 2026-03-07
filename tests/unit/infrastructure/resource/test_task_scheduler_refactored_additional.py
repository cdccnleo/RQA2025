import threading
from unittest.mock import MagicMock

import pytest

from src.infrastructure.resource.scheduling.task_scheduler_refactored import (
    TaskManager,
    TaskQueueManager,
    TaskMonitor,
    TaskSchedulerCore,
    TaskSchedulerFacade,
    TaskPriority,
    TaskStatus,
    Task,
)


@pytest.fixture
def task_manager():
    return TaskManager()


@pytest.fixture
def queue_manager():
    return TaskQueueManager(max_size=10)


@pytest.fixture
def monitor():
    mon = TaskMonitor()
    mon.start_time -= 10
    return mon


@pytest.fixture
def worker_manager(monkeypatch):
    from src.infrastructure.resource.scheduling.task_scheduler_refactored import TaskWorkerManager

    created = []

    class DummyThread:
        def __init__(self, target=None, name=None, daemon=None):
            self.target = target
            self.name = name
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            return

    monkeypatch.setattr("threading.Thread", lambda target, name, daemon: DummyThread(target, name, daemon))
    manager = TaskWorkerManager(max_workers=2)
    return manager


def sample_task(x, y):
    return x + y


def failing_task():
    raise ValueError("boom")


def test_task_manager_lifecycle(task_manager):
    task = Task(id="1", name="test", func=sample_task, priority=TaskPriority.NORMAL)
    task_manager.add_task(task)
    retrieved = task_manager.get_task("1")
    assert retrieved is task

    task_manager.update_task_status("1", TaskStatus.RUNNING)
    assert task_manager.get_task("1").status == TaskStatus.RUNNING

    task_manager.update_task_status("1", TaskStatus.COMPLETED, result=42)
    saved = task_manager.get_task("1")
    assert saved.status == TaskStatus.COMPLETED
    assert saved.result == 42

    task_manager.remove_task("1")
    assert task_manager.get_task("1") is None


def test_task_queue_manager_priorities(queue_manager):
    high = Task(id="h", name="high", func=sample_task, priority=TaskPriority.HIGH)
    low = Task(id="l", name="low", func=sample_task, priority=TaskPriority.LOW)
    queue_manager.put_task(low)
    queue_manager.put_task(high)
    queue_manager.queue.size = queue_manager.queue.qsize
    first = queue_manager.get_next_task()
    assert first is low
    assert queue_manager.get_pending_count() == 1


def test_task_monitor_statistics(monitor):
    monitor.record_task_start()
    monitor.record_task_start()
    monitor.record_task_completion(2.0)
    monitor.record_task_completion(4.0)
    monitor.record_task_failure()
    monitor.record_task_cancellation()
    monitor.update_uptime()

    stats = monitor.get_stats()
    assert stats["total_tasks"] == 2
    assert stats["completed_tasks"] == 2
    assert stats["failed_tasks"] == 1
    assert stats["cancelled_tasks"] == 1
    assert stats["avg_execution_time"] == pytest.approx(3.0)
    assert stats["uptime"] >= 10


@pytest.fixture
def scheduler_core(task_manager, queue_manager, worker_manager, monitor):
    return TaskSchedulerCore(task_manager, queue_manager, worker_manager, monitor)


def test_scheduler_core_submission_and_execution(scheduler_core, task_manager):
    task_id = scheduler_core.submit_task("add", sample_task, TaskPriority.NORMAL, 60, 2, 3)
    task = task_manager.get_task(task_id)
    assert task.priority == TaskPriority.NORMAL
    assert task.status == TaskStatus.PENDING

    scheduler_core.execute_task(task)
    assert task_manager.get_task(task_id).status == TaskStatus.COMPLETED
    assert task_manager.get_task(task_id).result == 5


def test_scheduler_core_failure_and_cancel(scheduler_core, task_manager):
    task_id = scheduler_core.submit_task("fail", failing_task, TaskPriority.HIGH)
    task = task_manager.get_task(task_id)
    scheduler_core.execute_task(task)
    assert task_manager.get_task(task_id).status == TaskStatus.FAILED

    cancel_id = scheduler_core.submit_task("cancel", sample_task, TaskPriority.LOW)
    assert scheduler_core.cancel_task(cancel_id) is True
    assert scheduler_core.get_task_status(cancel_id) == TaskStatus.CANCELLED


def test_task_scheduler_facade_integration(monkeypatch):
    facade = TaskSchedulerFacade(max_workers=1, queue_size=5)
    facade.worker_manager.running = False

    facade.worker_manager.start_workers(lambda: None)
    assert facade.worker_manager.is_running() is True

    task_id = facade.submit_task("sum", sample_task, TaskPriority.HIGH, 60, 4, 6)
    task = facade.task_manager.get_task(task_id)
    facade.task_manager.update_task_status(task_id, TaskStatus.RUNNING)
    facade.task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=10)
    facade.monitor.record_task_completion(1.2)

    stats = facade.get_stats()
    assert stats["completed_tasks"] >= 1
    assert facade.get_task_status(task_id) == TaskStatus.COMPLETED
    assert facade.get_task_result(task_id) == 10

    facade.shutdown()
    assert facade.worker_manager.is_running() is False
