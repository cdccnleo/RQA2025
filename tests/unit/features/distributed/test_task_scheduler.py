#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征任务调度器测试
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.features.distributed.task_scheduler import (
    FeatureTaskScheduler,
    FeatureTask,
    TaskStatus,
    TaskPriority,
    get_task_scheduler,
    submit_task
)


class TestFeatureTask:
    """特征任务测试"""

    def test_task_creation(self):
        """测试任务创建"""
        task = FeatureTask(
            task_id="test_task",
            task_type="feature_processing",
            data={"test": "data"},
            priority=TaskPriority.NORMAL
        )
        
        assert task.task_id == "test_task"
        assert task.task_type == "feature_processing"
        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None

    def test_task_with_metadata(self):
        """测试带元数据的任务"""
        metadata = {"key": "value"}
        task = FeatureTask(
            task_id="test_task",
            task_type="feature_processing",
            data={},
            priority=TaskPriority.NORMAL,
            metadata=metadata
        )
        
        assert task.metadata == metadata


class TestFeatureTaskScheduler:
    """特征任务调度器测试"""

    @pytest.fixture
    def scheduler(self):
        """创建任务调度器"""
        return FeatureTaskScheduler(max_queue_size=100)

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return {"test": "data"}

    def test_init(self, scheduler):
        """测试初始化"""
        assert scheduler.max_queue_size == 100
        assert scheduler._running is False
        assert scheduler._stats["total_tasks"] == 0

    def test_submit_task(self, scheduler, sample_data):
        """测试提交任务"""
        task_id = scheduler.submit_task(
            task_type="feature_processing",
            data=sample_data,
            priority=TaskPriority.NORMAL
        )
        
        assert task_id is not None
        assert isinstance(task_id, str)
        assert scheduler._stats["total_tasks"] == 1
        assert scheduler._stats["pending_tasks"] == 1

    def test_submit_task_with_metadata(self, scheduler, sample_data):
        """测试提交带元数据的任务"""
        metadata = {"config": {"key": "value"}}
        task_id = scheduler.submit_task(
            task_type="feature_processing",
            data=sample_data,
            priority=TaskPriority.HIGH,
            metadata=metadata
        )
        
        task = scheduler._tasks[task_id]
        assert task.metadata == metadata
        assert task.priority == TaskPriority.HIGH

    def test_submit_task_queue_full(self, scheduler, sample_data):
        """测试队列满时提交任务"""
        # 创建小队列
        small_scheduler = FeatureTaskScheduler(max_queue_size=2)
        
        # 填满队列
        small_scheduler.submit_task("type1", sample_data)
        small_scheduler.submit_task("type2", sample_data)
        
        # 应该抛出异常
        with pytest.raises(ValueError, match="任务队列已满"):
            small_scheduler.submit_task("type3", sample_data)

    def test_get_task(self, scheduler, sample_data):
        """测试获取任务"""
        # 注册工作节点
        scheduler.register_worker("worker1", {"cpu": 4})
        
        # 提交任务
        task_id = scheduler.submit_task("feature_processing", sample_data)
        
        # 获取任务
        task = scheduler.get_task("worker1")
        
        assert task is not None
        assert task.task_id == task_id
        assert task.status == TaskStatus.RUNNING
        assert task.worker_id == "worker1"
        assert task.started_at is not None

    def test_get_task_empty_queue(self, scheduler):
        """测试空队列获取任务"""
        scheduler.register_worker("worker1", {"cpu": 4})
        task = scheduler.get_task("worker1")
        assert task is None

    def test_complete_task_success(self, scheduler, sample_data):
        """测试完成任务-成功"""
        task_id = scheduler.submit_task("feature_processing", sample_data)
        scheduler.register_worker("worker1", {"cpu": 4})
        scheduler.get_task("worker1")  # 分配任务
        
        scheduler.complete_task(task_id, result={"result": "success"})
        
        task = scheduler._tasks[task_id]
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"result": "success"}
        assert task.completed_at is not None
        assert scheduler._stats["completed_tasks"] == 1

    def test_complete_task_failure(self, scheduler, sample_data):
        """测试完成任务-失败"""
        task_id = scheduler.submit_task("feature_processing", sample_data)
        scheduler.register_worker("worker1", {"cpu": 4})
        scheduler.get_task("worker1")
        
        scheduler.complete_task(task_id, result=None, error="处理失败")
        
        task = scheduler._tasks[task_id]
        assert task.status == TaskStatus.FAILED
        assert task.error == "处理失败"
        assert scheduler._stats["failed_tasks"] == 1

    def test_complete_task_not_found(self, scheduler):
        """测试完成任务-任务不存在"""
        scheduler.complete_task("nonexistent_task", result={})
        # 应该记录警告但不报错

    def test_cancel_task(self, scheduler, sample_data):
        """测试取消任务"""
        task_id = scheduler.submit_task("feature_processing", sample_data)
        
        cancelled = scheduler.cancel_task(task_id)
        assert cancelled is True
        
        task = scheduler._tasks[task_id]
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_task_completed(self, scheduler, sample_data):
        """测试取消已完成的任务"""
        task_id = scheduler.submit_task("feature_processing", sample_data)
        scheduler.register_worker("worker1", {"cpu": 4})
        scheduler.get_task("worker1")
        scheduler.complete_task(task_id, result={})
        
        cancelled = scheduler.cancel_task(task_id)
        assert cancelled is False  # 已完成的任务不能取消

    def test_cancel_task_not_found(self, scheduler):
        """测试取消不存在的任务"""
        cancelled = scheduler.cancel_task("nonexistent_task")
        assert cancelled is False

    def test_get_task_status(self, scheduler, sample_data):
        """测试获取任务状态"""
        task_id = scheduler.submit_task("feature_processing", sample_data)
        
        status = scheduler.get_task_status(task_id)
        assert status == TaskStatus.PENDING
        
        scheduler.register_worker("worker1", {"cpu": 4})
        scheduler.get_task("worker1")
        
        status = scheduler.get_task_status(task_id)
        assert status == TaskStatus.RUNNING

    def test_get_task_status_not_found(self, scheduler):
        """测试获取不存在任务的状态"""
        status = scheduler.get_task_status("nonexistent_task")
        assert status is None

    def test_get_task_result(self, scheduler, sample_data):
        """测试获取任务结果"""
        task_id = scheduler.submit_task("feature_processing", sample_data)
        scheduler.register_worker("worker1", {"cpu": 4})
        scheduler.get_task("worker1")
        scheduler.complete_task(task_id, result={"result": "data"})
        
        result = scheduler.get_task_result(task_id)
        assert result == {"result": "data"}

    def test_get_task_result_not_completed(self, scheduler, sample_data):
        """测试获取未完成任务的结果"""
        task_id = scheduler.submit_task("feature_processing", sample_data)
        result = scheduler.get_task_result(task_id)
        assert result is None

    def test_register_worker(self, scheduler):
        """测试注册工作节点"""
        scheduler.register_worker("worker1", {"cpu": 4, "memory": 8192})
        
        assert "worker1" in scheduler._workers
        assert scheduler._workers["worker1"]["capabilities"] == {"cpu": 4, "memory": 8192}

    def test_unregister_worker(self, scheduler):
        """测试注销工作节点"""
        scheduler.register_worker("worker1", {"cpu": 4})
        scheduler.unregister_worker("worker1")
        
        assert "worker1" not in scheduler._workers

    def test_update_worker_heartbeat(self, scheduler):
        """测试更新工作节点心跳"""
        scheduler.register_worker("worker1", {"cpu": 4})
        
        old_heartbeat = scheduler._workers["worker1"]["last_heartbeat"]
        time.sleep(0.1)
        scheduler.update_worker_heartbeat("worker1")
        
        new_heartbeat = scheduler._workers["worker1"]["last_heartbeat"]
        assert new_heartbeat > old_heartbeat

    def test_get_available_workers(self, scheduler):
        """测试获取可用工作节点"""
        scheduler.register_worker("worker1", {"cpu": 4})
        scheduler.update_worker_heartbeat("worker1")
        
        available = scheduler.get_available_workers()
        assert "worker1" in available

    def test_get_available_workers_timeout(self, scheduler):
        """测试获取超时工作节点"""
        scheduler.register_worker("worker1", {"cpu": 4})
        # 设置旧的心跳时间
        scheduler._workers["worker1"]["last_heartbeat"] = datetime.now() - timedelta(seconds=60)
        
        available = scheduler.get_available_workers()
        assert "worker1" not in available

    def test_get_scheduler_stats(self, scheduler, sample_data):
        """测试获取调度器统计"""
        scheduler.submit_task("feature_processing", sample_data)
        scheduler.register_worker("worker1", {"cpu": 4})
        
        stats = scheduler.get_scheduler_stats()
        
        assert "total_tasks" in stats
        assert "completed_tasks" in stats
        assert "queue_size" in stats
        assert "active_workers" in stats
        assert stats["total_tasks"] == 1

    def test_get_task_history(self, scheduler, sample_data):
        """测试获取任务历史"""
        task_id1 = scheduler.submit_task("type1", sample_data)
        task_id2 = scheduler.submit_task("type2", sample_data)
        
        history = scheduler.get_task_history()
        
        assert len(history) == 2
        assert any(t.task_id == task_id1 for t in history)
        assert any(t.task_id == task_id2 for t in history)

    def test_get_task_history_with_limit(self, scheduler, sample_data):
        """测试获取任务历史-带限制"""
        for i in range(5):
            scheduler.submit_task(f"type{i}", sample_data)
        
        history = scheduler.get_task_history(limit=3)
        assert len(history) == 3

    def test_clear_completed_tasks(self, scheduler, sample_data):
        """测试清理已完成的任务"""
        task_id = scheduler.submit_task("feature_processing", sample_data)
        scheduler.register_worker("worker1", {"cpu": 4})
        scheduler.get_task("worker1")
        scheduler.complete_task(task_id, result={})
        
        # 设置完成时间为过去
        scheduler._tasks[task_id].completed_at = datetime.now() - timedelta(hours=25)
        
        cleared = scheduler.clear_completed_tasks(older_than_hours=24)
        assert cleared == 1
        assert task_id not in scheduler._tasks

    def test_start_stop(self, scheduler):
        """测试启动和停止"""
        scheduler.start()
        assert scheduler._running is True
        
        time.sleep(0.1)
        
        scheduler.stop()
        assert scheduler._running is False

    def test_start_already_running(self, scheduler):
        """测试重复启动"""
        scheduler.start()
        scheduler.start()  # 应该不报错
        scheduler.stop()


class TestGlobalFunctions:
    """全局函数测试"""

    def test_get_task_scheduler(self):
        """测试获取全局任务调度器"""
        scheduler = get_task_scheduler()
        assert isinstance(scheduler, FeatureTaskScheduler)

    def test_submit_task_function(self):
        """测试提交任务函数"""
        task_id = submit_task("test_type", {"data": "test"}, TaskPriority.NORMAL)
        assert task_id is not None
        assert isinstance(task_id, str)

