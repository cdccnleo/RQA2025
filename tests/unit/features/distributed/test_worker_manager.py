#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工作节点管理器测试
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.features.distributed.worker_manager import (
    FeatureWorkerManager,
    WorkerInfo,
    WorkerStatus,
    get_worker_manager
)


class TestWorkerInfo:
    """工作节点信息测试"""

    def test_worker_info_creation(self):
        """测试工作节点信息创建"""
        from datetime import datetime
        worker = WorkerInfo(
            worker_id="worker1",
            status=WorkerStatus.IDLE,
            capabilities={"cpu": 4, "memory": 8192},
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        
        assert worker.worker_id == "worker1"
        assert worker.status == WorkerStatus.IDLE
        assert worker.capabilities == {"cpu": 4, "memory": 8192}
        assert worker.current_task is None
        assert worker.completed_tasks == 0


class TestFeatureWorkerManager:
    """特征工作节点管理器测试"""

    @pytest.fixture
    def manager(self):
        """创建工作节点管理器"""
        return FeatureWorkerManager()

    def test_init(self, manager):
        """测试初始化"""
        assert manager._running is False
        assert manager._stats["total_workers"] == 0
        assert manager._stats["active_workers"] == 0

    def test_register_worker(self, manager):
        """测试注册工作节点"""
        success = manager.register_worker(
            worker_id="worker1",
            capabilities={"cpu": 4, "memory": 8192}
        )
        
        assert success is True
        assert "worker1" in manager._workers
        assert manager._stats["total_workers"] == 1
        assert manager._stats["active_workers"] == 1
        assert manager._stats["idle_workers"] == 1

    def test_register_worker_duplicate(self, manager):
        """测试重复注册工作节点"""
        manager.register_worker("worker1", {"cpu": 4})
        success = manager.register_worker("worker1", {"cpu": 8})
        
        assert success is False  # 重复注册应该失败

    def test_unregister_worker(self, manager):
        """测试注销工作节点"""
        manager.register_worker("worker1", {"cpu": 4})
        success = manager.unregister_worker("worker1")
        
        assert success is True
        assert "worker1" not in manager._workers
        assert manager._stats["total_workers"] == 0

    def test_unregister_worker_not_found(self, manager):
        """测试注销不存在的工作节点"""
        success = manager.unregister_worker("nonexistent")
        assert success is False

    def test_update_worker_heartbeat(self, manager):
        """测试更新工作节点心跳"""
        manager.register_worker("worker1", {"cpu": 4})
        
        old_heartbeat = manager._workers["worker1"].last_heartbeat
        time.sleep(0.1)
        success = manager.update_worker_heartbeat("worker1")
        
        assert success is True
        new_heartbeat = manager._workers["worker1"].last_heartbeat
        assert new_heartbeat > old_heartbeat

    def test_update_worker_heartbeat_not_found(self, manager):
        """测试更新不存在工作节点的心跳"""
        success = manager.update_worker_heartbeat("nonexistent")
        assert success is False

    def test_update_worker_heartbeat_reconnect(self, manager):
        """测试离线节点重新上线"""
        manager.register_worker("worker1", {"cpu": 4})
        manager.update_worker_status("worker1", WorkerStatus.OFFLINE)
        
        # 更新心跳应该使节点重新上线
        manager.update_worker_heartbeat("worker1")
        
        worker = manager._workers["worker1"]
        assert worker.status == WorkerStatus.IDLE

    def test_update_worker_status(self, manager):
        """测试更新工作节点状态"""
        manager.register_worker("worker1", {"cpu": 4})
        
        success = manager.update_worker_status("worker1", WorkerStatus.BUSY)
        assert success is True
        
        worker = manager._workers["worker1"]
        assert worker.status == WorkerStatus.BUSY
        assert manager._stats["busy_workers"] == 1
        assert manager._stats["idle_workers"] == 0

    def test_update_worker_status_not_found(self, manager):
        """测试更新不存在工作节点的状态"""
        success = manager.update_worker_status("nonexistent", WorkerStatus.BUSY)
        assert success is False

    def test_assign_task_to_worker(self, manager):
        """测试分配任务给工作节点"""
        manager.register_worker("worker1", {"cpu": 4})
        
        success = manager.assign_task_to_worker("worker1", "task1")
        assert success is True
        
        worker = manager._workers["worker1"]
        assert worker.current_task == "task1"
        assert worker.status == WorkerStatus.BUSY

    def test_assign_task_to_busy_worker(self, manager):
        """测试分配任务给忙碌的工作节点"""
        manager.register_worker("worker1", {"cpu": 4})
        manager.update_worker_status("worker1", WorkerStatus.BUSY)
        
        success = manager.assign_task_to_worker("worker1", "task1")
        assert success is False  # 忙碌节点不能分配任务

    def test_complete_task(self, manager):
        """测试完成任务"""
        manager.register_worker("worker1", {"cpu": 4})
        manager.assign_task_to_worker("worker1", "task1")
        
        success = manager.complete_task("worker1", processing_time=1.5)
        assert success is True
        
        worker = manager._workers["worker1"]
        assert worker.current_task is None
        assert worker.status == WorkerStatus.IDLE
        assert worker.completed_tasks == 1
        assert worker.total_processing_time == 1.5

    def test_fail_task(self, manager):
        """测试任务失败"""
        manager.register_worker("worker1", {"cpu": 4})
        manager.assign_task_to_worker("worker1", "task1")
        
        success = manager.fail_task("worker1")
        assert success is True
        
        worker = manager._workers["worker1"]
        assert worker.current_task is None
        assert worker.status == WorkerStatus.IDLE
        assert worker.failed_tasks == 1

    def test_check_worker_health(self, manager):
        """测试检查工作节点健康状态"""
        manager.register_worker("worker1", {"cpu": 4})
        # 设置旧的心跳时间
        manager._workers["worker1"].last_heartbeat = datetime.now() - timedelta(minutes=10)
        
        unhealthy = manager.check_worker_health(timeout_minutes=5)
        assert "worker1" in unhealthy
        
        worker = manager._workers["worker1"]
        assert worker.status == WorkerStatus.OFFLINE

    def test_cleanup_offline_workers(self, manager):
        """测试清理离线工作节点"""
        manager.register_worker("worker1", {"cpu": 4})
        manager.update_worker_status("worker1", WorkerStatus.OFFLINE)
        
        cleaned = manager.cleanup_offline_workers()
        assert cleaned == 1
        assert "worker1" not in manager._workers

    def test_get_available_workers(self, manager):
        """测试获取可用工作节点"""
        manager.register_worker("worker1", {"cpu": 4})
        manager.register_worker("worker2", {"cpu": 8})
        manager.update_worker_status("worker2", WorkerStatus.BUSY)
        
        available = manager.get_available_workers()
        assert "worker1" in available
        assert "worker2" not in available

    def test_get_worker_info(self, manager):
        """测试获取工作节点信息"""
        manager.register_worker("worker1", {"cpu": 4})
        
        info = manager.get_worker_info("worker1")
        assert info is not None
        assert info.worker_id == "worker1"

    def test_get_worker_info_not_found(self, manager):
        """测试获取不存在工作节点的信息"""
        info = manager.get_worker_info("nonexistent")
        assert info is None

    def test_get_all_workers(self, manager):
        """测试获取所有工作节点"""
        manager.register_worker("worker1", {"cpu": 4})
        manager.register_worker("worker2", {"cpu": 8})
        
        all_workers = manager.get_all_workers()
        assert len(all_workers) == 2
        assert all(isinstance(w, WorkerInfo) for w in all_workers)

    def test_get_worker_stats(self, manager):
        """测试获取工作节点统计"""
        manager.register_worker("worker1", {"cpu": 4})
        manager.complete_task("worker1", processing_time=1.0)
        
        stats = manager.get_worker_stats()
        
        assert "total_workers" in stats
        assert "active_workers" in stats
        assert "avg_processing_time" in stats
        assert "success_rate" in stats
        assert stats["total_workers"] == 1

    def test_find_best_worker(self, manager):
        """测试找到最适合的工作节点"""
        manager.register_worker("worker1", {"cpu": 4, "max_memory": 8192})
        manager.register_worker("worker2", {"cpu": 8, "max_memory": 16384})
        
        # 查找适合高CPU需求任务的节点
        best = manager.find_best_worker({"cpu_cores": 6})
        assert best is not None
        assert best in ["worker1", "worker2"]

    def test_find_best_worker_no_available(self, manager):
        """测试没有可用工作节点"""
        best = manager.find_best_worker({"cpu_cores": 4})
        assert best is None

    def test_start_stop_monitoring(self, manager):
        """测试启动和停止监控"""
        manager.start_monitoring()
        assert manager._running is True
        
        time.sleep(0.1)
        
        manager.stop_monitoring()
        assert manager._running is False

    def test_start_monitoring_already_running(self, manager):
        """测试重复启动监控"""
        manager.start_monitoring()
        manager.start_monitoring()  # 应该不报错
        manager.stop_monitoring()


class TestGlobalFunctions:
    """全局函数测试"""

    def test_get_worker_manager(self):
        """测试获取全局工作节点管理器"""
        manager = get_worker_manager()
        assert isinstance(manager, FeatureWorkerManager)

