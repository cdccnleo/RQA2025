#!/usr/bin/env python3
"""
历史数据调度器单元测试
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.core.orchestration.historical_data_scheduler import (
    HistoricalDataScheduler,
    SchedulerStatus,
    get_historical_data_scheduler
)
from src.core.monitoring.historical_data_monitor import HistoricalTaskStatus


@pytest.fixture(scope="function")
def scheduler():
    """
    创建测试用的调度器实例
    使用同步方式创建，避免async fixture配置问题
    """
    return HistoricalDataScheduler()


class TestHistoricalDataScheduler:
    """
    历史数据调度器单元测试
    """
    
    @pytest.mark.asyncio
    async def test_scheduler_initialization(self, scheduler):
        """
        测试调度器初始化
        """
        # 启动调度器
        await scheduler.start()
        
        try:
            assert scheduler.status == SchedulerStatus.RUNNING
            assert scheduler.worker_nodes is not None
            assert len(scheduler.worker_nodes) > 0
        finally:
            # 停止调度器
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_start_stop_scheduler(self):
        """
        测试调度器启动和停止
        """
        scheduler = HistoricalDataScheduler()
        
        # 测试启动
        start_result = await scheduler.start()
        assert start_result is True
        assert scheduler.status == SchedulerStatus.RUNNING
        
        # 测试停止
        stop_result = await scheduler.stop()
        assert stop_result is True
        assert scheduler.status == SchedulerStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_scheduler_singleton(self):
        """
        测试调度器单例模式
        """
        scheduler1 = get_historical_data_scheduler()
        scheduler2 = get_historical_data_scheduler()
        
        assert scheduler1 is scheduler2
    
    @pytest.mark.asyncio
    @patch('src.core.orchestration.historical_data_scheduler.HistoricalDataScheduler._execute_task')
    async def test_task_scheduling(self, mock_execute_task, scheduler):
        """
        测试任务调度
        """
        # 启动调度器
        await scheduler.start()
        
        try:
            # 模拟任务执行
            mock_execute_task.return_value = asyncio.Future()
            mock_execute_task.return_value.set_result(None)
            
            # 调度一个测试任务
            task_id = scheduler.schedule_task(
                symbol="000001",
                start_date="2023-01-01",
                end_date="2023-01-02",
                data_types=["price"]
            )
            
            assert task_id is not None
            assert len(scheduler.pending_tasks) == 1
            
            # 等待调度循环执行
            await asyncio.sleep(0.5)
        finally:
            # 停止调度器
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_worker_management(self, scheduler):
        """
        测试工作进程管理
        """
        # 启动调度器
        await scheduler.start()
        
        try:
            # 检查初始工作进程
            assert len(scheduler.worker_nodes) > 0
            
            # 注册一个新的工作进程
            worker_id = "test_worker"
            success = scheduler.register_worker(worker_id, "localhost", 8001, ["historical_data"])
            assert success is True
            assert worker_id in scheduler.worker_nodes
            
            # 注销工作进程
            success = scheduler.unregister_worker(worker_id)
            assert success is True
            assert worker_id not in scheduler.worker_nodes
        finally:
            # 停止调度器
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_task_completion(self, scheduler):
        """
        测试任务完成
        """
        # 启动调度器
        await scheduler.start()
        
        try:
            # 模拟一个任务
            task_id = "test_task_123"
            
            # 确保有工作进程
            worker_id = list(scheduler.worker_nodes.keys())[0]
            
            # 添加到运行任务
            scheduler.running_tasks[task_id] = {
                "symbol": "000001",
                "start_date": "2023-01-01",
                "end_date": "2023-01-02",
                "data_types": ["price"],
                "priority": MagicMock(value=1),
                "worker_id": worker_id,
                "assigned_at": datetime.now().timestamp()
            }
            
            # 添加到监控器
            mock_task = MagicMock()
            scheduler.monitor.tasks[task_id] = mock_task
            
            # 完成任务
            scheduler.complete_task(task_id, records_collected=100)
            
            # 验证任务已从运行列表中移除
            assert task_id not in scheduler.running_tasks
            assert scheduler.stats['tasks_completed'] == 1
        finally:
            # 停止调度器
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_task_failure(self, scheduler):
        """
        测试任务失败处理
        """
        # 启动调度器
        await scheduler.start()
        
        try:
            # 模拟一个任务
            task_id = "test_task_456"
            
            # 确保有工作进程
            worker_id = list(scheduler.worker_nodes.keys())[0]
            
            # 添加到运行任务
            scheduler.running_tasks[task_id] = {
                "symbol": "000001",
                "start_date": "2023-01-01",
                "end_date": "2023-01-02",
                "data_types": ["price"],
                "priority": MagicMock(value=1),
                "worker_id": worker_id,
                "assigned_at": datetime.now().timestamp()
            }
            
            # 模拟监控器任务
            mock_task = MagicMock()
            scheduler.monitor.tasks[task_id] = mock_task
            
            # 完成失败任务
            error_message = "测试错误"
            scheduler.complete_task(task_id, error_message=error_message)
            
            # 验证任务已从运行列表中移除
            assert task_id not in scheduler.running_tasks
            assert scheduler.stats['tasks_failed'] == 1
        finally:
            # 停止调度器
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_task_execution(self, scheduler):
        """
        测试任务执行
        """
        # 启动调度器
        await scheduler.start()
        
        try:
            # 模拟一个任务
            task_info = {
                "symbol": "000001",
                "start_date": "2023-01-01",
                "end_date": "2023-01-02",
                "data_types": ["price"]
            }
            
            task_id = "test_task_789"
            
            # 模拟执行任务
            await scheduler._execute_task(task_id, task_info)
            
            # 验证任务已完成
            assert task_id not in scheduler.running_tasks
        finally:
            # 停止调度器
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_tasks(self, scheduler):
        """
        测试清理过期任务
        """
        # 启动调度器
        await scheduler.start()
        
        try:
            # 创建一个过期任务
            expired_time = datetime.now() - timedelta(hours=2)
            task_id = "expired_task"
            
            # 确保有工作进程
            worker_id = list(scheduler.worker_nodes.keys())[0]
            
            # 添加到运行任务
            scheduler.running_tasks[task_id] = {
                "symbol": "000001",
                "start_date": "2023-01-01",
                "end_date": "2023-01-02",
                "data_types": ["price"],
                "priority": MagicMock(value=1),
                "worker_id": worker_id,
                "assigned_at": expired_time.timestamp()
            }
            
            # 运行清理
            await scheduler._cleanup_expired_tasks()
            
            # 验证任务已被清理
            assert task_id not in scheduler.running_tasks
        finally:
            # 停止调度器
            await scheduler.stop()


if __name__ == "__main__":
    pytest.main([__file__])
