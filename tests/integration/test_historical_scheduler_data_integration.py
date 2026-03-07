#!/usr/bin/env python3
"""
历史数据调度器数据层集成测试
测试调度器与数据采集、缓存、质量监控等组件的集成
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import asyncio

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.historical_data_scheduler import (
    HistoricalDataScheduler,
    SchedulerStatus
)
from src.core.monitoring.historical_data_monitor import HistoricalTaskStatus


class TestHistoricalSchedulerDataIntegration:
    """历史数据调度器数据层集成测试类"""

    def setup_method(self):
        """
        测试前准备
        """
        self.scheduler = HistoricalDataScheduler()
        self.loop = asyncio.get_event_loop()
        # 启动调度器
        self.loop.run_until_complete(self.scheduler.start())

    def teardown_method(self):
        """
        测试后清理
        """
        # 停止调度器
        self.loop.run_until_complete(self.scheduler.stop())

    def test_scheduler_with_akshare_integration(self):
        """
        测试调度器与AKShare数据源的集成
        """
        # 模拟AKShare数据源可用
        with patch('src.core.orchestration.historical_data_scheduler.akshare_available', True):
            with patch('src.core.orchestration.historical_data_scheduler.ak.stock_zh_a_hist') as mock_ak_hist:
                # 简化模拟，不使用复杂的DataFrame模拟
                mock_ak_hist.return_value = Mock()
                mock_ak_hist.return_value.iterrows.return_value = []

                # 模拟基本面数据
                with patch('src.core.orchestration.historical_data_scheduler.ak.stock_individual_info_em') as mock_ak_info:
                    mock_info = {'股票简称': '平安银行', '所属行业': '银行'}
                    mock_ak_info.return_value = mock_info

                    # 调度一个任务
                    task_id = self.scheduler.schedule_task(
                        symbol="000001",
                        start_date="2020-01-01",
                        end_date="2020-01-02",
                        data_types=["price"]
                    )

                    # 等待任务执行
                    self.loop.run_until_complete(asyncio.sleep(1))

                    # 验证任务状态
                    task_status = self.scheduler.monitor.tasks.get(task_id)
                    assert task_status is not None
                    assert task_status.status in [HistoricalTaskStatus.RUNNING, HistoricalTaskStatus.COMPLETED]

    def test_scheduler_with_cache_integration(self):
        """
        测试调度器与缓存机制的集成
        """
        # 调度第一个任务（缓存未命中）
        task_id1 = self.scheduler.schedule_task(
            symbol="000001",
            start_date="2020-01-01",
            end_date="2020-01-02",
            data_types=["price"]
        )

        # 等待第一个任务完成
        self.loop.run_until_complete(asyncio.sleep(1.5))

        # 调度第二个相同的任务（应该缓存命中）
        task_id2 = self.scheduler.schedule_task(
            symbol="000001",
            start_date="2020-01-01",
            end_date="2020-01-02",
            data_types=["price"]
        )

        # 等待第二个任务完成
        self.loop.run_until_complete(asyncio.sleep(1))

        # 验证两个任务都已完成
        completed_task_ids = [t.task_id for t in self.scheduler.monitor.tasks.values() if t.status == HistoricalTaskStatus.COMPLETED]
        assert task_id1 in completed_task_ids
        assert task_id2 in completed_task_ids

    def test_scheduler_data_quality_integration(self):
        """
        测试调度器与数据质量监控的集成
        """
        # 调度一个任务
        task_id = self.scheduler.schedule_task(
            symbol="000001",
            start_date="2020-01-01",
            end_date="2020-01-02",
            data_types=["price"]
        )

        # 等待任务执行
        self.loop.run_until_complete(asyncio.sleep(1))

        # 验证任务不会因为质量监控错误而失败
        task_status = self.scheduler.monitor.tasks.get(task_id)
        assert task_status is not None
        assert task_status.status in [HistoricalTaskStatus.RUNNING, HistoricalTaskStatus.COMPLETED]

    def test_scheduler_complete_workflow(self):
        """
        测试调度器完整工作流程
        """
        # 调度一个任务
        task_id = self.scheduler.schedule_task(
            symbol="000001",
            start_date="2020-01-01",
            end_date="2020-01-02",
            data_types=["price", "volume"]
        )

        # 等待任务完成
        self.loop.run_until_complete(asyncio.sleep(2))

        # 验证任务状态
        task_status = self.scheduler.monitor.tasks.get(task_id)
        assert task_status is not None
        assert task_status.status == HistoricalTaskStatus.COMPLETED
        assert task_status.records_collected >= 0

    def test_scheduler_task_lifecycle(self):
        """
        测试调度器任务生命周期
        """
        # 调度一个任务
        task_id = self.scheduler.schedule_task(
            symbol="000001",
            start_date="2020-01-01",
            end_date="2020-01-02",
            data_types=["price"]
        )

        # 验证任务已创建
        assert task_id in self.scheduler.monitor.tasks
        assert self.scheduler.monitor.tasks[task_id].status == HistoricalTaskStatus.PENDING

        # 等待任务开始执行
        self.loop.run_until_complete(asyncio.sleep(0.5))

        # 验证任务正在执行
        assert self.scheduler.monitor.tasks[task_id].status == HistoricalTaskStatus.RUNNING

        # 等待任务完成
        self.loop.run_until_complete(asyncio.sleep(1.5))

        # 验证任务已完成
        assert self.scheduler.monitor.tasks[task_id].status == HistoricalTaskStatus.COMPLETED

    def test_scheduler_with_multiple_data_types(self):
        """
        测试调度器处理多种数据类型
        """
        # 调度一个包含多种数据类型的任务
        task_id = self.scheduler.schedule_task(
            symbol="000001",
            start_date="2020-01-01",
            end_date="2020-01-02",
            data_types=["price", "volume", "fundamental"]
        )

        # 等待任务完成
        self.loop.run_until_complete(asyncio.sleep(2))

        # 验证任务已完成
        assert self.scheduler.monitor.tasks[task_id].status == HistoricalTaskStatus.COMPLETED

    def test_scheduler_worker_management(self):
        """
        测试调度器与工作进程管理的集成
        """
        # 验证默认工作进程已注册
        assert len(self.scheduler.worker_nodes) > 0

        # 注册一个新的工作进程
        worker_id = "test_integration_worker"
        success = self.scheduler.register_worker(worker_id, "localhost", 8001, ["historical_data"])
        assert success is True
        assert worker_id in self.scheduler.worker_nodes

        # 调度一个任务
        task_id = self.scheduler.schedule_task(
            symbol="000001",
            start_date="2020-01-01",
            end_date="2020-01-02",
            data_types=["price"]
        )

        # 等待任务分配
        self.loop.run_until_complete(asyncio.sleep(0.5))

        # 验证任务已分配到工作进程
        task = self.scheduler.monitor.tasks.get(task_id)
        assert task is not None
        assert task.worker_id is not None

        # 注销工作进程
        success = self.scheduler.unregister_worker(worker_id)
        assert success is True
        assert worker_id not in self.scheduler.worker_nodes

    def test_scheduler_stats_integration(self):
        """
        测试调度器统计信息集成
        """
        # 调度多个任务
        for i in range(3):
            self.scheduler.schedule_task(
                symbol=f"00000{i+1}",
                start_date="2020-01-01",
                end_date="2020-01-02",
                data_types=["price"]
            )

        # 等待任务完成
        self.loop.run_until_complete(asyncio.sleep(2))

        # 验证统计信息
        stats = self.scheduler.get_scheduler_status()
        assert stats['tasks_scheduled'] >= 3
        assert stats['workers']['total'] > 0

    def test_scheduler_with_fallback_data(self):
        """
        测试调度器使用模拟数据的情况
        """
        # 模拟AKShare不可用
        with patch('src.core.orchestration.historical_data_scheduler.akshare_available', False):
            # 调度一个任务
            task_id = self.scheduler.schedule_task(
                symbol="000001",
                start_date="2020-01-01",
                end_date="2020-01-02",
                data_types=["price"]
            )

            # 等待任务完成
            self.loop.run_until_complete(asyncio.sleep(1.5))

            # 验证任务已完成
            assert self.scheduler.monitor.tasks[task_id].status == HistoricalTaskStatus.COMPLETED
            assert self.scheduler.monitor.tasks[task_id].records_collected > 0

    def test_scheduler_concurrent_tasks(self):
        """
        测试调度器处理并发任务
        """
        # 调度多个并发任务
        task_ids = []
        for i in range(5):
            task_id = self.scheduler.schedule_task(
                symbol=f"00000{i+1}",
                start_date="2020-01-01",
                end_date="2020-01-02",
                data_types=["price"]
            )
            task_ids.append(task_id)

        # 等待任务完成
        self.loop.run_until_complete(asyncio.sleep(3))

        # 验证所有任务都已完成或正在执行
        completed_tasks = 0
        for task_id in task_ids:
            task = self.scheduler.monitor.tasks.get(task_id)
            assert task is not None
            if task.status == HistoricalTaskStatus.COMPLETED:
                completed_tasks += 1

        # 至少有一些任务应该已完成
        assert completed_tasks > 0


if __name__ == '__main__':
    pytest.main([__file__])
