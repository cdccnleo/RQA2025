"""
立即行动阶段功能单元测试

测试内容：
1. 量化交易专用任务类型
2. 任务超时机制
3. 任务重试机制
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.orchestration.scheduler.base import (
    Task, TaskStatus, JobType, generate_task_id
)
from src.core.orchestration.scheduler.task_manager import TaskManager
from src.core.orchestration.scheduler.unified_scheduler import UnifiedScheduler


class TestJobTypeExtension:
    """测试任务类型扩展"""

    def test_quantitative_trading_job_types_exist(self):
        """测试量化交易专用任务类型是否存在"""
        # 信号层任务
        assert JobType.SIGNAL_GENERATION.value == "signal_generation"
        assert JobType.SIGNAL_FILTERING.value == "signal_filtering"
        assert JobType.SIGNAL_AGGREGATION.value == "signal_aggregation"

        # 交易执行层任务
        assert JobType.ORDER_PREPARATION.value == "order_preparation"
        assert JobType.ORDER_VALIDATION.value == "order_validation"
        assert JobType.ORDER_EXECUTION.value == "order_execution"
        assert JobType.ORDER_CONFIRMATION.value == "order_confirmation"

        # 风险控制层任务
        assert JobType.RISK_CALCULATION.value == "risk_calculation"
        assert JobType.RISK_MONITORING.value == "risk_monitoring"
        assert JobType.RISK_ALERTING.value == "risk_alerting"
        assert JobType.POSITION_LIMIT_CHECK.value == "position_limit_check"

        # 组合管理层任务
        assert JobType.PORTFOLIO_CONSTRUCTION.value == "portfolio_construction"
        assert JobType.PORTFOLIO_REBALANCING.value == "portfolio_rebalancing"
        assert JobType.PORTFOLIO_ANALYSIS.value == "portfolio_analysis"

    def test_job_type_layer_organization(self):
        """测试任务类型按层级组织"""
        # 数据层
        data_layer = [JobType.DATA_COLLECTION, JobType.DATA_CLEANING, JobType.DATA_VALIDATION]
        assert all(jt.value.startswith("data_") for jt in data_layer)

        # 特征层
        feature_layer = [JobType.FEATURE_EXTRACTION, JobType.FEATURE_ENGINEERING, JobType.FEATURE_VALIDATION]
        assert all(jt.value.startswith("feature_") for jt in feature_layer)

        # 模型层
        model_layer = [JobType.MODEL_TRAINING, JobType.MODEL_VALIDATION, JobType.MODEL_INFERENCE, JobType.MODEL_DEPLOYMENT]
        assert all(jt.value.startswith("model_") for jt in model_layer)

        # 策略层
        strategy_layer = [JobType.STRATEGY_BACKTEST, JobType.STRATEGY_OPTIMIZATION, JobType.STRATEGY_VALIDATION]
        assert all(jt.value.startswith("strategy_") for jt in strategy_layer)


class TestTaskTimeout:
    """测试任务超时机制"""

    @pytest.mark.asyncio
    async def test_task_creation_with_timeout(self):
        """测试创建带超时时间的任务"""
        task_manager = TaskManager()

        task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test"},
            priority=5,
            timeout_seconds=30,
            max_retries=2
        )

        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.timeout_seconds == 30
        assert task.max_retries == 2
        assert task.deadline is not None
        assert task.deadline > task.created_at

    @pytest.mark.asyncio
    async def test_task_timeout_detection(self):
        """测试任务超时检测"""
        task_manager = TaskManager()

        # 创建一个已超时的任务
        task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test"},
            timeout_seconds=1  # 1秒超时
        )

        # 等待任务超时
        await asyncio.sleep(1.5)

        task = task_manager.get_task(task_id)
        assert task.is_timeout() is True

    @pytest.mark.asyncio
    async def test_task_not_timeout(self):
        """测试未超时任务"""
        task_manager = TaskManager()

        task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test"},
            timeout_seconds=60  # 60秒超时
        )

        task = task_manager.get_task(task_id)
        assert task.is_timeout() is False
        assert task.get_remaining_time() > 50  # 剩余时间应该大于50秒

    @pytest.mark.asyncio
    async def test_get_timeout_tasks(self):
        """测试获取所有超时任务"""
        task_manager = TaskManager()

        # 创建超时任务
        timeout_task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test1"},
            timeout_seconds=1
        )

        # 创建未超时任务
        normal_task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test2"},
            timeout_seconds=60
        )

        # 等待超时
        await asyncio.sleep(1.5)

        timeout_tasks = task_manager.get_timeout_tasks()
        assert len(timeout_tasks) == 1
        assert timeout_tasks[0].id == timeout_task_id

    @pytest.mark.asyncio
    async def test_mark_task_timeout(self):
        """测试标记任务为超时"""
        task_manager = TaskManager()

        task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test"},
            timeout_seconds=60
        )

        # 更新状态为运行中
        await task_manager.update_task_status(task_id, TaskStatus.RUNNING)

        # 标记为超时
        success = await task_manager.mark_task_timeout(task_id)
        assert success is True

        # 验证任务状态
        task = task_manager.get_task(task_id)
        # 超时任务应该被移动到历史记录
        assert task is None or task.status == TaskStatus.FAILED


class TestTaskRetry:
    """测试任务重试机制"""

    @pytest.mark.asyncio
    async def test_task_retry_creation(self):
        """测试重试任务创建"""
        task_manager = TaskManager()

        # 创建失败的任务
        task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test"},
            max_retries=3,
            retry_delay_seconds=1
        )

        # 模拟任务失败
        await task_manager.update_task_status(
            task_id,
            TaskStatus.FAILED,
            error="Test error"
        )

        # 重试任务
        new_task_id = await task_manager.retry_task(task_id)
        assert new_task_id is not None
        assert new_task_id != task_id

        # 验证新任务
        new_task = task_manager.get_task(new_task_id)
        assert new_task is not None
        assert new_task.retry_count == 1
        assert new_task.max_retries == 3
        assert new_task.payload.get("_retry_info") is not None
        assert new_task.payload["_retry_info"]["original_task_id"] == task_id

    @pytest.mark.asyncio
    async def test_retry_exceeds_max_retries(self):
        """测试超过最大重试次数"""
        task_manager = TaskManager()

        # 创建任务，最大重试1次
        task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test"},
            max_retries=1
        )

        # 模拟任务失败
        await task_manager.update_task_status(task_id, TaskStatus.FAILED, error="Error 1")

        # 第一次重试
        new_task_id = await task_manager.retry_task(task_id)
        assert new_task_id is not None

        # 模拟新任务也失败
        await task_manager.update_task_status(new_task_id, TaskStatus.FAILED, error="Error 2")

        # 第二次重试应该失败（已达到最大重试次数）
        second_retry_id = await task_manager.retry_task(new_task_id)
        assert second_retry_id is None

    @pytest.mark.asyncio
    async def test_retry_non_failed_task(self):
        """测试重试非失败任务"""
        task_manager = TaskManager()

        # 创建成功的任务
        task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test"},
            max_retries=3
        )

        await task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result="success")

        # 尝试重试成功的任务应该失败
        new_task_id = await task_manager.retry_task(task_id)
        assert new_task_id is None

    @pytest.mark.asyncio
    async def test_should_retry_logic(self):
        """测试是否应该重试的逻辑"""
        task_manager = TaskManager()

        # 创建任务
        task_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test"},
            max_retries=2
        )

        task = task_manager.get_task(task_id)
        assert task.should_retry() is False  # 任务未失败，不应该重试

        # 模拟失败
        await task_manager.update_task_status(task_id, TaskStatus.FAILED, error="Error")

        # 从历史记录中获取任务
        task_from_history = None
        for t in task_manager._task_history:
            if t.id == task_id:
                task_from_history = t
                break

        assert task_from_history is not None
        assert task_from_history.should_retry() is True  # 失败且未超过重试次数

    @pytest.mark.asyncio
    async def test_get_tasks_needing_retry(self):
        """测试获取需要重试的任务列表"""
        task_manager = TaskManager()

        # 创建两个可重试的任务
        task1_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test1"},
            max_retries=2
        )
        await task_manager.update_task_status(task1_id, TaskStatus.FAILED, error="Error 1")

        task2_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test2"},
            max_retries=2
        )
        await task_manager.update_task_status(task2_id, TaskStatus.FAILED, error="Error 2")

        # 创建一个不可重试的任务（max_retries=0）
        task3_id = await task_manager.create_task(
            task_type="data_collection",
            payload={"source": "test3"},
            max_retries=0
        )
        await task_manager.update_task_status(task3_id, TaskStatus.FAILED, error="Error 3")

        retry_tasks = task_manager.get_tasks_needing_retry()
        assert len(retry_tasks) == 2
        task_ids = [t.id for t in retry_tasks]
        assert task1_id in task_ids
        assert task2_id in task_ids
        assert task3_id not in task_ids


class TestUnifiedSchedulerIntegration:
    """测试统一调度器集成"""

    @pytest.fixture
    def scheduler(self):
        """创建调度器实例"""
        return UnifiedScheduler(max_workers=2, max_task_history=100)

    @pytest.mark.asyncio
    async def test_submit_task_with_timeout_and_retry(self, scheduler):
        """测试提交带超时和重试参数的任务"""
        task_id = await scheduler.submit_task(
            task_type="signal_generation",
            payload={"strategy": "test"},
            priority=3,
            timeout_seconds=30,
            max_retries=2,
            retry_delay_seconds=5
        )

        assert task_id is not None
        assert task_id.startswith("task-")

        task = scheduler.get_task_detail(task_id)
        assert task is not None
        assert task["timeout_seconds"] == 30
        assert task["max_retries"] == 2
        assert task["retry_delay_seconds"] == 5

    @pytest.mark.asyncio
    async def test_retry_task_api(self, scheduler):
        """测试重试任务API"""
        # 先提交一个任务
        task_id = await scheduler.submit_task(
            task_type="order_execution",
            payload={"order_id": "123"},
            max_retries=2
        )

        # 模拟任务失败
        await scheduler._task_manager.update_task_status(
            task_id, TaskStatus.FAILED, error="Execution failed"
        )

        # 重试任务
        new_task_id = await scheduler.retry_task(task_id)
        assert new_task_id is not None

        # 验证新任务状态
        new_task = scheduler.get_task_detail(new_task_id)
        assert new_task is not None
        assert new_task["status"] == "running"


class TestTaskDataClass:
    """测试Task数据类的新功能"""

    def test_task_with_timeout_fields(self):
        """测试带超时字段的任务"""
        task = Task(
            id="task-test-001",
            type="data_collection",
            status=TaskStatus.PENDING,
            priority=5,
            created_at=datetime.now(),
            timeout_seconds=60,
            max_retries=3,
            retry_count=0,
            retry_delay_seconds=5
        )

        assert task.timeout_seconds == 60
        assert task.max_retries == 3
        assert task.retry_count == 0
        assert task.retry_delay_seconds == 5
        assert task.deadline is not None

    def test_task_to_dict_includes_timeout_info(self):
        """测试to_dict包含超时信息"""
        task = Task(
            id="task-test-002",
            type="signal_generation",
            status=TaskStatus.RUNNING,
            priority=5,
            created_at=datetime.now(),
            timeout_seconds=30,
            max_retries=2
        )

        task_dict = task.to_dict()
        assert "timeout_seconds" in task_dict
        assert "max_retries" in task_dict
        assert "retry_count" in task_dict
        assert "retry_delay_seconds" in task_dict
        assert "deadline" in task_dict
        assert "is_timeout" in task_dict
        assert "remaining_time" in task_dict

    def test_task_without_timeout(self):
        """测试无超时设置的任务"""
        task = Task(
            id="task-test-003",
            type="data_collection",
            status=TaskStatus.PENDING,
            priority=5,
            created_at=datetime.now()
        )

        assert task.timeout_seconds is None
        assert task.deadline is None
        assert task.is_timeout() is False
        assert task.get_remaining_time() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
