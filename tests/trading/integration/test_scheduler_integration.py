# -*- coding: utf-8 -*-
"""
交易层统一调度器集成测试

测试交易层与统一调度器的集成能力，包括任务提交、状态查询、结果回调等功能。

函数级注释:
- 所有测试函数均包含详细的中文注释
- 使用pytest框架进行测试
- 支持异步测试

作者: AI系统集成助手
日期: 2026-03-22
版本: 1.0.0
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# 导入被测试模块
from src.trading.integration.scheduler_integration import (
    TradingSchedulerIntegration,
    TradingTaskType,
    TradingTaskConfig,
    TradingTaskResult,
    get_trading_scheduler_integration,
)
from src.trading.execution.order_manager import Order, OrderType, OrderSide, OrderStatus
from src.trading.execution.execution_types import ExecutionMode
from src.core.orchestration.scheduler import TaskStatus


class TestTradingSchedulerIntegration:
    """交易调度器集成测试类"""
    
    @pytest.fixture
    def integration(self):
        """创建集成实例"""
        # 重置单例状态
        TradingSchedulerIntegration._instance = None
        return TradingSchedulerIntegration()
    
    @pytest.fixture
    def mock_scheduler(self):
        """创建模拟调度器"""
        scheduler = Mock()
        scheduler.is_running.return_value = True
        scheduler.submit_task = AsyncMock(return_value="task_123")
        scheduler.get_task_detail = Mock(return_value=None)
        scheduler.cancel_task = AsyncMock(return_value=True)
        scheduler.get_status.return_value = {"workers": {}}
        scheduler.get_statistics.return_value = {"total": 0}
        return scheduler
    
    @pytest.fixture
    def sample_order(self):
        """创建示例订单"""
        return Order(
            order_id="order_123",
            symbol="000001.SZ",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            price=10.5,
            strategy_id="strategy_001",
            account_id="account_001"
        )
    
    @pytest.mark.asyncio
    async def test_singleton_pattern(self, integration):
        """测试单例模式"""
        # 获取两个实例
        instance1 = get_trading_scheduler_integration()
        instance2 = get_trading_scheduler_integration()
        
        # 验证是同一个实例
        assert instance1 is instance2
        assert isinstance(instance1, TradingSchedulerIntegration)
    
    @pytest.mark.asyncio
    async def test_submit_task_success(self, integration, mock_scheduler):
        """测试成功提交任务"""
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            payload = {"test": "data"}
            config = TradingTaskConfig(priority=2, max_retries=3)
            
            task_id = await integration.submit_task(
                task_type=TradingTaskType.ORDER_EXECUTION,
                payload=payload,
                config=config
            )
            
            assert task_id == "task_123"
            mock_scheduler.submit_task.assert_called_once()
            
            # 验证调用参数
            call_args = mock_scheduler.submit_task.call_args
            assert call_args[1]['task_type'] == "order_execution"
            assert call_args[1]['priority'] == 2
            assert call_args[1]['max_retries'] == 3
    
    @pytest.mark.asyncio
    async def test_submit_task_failure(self, integration, mock_scheduler):
        """测试提交任务失败"""
        mock_scheduler.submit_task = AsyncMock(side_effect=Exception("调度器错误"))
        
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            with pytest.raises(RuntimeError) as exc_info:
                await integration.submit_task(
                    task_type=TradingTaskType.ORDER_EXECUTION,
                    payload={}
                )
            
            assert "提交交易任务失败" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_submit_order_execution(self, integration, mock_scheduler, sample_order):
        """测试提交订单执行任务"""
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            task_id = await integration.submit_order_execution(
                order=sample_order,
                execution_mode=ExecutionMode.MARKET
            )
            
            assert task_id == "task_123"
            
            # 验证调用参数
            call_args = mock_scheduler.submit_task.call_args
            payload = call_args[1]['payload']['payload']
            assert payload['order_id'] == "order_123"
            assert payload['symbol'] == "000001.SZ"
            assert payload['side'] == "buy"
            assert call_args[1]['priority'] == 2  # 订单执行使用高优先级
    
    @pytest.mark.asyncio
    async def test_submit_order_validation(self, integration, mock_scheduler, sample_order):
        """测试提交订单验证任务"""
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            task_id = await integration.submit_order_validation(order=sample_order)
            
            assert task_id == "task_123"
            
            # 验证调用参数
            call_args = mock_scheduler.submit_task.call_args
            assert call_args[1]['task_type'] == "order_validation"
            assert call_args[1]['priority'] == 1  # 验证使用最高优先级
            assert call_args[1]['timeout_seconds'] == 5  # 验证使用短超时
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, integration, mock_scheduler):
        """测试获取任务状态"""
        mock_task = Mock()
        mock_task.status = TaskStatus.RUNNING
        mock_scheduler.get_task_detail.return_value = mock_task
        
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            status = await integration.get_task_status("task_123")
            
            assert status == TaskStatus.RUNNING
            mock_scheduler.get_task_detail.assert_called_once_with("task_123")
    
    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, integration, mock_scheduler):
        """测试获取不存在的任务状态"""
        mock_scheduler.get_task_detail.return_value = None
        
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            status = await integration.get_task_status("non_existent_task")
            
            assert status is None
    
    @pytest.mark.asyncio
    async def test_wait_for_task_completion_success(self, integration, mock_scheduler):
        """测试等待任务完成成功"""
        mock_task = Mock()
        mock_task.status = TaskStatus.COMPLETED
        mock_scheduler.get_task_detail.return_value = mock_task
        
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            result = await integration.wait_for_task_completion("task_123", timeout_seconds=1.0)
            
            assert result.task_id == "task_123"
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_wait_for_task_completion_timeout(self, integration, mock_scheduler):
        """测试等待任务完成超时"""
        mock_task = Mock()
        mock_task.status = TaskStatus.PENDING
        mock_scheduler.get_task_detail.return_value = mock_task
        
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            with pytest.raises(TimeoutError):
                await integration.wait_for_task_completion("task_123", timeout_seconds=0.1, poll_interval=0.01)
    
    @pytest.mark.asyncio
    async def test_cancel_task_success(self, integration, mock_scheduler):
        """测试成功取消任务"""
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            success = await integration.cancel_task("task_123")
            
            assert success is True
            mock_scheduler.cancel_task.assert_called_once_with("task_123")
    
    @pytest.mark.asyncio
    async def test_cancel_task_failure(self, integration, mock_scheduler):
        """测试取消任务失败"""
        mock_scheduler.cancel_task = AsyncMock(side_effect=Exception("取消失败"))
        
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            success = await integration.cancel_task("task_123")
            
            assert success is False
    
    def test_register_task_result(self, integration):
        """测试注册任务结果"""
        result = TradingTaskResult(
            task_id="task_123",
            success=True,
            data={"filled_quantity": 100}
        )
        
        integration.register_task_result("task_123", result)
        
        assert integration._task_results["task_123"] == result
    
    def test_register_task_result_with_callback(self, integration):
        """测试注册任务结果并触发回调"""
        callback_called = False
        received_result = None
        
        def callback(result):
            nonlocal callback_called, received_result
            callback_called = True
            received_result = result
        
        # 注册回调
        integration._task_callbacks["task_123"] = callback
        
        result = TradingTaskResult(
            task_id="task_123",
            success=True,
            data={"filled_quantity": 100}
        )
        
        integration.register_task_result("task_123", result)
        
        assert callback_called is True
        assert received_result == result
        assert "task_123" not in integration._task_callbacks  # 回调后清理
    
    @pytest.mark.asyncio
    async def test_get_scheduler_status(self, integration, mock_scheduler):
        """测试获取调度器状态"""
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            status = await integration.get_scheduler_status()
            
            assert status["is_running"] is True
            assert "status" in status
            assert "statistics" in status
    
    def test_trading_task_config_defaults(self):
        """测试交易任务配置默认值"""
        config = TradingTaskConfig()
        
        assert config.priority == 5
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1
        assert config.enable_callback is True
    
    def test_trading_task_config_custom(self):
        """测试交易任务配置自定义值"""
        config = TradingTaskConfig(
            priority=1,
            timeout_seconds=10,
            max_retries=5,
            retry_delay_seconds=2,
            enable_callback=False
        )
        
        assert config.priority == 1
        assert config.timeout_seconds == 10
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2
        assert config.enable_callback is False
    
    def test_trading_task_result(self):
        """测试交易任务结果"""
        result = TradingTaskResult(
            task_id="task_123",
            success=True,
            data={"filled_quantity": 100, "avg_price": 10.5},
            error_message=None,
            execution_time_ms=15.5
        )
        
        assert result.task_id == "task_123"
        assert result.success is True
        assert result.data["filled_quantity"] == 100
        assert result.execution_time_ms == 15.5


class TestTradingTaskTypeMapping:
    """交易任务类型映射测试类"""
    
    @pytest.fixture
    def integration(self):
        """创建集成实例"""
        TradingSchedulerIntegration._instance = None
        return TradingSchedulerIntegration()
    
    def test_order_preparation_mapping(self, integration):
        """测试订单准备任务类型映射"""
        job_type = integration._map_to_job_type(TradingTaskType.ORDER_PREPARATION)
        assert job_type == "order_preparation"
    
    def test_order_validation_mapping(self, integration):
        """测试订单验证任务类型映射"""
        job_type = integration._map_to_job_type(TradingTaskType.ORDER_VALIDATION)
        assert job_type == "order_validation"
    
    def test_order_execution_mapping(self, integration):
        """测试订单执行任务类型映射"""
        job_type = integration._map_to_job_type(TradingTaskType.ORDER_EXECUTION)
        assert job_type == "order_execution"
    
    def test_order_confirmation_mapping(self, integration):
        """测试订单确认任务类型映射"""
        job_type = integration._map_to_job_type(TradingTaskType.ORDER_CONFIRMATION)
        assert job_type == "order_confirmation"
    
    def test_trade_processing_mapping(self, integration):
        """测试交易处理任务类型映射"""
        job_type = integration._map_to_job_type(TradingTaskType.TRADE_PROCESSING)
        assert job_type == "trade_processing"
    
    def test_portfolio_rebalance_mapping(self, integration):
        """测试组合再平衡任务类型映射"""
        job_type = integration._map_to_job_type(TradingTaskType.PORTFOLIO_REBALANCE)
        assert job_type == "portfolio_rebalancing"


class TestTaskPriorityMapping:
    """任务优先级映射测试类"""
    
    @pytest.fixture
    def integration(self):
        """创建集成实例"""
        TradingSchedulerIntegration._instance = None
        return TradingSchedulerIntegration()
    
    def test_critical_priority_mapping(self, integration):
        """测试关键优先级映射"""
        from src.core.orchestration.scheduler import TaskPriority
        priority = integration._map_to_task_priority(1)
        assert priority == TaskPriority.CRITICAL
        
        priority = integration._map_to_task_priority(2)
        assert priority == TaskPriority.CRITICAL
    
    def test_high_priority_mapping(self, integration):
        """测试高优先级映射"""
        from src.core.orchestration.scheduler import TaskPriority
        priority = integration._map_to_task_priority(3)
        assert priority == TaskPriority.HIGH
        
        priority = integration._map_to_task_priority(4)
        assert priority == TaskPriority.HIGH
    
    def test_normal_priority_mapping(self, integration):
        """测试普通优先级映射"""
        from src.core.orchestration.scheduler import TaskPriority
        priority = integration._map_to_task_priority(5)
        assert priority == TaskPriority.NORMAL
        
        priority = integration._map_to_task_priority(6)
        assert priority == TaskPriority.NORMAL
    
    def test_low_priority_mapping(self, integration):
        """测试低优先级映射"""
        from src.core.orchestration.scheduler import TaskPriority
        priority = integration._map_to_task_priority(7)
        assert priority == TaskPriority.LOW
        
        priority = integration._map_to_task_priority(10)
        assert priority == TaskPriority.LOW


# 性能测试
@pytest.mark.asyncio
async def test_concurrent_task_submission():
    """测试并发任务提交性能"""
    integration = TradingSchedulerIntegration()
    TradingSchedulerIntegration._instance = None
    
    mock_scheduler = Mock()
    mock_scheduler.is_running.return_value = True
    mock_scheduler.submit_task = AsyncMock(return_value="task_id")
    
    with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
        # 并发提交100个任务
        tasks = []
        for i in range(100):
            task = integration.submit_task(
                task_type=TradingTaskType.ORDER_EXECUTION,
                payload={"index": i}
            )
            tasks.append(task)
        
        start_time = datetime.now()
        results = await asyncio.gather(*tasks)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # 验证所有任务提交成功
        assert len(results) == 100
        assert all(r == "task_id" for r in results)
        
        # 验证性能(100个任务应在1秒内完成)
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
