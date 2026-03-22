# -*- coding: utf-8 -*-
"""
系统集成测试套件

测试系统各层与统一调度器及事件总线的集成情况。

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
from src.trading.integration import (
    get_trading_scheduler_integration,
    get_risk_intercept_integration,
    TradingTaskType,
)
from src.data.integration import (
    get_data_scheduler_integration,
    DataTaskType,
)
from src.strategy.integration import (
    get_strategy_event_bus_integration,
    StrategySignal,
    SignalType,
)
from src.trading.events import (
    get_order_state_event_publisher,
    OrderState,
)
from src.core.performance import get_integration_performance_monitor
from src.core.compatibility import get_integration_compatibility_manager


class TestTradingSchedulerIntegration:
    """交易调度器集成测试"""
    
    @pytest.fixture
    def mock_scheduler(self):
        """创建模拟调度器"""
        scheduler = Mock()
        scheduler.is_running.return_value = True
        scheduler.submit_task = AsyncMock(return_value="task_123")
        scheduler.get_task_detail = Mock(return_value=None)
        return scheduler
    
    @pytest.mark.asyncio
    async def test_trading_scheduler_submit_task(self, mock_scheduler):
        """测试交易调度器提交任务"""
        with patch('src.trading.integration.scheduler_integration.get_unified_scheduler', return_value=mock_scheduler):
            integration = get_trading_scheduler_integration()
            
            task_id = await integration.submit_task(
                task_type=TradingTaskType.ORDER_EXECUTION,
                payload={"order_id": "ORD-001"}
            )
            
            assert task_id == "task_123"
            mock_scheduler.submit_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trading_scheduler_integration_end_to_end(self):
        """测试交易调度器端到端集成"""
        # 模拟完整的交易流程
        integration = get_trading_scheduler_integration()
        
        # 提交订单执行任务
        with patch.object(integration, '_scheduler') as mock_scheduler:
            mock_scheduler.submit_task = AsyncMock(return_value="task_exec_001")
            
            task_id = await integration.submit_order_execution(
                order=Mock(order_id="ORD-001", symbol="000001.SZ"),
                execution_mode="market"
            )
            
            assert task_id == "task_exec_001"


class TestDataSchedulerIntegration:
    """数据调度器集成测试"""
    
    @pytest.mark.asyncio
    async def test_data_scheduler_submit_collection(self):
        """测试数据调度器提交采集任务"""
        integration = get_data_scheduler_integration()
        
        with patch.object(integration, '_scheduler') as mock_scheduler:
            mock_scheduler.submit_task = AsyncMock(return_value="task_data_001")
            
            task_id = await integration.submit_collection_task(
                symbols=["000001.SZ", "000002.SZ"],
                data_source="tushare"
            )
            
            assert task_id == "task_data_001"
    
    @pytest.mark.asyncio
    async def test_data_scheduler_create_scheduled_job(self):
        """测试数据调度器创建定时任务"""
        integration = get_data_scheduler_integration()
        
        with patch.object(integration, '_scheduler') as mock_scheduler:
            mock_scheduler.create_job = AsyncMock(return_value="job_001")
            
            job_id = await integration.create_scheduled_collection(
                symbols=["000001.SZ"],
                cron="0 9 * * *"
            )
            
            assert job_id == "job_001"


class TestStrategyEventBusIntegration:
    """策略事件总线集成测试"""
    
    @pytest.mark.asyncio
    async def test_strategy_publish_signal(self):
        """测试策略信号发布"""
        integration = get_strategy_event_bus_integration()
        
        with patch.object(integration, '_event_bus') as mock_event_bus:
            mock_event_bus.publish = AsyncMock()
            
            signal = StrategySignal(
                signal_id="SIG-001",
                strategy_id="STRAT-001",
                symbol="000001.SZ",
                signal_type=SignalType.BUY,
                strength=0.85
            )
            
            success = await integration.publish_signal(signal)
            
            assert success is True
            mock_event_bus.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_strategy_subscribe_to_signals(self):
        """测试策略信号订阅"""
        integration = get_strategy_event_bus_integration()
        
        with patch.object(integration, '_event_bus') as mock_event_bus:
            mock_event_bus.subscribe = AsyncMock(return_value="sub_001")
            
            callback = Mock()
            subscription_id = await integration.subscribe_to_trading_signals(
                callback=callback,
                symbols=["000001.SZ"]
            )
            
            assert subscription_id == "sub_001"


class TestRiskInterceptIntegration:
    """风险拦截集成测试"""
    
    @pytest.mark.asyncio
    async def test_risk_intercept_subscription(self):
        """测试风险拦截事件订阅"""
        integration = get_risk_intercept_integration()
        
        with patch.object(integration, '_event_bus') as mock_event_bus:
            mock_event_bus.subscribe = AsyncMock(return_value="risk_sub_001")
            
            await integration.start()
            
            assert integration._subscription_id == "risk_sub_001"


class TestOrderStateEvents:
    """订单状态事件测试"""
    
    @pytest.mark.asyncio
    async def test_order_state_transition(self):
        """测试订单状态转换"""
        from src.trading.events.order_state_events import OrderStateMachine
        
        sm = OrderStateMachine(order_id="ORD-001")
        
        # 验证初始状态
        assert sm.current_state == OrderState.CREATED
        
        # 验证允许的状态转换
        assert sm.can_transition_to(OrderState.VALIDATING) is True
        
        # 执行状态转换
        success, transition = await sm.transition_to(
            target_state=OrderState.VALIDATING,
            reason="开始验证"
        )
        
        assert success is True
        assert sm.current_state == OrderState.VALIDATING
    
    @pytest.mark.asyncio
    async def test_order_state_event_publish(self):
        """测试订单状态事件发布"""
        publisher = get_order_state_event_publisher()
        
        with patch.object(publisher, '_event_bus') as mock_event_bus:
            mock_event_bus.publish = AsyncMock()
            
            success = await publisher.publish_state_changed(
                order_id="ORD-001",
                from_state=OrderState.CREATED,
                to_state=OrderState.VALIDATING,
                reason="开始验证"
            )
            
            assert success is True
            mock_event_bus.publish.assert_called_once()


class TestPerformanceMonitor:
    """性能监控测试"""
    
    @pytest.mark.asyncio
    async def test_performance_monitor_record(self):
        """测试性能监控记录"""
        monitor = get_integration_performance_monitor()
        
        # 记录延迟
        await monitor.record_latency(
            integration_name="test_integration",
            latency_ms=15.5,
            operation="test_op",
            success=True
        )
        
        # 获取统计
        stats = monitor.get_stats("test_integration")
        
        assert stats is not None
        assert stats.total_calls == 1
        assert stats.success_calls == 1
    
    def test_performance_monitor_decorator(self):
        """测试性能监控装饰器"""
        from src.core.performance.integration_monitor import performance_monitor
        
        @performance_monitor("test_integration", "test_op")
        def test_function():
            return "result"
        
        result = test_function()
        assert result == "result"


class TestCompatibilityManager:
    """兼容性管理测试"""
    
    def test_compatibility_check(self):
        """测试兼容性检查"""
        from src.core.compatibility.integration_adapter import CompatibilityLevel
        
        manager = get_integration_compatibility_manager()
        
        # 测试完全兼容
        level = manager.check_compatibility("1.0.0", "1.0.1")
        assert level == CompatibilityLevel.FULL
        
        # 测试向后兼容
        level = manager.check_compatibility("1.0.0", "1.1.0")
        assert level == CompatibilityLevel.BACKWARD
        
        # 测试不兼容
        level = manager.check_compatibility("1.0.0", "2.0.0")
        assert level == CompatibilityLevel.BREAKING
    
    def test_version_registration(self):
        """测试版本注册"""
        from src.core.compatibility.integration_adapter import VersionInfo, CompatibilityLevel
        
        manager = get_integration_compatibility_manager()
        
        version = VersionInfo(
            major=1,
            minor=0,
            patch=0,
            compatibility=CompatibilityLevel.FULL,
            release_date=datetime.now(),
            changes=["初始版本"]
        )
        
        manager.register_version("test_integration", version)
        
        history = manager.get_version_history("test_integration")
        assert len(history) == 1
        assert str(history[0]) == "1.0.0"


class TestCrossLayerIntegration:
    """跨层集成测试"""
    
    @pytest.mark.asyncio
    async def test_data_to_strategy_event_flow(self):
        """测试数据层到策略层的事件流"""
        # 模拟数据层发布数据事件
        data_integration = get_data_scheduler_integration()
        strategy_integration = get_strategy_event_bus_integration()
        
        with patch.object(data_integration, '_event_bus') as mock_event_bus:
            mock_event_bus.publish = AsyncMock()
            
            # 数据层发布数据更新事件
            await data_integration.publish_data_event(
                event_type="DATA_UPDATED",
                data={"symbol": "000001.SZ", "price": 10.5}
            )
            
            mock_event_bus.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_strategy_to_trading_signal_flow(self):
        """测试策略层到交易层的信号流"""
        strategy_integration = get_strategy_event_bus_integration()
        trading_integration = get_trading_scheduler_integration()
        
        with patch.object(strategy_integration, '_event_bus') as mock_event_bus:
            mock_event_bus.publish = AsyncMock()
            mock_event_bus.subscribe = AsyncMock(return_value="sub_001")
            
            # 策略层发布信号
            signal = StrategySignal(
                signal_id="SIG-001",
                strategy_id="STRAT-001",
                symbol="000001.SZ",
                signal_type=SignalType.BUY
            )
            
            await strategy_integration.publish_signal(signal)
            
            # 验证事件已发布
            mock_event_bus.publish.assert_called_once()


class TestEndToEndIntegration:
    """端到端集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self):
        """测试完整交易流程"""
        # 1. 策略生成信号
        strategy_integration = get_strategy_event_bus_integration()
        
        # 2. 交易层接收信号并创建订单
        trading_integration = get_trading_scheduler_integration()
        
        # 3. 订单状态变更
        state_publisher = get_order_state_event_publisher()
        
        # 4. 风险检查
        risk_integration = get_risk_intercept_integration()
        
        with patch.multiple(
            strategy_integration, _event_bus=AsyncMock()
        ), patch.multiple(
            trading_integration, _scheduler=AsyncMock()
        ), patch.multiple(
            state_publisher, _event_bus=AsyncMock()
        ):
            # 模拟完整流程
            signal = StrategySignal(
                signal_id="SIG-001",
                strategy_id="STRAT-001",
                symbol="000001.SZ",
                signal_type=SignalType.BUY
            )
            
            # 发布信号
            await strategy_integration.publish_signal(signal)
            
            # 验证各层交互
            strategy_integration._event_bus.publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_data_to_trading_workflow(self):
        """测试数据到交易的完整流程"""
        data_integration = get_data_scheduler_integration()
        strategy_integration = get_strategy_event_bus_integration()
        trading_integration = get_trading_scheduler_integration()
        
        with patch.multiple(
            data_integration, _scheduler=AsyncMock()
        ), patch.multiple(
            strategy_integration, _event_bus=AsyncMock()
        ), patch.multiple(
            trading_integration, _scheduler=AsyncMock()
        ):
            # 数据采集
            await data_integration.submit_collection_task(
                symbols=["000001.SZ"],
                data_source="tushare"
            )
            
            # 策略生成信号
            signal = StrategySignal(
                signal_id="SIG-001",
                strategy_id="STRAT-001",
                symbol="000001.SZ",
                signal_type=SignalType.BUY
            )
            await strategy_integration.publish_signal(signal)
            
            # 验证流程
            data_integration._scheduler.submit_task.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
