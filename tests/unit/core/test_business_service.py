#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试核心业务服务

测试目标：提升business_service.py的覆盖率到100%
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any, Optional

from src.core.core_services.core.business_service import (
    BusinessProcessStatus,
    BusinessProcessType,
    BusinessProcess,
    TradingStrategy,
    StrategyService,
    OrderService,
    PortfolioService,
    ProcessService,
    DataAnalysisService,
    BusinessService
)


class TestBusinessEnums:
    """测试业务枚举"""

    def test_business_process_status_values(self):
        """测试业务流程状态枚举值"""
        assert BusinessProcessStatus.PENDING.value == "pending"
        assert BusinessProcessStatus.RUNNING.value == "running"
        assert BusinessProcessStatus.COMPLETED.value == "completed"
        assert BusinessProcessStatus.FAILED.value == "failed"
        assert BusinessProcessStatus.CANCELLED.value == "cancelled"

    def test_business_process_type_values(self):
        """测试业务流程类型枚举值"""
        assert BusinessProcessType.STRATEGY_EXECUTION.value == "strategy_execution"
        assert BusinessProcessType.ORDER_PROCESSING.value == "order_processing"
        assert BusinessProcessType.PORTFOLIO_REBALANCE.value == "portfolio_rebalance"
        assert BusinessProcessType.RISK_ASSESSMENT.value == "risk_assessment"
        assert BusinessProcessType.DATA_PROCESSING.value == "data_processing"
        assert BusinessProcessType.MARKET_ANALYSIS.value == "market_analysis"


class TestBusinessDataStructures:
    """测试业务数据结构"""

    def test_business_process_creation(self):
        """测试业务流程创建"""
        process = BusinessProcess(
            process_id="test_process_001",
            process_type=BusinessProcessType.STRATEGY_EXECUTION,
            user_id=12345
        )

        assert process.process_id == "test_process_001"
        assert process.process_type == BusinessProcessType.STRATEGY_EXECUTION
        assert process.user_id == 12345
        assert process.status == BusinessProcessStatus.PENDING
        assert isinstance(process.parameters, dict)
        assert isinstance(process.results, dict)
        assert process.progress == 0.0
        assert isinstance(process.steps, list)

    def test_business_process_with_custom_data(self):
        """测试业务流程自定义数据"""
        created_at = datetime.now()
        started_at = created_at + timedelta(minutes=5)
        completed_at = started_at + timedelta(minutes=10)

        process = BusinessProcess(
            process_id="test_process_002",
            process_type=BusinessProcessType.ORDER_PROCESSING,
            user_id=67890,
            status=BusinessProcessStatus.COMPLETED,
            parameters={"param1": "value1"},
            results={"result1": "output1"},
            error_message=None,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            progress=1.0,
            steps=[{"step": 1, "status": "completed"}]
        )

        assert process.status == BusinessProcessStatus.COMPLETED
        assert process.parameters == {"param1": "value1"}
        assert process.results == {"result1": "output1"}
        assert process.progress == 1.0
        assert len(process.steps) == 1

    def test_trading_strategy_creation(self):
        """测试交易策略创建"""
        strategy = TradingStrategy(
            strategy_id="test_strategy_001",
            name="Test Strategy",
            description="A momentum trading strategy",
            user_id=12345,
            parameters={"period": 20, "threshold": 0.05},
            is_active=True
        )

        assert strategy.strategy_id == "test_strategy_001"
        assert strategy.name == "Test Strategy"
        assert strategy.description == "A momentum trading strategy"
        assert strategy.user_id == 12345
        assert strategy.parameters == {"period": 20, "threshold": 0.05}
        assert strategy.is_active == True
        assert isinstance(strategy.created_at, datetime)
        assert isinstance(strategy.performance_metrics, dict)


class TestStrategyService:
    """测试策略服务"""

    @pytest.fixture
    def mock_config_manager(self):
        """创建模拟配置管理器"""
        return Mock()

    @pytest.fixture
    def mock_db_service(self):
        """创建模拟数据库服务"""
        return Mock()

    @pytest.fixture
    def strategy_service(self, mock_config_manager, mock_db_service):
        """创建策略服务实例"""
        return StrategyService(mock_config_manager, mock_db_service)

    @pytest.mark.asyncio
    async def test_create_strategy(self, strategy_service, mock_db_service):
        """测试创建策略"""
        strategy_data = {
            "name": "Test Momentum Strategy",
            "strategy_type": "momentum",
            "parameters": {"period": 20}
        }

        # 模拟数据库操作
        mock_db_service.create_strategy = AsyncMock(return_value="strategy_001")

        result = await strategy_service.create_strategy(12345, strategy_data)

        assert isinstance(result, dict)
        assert result["strategy_id"] == "strategy_001"
        mock_db_service.create_strategy.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_strategy(self, strategy_service, mock_db_service):
        """测试执行策略"""
        market_data = {
            "symbols": ["000001", "000002"],
            "prices": {"000001": 10.0, "000002": 20.0}
        }

        # 模拟数据库和策略执行
        mock_db_service.get_strategy = AsyncMock(return_value={
            "strategy_id": "strategy_001",
            "name": "Test Strategy",
            "parameters": {"period": 20}
        })

        result = await strategy_service.execute_strategy("strategy_001", market_data)

        assert isinstance(result, dict)
        mock_db_service.get_strategy.assert_called_once_with("strategy_001")


class TestOrderService:
    """测试订单服务"""

    @pytest.fixture
    def order_service(self, mock_config_manager, mock_db_service):
        """创建订单服务实例"""
        return OrderService(mock_config_manager, mock_db_service)

    @pytest.mark.asyncio
    async def test_process_order(self, order_service, mock_db_service):
        """测试处理订单"""
        order_data = {
            "symbol": "000001",
            "order_type": "LIMIT",
            "direction": "BUY",
            "quantity": 100,
            "price": 10.0
        }

        # 模拟订单处理
        mock_db_service.create_order = AsyncMock(return_value="order_001")

        result = await order_service.process_order(12345, order_data)

        assert isinstance(result, dict)
        assert result["order_id"] == "order_001"
        mock_db_service.create_order.assert_called_once()


class TestPortfolioService:
    """测试投资组合服务"""

    @pytest.fixture
    def portfolio_service(self, mock_config_manager, mock_db_service):
        """创建投资组合服务实例"""
        return PortfolioService(mock_config_manager, mock_db_service)

    @pytest.mark.asyncio
    async def test_rebalance_portfolio(self, portfolio_service, mock_db_service):
        """测试重新平衡投资组合"""
        target_allocation = {
            "000001": 0.5,
            "000002": 0.3,
            "000003": 0.2
        }

        # 模拟重新平衡操作
        mock_db_service.rebalance_portfolio = AsyncMock(return_value={
            "status": "success",
            "orders": ["order_001", "order_002"]
        })

        result = await portfolio_service.rebalance_portfolio(12345, target_allocation)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        mock_db_service.rebalance_portfolio.assert_called_once()


class TestProcessService:
    """测试流程服务"""

    @pytest.fixture
    def process_service(self, mock_config_manager, mock_db_service):
        """创建流程服务实例"""
        return ProcessService(mock_config_manager, mock_db_service)

    @pytest.mark.asyncio
    async def test_get_process_status(self, process_service, mock_db_service):
        """测试获取流程状态"""
        mock_db_service.get_process = AsyncMock(return_value={
            "process_id": "process_001",
            "status": "running",
            "progress": 0.5
        })

        result = await process_service.get_process_status("process_001")

        assert isinstance(result, dict)
        assert result["process_id"] == "process_001"
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_cancel_process(self, process_service, mock_db_service):
        """测试取消流程"""
        mock_db_service.cancel_process = AsyncMock(return_value=True)

        result = await process_service.cancel_process("process_001")

        assert result == True
        mock_db_service.cancel_process.assert_called_once_with("process_001")

    @pytest.mark.asyncio
    async def test_get_user_processes(self, process_service, mock_db_service):
        """测试获取用户流程"""
        mock_db_service.get_user_processes = AsyncMock(return_value=[
            {"process_id": "process_001", "status": "completed"},
            {"process_id": "process_002", "status": "running"}
        ])

        result = await process_service.get_user_processes(12345, "completed")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["status"] == "completed"


class TestDataAnalysisService:
    """测试数据分析服务"""

    @pytest.fixture
    def data_analysis_service(self, mock_config_manager, mock_db_service):
        """创建数据分析服务实例"""
        return DataAnalysisService(mock_config_manager, mock_db_service)

    @pytest.mark.asyncio
    async def test_analyze_market_data(self, data_analysis_service, mock_db_service):
        """测试分析市场数据"""
        symbols = ["000001", "000002"]

        # 模拟数据分析
        mock_db_service.analyze_market_data = AsyncMock(return_value={
            "analysis_type": "technical",
            "results": {
                "000001": {"rsi": 65.5, "macd": 0.12},
                "000002": {"rsi": 45.2, "macd": -0.08}
            }
        })

        result = await data_analysis_service.analyze_market_data(symbols, "technical")

        assert isinstance(result, dict)
        assert result["analysis_type"] == "technical"
        assert "results" in result
        mock_db_service.analyze_market_data.assert_called_once()


class TestBusinessService:
    """测试业务服务主类"""

    @pytest.fixture
    def business_service(self):
        """创建业务服务实例"""
        return BusinessService()

    def test_business_service_initialization(self, business_service):
        """测试业务服务初始化"""
        assert hasattr(business_service, 'strategy_service')
        assert hasattr(business_service, 'order_service')
        assert hasattr(business_service, 'portfolio_service')
        assert hasattr(business_service, 'process_service')
        assert hasattr(business_service, 'data_analysis_service')

    @pytest.mark.asyncio
    async def test_initialize(self, business_service):
        """测试业务服务初始化"""
        # Mock各个子服务的初始化
        with patch.object(business_service.strategy_service, 'initialize', new_callable=AsyncMock) as mock_strategy_init, \
             patch.object(business_service.order_service, 'initialize', new_callable=AsyncMock) as mock_order_init, \
             patch.object(business_service.portfolio_service, 'initialize', new_callable=AsyncMock) as mock_portfolio_init, \
             patch.object(business_service.process_service, 'initialize', new_callable=AsyncMock) as mock_process_init, \
             patch.object(business_service.data_analysis_service, 'initialize', new_callable=AsyncMock) as mock_analysis_init:

            await business_service.initialize()

            mock_strategy_init.assert_called_once()
            mock_order_init.assert_called_once()
            mock_portfolio_init.assert_called_once()
            mock_process_init.assert_called_once()
            mock_analysis_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_strategy(self, business_service):
        """测试创建策略"""
        strategy_data = {
            "name": "Test Strategy",
            "strategy_type": "momentum",
            "parameters": {"period": 20}
        }

        with patch.object(business_service.strategy_service, 'create_strategy', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = {"strategy_id": "strategy_001"}

            result = await business_service.create_strategy(12345, strategy_data)

            assert result["strategy_id"] == "strategy_001"
            mock_create.assert_called_once_with(12345, strategy_data)

    @pytest.mark.asyncio
    async def test_execute_strategy(self, business_service):
        """测试执行策略"""
        market_data = {"symbols": ["000001"], "prices": {"000001": 10.0}}

        with patch.object(business_service.strategy_service, 'execute_strategy', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"status": "success", "signals": []}

            result = await business_service.execute_strategy("strategy_001", market_data)

            assert result["status"] == "success"
            mock_execute.assert_called_once_with("strategy_001", market_data)


class TestBusinessServiceIntegration:
    """测试业务服务集成场景"""

    @pytest.fixture
    def business_service(self):
        """创建完整的业务服务"""
        return BusinessService()

    @pytest.mark.asyncio
    async def test_complete_business_workflow(self, business_service):
        """测试完整业务工作流程"""
        # 1. 初始化服务
        await business_service.initialize()

        # 2. 创建策略
        strategy_data = {
            "name": "Integration Test Strategy",
            "strategy_type": "momentum",
            "parameters": {"period": 20}
        }

        with patch.object(business_service.strategy_service, 'create_strategy', new_callable=AsyncMock) as mock_create_strategy:
            mock_create_strategy.return_value = {"strategy_id": "integration_strategy_001"}

            strategy_result = await business_service.create_strategy(12345, strategy_data)
            assert strategy_result["strategy_id"] == "integration_strategy_001"

        # 3. 执行策略
        market_data = {
            "symbols": ["000001", "000002"],
            "prices": {"000001": 10.0, "000002": 20.0}
        }

        with patch.object(business_service.strategy_service, 'execute_strategy', new_callable=AsyncMock) as mock_execute_strategy:
            mock_execute_strategy.return_value = {
                "status": "success",
                "signals": [
                    {"symbol": "000001", "signal": "BUY", "quantity": 100}
                ]
            }

            execution_result = await business_service.execute_strategy("integration_strategy_001", market_data)
            assert execution_result["status"] == "success"
            assert len(execution_result["signals"]) == 1

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, business_service):
        """测试并发操作"""
        await business_service.initialize()

        # 创建多个并发任务
        async def create_and_execute_strategy(user_id: int, strategy_id: str):
            strategy_data = {
                "name": f"Concurrent Strategy {user_id}",
                "strategy_type": "momentum",
                "parameters": {"period": 20}
            }

            with patch.object(business_service.strategy_service, 'create_strategy', new_callable=AsyncMock) as mock_create:
                mock_create.return_value = {"strategy_id": f"strategy_{user_id}"}

                create_result = await business_service.create_strategy(user_id, strategy_data)

                market_data = {"symbols": ["000001"], "prices": {"000001": 10.0}}
                with patch.object(business_service.strategy_service, 'execute_strategy', new_callable=AsyncMock) as mock_execute:
                    mock_execute.return_value = {"status": "success"}

                    execute_result = await business_service.execute_strategy(create_result["strategy_id"], market_data)

                    return create_result, execute_result

        # 并发执行多个操作
        tasks = [
            create_and_execute_strategy(i, f"strategy_{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for create_result, execute_result in results:
            assert "strategy_id" in create_result
            assert execute_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, business_service):
        """测试错误处理和恢复"""
        await business_service.initialize()

        # 测试策略创建失败的情况
        strategy_data = {
            "name": "Error Test Strategy",
            "strategy_type": "invalid_type",
            "parameters": {}
        }

        with patch.object(business_service.strategy_service, 'create_strategy', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("Database connection failed")

            try:
                await business_service.create_strategy(12345, strategy_data)
                assert False, "Should have raised an exception"
            except Exception as e:
                assert "Database connection failed" in str(e)

        # 验证服务仍然可以继续工作
        with patch.object(business_service.strategy_service, 'create_strategy', new_callable=AsyncMock) as mock_create_recovery:
            mock_create_recovery.return_value = {"strategy_id": "recovery_strategy_001"}

            recovery_result = await business_service.create_strategy(12345, strategy_data)
            assert recovery_result["strategy_id"] == "recovery_strategy_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
