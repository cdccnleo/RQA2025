"""
交易执行引擎增强测试
测试TradeExecutionEngine的各种功能和边界情况
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.trading.core.execution.trade_execution_engine import TradeExecutionEngine, ExecutionAlgorithm
from src.trading.core.execution.execution_result import ExecutionResult, ExecutionResultStatus
from src.trading.core.execution.execution_context import ExecutionContext


class TestTradeExecutionEngineEnhanced:
    """交易执行引擎增强测试"""

    @pytest.fixture
    def execution_engine(self):
        """创建交易执行引擎实例"""
        return TradeExecutionEngine()

    @pytest.fixture
    def execution_engine_with_config(self):
        """创建带配置的交易执行引擎"""
        config = {
            'max_slippage': 0.002,
            'commission_rate': 0.0003,
            'min_order_size': 100,
            'max_order_size': 10000
        }
        return TradeExecutionEngine(config)

    def test_execution_engine_initialization(self, execution_engine):
        """测试执行引擎初始化"""
        assert execution_engine is not None
        assert hasattr(execution_engine, 'execution_algorithms')

    def test_execution_engine_with_config(self, execution_engine_with_config):
        """测试带配置的执行引擎"""
        engine = execution_engine_with_config
        assert engine.max_slippage == 0.002
        assert engine.commission_rate == 0.0003

    def test_execute_order_market_order(self, execution_engine):
        """测试市价单执行"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'direction': 'buy'
        }

        with patch.object(execution_engine, '_execute_market_order') as mock_execute:
            mock_result = ExecutionResult(
                execution_id='exec_test',
                order_id='test_order_001',
                symbol='000001',
                requested_quantity=1000,
                executed_quantity=1000,
                average_price=10.5,
                status=ExecutionResultStatus.SUCCESS
            )
            mock_execute.return_value = mock_result

            result = execution_engine.execute_order(order)
            assert result is not None
            assert result.status == ExecutionResultStatus.SUCCESS

    def test_execute_order_limit_order(self, execution_engine):
        """测试限价单执行"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'limit',
            'direction': 'buy',
            'price': 10.5
        }

        with patch.object(execution_engine, '_execute_limit_order') as mock_execute:
            mock_result = ExecutionResult(
                execution_id='exec_test',
                order_id='test_limit_order_001',
                symbol='000001',
                requested_quantity=1000,
                executed_quantity=500,  # 部分成交
                average_price=10.5,
                status=ExecutionResultStatus.PARTIAL
            )
            mock_execute.return_value = mock_result

            result = execution_engine.execute_order(order)
            assert result is not None
            assert result.status == ExecutionResultStatus.PARTIAL

    def test_market_order_execution_logic(self, execution_engine):
        """测试市价单执行逻辑"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'direction': 'buy'
        }

        # Mock市场数据
        with patch.object(execution_engine, '_get_market_price') as mock_price:
            mock_price.return_value = 10.5

            result = execution_engine._execute_market_order(order)
            assert isinstance(result, ExecutionResult)
            assert result.symbol == '000001'
            assert result.requested_quantity == 1000

    def test_limit_order_execution_logic(self, execution_engine):
        """测试限价单执行逻辑"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'direction': 'buy',
            'price': 10.5
        }

        # Mock市场数据
        with patch.object(execution_engine, '_get_market_price') as mock_price, \
             patch.object(execution_engine, '_wait_for_price') as mock_wait:
            mock_price.return_value = 10.3  # 市场价低于限价
            mock_wait.return_value = True

            result = execution_engine._execute_limit_order(order)
            assert isinstance(result, ExecutionResult)

    def test_get_market_price(self, execution_engine):
        """测试获取市场价格"""
        symbol = '000001'

        with patch.object(execution_engine, '_fetch_market_data') as mock_fetch:
            mock_fetch.return_value = {'price': 10.5, 'volume': 1000}

            price = execution_engine._get_market_price(symbol)
            assert price == 10.5

    def test_execute_order_algorithm_detection_dict_order(self, execution_engine):
        """测试字典订单的算法自动检测"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'limit',  # 指定为限价单
            'price': 10.5,
            'direction': 'buy'
        }

        with patch.object(execution_engine, '_execute_limit_order') as mock_execute:
            mock_result = ExecutionResult(
                execution_id='exec_test',
                order_id='test_order_001',
                symbol='000001',
                requested_quantity=1000,
                executed_quantity=1000,
                average_price=10.5,
                status=ExecutionResultStatus.SUCCESS
            )
            mock_execute.return_value = mock_result

            result = execution_engine.execute_order(order)  # 不指定algorithm参数
            assert isinstance(result, ExecutionResult)
            mock_execute.assert_called_once()  # 应该调用限价执行方法

    def test_execute_order_algorithm_detection_object_order(self, execution_engine):
        """测试对象订单的算法自动检测"""
        # 创建一个简单的对象订单
        class SimpleOrder:
            def __init__(self):
                self.order_id = "test_order_001"
                self.symbol = "000001"
                self.quantity = 1000
                self.order_type = 'market'

        order = SimpleOrder()

        execution_id = execution_engine.execute_order(order)  # 不指定algorithm参数
        assert execution_id.startswith("exec_")
        assert execution_id not in execution_engine.active_executions  # 市价单应该立即完成

    def test_get_market_price_exception_handling(self, execution_engine):
        """测试获取市场价格的异常处理"""
        symbol = '000001'

        with patch.object(execution_engine, '_fetch_market_data') as mock_fetch:
            mock_fetch.side_effect = ConnectionError("Network error")

            price = execution_engine._get_market_price(symbol)
            assert price == 10.5  # 应该返回默认价格

    def test_calculate_slippage(self, execution_engine):
        """测试滑点计算"""
        expected_price = 10.5
        actual_price = 10.52

        slippage = execution_engine._calculate_slippage(expected_price, actual_price)
        assert isinstance(slippage, float)
        assert slippage == 0.0019  # (10.52 - 10.5) / 10.5

    def test_calculate_commission(self, execution_engine):
        """测试佣金计算"""
        quantity = 1000
        price = 10.5

        commission = execution_engine._calculate_commission(quantity, price)
        expected_commission = quantity * price * execution_engine.commission_rate
        assert commission == expected_commission

    def test_validate_order(self, execution_engine):
        """测试订单验证"""
        # 有效订单
        valid_order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'direction': 'buy'
        }

        is_valid, message = execution_engine._validate_order(valid_order)
        assert is_valid is True

        # 无效订单 - 缺少必要字段
        invalid_order = {
            'symbol': '000001'
            # 缺少quantity等字段
        }

        is_valid, message = execution_engine._validate_order(invalid_order)
        assert is_valid is False

    def test_validate_order_quantity(self, execution_engine):
        """测试订单数量验证"""
        # 正常数量
        valid_quantity = 1000
        assert execution_engine._validate_quantity(valid_quantity) is True

        # 过小数量
        small_quantity = 10
        assert execution_engine._validate_quantity(small_quantity) is False

        # 过大数量
        large_quantity = 100000
        assert execution_engine._validate_quantity(large_quantity) is False

    def test_execution_context_creation(self, execution_engine):
        """测试执行上下文创建"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'direction': 'buy'
        }

        context = execution_engine._create_execution_context(order)
        assert isinstance(context, ExecutionContext)
        assert context.symbol == '000001'
        assert context.quantity == 1000

    def test_execution_result_creation(self, execution_engine):
        """测试执行结果创建"""
        order_id = 'test_order_001'
        symbol = '000001'
        quantity = 1000
        executed_quantity = 1000
        price = 10.5

        result = execution_engine._create_execution_result(
            order_id, symbol, quantity, executed_quantity, price
        )

        assert isinstance(result, ExecutionResult)
        assert result.order_id == order_id
        assert result.symbol == symbol
        assert result.executed_quantity == executed_quantity
        assert result.average_price == price

    def test_partial_execution_handling(self, execution_engine):
        """测试部分成交处理"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'direction': 'buy'
        }

        # Mock部分成交
        with patch.object(execution_engine, '_get_market_price') as mock_price, \
             patch.object(execution_engine, '_check_liquidity') as mock_liquidity:
            mock_price.return_value = 10.5
            mock_liquidity.return_value = 500  # 只有500手的流动性

            result = execution_engine._execute_market_order(order)
            assert result.status in [ExecutionResultStatus.PARTIAL, ExecutionResultStatus.SUCCESS]
            assert result.executed_quantity <= order['quantity']

    def test_liquidity_check(self, execution_engine):
        """测试流动性检查"""
        symbol = '000001'
        quantity = 1000

        import random
        with patch('random.randint') as mock_randint:
            mock_randint.return_value = 800  # 模拟只有800手的流动性

            available_liquidity = execution_engine._check_liquidity(symbol, quantity)
            assert isinstance(available_liquidity, int)
            assert available_liquidity <= quantity
            assert available_liquidity == 800  # 应该返回mock的值

    def test_price_impact_calculation(self, execution_engine):
        """测试价格影响计算"""
        symbol = '000001'
        quantity = 5000
        market_price = 10.5

        impact = execution_engine._calculate_price_impact(symbol, quantity, market_price)
        assert isinstance(impact, float)
        assert impact >= 0

    def test_execution_algorithm_selection(self, execution_engine):
        """测试执行算法选择"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'urgency': 'high'
        }

        algorithm = execution_engine._select_execution_algorithm(order)
        assert isinstance(algorithm, ExecutionAlgorithm)

    def test_vwap_execution(self, execution_engine):
        """测试VWAP执行"""
        order = {
            'symbol': '000001',
            'quantity': 10000,
            'order_type': 'vwap',
            'direction': 'buy',
            'time_horizon': 60  # 60分钟
        }

        with patch.object(execution_engine, '_get_vwap_schedule') as mock_schedule, \
             patch.object(execution_engine, '_execute_scheduled_order') as mock_execute:
            mock_schedule.return_value = [{'time': 0, 'quantity': 2000}]
            mock_execute.return_value = ExecutionResult(
                execution_id='exec_test',
                order_id='vwap_order_001',
                symbol='000001',
                requested_quantity=2000,
                executed_quantity=2000,
                average_price=10.5,
                status=ExecutionResultStatus.SUCCESS
            )

            result = execution_engine._execute_vwap_order(order)
            assert isinstance(result, ExecutionResult)

    def test_twap_execution(self, execution_engine):
        """测试TWAP执行"""
        order = {
            'symbol': '000001',
            'quantity': 6000,
            'order_type': 'twap',
            'direction': 'sell',
            'time_horizon': 30  # 30分钟
        }

        with patch.object(execution_engine, '_get_twap_schedule') as mock_schedule:
            mock_schedule.return_value = [{'time': 0, 'quantity': 1200}]

            result = execution_engine._execute_twap_order(order)
            assert isinstance(result, ExecutionResult)

    def test_iceberg_execution(self, execution_engine):
        """测试冰山订单执行"""
        order = {
            'symbol': '000001',
            'quantity': 10000,
            'order_type': 'iceberg',
            'direction': 'buy',
            'visible_quantity': 1000  # 每次显示1000手
        }

        with patch.object(execution_engine, '_split_iceberg_order') as mock_split, \
             patch.object(execution_engine, '_execute_market_order') as mock_execute:
            mock_split.return_value = [{'quantity': 1000}, {'quantity': 1000}]
            mock_execute.return_value = ExecutionResult(
                execution_id='exec_test',
                order_id='iceberg_001',
                symbol='000001',
                requested_quantity=1000,
                executed_quantity=1000,
                average_price=10.5,
                status=ExecutionResultStatus.SUCCESS
            )

            result = execution_engine._execute_iceberg_order(order)
            assert isinstance(result, ExecutionResult)

    def test_execution_performance_monitoring(self, execution_engine):
        """测试执行性能监控"""
        execution_result = ExecutionResult(
            execution_id='exec_test',
            order_id='test_order_001',
            symbol='000001',
            requested_quantity=1000,
            executed_quantity=1000,
            average_price=10.5,
            status=ExecutionResultStatus.SUCCESS
        )

        execution_engine._record_execution_metrics(execution_result)

        # 验证指标已被记录
        assert hasattr(execution_engine, 'execution_metrics')

    def test_error_handling_and_recovery(self, execution_engine):
        """测试错误处理和恢复"""
        # 测试网络错误
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'direction': 'buy'
        }

        with patch.object(execution_engine, '_get_market_price') as mock_price:
            mock_price.side_effect = ConnectionError("Network error")

            result = execution_engine.execute_order(order)
            # 应该返回错误状态的结果
            assert isinstance(result, ExecutionResult)
            assert result.status.value in ['failed', 'error']

    def test_circuit_breaker_handling(self, execution_engine):
        """测试熔断机制处理"""
        # 当市场波动过大时
        with patch.object(execution_engine, '_check_market_conditions') as mock_check:
            mock_check.return_value = False  # 市场异常

            order = {
                'symbol': '000001',
                'quantity': 1000,
                'order_type': 'market',
                'direction': 'buy'
            }

            result = execution_engine.execute_order(order)
            assert isinstance(result, ExecutionResult)

    def test_execution_timeout_handling(self, execution_engine):
        """测试执行超时处理"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'limit',
            'direction': 'buy',
            'price': 10.5,
            'timeout': 30  # 30秒超时
        }

        with patch.object(execution_engine, '_wait_for_price') as mock_wait:
            mock_wait.return_value = False  # 超时

            result = execution_engine._execute_limit_order(order)
            assert isinstance(result, ExecutionResult)
            assert result.status.value in ['timeout', 'cancelled']

    def test_multi_asset_execution(self, execution_engine):
        """测试多资产执行"""
        orders = [
            {
                'symbol': '000001',
                'quantity': 1000,
                'order_type': 'market',
                'direction': 'buy'
            },
            {
                'symbol': '000002',
                'quantity': 500,
                'order_type': 'market',
                'direction': 'sell'
            }
        ]

        results = execution_engine.execute_orders_batch(orders)
        assert isinstance(results, list)
        assert len(results) == len(orders)

        for result in results:
            assert isinstance(result, ExecutionResult)

    def test_execution_cost_analysis(self, execution_engine):
        """测试执行成本分析"""
        execution_result = ExecutionResult(
            execution_id='exec_test',
            order_id='test_order_001',
            symbol='000001',
            requested_quantity=1000,
            executed_quantity=1000,
            average_price=10.52,  # 比期望价格高0.02
            status=ExecutionResultStatus.SUCCESS
        )

        expected_price = 10.5
        cost_analysis = execution_engine._analyze_execution_cost(execution_result, expected_price)

        assert isinstance(cost_analysis, dict)
        assert 'slippage_cost' in cost_analysis
        assert 'commission_cost' in cost_analysis

    def test_execution_strategy_optimization(self, execution_engine):
        """测试执行策略优化"""
        historical_executions = [
            {'symbol': '000001', 'quantity': 1000, 'slippage': 0.001, 'time': 30},
            {'symbol': '000001', 'quantity': 1000, 'slippage': 0.002, 'time': 45}
        ]

        optimized_params = execution_engine._optimize_execution_parameters(historical_executions)
        assert isinstance(optimized_params, dict)
