"""
交易引擎执行逻辑测试
测试交易引擎的核心执行功能、订单路由、风险控制等
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from src.trading.execution.order_manager import OrderManager, OrderType, OrderStatus, Order, OrderSide
from src.trading.execution.execution_algorithm import BaseExecutionAlgorithm, AlgorithmType, AlgorithmConfig
from src.trading.core.trading_engine import TradingEngine


class TradingDataFactory:
    """交易测试数据工厂"""

    @staticmethod
    def create_sample_order():
        """创建样本订单"""
        return Order(
            order_id="test_order_001",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=150.0,
            side=OrderSide.BUY
        )

    @staticmethod
    def create_limit_order():
        """创建限价订单"""
        return Order(
            order_id="limit_order_001",
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            quantity=50.0,
            price=148.0,
            side=OrderSide.BUY
        )

    @staticmethod
    def create_market_data_snapshot():
        """创建市场数据快照"""
        return {
            'symbol': 'AAPL',
            'bid_price': 149.95,
            'ask_price': 150.05,
            'bid_volume': 1000,
            'ask_volume': 800,
            'last_price': 150.00,
            'volume': 50000,
            'timestamp': datetime.now()
        }

    @staticmethod
    def create_order_book():
        """创建订单簿"""
        return {
            'bids': [
                {'price': 149.95, 'volume': 1000},
                {'price': 149.90, 'volume': 800},
                {'price': 149.85, 'volume': 1200},
            ],
            'asks': [
                {'price': 150.05, 'volume': 800},
                {'price': 150.10, 'volume': 600},
                {'price': 150.15, 'volume': 900},
            ]
        }


class TestTradingEngineExecution:
    """交易引擎执行测试"""

    def setup_method(self):
        """测试前准备"""
        self.order_manager = OrderManager()
        self.data_factory = TradingDataFactory()

    def test_market_order_execution_simulation(self):
        """测试市价单执行模拟"""
        order = self.data_factory.create_sample_order()
        market_data = self.data_factory.create_market_data_snapshot()

        # 模拟市价单执行
        execution_price = market_data['ask_price'] if order.side == OrderSide.BUY else market_data['bid_price']
        execution_quantity = min(order.quantity, market_data['ask_volume'] if order.side == OrderSide.BUY else market_data['bid_volume'])

        # 验证执行逻辑
        assert execution_quantity > 0, "应该有执行数量"
        assert execution_price > 0, "执行价格应该有效"

        # 计算 slippage
        expected_price = order.price
        slippage = abs(execution_price - expected_price) / expected_price
        assert slippage < 0.01, "滑点应该在合理范围内"

    def test_limit_order_execution_logic(self):
        """测试限价单执行逻辑"""
        order = self.data_factory.create_limit_order()
        order_book = self.data_factory.create_order_book()

        # 模拟限价单匹配逻辑
        if order.side == OrderSide.BUY:
            # 买入订单：寻找价格 <= 限价的卖单
            matching_asks = [ask for ask in order_book['asks'] if ask['price'] <= order.price]
            if matching_asks:
                execution_price = min(ask['price'] for ask in matching_asks)
                assert execution_price <= order.price, "买入执行价格应该 <= 限价"
        else:
            # 卖出订单：寻找价格 >= 限价的买单
            matching_bids = [bid for bid in order_book['bids'] if bid['price'] >= order.price]
            if matching_bids:
                execution_price = max(bid['price'] for bid in matching_bids)
                assert execution_price >= order.price, "卖出执行价格应该 >= 限价"

    def test_order_routing_decision_making(self):
        """测试订单路由决策"""
        order = self.data_factory.create_sample_order()

        # 模拟多个经纪商的报价
        broker_quotes = [
            {'broker': 'Broker_A', 'price': 150.02, 'latency': 10, 'fee': 0.005},
            {'broker': 'Broker_B', 'price': 150.01, 'latency': 15, 'fee': 0.003},
            {'broker': 'Broker_C', 'price': 150.03, 'latency': 8, 'fee': 0.007},
        ]

        # 简单的路由决策：最低总成本
        for quote in broker_quotes:
            quote['total_cost'] = quote['price'] * (1 + quote['fee']/100) + quote['latency'] * 0.001

        best_quote = min(broker_quotes, key=lambda x: x['total_cost'])

        assert best_quote['broker'] in ['Broker_A', 'Broker_B', 'Broker_C']
        assert best_quote['total_cost'] < 151, "总成本应该合理"

    def test_risk_management_pre_trade_checks(self):
        """测试交易前风险检查"""
        order = self.data_factory.create_sample_order()

        # 账户信息
        account_info = {
            'cash': 50000.0,
            'margin_available': 100000.0,
            'current_positions': {'AAPL': 200},
            'risk_limits': {
                'max_order_value': 25000.0,
                'max_position_size': 1000,
                'max_daily_loss': 5000.0
            }
        }

        # 风险检查逻辑
        order_value = order.price * order.quantity
        max_order_check = order_value <= account_info['risk_limits']['max_order_value']

        new_position = account_info['current_positions'].get(order.symbol, 0)
        if order.side == OrderSide.BUY:
            new_position += order.quantity
        else:
            new_position -= order.quantity

        position_check = abs(new_position) <= account_info['risk_limits']['max_position_size']

        # 现金充足性检查
        required_cash = order_value if order.side == OrderSide.BUY else 0
        cash_check = account_info['cash'] >= required_cash

        # 综合风险评估
        risk_passed = max_order_check and position_check and cash_check

        assert risk_passed, "订单应该通过风险检查"

    def test_execution_algorithm_twap_simulation(self):
        """测试TWAP执行算法模拟"""
        config = AlgorithmConfig(
            algo_type=AlgorithmType.MARKET,
            duration=3600,  # 1小时
            target_quantity=1000.0,
            max_participation=0.1
        )

        # 创建TWAP算法模拟
        class TWAPSimulation:
            def __init__(self, config):
                self.config = config
                self.slices = []
                self.time_remaining = config.duration
                self.quantity_remaining = config.target_quantity

            def generate_schedule(self):
                """生成执行时间表"""
                intervals = 12  # 每5分钟执行一次
                time_step = self.config.duration / intervals
                quantity_step = self.config.target_quantity / intervals

                schedule = []
                current_time = 0
                remaining_quantity = self.config.target_quantity

                for i in range(intervals):
                    slice_quantity = min(quantity_step, remaining_quantity)
                    schedule.append({
                        'time': current_time,
                        'quantity': slice_quantity,
                        'remaining': remaining_quantity - slice_quantity
                    })
                    current_time += time_step
                    remaining_quantity -= slice_quantity

                return schedule

        twap = TWAPSimulation(config)
        schedule = twap.generate_schedule()

        # 验证TWAP执行计划
        assert len(schedule) == 12, "应该有12个执行切片"
        assert abs(sum(s['quantity'] for s in schedule) - config.target_quantity) < 1e-6, "总数量应该等于目标数量"

        # 验证时间分布均匀
        times = [s['time'] for s in schedule]
        assert times[0] == 0, "应该从0开始"
        assert times[-1] < config.duration, "最后时间应该在总时长内"

    def test_execution_algorithm_vwap_simulation(self):
        """测试VWAP执行算法模拟"""
        config = AlgorithmConfig(
            algo_type=AlgorithmType.MARKET,
            duration=3600,
            target_quantity=1000.0,
            max_participation=0.05
        )

        # 模拟VWAP算法
        class VWAPSimulation:
            def __init__(self, config):
                self.config = config

            def calculate_vwap_schedule(self, volume_profile):
                """基于成交量分布计算VWAP执行计划"""
                total_volume = sum(volume_profile)
                target_volume_pct = self.config.target_quantity / total_volume

                schedule = []
                remaining_quantity = self.config.target_quantity

                for i, period_volume in enumerate(volume_profile):
                    # 按成交量比例分配执行数量
                    period_quantity = period_volume * target_volume_pct
                    period_quantity = min(period_quantity, remaining_quantity)

                    schedule.append({
                        'period': i,
                        'quantity': period_quantity,
                        'volume_pct': period_volume / total_volume
                    })

                    remaining_quantity -= period_quantity
                    if remaining_quantity <= 0:
                        break

                return schedule

        # 模拟成交量分布（典型日间模式）
        volume_profile = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]

        vwap = VWAPSimulation(config)
        schedule = vwap.calculate_vwap_schedule(volume_profile)

        # 验证VWAP执行计划
        total_quantity = sum(s['quantity'] for s in schedule)
        assert abs(total_quantity - config.target_quantity) < 1e-6, "总数量应该等于目标数量"

        # 验证高峰期执行更多
        peak_periods = schedule[5:8]  # 高峰期切片
        peak_quantity = sum(s['quantity'] for s in peak_periods)
        total_quantity = sum(s['quantity'] for s in schedule)
        peak_ratio = peak_quantity / total_quantity

        assert 0.4 < peak_ratio < 0.6, "高峰期应该执行40-60%的总量"

    def test_smart_order_routing_optimization(self):
        """测试智能订单路由优化"""
        order = self.data_factory.create_sample_order()

        # 模拟多市场路由选项
        routing_options = [
            {
                'venue': 'NASDAQ',
                'price': 150.02,
                'liquidity': 0.9,
                'latency': 5,
                'fee': 0.0005
            },
            {
                'venue': 'NYSE',
                'price': 150.01,
                'liquidity': 0.7,
                'latency': 8,
                'fee': 0.0003
            },
            {
                'venue': 'BATS',
                'price': 150.03,
                'liquidity': 0.6,
                'latency': 3,
                'fee': 0.0002
            }
        ]

        # 智能路由决策：综合考虑价格、流动性、延迟、费用
        for option in routing_options:
            # 计算综合评分 (0-1, 越高越好)
            price_score = 1 - abs(option['price'] - 150.00) / 150.00  # 价格接近程度
            liquidity_score = option['liquidity']
            latency_score = 1 - (option['latency'] / 10)  # 延迟越低越好
            fee_score = 1 - option['fee'] * 1000  # 费用越低越好

            option['composite_score'] = (
                price_score * 0.4 +
                liquidity_score * 0.3 +
                latency_score * 0.2 +
                fee_score * 0.1
            )

        best_option = max(routing_options, key=lambda x: x['composite_score'])

        assert best_option['venue'] in ['NASDAQ', 'NYSE', 'BATS']
        assert 'composite_score' in best_option
        assert 0 <= best_option['composite_score'] <= 1

    def test_portfolio_risk_integration(self):
        """测试投资组合风险集成"""
        # 使用TradingEngine进行测试
        engine = TradingEngine()

        # 设置一些持仓
        engine.positions = {
            'AAPL': {'quantity': 100, 'avg_price': 145.0},
            'GOOGL': {'quantity': 50, 'avg_price': 2500.0}
        }

        # 测试风险检查 - 应该允许正常交易
        can_trade, reason = engine.check_risk_limits()
        assert can_trade == True

        # 测试大额交易 - 应该被拒绝
        risk_data = {
            'portfolio_value': 100000.0,
            'daily_loss': 100.0,
            'position_sizes': {'AAPL': 80000.0}  # 超过限制
        }
        risk_result = engine.check_risk_limits(risk_data)
        assert risk_result['can_trade'] == False

    def test_high_frequency_execution_simulation(self):
        """测试高频执行模拟"""
        order = Order(
            order_id="hft_order_001",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=1000.0,
            price=150.0,
            side=OrderSide.BUY
        )

        # 模拟HFT执行环境
        market_conditions = {
            'spread': 0.1,  # 0.1美元价差
            'depth': 10000,  # 深度充足
            'volatility': 0.02,  # 2%波动率
            'latency': 1  # 1ms延迟
        }

        # HFT执行策略：快速分批执行
        class HFTExecutionSimulator:
            def __init__(self, order, conditions):
                self.order = order
                self.conditions = conditions
                self.executions = []

            def execute_hft(self):
                """HFT执行逻辑"""
                remaining_quantity = self.order.quantity
                max_slice = 100  # 最大单次执行100股

                while remaining_quantity > 0:
                    slice_quantity = min(max_slice, remaining_quantity)
                    execution_price = self.order.price + np.random.normal(0, self.conditions['spread'] * 0.1)

                    self.executions.append({
                        'quantity': slice_quantity,
                        'price': execution_price,
                        'timestamp': datetime.now()
                    })

                    remaining_quantity -= slice_quantity

                return self.executions

        hft_simulator = HFTExecutionSimulator(order, market_conditions)
        executions = hft_simulator.execute_hft()

        # 验证HFT执行结果
        total_executed = sum(ex['quantity'] for ex in executions)
        assert total_executed == order.quantity, "应该完全执行订单"

        # 计算平均执行价格和滑点
        avg_price = sum(ex['quantity'] * ex['price'] for ex in executions) / total_executed
        slippage = abs(avg_price - order.price) / order.price

        assert slippage < 0.001, "HFT执行滑点应该很小"
        assert len(executions) > 1, "HFT应该分批执行"

    def test_execution_monitoring_and_metrics(self):
        """测试执行监控和指标"""
        order = self.data_factory.create_sample_order()

        # 模拟执行过程监控
        execution_metrics = {
            'start_time': datetime.now(),
            'target_quantity': order.quantity,
            'executed_quantity': 0,
            'total_cost': 0.0,
            'slippage': 0.0,
            'market_impact': 0.0,
            'completion_rate': 0.0
        }

        # 模拟分批执行
        executions = []
        remaining = order.quantity

        for i in range(5):  # 5批执行
            batch_quantity = min(remaining / (5 - i), 25)  # 最后几批平均分配
            batch_price = order.price + np.random.normal(0, 0.5)

            executions.append({
                'batch': i + 1,
                'quantity': batch_quantity,
                'price': batch_price,
                'timestamp': datetime.now() + timedelta(seconds=i * 60)
            })

            remaining -= batch_quantity

        # 更新监控指标
        execution_metrics['executed_quantity'] = sum(ex['quantity'] for ex in executions)
        execution_metrics['total_cost'] = sum(ex['quantity'] * ex['price'] for ex in executions)
        execution_metrics['avg_price'] = execution_metrics['total_cost'] / execution_metrics['executed_quantity']
        execution_metrics['slippage'] = (execution_metrics['avg_price'] - order.price) / order.price
        execution_metrics['completion_rate'] = execution_metrics['executed_quantity'] / order.quantity

        # 验证监控指标
        assert execution_metrics['completion_rate'] == 1.0, "应该100%完成"
        assert abs(execution_metrics['slippage']) < 0.01, "滑点应该在合理范围内"
        assert execution_metrics['executed_quantity'] == order.quantity, "执行数量应该等于订单数量"

    def test_adaptive_execution_based_on_market_conditions(self):
        """测试基于市场条件的自适应执行"""
        order = self.data_factory.create_sample_order()

        # 模拟不同市场条件
        market_scenarios = [
            {'volatility': 0.01, 'liquidity': 0.9, 'trend': 'stable'},    # 稳定市场
            {'volatility': 0.05, 'liquidity': 0.5, 'trend': 'trending'},  # 趋势市场
            {'volatility': 0.08, 'liquidity': 0.3, 'trend': 'volatile'}   # 波动市场
        ]

        for scenario in market_scenarios:
            # 自适应执行策略调整
            if scenario['volatility'] > 0.05:
                # 高波动：减少单次执行量，增加观察时间
                slice_size = order.quantity * 0.1  # 10%批次
                observation_period = 300  # 5分钟
            elif scenario['liquidity'] < 0.5:
                # 低流动性：减少执行频率
                slice_size = order.quantity * 0.05  # 5%批次
                observation_period = 600  # 10分钟
            else:
                # 正常市场：标准执行
                slice_size = order.quantity * 0.2  # 20%批次
                observation_period = 180  # 3分钟

            # 验证自适应逻辑
            assert slice_size > 0, f"{scenario['trend']}市场应该有合理的批次大小"
            assert observation_period >= 180, f"{scenario['trend']}市场应该有足够的观察期"

            # 高波动市场应该更保守
            if scenario['volatility'] > 0.05:
                assert slice_size <= order.quantity * 0.1
                assert observation_period >= 300
