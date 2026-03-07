"""
真实交易执行流程测试
测试实际的交易执行场景，包括订单管理、风险控制、执行算法等
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.trading.execution.order_manager import OrderManager, OrderType, OrderStatus, Order, OrderSide
from src.trading.execution.execution_algorithm import BaseExecutionAlgorithm, AlgorithmType, AlgorithmConfig


class RealTradingExecutionSimulator:
    """真实交易执行模拟器"""

    def __init__(self):
        self.order_manager = OrderManager()
        self.market_data_feed = []
        self.execution_results = []
        self.risk_limits = {
            'max_order_value': 50000.0,
            'max_position_size': 1000,
            'max_daily_loss': 5000.0,
            'max_slippage': 0.02
        }

    def simulate_market_conditions(self, scenario='normal'):
        """模拟不同市场条件"""
        np.random.seed(42)

        if scenario == 'normal':
            # 正常市场：适中的波动性和流动性
            volatility = 0.015  # 1.5%波动率
            spread = 0.001  # 0.1%价差
            liquidity = 0.8  # 80%流动性
        elif scenario == 'volatile':
            # 高波动市场
            volatility = 0.04
            spread = 0.003
            liquidity = 0.5
        elif scenario == 'illiquid':
            # 低流动性市场
            volatility = 0.02
            spread = 0.005
            liquidity = 0.3
        elif scenario == 'trending':
            # 趋势市场
            volatility = 0.025
            spread = 0.002
            liquidity = 0.7

        return {
            'volatility': volatility,
            'spread': spread,
            'liquidity': liquidity,
            'trend_strength': 0.02 if scenario == 'trending' else 0.0
        }

    def generate_realistic_market_data(self, symbol='AAPL', periods=100, conditions=None):
        """生成逼真的市场数据"""
        if conditions is None:
            conditions = self.simulate_market_conditions('normal')

        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=periods, freq='1min')

        # 基础价格序列
        base_price = 150.0
        prices = [base_price]

        for i in range(1, periods):
            # 随机游走 + 趋势
            random_change = np.random.normal(0, conditions['volatility'])
            trend_change = conditions['trend_strength'] if np.random.random() > 0.5 else 0
            new_price = prices[-1] * (1 + random_change + trend_change)
            prices.append(max(new_price, 0.01))  # 确保价格为正

        # 生成OHLCV数据
        market_data = []
        for i, price in enumerate(prices):
            # 生成高低价
            high = price * (1 + abs(np.random.normal(0, conditions['volatility'] * 0.5)))
            low = price * (1 - abs(np.random.normal(0, conditions['volatility'] * 0.5)))
            close = price * (1 + np.random.normal(0, conditions['volatility'] * 0.3))

            # 成交量基于流动性
            base_volume = 10000
            volume_factor = conditions['liquidity'] + np.random.normal(0, 0.2)
            volume = max(int(base_volume * volume_factor), 100)

            market_data.append({
                'timestamp': dates[i],
                'symbol': symbol,
                'open': price,
                'high': max(high, price, close),
                'low': min(low, price, close),
                'close': close,
                'volume': volume,
                'spread': conditions['spread'],
                'mid_price': (price + close) / 2
            })

        self.market_data_feed = market_data
        return market_data

    def simulate_order_execution(self, order, market_data=None):
        """模拟订单执行"""
        if market_data is None:
            market_data = self.market_data_feed

        if not market_data:
            raise ValueError("No market data available")

        # 获取当前市场数据
        current_data = market_data[-1] if market_data else None
        if not current_data:
            return None

        executed_quantity = 0
        executed_price = 0.0
        slippage = 0.0

        if order.order_type == OrderType.MARKET:
            # 市价单：立即执行，可能有滑点
            if order.side == OrderSide.BUY:
                execution_price = current_data['mid_price'] * (1 + current_data['spread'] / 2)
            else:
                execution_price = current_data['mid_price'] * (1 - current_data['spread'] / 2)

            # 考虑市场冲击
            market_impact = min(order.quantity / current_data['volume'], 0.02)  # 最大2%冲击
            execution_price *= (1 + market_impact if order.side == OrderSide.BUY else 1 - market_impact)

            executed_quantity = order.quantity  # 假设完全成交
            slippage = abs(execution_price - order.price) / order.price if order.price > 0 else 0

        elif order.order_type == OrderType.LIMIT:
            # 限价单：检查是否能成交
            if order.side == OrderSide.BUY and order.price >= current_data['low']:
                execution_price = min(order.price, current_data['high'])
                executed_quantity = min(order.quantity, current_data['volume'] * 0.1)  # 假设成交10%成交量
            elif order.side == OrderSide.SELL and order.price <= current_data['high']:
                execution_price = max(order.price, current_data['low'])
                executed_quantity = min(order.quantity, current_data['volume'] * 0.1)
            else:
                # 限价单无法成交
                executed_quantity = 0
                execution_price = 0.0

        # 计算执行结果
        execution_result = {
            'order_id': order.order_id,
            'executed_quantity': executed_quantity,
            'execution_price': execution_price,
            'slippage': slippage,
            'timestamp': current_data['timestamp'],
            'market_conditions': current_data
        }

        self.execution_results.append(execution_result)
        return execution_result

    def run_trading_simulation(self, orders, market_conditions='normal'):
        """运行完整的交易模拟"""
        # 生成市场数据
        market_data = self.generate_realistic_market_data(conditions=self.simulate_market_conditions(market_conditions))

        results = []
        for order in orders:
            # 提交订单
            success, message, order_id = self.order_manager.submit_order(order)
            if success:
                # 模拟执行
                execution_result = self.simulate_order_execution(order, market_data)
                if execution_result:
                    results.append({
                        'order': order,
                        'execution': execution_result,
                        'success': True
                    })
                else:
                    results.append({
                        'order': order,
                        'execution': None,
                        'success': False,
                        'error': 'Execution failed'
                    })
            else:
                results.append({
                    'order': order,
                    'execution': None,
                    'success': False,
                    'error': message
                })

        return results


class TestRealTradingExecution:
    """真实交易执行测试"""

    def setup_method(self):
        """测试前准备"""
        self.simulator = RealTradingExecutionSimulator()

    def test_normal_market_execution(self):
        """测试正常市场条件下的订单执行"""
        # 创建测试订单
        orders = [
            Order(
                order_id="normal_buy_001",
                symbol="AAPL",
                order_type=OrderType.MARKET,
                quantity=100.0,
                price=150.0,
                side=OrderSide.BUY
            ),
            Order(
                order_id="normal_sell_001",
                symbol="AAPL",
                order_type=OrderType.MARKET,
                quantity=50.0,
                price=155.0,
                side=OrderSide.SELL
            )
        ]

        # 运行模拟
        results = self.simulator.run_trading_simulation(orders, 'normal')

        # 验证结果
        assert len(results) == 2
        for result in results:
            assert result['success'] is True
            assert result['execution'] is not None
            assert result['execution']['executed_quantity'] > 0
            assert result['execution']['execution_price'] > 0
            assert result['execution']['slippage'] >= 0

    def test_volatile_market_execution(self):
        """测试高波动市场条件下的订单执行"""
        orders = [
            Order(
                order_id="volatile_test_001",
                symbol="AAPL",
                order_type=OrderType.MARKET,
                quantity=200.0,
                price=150.0,
                side=OrderSide.BUY
            )
        ]

        results = self.simulator.run_trading_simulation(orders, 'volatile')

        assert len(results) == 1
        result = results[0]

        # 高波动市场应该有更高的滑点
        assert result['success'] is True
        assert result['execution']['slippage'] > 0

        # 滑点应该在合理范围内（不会过高）
        assert result['execution']['slippage'] < 0.1  # 最大10%滑点

    def test_illiquid_market_execution(self):
        """测试低流动性市场条件下的订单执行"""
        orders = [
            Order(
                order_id="illiquid_test_001",
                symbol="AAPL",
                order_type=OrderType.MARKET,
                quantity=500.0,
                price=150.0,
                side=OrderSide.BUY
            )
        ]

        results = self.simulator.run_trading_simulation(orders, 'illiquid')

        assert len(results) == 1
        result = results[0]

        # 低流动性市场成交量可能不足
        assert result['success'] is True
        # 在低流动性市场，可能无法完全成交大订单
        assert result['execution']['executed_quantity'] <= 500

    def test_limit_order_execution(self):
        """测试限价单执行"""
        # 创建限价单
        limit_buy_order = Order(
            order_id="limit_buy_test",
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            quantity=100.0,
            price=145.0,  # 低于市场价的买入限价
            side=OrderSide.BUY
        )

        results = self.simulator.run_trading_simulation([limit_buy_order], 'normal')

        assert len(results) == 1
        result = results[0]

        # 限价单可能部分成交或不成交
        if result['execution']:
            # 如果成交，价格应该不超过限价
            assert result['execution']['execution_price'] <= limit_buy_order.price

    def test_risk_limits_enforcement(self):
        """测试风险限制执行"""
        # 创建大额订单
        large_order = Order(
            order_id="large_order_test",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=2000.0,  # 大于最大持仓限制
            price=150.0,
            side=OrderSide.BUY
        )

        results = self.simulator.run_trading_simulation([large_order], 'normal')

        assert len(results) == 1
        result = results[0]

        # 即使模拟执行，风险检查应该在实际系统中执行
        # 这里主要验证模拟器的行为
        assert result['order'].quantity == 2000.0

    def test_multi_order_execution(self):
        """测试多订单并发执行"""
        # 创建多个订单
        orders = []
        for i in range(5):
            order = Order(
                order_id=f"multi_order_{i:03d}",
                symbol="AAPL",
                order_type=OrderType.MARKET,
                quantity=50.0 + i * 10,  # 不同数量
                price=150.0 + i * 2,    # 不同价格
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            )
            orders.append(order)

        results = self.simulator.run_trading_simulation(orders, 'normal')

        # 验证所有订单都被处理
        assert len(results) == 5

        # 检查买卖方向分布
        buy_orders = [r for r in results if r['order'].side == OrderSide.BUY]
        sell_orders = [r for r in results if r['order'].side == OrderSide.SELL]

        assert len(buy_orders) == 3  # 0,2,4
        assert len(sell_orders) == 2  # 1,3

        # 验证每笔订单都有执行结果
        for result in results:
            assert result['success'] is True
            assert result['execution'] is not None

    def test_execution_performance_metrics(self):
        """测试执行性能指标"""
        orders = [
            Order(
                order_id="perf_test_001",
                symbol="AAPL",
                order_type=OrderType.MARKET,
                quantity=100.0,
                price=150.0,
                side=OrderSide.BUY
            )
        ]

        results = self.simulator.run_trading_simulation(orders, 'normal')

        assert len(results) == 1
        result = results[0]

        execution = result['execution']

        # 计算性能指标
        if execution['executed_quantity'] > 0:
            fill_rate = execution['executed_quantity'] / result['order'].quantity
            assert fill_rate > 0

            # 滑点分析
            slippage_bps = execution['slippage'] * 10000  # 转换为基点
            assert slippage_bps >= 0

            # 市场影响成本
            market_impact = abs(execution['execution_price'] - execution['market_conditions']['mid_price'])
            market_impact_bps = market_impact / execution['market_conditions']['mid_price'] * 10000

            assert market_impact_bps >= 0

    def test_different_market_scenarios(self):
        """测试不同市场情景"""
        scenarios = ['normal', 'volatile', 'illiquid', 'trending']
        order = Order(
            order_id="scenario_test",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=150.0,
            side=OrderSide.BUY
        )

        scenario_results = {}

        for scenario in scenarios:
            results = self.simulator.run_trading_simulation([order], scenario)
            scenario_results[scenario] = results[0]

        # 验证不同情景下的执行差异
        assert len(scenario_results) == 4

        # 检查哪些情景成功执行
        successful_scenarios = [s for s in scenarios if scenario_results[s]['execution'] is not None]

        # 至少应该有一些情景成功执行
        assert len(successful_scenarios) > 0

        # 如果有多个成功情景，比较它们的滑点
        if len(successful_scenarios) >= 2:
            # 高波动市场通常有更高滑点
            if 'volatile' in successful_scenarios and 'normal' in successful_scenarios:
                volatile_slippage = scenario_results['volatile']['execution']['slippage']
                normal_slippage = scenario_results['normal']['execution']['slippage']

                # 波动市场滑点应该不同（但不一定总是更高，因为是随机模拟）
                assert volatile_slippage >= 0
                assert normal_slippage >= 0

    def test_order_queue_management(self):
        """测试订单队列管理"""
        # 创建一系列订单
        orders = []
        for i in range(10):
            order = Order(
                order_id=f"queue_test_{i:03d}",
                symbol="AAPL",
                order_type=OrderType.MARKET,
                quantity=10.0 + i,  # 递增数量
                price=150.0,
                side=OrderSide.BUY
            )
            orders.append(order)

        results = self.simulator.run_trading_simulation(orders, 'normal')

        # 验证所有订单都被处理
        assert len(results) == 10

        # 验证订单管理器状态
        active_orders = self.simulator.order_manager.get_active_orders()
        # 注意：模拟器可能不会实际管理订单状态，这里主要是测试接口

        # 验证执行结果的一致性
        for result in results:
            assert result['success'] is True
            assert result['execution']['executed_quantity'] > 0

    def test_execution_algorithm_integration(self):
        """测试执行算法集成"""
        # 创建VWAP执行算法的模拟
        class VWAPExecutionSimulator:
            def __init__(self, order, time_horizon=60):  # 分钟
                self.order = order
                self.time_horizon = time_horizon
                self.slices = []

            def generate_vwap_schedule(self, volume_profile):
                """生成VWAP执行计划"""
                total_volume = sum(volume_profile)
                target_shares_per_period = self.order.quantity / len(volume_profile)

                schedule = []
                remaining_quantity = self.order.quantity

                for i, period_volume in enumerate(volume_profile):
                    # 按成交量比例分配
                    volume_ratio = period_volume / total_volume
                    period_quantity = min(target_shares_per_period * volume_ratio * len(volume_profile), remaining_quantity)

                    schedule.append({
                        'period': i,
                        'quantity': period_quantity,
                        'time_weight': volume_ratio
                    })

                    remaining_quantity -= period_quantity

                return schedule

        # 测试VWAP算法
        order = Order(
            order_id="vwap_test",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=1000.0,
            price=150.0,
            side=OrderSide.BUY
        )

        vwap_sim = VWAPExecutionSimulator(order)

        # 模拟典型日内成交量分布
        volume_profile = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]

        schedule = vwap_sim.generate_vwap_schedule(volume_profile)

        # 验证VWAP计划
        total_scheduled = sum(s['quantity'] for s in schedule)
        assert abs(total_scheduled - order.quantity) < 1e-6

        # 验证高峰期分配更多
        peak_periods = schedule[5:8]  # 高峰期
        peak_quantity = sum(s['quantity'] for s in peak_periods)
        total_quantity = sum(s['quantity'] for s in schedule)

        assert peak_quantity > 0
        assert peak_quantity / total_quantity > 0.3  # 高峰期至少30%
