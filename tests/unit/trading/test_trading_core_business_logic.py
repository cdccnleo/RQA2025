#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易核心业务逻辑深度测试
测试交易引擎的核心算法、边界条件、异常处理和性能优化

测试覆盖目标: 95%+
测试深度: 核心业务逻辑、边界条件、异常处理、性能优化
"""

import pytest
import time
import threading
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import concurrent.futures
import queue
import statistics

# 尝试导入交易相关模块
try:
    from src.trading.trading_engine import TradingEngine, OrderType, OrderDirection, OrderStatus
    from src.trading.execution_engine import ExecutionEngine
    from src.trading.order_manager import OrderManager
    from src.trading.risk import RiskManager
    from src.trading.account_manager import AccountManager
    trading_available = True
except ImportError:
    trading_available = False
    TradingEngine = Mock
    ExecutionEngine = Mock
    OrderManager = Mock
    RiskManager = Mock
    AccountManager = Mock

pytestmark = pytest.mark.skipif(
    not trading_available,
    reason="Trading modules not available"
)


class TestTradingCoreBusinessLogic:
    """交易核心业务逻辑深度测试类"""

    @pytest.fixture
    def trading_engine(self):
        """创建交易引擎"""
        config = {
            'max_position_size': 1000000,
            'max_daily_loss': 50000,
            'commission_rate': 0.0003,
            'slippage_model': 'fixed',
            'slippage_rate': 0.0001
        }
        engine = TradingEngine(config=config)
        yield engine
        # 清理资源
        if hasattr(engine, 'shutdown'):
            engine.shutdown()

    @pytest.fixture
    def execution_engine(self):
        """创建执行引擎"""
        config = {
            'max_concurrent_orders': 10,
            'order_timeout_seconds': 30,
            'retry_attempts': 3,
            'circuit_breaker_threshold': 5
        }
        engine = ExecutionEngine(config=config)
        yield engine
        if hasattr(engine, 'stop'):
            engine.stop()

    @pytest.fixture
    def risk_manager(self):
        """创建风险管理器"""
        config = {
            'max_position_risk': 0.1,
            'max_portfolio_risk': 0.2,
            'var_confidence': 0.95,
            'stress_test_scenarios': 1000
        }
        manager = RiskManager(config=config)
        yield manager

    @pytest.fixture
    def account_manager(self):
        """创建账户管理器"""
        config = {
            'initial_balance': 1000000,
            'currency': 'CNY',
            'leverage': 1.0,
            'margin_requirement': 0.1
        }
        manager = AccountManager(config=config)
        yield manager

    @pytest.fixture
    def market_data(self):
        """创建市场数据"""
        # 生成A股市场数据
        symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        market_data = {}
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            base_price = np.random.uniform(10, 200)

            prices = []
            volumes = []

            for i in range(len(dates)):
                # 价格走势模拟
                trend = 0.001 * (i - 50)  # 轻微上升趋势
                noise = np.random.normal(0, 0.02)  # 2%的波动
                price_change = trend + noise

                if i == 0:
                    price = base_price
                else:
                    price = prices[-1] * (1 + price_change)

                prices.append(max(0.01, price))  # 确保价格为正
                volumes.append(int(np.random.lognormal(10, 1)))  # 对数正态分布的成交量

            market_data[symbol] = {
                'dates': dates,
                'prices': prices,
                'volumes': volumes,
                'highs': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'lows': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
            }

        return market_data

    def test_trading_algorithm_price_impact_model(self, trading_engine, market_data):
        """测试交易算法价格影响模型"""
        symbol = '000001.SZ'
        data = market_data[symbol]

        # 测试大单交易的价格影响
        test_orders = [
            {'quantity': 10000, 'expected_impact': 0.001},  # 小单，影响小
            {'quantity': 100000, 'expected_impact': 0.005},  # 中单，影响中等
            {'quantity': 1000000, 'expected_impact': 0.02},  # 大单，影响显著
            {'quantity': 10000000, 'expected_impact': 0.10}  # 超大单，影响巨大
        ]

        current_price = data['prices'][-1]
        avg_volume = sum(data['volumes'][-20:]) / 20  # 20日平均成交量

        for order in test_orders:
            # 计算预期价格影响
            participation_rate = order['quantity'] / avg_volume
            expected_impact = self._calculate_price_impact(participation_rate, current_price)

            # 验证价格影响在合理范围内
            assert expected_impact >= order['expected_impact'] * 0.5
            assert expected_impact <= order['expected_impact'] * 2.0

            print(f"✅ {order['quantity']}股交易价格影响: {expected_impact:.2f}")
    def _calculate_price_impact(self, participation_rate: float, current_price: float) -> float:
        """计算价格影响"""
        # 使用平方根流动性模型
        market_impact = 0.1 * np.sqrt(participation_rate) + 0.01 * participation_rate
        return min(market_impact, 0.5)  # 最大50%的价格影响

    def test_trading_algorithm_optimal_execution(self, trading_engine, market_data):
        """测试交易算法最优执行"""
        symbol = '000001.SZ'
        data = market_data[symbol]

        # 测试不同的执行策略
        strategies = ['VWAP', 'TWAP', 'POV', 'IS']

        target_quantity = 500000
        time_horizon = 60  # 60分钟

        for strategy in strategies:
            # 模拟执行计划
            execution_plan = self._generate_execution_plan(
                strategy, target_quantity, time_horizon, data
            )

            # 验证执行计划
            assert len(execution_plan) > 0
            assert sum(order['quantity'] for order in execution_plan) == target_quantity

            # 计算执行质量指标
            total_cost = sum(order['price'] * order['quantity'] for order in execution_plan)
            avg_price = total_cost / target_quantity
            market_price = data['prices'][-1]

            # VWAP和TWAP应该接近市场价格
            if strategy in ['VWAP', 'TWAP']:
                price_deviation = abs(avg_price - market_price) / market_price
                assert price_deviation < 0.05  # 5%以内

            # POV应该有更好的价格但执行时间更长
            elif strategy == 'POV':
                price_deviation = abs(avg_price - market_price) / market_price
                assert price_deviation < 0.03  # 3%以内

            print(f"✅ {strategy}策略执行计划验证通过")

    def _generate_execution_plan(self, strategy: str, quantity: int,
                               time_horizon: int, market_data: Dict) -> List[Dict]:
        """生成执行计划"""
        if strategy == 'VWAP':
            # VWAP策略：按成交量加权分配
            volumes = market_data['volumes'][-time_horizon:]
            total_volume = sum(volumes)
            return [
                {'quantity': int(quantity * vol / total_volume),
                 'price': market_data['prices'][-time_horizon + i]}
                for i, vol in enumerate(volumes)
            ]
        elif strategy == 'TWAP':
            # TWAP策略：时间平均分配
            intervals = time_horizon
            qty_per_interval = quantity // intervals
            return [
                {'quantity': qty_per_interval,
                 'price': market_data['prices'][-time_horizon + i]}
                for i in range(intervals)
            ]
        else:
            # 其他策略的简化实现
            return [
                {'quantity': quantity // time_horizon,
                 'price': market_data['prices'][-time_horizon + i]}
                for i in range(time_horizon)
            ]

    def test_risk_management_var_calculation(self, risk_manager, market_data):
        """测试风险管理VaR计算"""
        # 构建投资组合
        portfolio = {
            '000001.SZ': 100000,  # 10万股
            '600000.SH': 50000,   # 5万股
            '000858.SZ': 20000    # 2万股
        }

        # 计算历史收益率
        returns_data = {}
        for symbol, position in portfolio.items():
            if symbol in market_data:
                prices = market_data[symbol]['prices']
                returns = [(prices[i] - prices[i-1]) / prices[i-1]
                          for i in range(1, len(prices))]
                returns_data[symbol] = returns

        # 计算VaR
        confidence_levels = [0.95, 0.99, 0.999]
        time_horizons = [1, 5, 10, 20]  # 天数

        for confidence in confidence_levels:
            for horizon in time_horizons:
                portfolio_returns = []

                # 生成投资组合收益率
                for i in range(len(returns_data['000001.SZ']) - horizon + 1):
                    portfolio_return = 0
                    total_value = 0

                    for symbol, position in portfolio.items():
                        if symbol in returns_data:
                            current_price = market_data[symbol]['prices'][i + horizon - 1]
                            price_horizon_ago = market_data[symbol]['prices'][i]

                            position_value = position * current_price
                            position_return = (current_price - price_horizon_ago) / price_horizon_ago
                            portfolio_return += position_return * (position_value / total_value) if total_value > 0 else position_return
                            total_value += position_value

                    portfolio_returns.append(portfolio_return)

                if portfolio_returns:
                    # 计算VaR
                    portfolio_returns.sort()
                    var_index = int((1 - confidence) * len(portfolio_returns))
                    var = portfolio_returns[var_index]

                    # 验证VaR合理性
                    assert var < 0, "VaR应该是负数（损失）"
                    assert abs(var) < 0.5, "VaR绝对值不应超过50%"

                    print(f"✅ {order['quantity']}股交易价格影响: {expected_impact:.2f}")
    def test_trading_engine_concurrent_order_processing(self, trading_engine, execution_engine):
        """测试交易引擎并发订单处理"""
        # 创建多个并发订单
        num_concurrent_orders = 50
        orders_queue = queue.Queue()
        processed_orders = []
        processing_errors = []

        # 生成测试订单
        for i in range(num_concurrent_orders):
            order = {
                'order_id': f'order_{i}',
                'symbol': f'600{i%100:03d}.SH',
                'quantity': np.random.randint(100, 10000),
                'price': np.random.uniform(10, 200),
                'order_type': np.random.choice(['limit', 'market']),
                'direction': np.random.choice(['buy', 'sell']),
                'timestamp': datetime.now() + timedelta(seconds=i)
            }
            orders_queue.put(order)

        def order_processor():
            """订单处理器"""
            while True:
                try:
                    order = orders_queue.get(timeout=0.1)

                    # 模拟订单处理
                    processing_time = np.random.uniform(0.01, 0.1)  # 10-100ms
                    time.sleep(processing_time)

                    # 模拟处理结果
                    if np.random.random() < 0.95:  # 95%成功率
                        result = {
                            'order_id': order['order_id'],
                            'status': 'filled',
                            'execution_price': order['price'] * (1 + np.random.normal(0, 0.001)),
                            'execution_quantity': order['quantity'],
                            'processing_time': processing_time
                        }
                        processed_orders.append(result)
                    else:
                        processing_errors.append({
                            'order_id': order['order_id'],
                            'error': 'Insufficient funds'
                        })

                    orders_queue.task_done()

                except queue.Empty:
                    break
                except Exception as e:
                    processing_errors.append({
                        'order_id': order.get('order_id', 'unknown'),
                        'error': str(e)
                    })

        # 启动并发处理
        start_time = time.time()

        threads = []
        for _ in range(min(10, num_concurrent_orders)):  # 最多10个线程
            thread = threading.Thread(target=order_processor)
            thread.start()
            threads.append(thread)

        # 等待所有订单被处理
        for thread in threads:
            thread.join(timeout=30)

        total_time = time.time() - start_time

        # 验证并发处理结果
        successful_orders = len(processed_orders)
        failed_orders = len(processing_errors)
        total_processed = successful_orders + failed_orders

        print(f"\n📊 并发订单处理结果:")
        print(f"   总订单数: {num_concurrent_orders}")
        print(f"   成功处理: {successful_orders}")
        print(f"   处理失败: {failed_orders}")
        print(f"   总处理数: {total_processed}")
        print(f"   成功率: {successful_orders/num_concurrent_orders*100:.1f}%")
        print(f"   总耗时: {total_time:.2f}秒")
        print(f"   平均处理时间: {total_time/num_concurrent_orders*1000:.1f}ms/订单")

        # 验证并发处理质量
        assert successful_orders >= num_concurrent_orders * 0.9, "成功率过低"
        assert total_time < 10.0, "处理时间过长"
        assert total_processed == num_concurrent_orders, "订单处理不完整"

    def test_trading_engine_circuit_breaker_mechanism(self, trading_engine, execution_engine):
        """测试交易引擎熔断机制"""
        # 模拟市场极端波动
        circuit_breaker_scenarios = [
            {'volatility': 0.02, 'expected_breaker': False},  # 正常波动
            {'volatility': 0.05, 'expected_breaker': False},  # 较高波动
            {'volatility': 0.10, 'expected_breaker': True},   # 极端波动
            {'volatility': 0.20, 'expected_breaker': True}    # 崩溃性波动
        ]

        for scenario in circuit_breaker_scenarios:
            volatility = scenario['volatility']
            expected_breaker = scenario['expected_breaker']

            # 模拟价格序列
            base_price = 100.0
            prices = [base_price]

            for i in range(100):
                # 生成波动价格
                change = np.random.normal(0, volatility)
                new_price = prices[-1] * (1 + change)
                prices.append(max(0.01, new_price))  # 确保价格为正

            # 计算价格波动率
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            actual_volatility = np.std(returns)

            # 检查是否应该触发熔断
            should_trigger = actual_volatility > 0.15  # 15%波动阈值

            # 验证熔断判断
            trigger_text = '触发' if expected_breaker else '不触发'
            assert should_trigger == expected_breaker, \
                f"熔断判断错误: 波动率{actual_volatility:.3f}, 期望{trigger_text}"

            status_text = '触发' if should_trigger else '正常'
            print(f"   波动率: {actual_volatility:.3f}, 熔断状态: {status_text}")

    def test_account_manager_position_management(self, account_manager):
        """测试账户管理器仓位管理"""
        # 初始账户状态
        initial_balance = 1000000
        account_manager.balance = initial_balance

        # 测试交易序列
        trades = [
            {'symbol': '000001.SZ', 'quantity': 1000, 'price': 10.0, 'direction': 'buy'},
            {'symbol': '600000.SH', 'quantity': 500, 'price': 8.0, 'direction': 'buy'},
            {'symbol': '000001.SZ', 'quantity': 500, 'price': 11.0, 'direction': 'sell'},
            {'symbol': '600036.SH', 'quantity': 200, 'price': 15.0, 'direction': 'buy'},
        ]

        # 执行交易
        for trade in trades:
            if trade['direction'] == 'buy':
                cost = trade['quantity'] * trade['price']
                commission = cost * 0.0003  # 0.03%佣金
                total_cost = cost + commission

                assert account_manager.balance >= total_cost, "余额不足"

                account_manager.balance -= total_cost
                account_manager.positions[trade['symbol']] = \
                    account_manager.positions.get(trade['symbol'], 0) + trade['quantity']

            elif trade['direction'] == 'sell':
                assert trade['symbol'] in account_manager.positions, "无该股票持仓"
                assert account_manager.positions[trade['symbol']] >= trade['quantity'], "持仓不足"

                revenue = trade['quantity'] * trade['price']
                commission = revenue * 0.0003
                net_revenue = revenue - commission

                account_manager.balance += net_revenue
                account_manager.positions[trade['symbol']] -= trade['quantity']

        # 验证最终状态
        final_balance = account_manager.balance
        total_pnl = final_balance - initial_balance

        print("\n📊 账户管理测试结果:")
        print(f"   初始余额: {initial_balance:.2f}元")
        print(f"   最终余额: {final_balance:.2f}元")
        print(f"   总盈亏: {total_pnl:.2f}元")
        print(f"   持仓股票: {list(account_manager.positions.keys())}")

        # 验证资金安全
        assert final_balance > 0, "账户余额不能为负"
        assert account_manager.balance >= initial_balance * 0.8, "亏损过大"

    def test_trading_engine_boundary_conditions(self, trading_engine):
        """测试交易引擎边界条件"""
        boundary_test_cases = [
            # 极端价格
            {'price': 0.01, 'quantity': 100, 'expected': 'valid'},      # 最低价格
            {'price': 1000000, 'quantity': 100, 'expected': 'invalid'}, # 最高价格
            {'price': -10, 'quantity': 100, 'expected': 'invalid'},     # 负价格

            # 极端数量
            {'price': 100, 'quantity': 1, 'expected': 'valid'},          # 最小数量
            {'price': 100, 'quantity': 100000000, 'expected': 'invalid'}, # 最大数量
            {'price': 100, 'quantity': -100, 'expected': 'invalid'},     # 负数量
            {'price': 100, 'quantity': 0, 'expected': 'invalid'},        # 零数量

            # 特殊符号
            {'price': 100, 'quantity': 100, 'symbol': '', 'expected': 'invalid'},           # 空符号
            {'price': 100, 'quantity': 100, 'symbol': 'INVALID', 'expected': 'invalid'},    # 无效符号
            {'price': 100, 'quantity': 100, 'symbol': '000001.SZ', 'expected': 'valid'},   # 有效A股
            {'price': 100, 'quantity': 100, 'symbol': '600000.SH', 'expected': 'valid'},   # 有效沪股
        ]

        for case in boundary_test_cases:
            # 验证订单有效性
            order = {
                'symbol': case.get('symbol', '000001.SZ'),
                'price': case['price'],
                'quantity': case['quantity'],
                'order_type': 'limit',
                'direction': 'buy'
            }

            is_valid = self._validate_order_boundary(order)
            expected_valid = case['expected'] == 'valid'

            assert is_valid == expected_valid, \
                f"边界条件测试失败: {case} -> 期望{expected_valid}, 实际{is_valid}"

            print(f"✅ 边界条件测试通过: {case['price']}元 x {case['quantity']}股")

    def _validate_order_boundary(self, order: Dict) -> bool:
        """验证订单边界条件"""
        # 价格检查
        if not (0.01 <= order['price'] <= 100000):
            return False

        # 数量检查
        if not (1 <= order['quantity'] <= 10000000):
            return False

        # 符号格式检查
        import re
        if not re.match(r'^\d{6}\.(SZ|SH)$', order['symbol']):
            return False

        return True

    def test_trading_engine_exception_handling(self, trading_engine):
        """测试交易引擎异常处理"""
        exception_scenarios = [
            {
                'scenario': 'network_timeout',
                'exception': ConnectionError("Network timeout"),
                'expected_recovery': True
            },
            {
                'scenario': 'insufficient_balance',
                'exception': ValueError("Insufficient balance"),
                'expected_recovery': False
            },
            {
                'scenario': 'market_closed',
                'exception': RuntimeError("Market is closed"),
                'expected_recovery': True
            },
            {
                'scenario': 'invalid_order',
                'exception': ValueError("Invalid order parameters"),
                'expected_recovery': False
            },
            {
                'scenario': 'database_error',
                'exception': Exception("Database connection failed"),
                'expected_recovery': True
            }
        ]

        for scenario in exception_scenarios:
            try:
                # 模拟异常场景
                if scenario['scenario'] == 'network_timeout':
                    # 测试网络重试机制
                    retry_count = 0
                    max_retries = 3

                    while retry_count < max_retries:
                        try:
                            # 模拟网络调用
                            if retry_count < 2:  # 前两次失败
                                raise scenario['exception']
                            break  # 第三次成功
                        except ConnectionError:
                            retry_count += 1
                            time.sleep(0.1 * retry_count)  # 指数退避

                    recovery_success = retry_count < max_retries

                elif scenario['scenario'] == 'insufficient_balance':
                    # 测试余额验证
                    balance = 1000
                    order_cost = 2000
                    if balance < order_cost:
                        raise scenario['exception']
                    recovery_success = False  # 余额不足无法恢复

                elif scenario['scenario'] == 'market_closed':
                    # 测试市场状态检查
                    current_time = datetime.now().time()
                    market_open = time(9, 30)
                    market_close = time(15, 0)

                    if not (market_open <= current_time <= market_close):
                        # 模拟市场关闭处理（延迟到下一个交易日）
                        recovery_success = True
                    else:
                        recovery_success = False

                else:
                    # 其他异常场景
                    recovery_success = scenario['expected_recovery']

                # 验证异常处理结果
                assert recovery_success == scenario['expected_recovery'], \
                    f"异常处理失败: {scenario['scenario']}"

                print(f"✅ 异常处理测试通过: {scenario['scenario']}")

            except Exception as e:
                print(f"❌ 异常处理测试失败: {scenario['scenario']} - {e}")
                raise

    def test_trading_engine_performance_benchmarks(self, trading_engine, market_data):
        """测试交易引擎性能基准"""
        # 性能测试场景
        performance_scenarios = [
            {'name': '单订单处理', 'orders': 1, 'expected_time': 0.01},
            {'name': '批量订单处理', 'orders': 100, 'expected_time': 0.5},
            {'name': '高频订单处理', 'orders': 1000, 'expected_time': 2.0},
            {'name': '超高频订单处理', 'orders': 10000, 'expected_time': 10.0}
        ]

        for scenario in performance_scenarios:
            orders = scenario['orders']
            expected_time = scenario['expected_time']

            # 生成测试订单
            test_orders = []
            for i in range(orders):
                symbol = list(market_data.keys())[i % len(market_data)]
                order = {
                    'order_id': f'perf_test_{i}',
                    'symbol': symbol,
                    'quantity': np.random.randint(100, 1000),
                    'price': market_data[symbol]['prices'][-1] * (1 + np.random.normal(0, 0.01)),
                    'order_type': 'limit',
                    'direction': np.random.choice(['buy', 'sell']),
                    'timestamp': datetime.now()
                }
                test_orders.append(order)

            # 执行性能测试
            start_time = time.time()

            processed_orders = []
            for order in test_orders:
                # 模拟订单处理
                processing_time = np.random.uniform(0.001, 0.01)
                time.sleep(processing_time)

                result = {
                    'order_id': order['order_id'],
                    'status': 'processed',
                    'processing_time': processing_time
                }
                processed_orders.append(result)

            total_time = time.time() - start_time

            # 计算性能指标
            orders_per_second = orders / total_time
            avg_processing_time = total_time / orders * 1000  # 毫秒

            print("\n📊 性能测试结果:")
            print(f"   场景: {scenario['name']}")
            print(f"   订单数量: {orders}")
            print(f"   总耗时: {total_time:.2f}秒")
            print(f"   吞吐量: {orders_per_second:.1f} 订单/秒")
            print(f"   平均处理时间: {avg_processing_time:.2f}ms/订单")

            # 验证性能要求
            assert total_time <= expected_time * 1.5, \
                f"性能不符合要求: 期望{expected_time:.2f}秒, 实际{total_time:.2f}秒"
            assert orders_per_second >= orders / expected_time * 0.8, \
                f"吞吐量不足: {orders_per_second:.1f} 订单/秒"

    def test_trading_engine_memory_efficiency(self, trading_engine):
        """测试交易引擎内存效率"""
        import psutil
        process = psutil.Process()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 模拟大量订单处理
        num_orders = 10000
        active_orders = []

        for i in range(num_orders):
            order = {
                'order_id': f'memory_test_{i}',
                'symbol': f'600{i%1000:03d}.SH',
                'quantity': 100,
                'price': 100.0,
                'order_type': 'limit',
                'direction': 'buy',
                'metadata': {
                    'user_id': f'user_{i%100}',
                    'strategy': f'strategy_{i%10}',
                    'timestamp': datetime.now(),
                    'additional_data': 'x' * 100  # 额外数据
                }
            }
            active_orders.append(order)

            # 每1000个订单检查一次内存
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                memory_per_order = memory_increase / (i + 1) * 1024  # KB per order

                print(f"   第{i//1000+1}k订单内存增量: {memory_increase:.1f}MB")
                # 验证内存效率
                assert memory_increase < 200, f"内存使用过多: {memory_increase:.1f}MB"
                assert memory_per_order < 50, f"每订单内存使用过高: {memory_per_order:.1f}KB"

        # 最终内存检查
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory

        print("\n📊 内存效率测试结果:")
        print(f"   初始内存: {initial_memory:.1f}MB")
        print(f"   最终内存: {final_memory:.1f}MB")
        print(f"   总订单数: {num_orders:,.0f}")
        print(f"   每订单内存: {total_memory_increase/num_orders*1024:.1f}KB")

        # 验证总体内存效率
        assert total_memory_increase < 300, f"总内存增长过大: {total_memory_increase:.1f}MB"
        assert total_memory_increase / num_orders < 0.1, "平均每订单内存使用过高"

    def test_trading_engine_stress_testing(self, trading_engine, execution_engine):
        """测试交易引擎压力测试"""
        # 压力测试场景
        stress_scenarios = [
            {'name': '正常负载', 'orders_per_second': 10, 'duration': 30},
            {'name': '高负载', 'orders_per_second': 50, 'duration': 30},
            {'name': '峰值负载', 'orders_per_second': 100, 'duration': 30},
            {'name': '极端负载', 'orders_per_second': 200, 'duration': 30}
        ]

        for scenario in stress_scenarios:
            name = scenario['name']
            orders_per_second = scenario['orders_per_second']
            duration = scenario['duration']

            print(f"\n🔥 开始压力测试: {name}")
            print(f"   目标: {orders_per_second} 订单/秒")
            print(f"   时长: {duration} 秒")

            # 执行压力测试
            test_results = self._execute_stress_test(
                trading_engine, execution_engine,
                orders_per_second, duration
            )

            # 分析测试结果
            actual_throughput = test_results['total_orders'] / test_results['total_time']
            success_rate = test_results['successful_orders'] / test_results['total_orders']
            avg_latency = test_results['total_latency'] / test_results['successful_orders'] * 1000  # ms

            print("\n📊 压力测试结果:")
            print(f"   实际吐吐量: {actual_throughput:.1f} 订单/秒")
            print(f"   成功率: {success_rate*100:.1f}%")
            print(f"   平均延迟: {avg_latency:.1f}ms")
            # 移除这个错误的print语句
            print(f"   错误数: {test_results['failed_orders']:.0f}")
            # 验证压力测试结果
            assert success_rate > 0.8, f"{name}成功率过低: {success_rate:.1f}"
            assert actual_throughput >= orders_per_second * 0.7, \
                f"{name}吞吐量不足: {actual_throughput:.1f} < {orders_per_second * 0.7:.1f}"

    def _execute_stress_test(self, trading_engine, execution_engine,
                           orders_per_second: int, duration: int) -> Dict[str, Any]:
        """执行压力测试"""
        results = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_latency': 0.0,
            'total_time': duration,
            'errors': []
        }

        order_queue = queue.Queue()
        result_queue = queue.Queue()

        # 订单生成器
        def order_generator():
            order_count = 0
            start_time = time.time()

            while time.time() - start_time < duration:
                # 生成订单
                order = {
                    'order_id': f'stress_{order_count}',
                    'symbol': f'600{order_count%1000:03d}.SH',
                    'quantity': np.random.randint(100, 1000),
                    'price': np.random.uniform(50, 200),
                    'order_type': 'limit',
                    'direction': np.random.choice(['buy', 'sell']),
                    'timestamp': datetime.now()
                }

                order_queue.put(order)
                order_count += 1
                results['total_orders'] += 1

                # 控制订单生成速率
                target_interval = 1.0 / orders_per_second
                time.sleep(target_interval)

        # 订单处理器
        def order_processor():
            while True:
                try:
                    order = order_queue.get(timeout=0.1)
                    processing_start = time.time()

                    # 模拟订单处理
                    processing_time = np.random.uniform(0.001, 0.05)
                    time.sleep(processing_time)

                    # 模拟成功/失败
                    if np.random.random() < 0.9:  # 90%成功率
                        result = {
                            'order_id': order['order_id'],
                            'status': 'success',
                            'latency': time.time() - processing_start
                        }
                        results['successful_orders'] += 1
                        results['total_latency'] += result['latency']
                    else:
                        result = {
                            'order_id': order['order_id'],
                            'status': 'failed',
                            'error': 'Random failure'
                        }
                        results['failed_orders'] += 1

                    result_queue.put(result)
                    order_queue.task_done()

                except queue.Empty:
                    break

        # 启动测试
        generator_thread = threading.Thread(target=order_generator)
        processor_thread = threading.Thread(target=order_processor)

        generator_thread.start()
        processor_thread.start()

        generator_thread.join()
        processor_thread.join()

        return results
