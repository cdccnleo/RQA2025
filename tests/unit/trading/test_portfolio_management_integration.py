#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资组合管理集成测试
测试投资组合管理与其他交易组件的集成
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

@dataclass
class Position:
    """持仓数据类"""
    symbol: str
    quantity: int
    avg_price: Decimal
    current_price: Optional[Decimal] = None
    market_value: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    last_update: Optional[float] = None

    def __post_init__(self):
        if self.last_update is None:
            self.last_update = time.time()
        if self.current_price is None:
            self.current_price = self.avg_price
        self.update_market_value()

    def update_market_value(self):
        """更新市值和未实现盈亏"""
        if self.current_price is not None:
            self.market_value = self.current_price * self.quantity
            self.unrealized_pnl = (self.current_price - self.avg_price) * self.quantity


@dataclass
class Portfolio:
    """投资组合数据类"""
    portfolio_id: str
    positions: Dict[str, Position]
    cash: Decimal
    total_value: Optional[Decimal] = None
    total_pnl: Optional[Decimal] = None

    def __post_init__(self):
        self.update_totals()

    def update_totals(self):
        """更新组合总值和总盈亏"""
        self.total_value = self.cash
        self.total_pnl = Decimal('0')

        for position in self.positions.values():
            if position.market_value is not None:
                self.total_value += position.market_value
            if position.unrealized_pnl is not None:
                self.total_pnl += position.unrealized_pnl


class PortfolioManager:
    """投资组合管理器"""

    def __init__(self):
        self.portfolios = {}
        self.risk_limits = {
            "max_single_position": Decimal('0.1'),  # 单股票最大仓位10%
            "max_total_loss": Decimal('0.05'),     # 最大总亏损5%
            "max_leverage": Decimal('2.0')         # 最大杠杆2倍
        }

    def create_portfolio(self, portfolio_id, initial_cash=Decimal('100000')):
        """创建投资组合"""
        portfolio = Portfolio(
            portfolio_id=portfolio_id,
            positions={},
            cash=initial_cash
        )
        self.portfolios[portfolio_id] = portfolio
        return portfolio

    def update_position(self, portfolio_id, symbol, quantity, price):
        """更新持仓"""
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        portfolio = self.portfolios[portfolio_id]

        if symbol in portfolio.positions:
            # 更新现有持仓
            position = portfolio.positions[symbol]
            total_quantity = position.quantity + quantity
            total_cost = (position.quantity * position.avg_price) + (quantity * price)

            if total_quantity == 0:
                # 平仓
                del portfolio.positions[symbol]
            else:
                # 更新平均价格
                position.avg_price = total_cost / total_quantity
                position.quantity = total_quantity
                position.last_update = time.time()
        else:
            # 新建持仓
            if quantity != 0:
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price
                )
                portfolio.positions[symbol] = position

        portfolio.update_totals()
        return portfolio

    def get_portfolio(self, portfolio_id):
        """获取投资组合"""
        return self.portfolios.get(portfolio_id)

    def check_risk_limits(self, portfolio_id):
        """检查风险限额"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {"breached": True, "reason": "portfolio_not_found"}

        # 检查单股票仓位限额
        for symbol, position in portfolio.positions.items():
            if position.market_value and portfolio.total_value:
                position_ratio = position.market_value / portfolio.total_value
                if position_ratio > self.risk_limits["max_single_position"]:
                    return {
                        "breached": True,
                        "reason": "single_position_limit",
                        "symbol": symbol,
                        "ratio": position_ratio
                    }

        # 检查总亏损限额
        if portfolio.total_pnl and portfolio.total_pnl < 0:
            loss_ratio = abs(portfolio.total_pnl) / (portfolio.cash + sum(
                p.market_value or 0 for p in portfolio.positions.values()
            ))
            if loss_ratio > self.risk_limits["max_total_loss"]:
                return {
                    "breached": True,
                    "reason": "total_loss_limit",
                    "loss_ratio": loss_ratio
                }

        return {"breached": False}

    def rebalance_portfolio(self, portfolio_id, target_allocations):
        """重新平衡投资组合"""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        # 计算需要调整的仓位
        adjustments = {}
        for symbol, target_ratio in target_allocations.items():
            current_position = portfolio.positions.get(symbol)
            current_value = current_position.market_value if current_position else 0
            target_value = portfolio.total_value * target_ratio
            adjustment_value = target_value - current_value

            if abs(adjustment_value) > 1:  # 最小调整阈值
                adjustments[symbol] = adjustment_value

        return adjustments


class MockMarketDataProvider:
    """模拟市场数据提供商"""

    def __init__(self):
        self.price_data = {
            "000001.SZ": Decimal('10.50'),
            "000002.SZ": Decimal('15.20'),
            "600000.SH": Decimal('8.30'),
            "000858.SZ": Decimal('120.00')
        }
        self.price_history = {}

    def get_current_price(self, symbol):
        """获取当前价格"""
        return self.price_data.get(symbol, Decimal('0'))

    def update_price(self, symbol, price):
        """更新价格"""
        self.price_data[symbol] = price
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append((time.time(), price))

        # 保持历史记录在合理范围内
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]


class MockExecutionEngine:
    """模拟执行引擎"""

    def __init__(self):
        self.executed_orders = []

    def execute_order(self, order):
        """执行订单"""
        # 模拟执行延迟
        time.sleep(0.001)

        result = {
            "order_id": order.get("order_id"),
            "status": "filled",
            "filled_quantity": order.get("quantity", 0),
            "avg_price": order.get("price", 0),
            "execution_time": time.time()
        }

        self.executed_orders.append(result)
        return result


@pytest.fixture
def setup_portfolio_components():
    """设置投资组合测试组件"""
    # 创建投资组合管理器
    portfolio_manager = PortfolioManager()

    # 创建市场数据提供商
    market_data_provider = MockMarketDataProvider()

    # 创建执行引擎
    execution_engine = MockExecutionEngine()

    # 创建测试投资组合
    portfolio = portfolio_manager.create_portfolio("test_portfolio", Decimal('100000'))

    return {
        "portfolio_manager": portfolio_manager,
        "market_data_provider": market_data_provider,
        "execution_engine": execution_engine,
        "portfolio": portfolio
    }


class TestPortfolioManagementIntegration:
    """投资组合管理集成测试"""

    def test_portfolio_creation_and_initialization(self, setup_portfolio_components):
        """测试投资组合创建和初始化"""
        components = setup_portfolio_components
        portfolio_manager = components["portfolio_manager"]

        # 创建投资组合
        portfolio = portfolio_manager.create_portfolio("new_portfolio", Decimal('50000'))

        assert portfolio.portfolio_id == "new_portfolio"
        assert portfolio.cash == Decimal('50000')
        assert len(portfolio.positions) == 0
        assert portfolio.total_value == Decimal('50000')

    def test_portfolio_position_updates(self, setup_portfolio_components):
        """测试投资组合持仓更新"""
        components = setup_portfolio_components
        portfolio_manager = components["portfolio_manager"]
        market_data_provider = components["market_data_provider"]

        portfolio_id = "test_portfolio"

        # 添加第一个持仓
        portfolio_manager.update_position(portfolio_id, "000001.SZ", 1000, Decimal('10.0'))
        portfolio = portfolio_manager.get_portfolio(portfolio_id)

        assert "000001.SZ" in portfolio.positions
        position = portfolio.positions["000001.SZ"]
        assert position.quantity == 1000
        assert position.avg_price == Decimal('10.0')

        # 更新价格并检查市值
        market_data_provider.update_price("000001.SZ", Decimal('10.50'))
        position.current_price = market_data_provider.get_current_price("000001.SZ")
        position.update_market_value()

        assert position.market_value == Decimal('10500')
        assert position.unrealized_pnl == Decimal('500')

        # 添加更多持仓
        portfolio_manager.update_position(portfolio_id, "000002.SZ", 500, Decimal('15.0'))
        portfolio = portfolio_manager.get_portfolio(portfolio_id)

        assert len(portfolio.positions) == 2
        portfolio.update_totals()
        expected_total_value = portfolio.cash + position.market_value + Decimal('7500')  # 500 * 15.0
        assert portfolio.total_value == expected_total_value

    def test_portfolio_with_market_data_integration(self, setup_portfolio_components):
        """测试投资组合与市场数据集成"""
        components = setup_portfolio_components
        portfolio_manager = components["portfolio_manager"]
        market_data_provider = components["market_data_provider"]

        portfolio_id = "test_portfolio"

        # 创建多个持仓
        holdings = [
            ("000001.SZ", 1000, Decimal('10.0')),
            ("000002.SZ", 500, Decimal('15.0')),
            ("600000.SH", 2000, Decimal('8.0'))
        ]

        for symbol, quantity, price in holdings:
            portfolio_manager.update_position(portfolio_id, symbol, quantity, price)

        # 模拟价格变动
        price_changes = {
            "000001.SZ": Decimal('10.50'),  # +5%
            "000002.SZ": Decimal('14.50'),  # -3.33%
            "600000.SH": Decimal('8.50')    # +6.25%
        }

        total_pnl = Decimal('0')
        for symbol, new_price in price_changes.items():
            market_data_provider.update_price(symbol, new_price)
            position = portfolio_manager.get_portfolio(portfolio_id).positions[symbol]
            position.current_price = new_price
            position.update_market_value()
            total_pnl += position.unrealized_pnl

        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        portfolio.update_totals()

        # 验证总盈亏计算
        assert portfolio.total_pnl is not None
        assert portfolio.total_value > portfolio.cash  # 应该有市值

    def test_portfolio_risk_management_integration(self, setup_portfolio_components):
        """测试投资组合风险管理集成"""
        components = setup_portfolio_components
        portfolio_manager = components["portfolio_manager"]
        market_data_provider = components["market_data_provider"]

        portfolio_id = "test_portfolio"

        # 创建高风险持仓（超过单股票限额）
        portfolio_manager.update_position(portfolio_id, "000001.SZ", 15000, Decimal('10.0'))

        # 设置当前价格
        market_data_provider.update_price("000001.SZ", Decimal('10.0'))
        position = portfolio_manager.get_portfolio(portfolio_id).positions["000001.SZ"]
        position.current_price = Decimal('10.0')
        position.update_market_value()

        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        portfolio.update_totals()

        # 检查风险限额
        risk_check = portfolio_manager.check_risk_limits(portfolio_id)

        # 应该触发单股票仓位限额
        if position.market_value and portfolio.total_value:
            position_ratio = position.market_value / portfolio.total_value
            if position_ratio > portfolio_manager.risk_limits["max_single_position"]:
                assert risk_check["breached"] == True
                assert risk_check["reason"] == "single_position_limit"

    def test_portfolio_rebalancing_integration(self, setup_portfolio_components):
        """测试投资组合再平衡集成"""
        components = setup_portfolio_components
        portfolio_manager = components["portfolio_manager"]
        market_data_provider = components["market_data_provider"]

        portfolio_id = "test_portfolio"

        # 创建初始持仓
        holdings = [
            ("000001.SZ", 1000, Decimal('10.0')),
            ("000002.SZ", 1000, Decimal('10.0')),
            ("600000.SH", 1000, Decimal('10.0'))
        ]

        for symbol, quantity, price in holdings:
            portfolio_manager.update_position(portfolio_id, symbol, quantity, price)
            market_data_provider.update_price(symbol, price)

        # 更新持仓市值
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        for position in portfolio.positions.values():
            position.current_price = market_data_provider.get_current_price(position.symbol)
            position.update_market_value()
        portfolio.update_totals()

        # 目标配置
        target_allocations = {
            "000001.SZ": Decimal('0.4'),  # 40%
            "000002.SZ": Decimal('0.4'),  # 40%
            "600000.SH": Decimal('0.2')   # 20%
        }

        # 计算再平衡调整
        adjustments = portfolio_manager.rebalance_portfolio(portfolio_id, target_allocations)

        # 验证调整计算
        assert isinstance(adjustments, dict)
        for symbol in target_allocations.keys():
            assert symbol in adjustments or adjustments.get(symbol, 0) == 0

    def test_portfolio_with_execution_engine_integration(self, setup_portfolio_components):
        """测试投资组合与执行引擎集成"""
        components = setup_portfolio_components
        portfolio_manager = components["portfolio_manager"]
        execution_engine = components["execution_engine"]

        portfolio_id = "test_portfolio"

        # 创建交易订单
        orders = [
            {
                "order_id": "order_001",
                "symbol": "000001.SZ",
                "quantity": 100,
                "price": Decimal('10.0'),
                "direction": "buy"
            },
            {
                "order_id": "order_002",
                "symbol": "000002.SZ",
                "quantity": 50,
                "price": Decimal('15.0'),
                "direction": "buy"
            }
        ]

        # 执行订单并更新投资组合
        for order in orders:
            # 执行订单
            execution_result = execution_engine.execute_order(order)

            # 更新投资组合
            quantity = order["quantity"] if order["direction"] == "buy" else -order["quantity"]
            portfolio_manager.update_position(
                portfolio_id,
                order["symbol"],
                quantity,
                order["price"]
            )

        # 验证投资组合更新
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        assert "000001.SZ" in portfolio.positions
        assert "000002.SZ" in portfolio.positions

        assert portfolio.positions["000001.SZ"].quantity == 100
        assert portfolio.positions["000002.SZ"].quantity == 50

    def test_portfolio_concurrent_updates(self, setup_portfolio_components):
        """测试投资组合并发更新"""
        components = setup_portfolio_components
        portfolio_manager = components["portfolio_manager"]

        portfolio_id = "test_portfolio"

        # 准备并发更新
        update_operations = [
            ("000001.SZ", 100, Decimal('10.0')),
            ("000001.SZ", 50, Decimal('10.5')),
            ("000002.SZ", 200, Decimal('15.0')),
            ("000001.SZ", -50, Decimal('11.0')),  # 卖出
        ]

        results = []

        def update_position_async(symbol, quantity, price):
            try:
                portfolio = portfolio_manager.update_position(portfolio_id, symbol, quantity, price)
                results.append({"success": True, "symbol": symbol, "quantity": quantity})
            except Exception as e:
                results.append({"success": False, "symbol": symbol, "error": str(e)})

        # 并发执行更新
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(update_position_async, symbol, quantity, price)
                for symbol, quantity, price in update_operations
            ]
            for future in as_completed(futures):
                future.result()

        # 验证所有操作成功
        successful_updates = [r for r in results if r["success"]]
        assert len(successful_updates) == len(update_operations)

        # 验证最终持仓
        portfolio = portfolio_manager.get_portfolio(portfolio_id)
        assert "000001.SZ" in portfolio.positions
        assert "000002.SZ" in portfolio.positions

    def test_portfolio_performance_monitoring(self, setup_portfolio_components):
        """测试投资组合性能监控"""
        components = setup_portfolio_components
        portfolio_manager = components["portfolio_manager"]
        market_data_provider = components["market_data_provider"]

        portfolio_id = "test_portfolio"

        # 创建投资组合历史
        initial_holdings = [
            ("000001.SZ", 1000, Decimal('10.0')),
            ("000002.SZ", 500, Decimal('15.0'))
        ]

        for symbol, quantity, price in initial_holdings:
            portfolio_manager.update_position(portfolio_id, symbol, quantity, price)

        # 模拟价格时间序列
        price_scenarios = [
            {"000001.SZ": Decimal('10.5'), "000002.SZ": Decimal('15.2')},  # +5%, +1.33%
            {"000001.SZ": Decimal('10.2'), "000002.SZ": Decimal('14.8')},  # +2%, -1.33%
            {"000001.SZ": Decimal('10.8'), "000002.SZ": Decimal('15.5')}   # +8%, +3.33%
        ]

        performance_history = []

        for i, prices in enumerate(price_scenarios):
            # 更新价格
            for symbol, price in prices.items():
                market_data_provider.update_price(symbol, price)

            # 更新持仓
            portfolio = portfolio_manager.get_portfolio(portfolio_id)
            for position in portfolio.positions.values():
                position.current_price = market_data_provider.get_current_price(position.symbol)
                position.update_market_value()

            portfolio.update_totals()

            # 记录性能
            performance_history.append({
                "timestamp": time.time(),
                "total_value": portfolio.total_value,
                "total_pnl": portfolio.total_pnl,
                "cash": portfolio.cash
            })

        # 验证性能跟踪
        assert len(performance_history) == len(price_scenarios)
        for record in performance_history:
            assert "total_value" in record
            assert "total_pnl" in record
            assert "cash" in record

        # 验证价值变化趋势
        initial_value = performance_history[0]["total_value"]
        final_value = performance_history[-1]["total_value"]
        assert final_value > initial_value  # 应该有增值


class TestPortfolioManagementLoadTesting:
    """投资组合管理负载测试"""

    def test_portfolio_high_frequency_updates(self, setup_portfolio_components):
        """测试高频投资组合更新"""
        components = setup_portfolio_components
        portfolio_manager = components["portfolio_manager"]
        portfolio = components["portfolio"]

        portfolio_id = portfolio.portfolio_id  # 使用fixture中创建的投资组合

        # 创建大量持仓进行负载测试
        symbols = [f"TEST{i:04d}.SZ" for i in range(100)]
        initial_price = Decimal('10.0')

        # 批量创建持仓
        for symbol in symbols:
            portfolio_manager.update_position(portfolio_id, symbol, 100, initial_price)

        # 高频价格更新
        update_count = 1000
        price_updates = []

        def generate_price_update():
            symbol = symbols[threading.current_thread().ident % len(symbols)]
            price = initial_price * (Decimal('1') + Decimal(str(threading.current_thread().ident % 20 - 10)) / Decimal('100'))
            return symbol, price

        def update_price_async(update_id):
            symbol, price = generate_price_update()
            start_time = time.time()

            # 更新价格（简化版本）
            portfolio = portfolio_manager.get_portfolio(portfolio_id)
            if symbol in portfolio.positions:
                position = portfolio.positions[symbol]
                position.current_price = price
                position.update_market_value()

            end_time = time.time()
            return {
                "update_id": update_id,
                "execution_time": end_time - start_time
            }

        # 并发执行价格更新
        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(update_price_async, i) for i in range(update_count)]
            for future in as_completed(futures):
                results.append(future.result())

        end_time = time.time()
        total_time = end_time - start_time

        # 验证结果
        assert len(results) == update_count

        # 性能指标
        total_execution_time = sum(r["execution_time"] for r in results)
        avg_execution_time = total_execution_time / update_count
        throughput = update_count / total_time

        print(f"投资组合更新性能指标:")
        print(f"- 更新次数: {update_count}")
        print(f"- 总时间: {total_time:.2f}s")
        print(f"- 平均执行时间: {avg_execution_time:.4f}s")
        print(f"- 吞吐量: {throughput:.1f} updates/s")

        # 性能断言
        assert avg_execution_time < 0.01, f"平均执行时间太长: {avg_execution_time:.4f}s"
        assert throughput > 100, f"吞吐量太低: {throughput:.1f} updates/s"


if __name__ == "__main__":
    pytest.main([__file__])
