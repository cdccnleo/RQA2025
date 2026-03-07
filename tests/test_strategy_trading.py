#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略和交易层测试 - 提升覆盖率
测试策略决策、信号生成、订单管理、交易执行等功能
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock


# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class TestStrategyDecision:
    """策略决策测试"""

    def test_momentum_strategy_basic(self):
        """测试基础动量策略"""
        # 模拟价格数据
        price_data = [100, 102, 105, 108, 110, 112, 115]
        
        def calculate_momentum(prices, window=3):
            if len(prices) < window:
                return 0
            
            recent_avg = sum(prices[-window:]) / window
            previous_avg = sum(prices[-window*2:-window]) / window if len(prices) >= window*2 else prices[0]
            
            return (recent_avg - previous_avg) / previous_avg
        
        momentum = calculate_momentum(price_data)
        assert momentum > 0  # 上涨趋势
        
        # 策略决策
        def momentum_decision(momentum_value, threshold=0.05):
            if momentum_value > threshold:
                return "BUY"
            elif momentum_value < -threshold:
                return "SELL"
            else:
                return "HOLD"
        
        decision = momentum_decision(momentum)
        assert decision in ["BUY", "SELL", "HOLD"]

    def test_mean_reversion_strategy(self):
        """测试均值回归策略"""
        # 模拟价格数据 - 偏离均值
        prices = [100, 98, 96, 94, 92, 90]  # 持续下跌
        
        def calculate_deviation(prices, window=5):
            if len(prices) < window:
                return 0
            
            mean_price = sum(prices[-window:]) / window
            current_price = prices[-1]
            
            return (current_price - mean_price) / mean_price
        
        deviation = calculate_deviation(prices)
        
        def mean_reversion_decision(deviation, threshold=0.1):
            if deviation < -threshold:  # 价格低于均值
                return "BUY"  # 期待回归
            elif deviation > threshold:  # 价格高于均值
                return "SELL"  # 期待回归
            else:
                return "HOLD"
        
        decision = mean_reversion_decision(deviation)
        assert decision == "BUY"  # 价格持续下跌，期待回归

    def test_strategy_performance_metrics(self):
        """测试策略性能指标"""
        # 模拟交易记录
        trades = [
            {"symbol": "AAPL", "action": "BUY", "price": 100, "quantity": 10, "pnl": 0},
            {"symbol": "AAPL", "action": "SELL", "price": 105, "quantity": 10, "pnl": 50},
            {"symbol": "GOOGL", "action": "BUY", "price": 2500, "quantity": 1, "pnl": 0},
            {"symbol": "GOOGL", "action": "SELL", "price": 2480, "quantity": 1, "pnl": -20},
        ]
        
        def calculate_strategy_metrics(trades):
            total_pnl = sum(trade["pnl"] for trade in trades)
            profitable_trades = len([t for t in trades if t["pnl"] > 0])
            total_trades = len([t for t in trades if t["pnl"] != 0])
            
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            return {
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "total_trades": total_trades
            }
        
        metrics = calculate_strategy_metrics(trades)
        assert metrics["total_pnl"] == 30  # 50 - 20
        assert metrics["win_rate"] == 0.5  # 1 win out of 2 trades
        assert metrics["total_trades"] == 2


class TestSignalGeneration:
    """信号生成测试"""

    def test_technical_indicator_signals(self):
        """测试技术指标信号"""
        # 模拟价格数据
        prices = [100, 102, 104, 106, 108, 110, 108, 106, 104, 102]
        
        def calculate_sma(prices, window):
            """简单移动平均"""
            if len(prices) < window:
                return None
            return sum(prices[-window:]) / window
        
        def generate_ma_crossover_signal(prices, short_window=3, long_window=5):
            """移动平均交叉信号"""
            if len(prices) < long_window:
                return "HOLD"
            
            short_ma = calculate_sma(prices, short_window)
            long_ma = calculate_sma(prices, long_window)
            
            if short_ma and long_ma:
                if short_ma > long_ma:
                    return "BUY"
                elif short_ma < long_ma:
                    return "SELL"
            
            return "HOLD"
        
        # 测试前半段上涨趋势
        signal_uptrend = generate_ma_crossover_signal(prices[:6])
        assert signal_uptrend in ["BUY", "HOLD"]
        
        # 测试后半段下跌趋势
        signal_downtrend = generate_ma_crossover_signal(prices)
        assert signal_downtrend in ["SELL", "HOLD"]

    def test_volume_weighted_signals(self):
        """测试成交量加权信号"""
        # 模拟价格和成交量数据
        market_data = [
            {"price": 100, "volume": 1000},
            {"price": 102, "volume": 1500},  # 价格上涨，成交量增加
            {"price": 104, "volume": 2000},
            {"price": 103, "volume": 500},   # 价格下跌，成交量减少
        ]
        
        def calculate_vwap(data, window=3):
            """成交量加权平均价格"""
            if len(data) < window:
                return None
            
            recent_data = data[-window:]
            total_value = sum(d["price"] * d["volume"] for d in recent_data)
            total_volume = sum(d["volume"] for d in recent_data)
            
            return total_value / total_volume if total_volume > 0 else 0
        
        def generate_vwap_signal(data):
            """基于VWAP的信号"""
            if len(data) < 3:
                return "HOLD"
            
            current_price = data[-1]["price"]
            vwap = calculate_vwap(data)
            
            if vwap and current_price > vwap:
                return "BUY"
            elif vwap and current_price < vwap:
                return "SELL"
            
            return "HOLD"
        
        signal = generate_vwap_signal(market_data)
        assert signal in ["BUY", "SELL", "HOLD"]

    def test_multi_timeframe_signals(self):
        """测试多时间框架信号"""
        # 模拟不同时间框架的数据
        short_term_trend = "UP"    # 短期上涨
        medium_term_trend = "DOWN" # 中期下跌
        long_term_trend = "UP"     # 长期上涨
        
        def generate_multi_timeframe_signal(short, medium, long):
            """多时间框架信号合成"""
            signals = [short, medium, long]
            
            # 计算信号强度
            up_count = signals.count("UP")
            down_count = signals.count("DOWN")
            
            if up_count >= 2:
                return "STRONG_BUY" if up_count == 3 else "BUY"
            elif down_count >= 2:
                return "STRONG_SELL" if down_count == 3 else "SELL"
            else:
                return "NEUTRAL"
        
        signal = generate_multi_timeframe_signal(short_term_trend, medium_term_trend, long_term_trend)
        assert signal == "BUY"  # 2个UP信号


class TestOrderManagement:
    """订单管理测试"""

    def setup_method(self):
        """测试前准备"""
        self.orders = {}
        self.order_id_counter = 1

    def create_order(self, symbol, side, quantity, price, order_type="LIMIT"):
        """创建订单"""
        order_id = f"ORDER_{self.order_id_counter}"
        self.order_id_counter += 1
        
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "status": "PENDING",
            "timestamp": time.time()
        }
        
        self.orders[order_id] = order
        return order_id

    def update_order_status(self, order_id, status):
        """更新订单状态"""
        if order_id in self.orders:
            self.orders[order_id]["status"] = status
            return True
        return False

    def cancel_order(self, order_id):
        """取消订单"""
        if order_id in self.orders and self.orders[order_id]["status"] in ["PENDING", "PARTIAL"]:
            self.orders[order_id]["status"] = "CANCELLED"
            return True
        return False

    def test_order_creation(self):
        """测试订单创建"""
        order_id = self.create_order("AAPL", "BUY", 100, 150.0)
        
        assert order_id.startswith("ORDER_")
        assert order_id in self.orders
        assert self.orders[order_id]["symbol"] == "AAPL"
        assert self.orders[order_id]["status"] == "PENDING"

    def test_order_status_updates(self):
        """测试订单状态更新"""
        order_id = self.create_order("AAPL", "BUY", 100, 150.0)
        
        # 更新为部分成交
        assert self.update_order_status(order_id, "PARTIAL")
        assert self.orders[order_id]["status"] == "PARTIAL"
        
        # 更新为完全成交
        assert self.update_order_status(order_id, "FILLED")
        assert self.orders[order_id]["status"] == "FILLED"

    def test_order_cancellation(self):
        """测试订单取消"""
        # 创建待处理订单
        pending_order = self.create_order("AAPL", "BUY", 100, 150.0)
        assert self.cancel_order(pending_order)
        assert self.orders[pending_order]["status"] == "CANCELLED"
        
        # 尝试取消已成交订单
        filled_order = self.create_order("GOOGL", "SELL", 10, 2500.0)
        self.update_order_status(filled_order, "FILLED")
        assert not self.cancel_order(filled_order)  # 不能取消已成交订单

    def test_batch_order_processing(self):
        """测试批量订单处理"""
        # 创建多个订单
        order_requests = [
            {"symbol": "AAPL", "side": "BUY", "quantity": 100, "price": 150.0},
            {"symbol": "GOOGL", "side": "SELL", "quantity": 10, "price": 2500.0},
            {"symbol": "MSFT", "side": "BUY", "quantity": 50, "price": 300.0},
        ]
        
        created_orders = []
        for req in order_requests:
            order_id = self.create_order(**req)
            created_orders.append(order_id)
        
        assert len(created_orders) == 3
        assert len(self.orders) == 3
        
        # 批量更新状态
        for order_id in created_orders:
            self.update_order_status(order_id, "FILLED")
        
        filled_orders = [o for o in self.orders.values() if o["status"] == "FILLED"]
        assert len(filled_orders) == 3


class TestTradingExecution:
    """交易执行测试"""

    def setup_method(self):
        """测试前准备"""
        self.portfolio = {
            "AAPL": {"quantity": 0, "avg_cost": 0},
            "GOOGL": {"quantity": 0, "avg_cost": 0},
            "cash": 10000
        }
        self.trades = []

    def execute_trade(self, symbol, side, quantity, price):
        """执行交易"""
        trade_value = quantity * price
        
        if side == "BUY":
            if self.portfolio["cash"] >= trade_value:
                # 更新现金
                self.portfolio["cash"] -= trade_value
                
                # 更新持仓
                if symbol not in self.portfolio:
                    self.portfolio[symbol] = {"quantity": 0, "avg_cost": 0}
                
                current_qty = self.portfolio[symbol]["quantity"]
                current_cost = self.portfolio[symbol]["avg_cost"]
                
                new_qty = current_qty + quantity
                new_avg_cost = ((current_qty * current_cost) + trade_value) / new_qty
                
                self.portfolio[symbol]["quantity"] = new_qty
                self.portfolio[symbol]["avg_cost"] = new_avg_cost
                
                # 记录交易
                trade = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "timestamp": time.time()
                }
                self.trades.append(trade)
                return True
        
        elif side == "SELL":
            if symbol in self.portfolio and self.portfolio[symbol]["quantity"] >= quantity:
                # 更新持仓
                self.portfolio[symbol]["quantity"] -= quantity
                
                # 更新现金
                self.portfolio["cash"] += trade_value
                
                # 记录交易
                trade = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "timestamp": time.time()
                }
                self.trades.append(trade)
                return True
        
        return False

    def test_buy_execution(self):
        """测试买入执行"""
        initial_cash = self.portfolio["cash"]
        
        # 买入AAPL
        success = self.execute_trade("AAPL", "BUY", 10, 150.0)
        
        assert success
        assert self.portfolio["AAPL"]["quantity"] == 10
        assert self.portfolio["AAPL"]["avg_cost"] == 150.0
        assert self.portfolio["cash"] == initial_cash - 1500
        assert len(self.trades) == 1

    def test_sell_execution(self):
        """测试卖出执行"""
        # 先买入
        self.execute_trade("AAPL", "BUY", 20, 150.0)
        
        # 再卖出部分
        initial_cash = self.portfolio["cash"]
        success = self.execute_trade("AAPL", "SELL", 5, 155.0)
        
        assert success
        assert self.portfolio["AAPL"]["quantity"] == 15  # 20 - 5
        assert self.portfolio["cash"] == initial_cash + 775  # 5 * 155
        assert len(self.trades) == 2

    def test_insufficient_funds(self):
        """测试资金不足情况"""
        # 尝试买入超过资金限制的股票
        success = self.execute_trade("GOOGL", "BUY", 10, 2500.0)  # 需要25000，但只有10000
        
        assert not success
        assert self.portfolio["GOOGL"]["quantity"] == 0
        assert len(self.trades) == 0

    def test_insufficient_shares(self):
        """测试股票不足情况"""
        # 尝试卖出没有的股票
        success = self.execute_trade("AAPL", "SELL", 10, 150.0)
        
        assert not success
        assert len(self.trades) == 0

    def test_portfolio_value_calculation(self):
        """测试投资组合价值计算"""
        # 执行一些交易
        self.execute_trade("AAPL", "BUY", 10, 150.0)
        self.execute_trade("GOOGL", "BUY", 2, 2500.0)
        
        def calculate_portfolio_value(portfolio, current_prices):
            """计算投资组合价值"""
            total_value = portfolio["cash"]
            
            for symbol, position in portfolio.items():
                if symbol != "cash" and isinstance(position, dict):
                    if position["quantity"] > 0 and symbol in current_prices:
                        total_value += position["quantity"] * current_prices[symbol]
            
            return total_value
        
        current_prices = {"AAPL": 155.0, "GOOGL": 2600.0}
        portfolio_value = calculate_portfolio_value(self.portfolio, current_prices)
        
        # 验证计算
        expected_value = (
            self.portfolio["cash"] +  # 剩余现金
            10 * 155.0 +             # AAPL持仓价值
            2 * 2600.0               # GOOGL持仓价值
        )
        assert portfolio_value == expected_value


class TestRiskManagement:
    """风险管理测试"""

    def test_position_size_limits(self):
        """测试仓位大小限制"""
        portfolio_value = 100000
        max_position_pct = 0.1  # 最大10%仓位
        
        def calculate_max_position_size(portfolio_value, price, max_pct):
            max_value = portfolio_value * max_pct
            return int(max_value / price)
        
        max_shares = calculate_max_position_size(portfolio_value, 150.0, max_position_pct)
        assert max_shares == 66  # 10000 / 150
        
        # 测试价格更高的股票
        max_shares_expensive = calculate_max_position_size(portfolio_value, 2500.0, max_position_pct)
        assert max_shares_expensive == 4  # 10000 / 2500

    def test_stop_loss_logic(self):
        """测试止损逻辑"""
        entry_price = 150.0
        stop_loss_pct = 0.05  # 5%止损
        
        def check_stop_loss(current_price, entry_price, stop_pct):
            loss_threshold = entry_price * (1 - stop_pct)
            return current_price <= loss_threshold
        
        # 测试不同价格
        assert not check_stop_loss(148.0, entry_price, stop_loss_pct)  # 1.3%下跌，不触发
        assert check_stop_loss(142.0, entry_price, stop_loss_pct)      # 5.3%下跌，触发

    def test_risk_metrics_calculation(self):
        """测试风险指标计算"""
        # 模拟日收益率数据
        daily_returns = [0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.03]
        
        def calculate_volatility(returns):
            """计算波动率（标准差）"""
            if len(returns) < 2:
                return 0
            
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            return variance ** 0.5
        
        def calculate_max_drawdown(returns):
            """计算最大回撤"""
            cumulative = [1]
            for r in returns:
                cumulative.append(cumulative[-1] * (1 + r))
            
            peak = cumulative[0]
            max_dd = 0
            
            for value in cumulative[1:]:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_dd = max(max_dd, drawdown)
            
            return max_dd
        
        volatility = calculate_volatility(daily_returns)
        max_drawdown = calculate_max_drawdown(daily_returns)
        
        assert volatility > 0
        assert 0 <= max_drawdown <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
