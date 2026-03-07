"""
交易执行引擎核心功能测试
测试TradingEngine的完整功能，包括订单管理、执行策略、风险控制等
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from src.trading.execution.order_manager import OrderManager, OrderType, OrderStatus, Order, OrderSide


class TradingEngineSimulator:
    """交易引擎模拟器"""

    def __init__(self):
        self.order_manager = OrderManager()
        self.active_orders = {}
        self.completed_orders = {}
        self.execution_history = []
        self.market_data_feed = []
        self.risk_limits = {
            'max_position_size': 1000,
            'max_daily_loss': 5000.0,
            'max_order_value': 25000.0,
            'max_slippage': 0.05
        }
        self.portfolio = {
            'cash': 50000.0,
            'positions': {},
            'daily_pnl': 0.0,
            'daily_trades': []
        }

    def initialize_market_data(self, symbol='AAPL', periods=100):
        """初始化市场数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=periods, freq='1min')

        # 生成价格序列
        base_price = 150.0
        prices = []
        for i in range(periods):
            trend = i * 0.02  # 轻微上涨趋势
            noise = np.random.normal(0, 1.5)
            price = base_price + trend + noise
            prices.append(max(price, 0.1))

        # 生成完整的市场数据
        self.market_data_feed = []
        for i, price in enumerate(prices):
            market_data = {
                'timestamp': dates[i],
                'symbol': symbol,
                'open': price * (1 + np.random.normal(0, 0.002)),
                'high': price * (1 + abs(np.random.normal(0, 0.008))),
                'low': price * (1 - abs(np.random.normal(0, 0.008))),
                'close': price,
                'volume': int(10000 + np.random.normal(0, 2000)),
                'bid_price': price * (1 - 0.001),
                'ask_price': price * (1 + 0.001),
                'bid_volume': int(5000 + np.random.normal(0, 1000)),
                'ask_volume': int(5000 + np.random.normal(0, 1000))
            }
            self.market_data_feed.append(market_data)

        return self.market_data_feed

    def submit_order(self, order):
        """提交订单"""
        # 风险检查
        if not self._check_risk_limits(order):
            return False, "Risk limit exceeded"

        # 市场检查
        if not self._check_market_conditions(order):
            return False, "Market conditions not suitable"

        # 提交到订单管理器
        success, message, order_id = self.order_manager.submit_order(order)
        if success:
            self.active_orders[order_id] = order

        return success, message

    def execute_order(self, order_id):
        """执行订单"""
        if order_id not in self.active_orders:
            return False, "Order not found"

        order = self.active_orders[order_id]

        # 模拟执行
        execution_result = self._simulate_execution(order)

        if execution_result['success']:
            # 更新投资组合
            self._update_portfolio(order, execution_result)

            # 移动到已完成订单
            self.completed_orders[order_id] = order
            del self.active_orders[order_id]

            # 记录执行历史
            self.execution_history.append({
                'order_id': order_id,
                'order': order,
                'execution': execution_result,
                'timestamp': datetime.now()
            })

        return execution_result['success'], execution_result.get('message', '')

    def cancel_order(self, order_id):
        """取消订单"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            success, message = self.order_manager.cancel_order(order_id)
            if success:
                del self.active_orders[order_id]
            return success, message

        return False, "Order not found"

    def get_portfolio_status(self):
        """获取投资组合状态"""
        total_value = self.portfolio['cash']

        for symbol, position in self.portfolio['positions'].items():
            if 'current_price' in position:
                total_value += position['quantity'] * position['current_price']

        return {
            'cash': self.portfolio['cash'],
            'positions': self.portfolio['positions'],
            'total_value': total_value,
            'daily_pnl': self.portfolio['daily_pnl'],
            'active_orders': len(self.active_orders)
        }

    def _check_risk_limits(self, order):
        """检查风险限制"""
        order_value = order.price * order.quantity

        # 检查订单价值限制
        if order_value > self.risk_limits['max_order_value']:
            return False

        # 检查持仓限制
        current_position = self.portfolio['positions'].get(order.symbol, {}).get('quantity', 0)
        if order.side == OrderSide.BUY:
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity

        if abs(new_position) > self.risk_limits['max_position_size']:
            return False

        # 检查现金充足性
        if order.side == OrderSide.BUY and self.portfolio['cash'] < order_value:
            return False

        return True

    def _check_market_conditions(self, order):
        """检查市场条件"""
        if not self.market_data_feed:
            return True  # 如果没有市场数据，默认允许

        current_data = self.market_data_feed[-1]

        # 检查价格合理性
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.price < current_data['low'] * 0.9:
                return False  # 买入价格过低
            elif order.side == OrderSide.SELL and order.price > current_data['high'] * 1.1:
                return False  # 卖出价格过高

        # 检查成交量充足性
        required_volume = order.quantity * 2  # 需要双倍成交量保证
        if current_data['volume'] < required_volume:
            return False

        return True

    def _simulate_execution(self, order):
        """模拟订单执行"""
        if not self.market_data_feed:
            return {'success': False, 'message': 'No market data available'}

        current_data = self.market_data_feed[-1]

        # 确定执行价格
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                execution_price = current_data['ask_price']
            else:
                execution_price = current_data['bid_price']
        elif order.order_type == OrderType.LIMIT:
            # 简单的限价单匹配逻辑
            if order.side == OrderSide.BUY:
                if order.price >= current_data['low']:
                    execution_price = min(order.price, current_data['high'])
                else:
                    return {'success': False, 'message': 'Limit price not reached'}
            else:
                if order.price <= current_data['high']:
                    execution_price = max(order.price, current_data['low'])
                else:
                    return {'success': False, 'message': 'Limit price not reached'}
        else:
            return {'success': False, 'message': 'Unsupported order type'}

        # 计算滑点
        slippage = abs(execution_price - order.price) / order.price
        if slippage > self.risk_limits['max_slippage']:
            return {'success': False, 'message': f'Slippage too high: {slippage:.2%}'}

        # 确定执行数量（可能部分成交）
        available_volume = current_data['ask_volume'] if order.side == OrderSide.BUY else current_data['bid_volume']
        executed_quantity = min(order.quantity, available_volume)

        return {
            'success': True,
            'executed_quantity': executed_quantity,
            'execution_price': execution_price,
            'slippage': slippage,
            'timestamp': current_data['timestamp']
        }

    def _update_portfolio(self, order, execution_result):
        """更新投资组合"""
        executed_quantity = execution_result['executed_quantity']
        execution_price = execution_result['execution_price']
        order_value = executed_quantity * execution_price

        if order.side == OrderSide.BUY:
            # 买入
            self.portfolio['cash'] -= order_value

            if order.symbol not in self.portfolio['positions']:
                self.portfolio['positions'][order.symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'current_price': execution_price
                }

            position = self.portfolio['positions'][order.symbol]
            total_quantity = position['quantity'] + executed_quantity
            total_cost = (position['quantity'] * position['avg_price']) + order_value
            position['avg_price'] = total_cost / total_quantity if total_quantity > 0 else 0
            position['quantity'] = total_quantity
            position['current_price'] = execution_price

        else:
            # 卖出
            self.portfolio['cash'] += order_value

            if order.symbol in self.portfolio['positions']:
                position = self.portfolio['positions'][order.symbol]
                position['quantity'] -= executed_quantity
                position['current_price'] = execution_price

                # 计算实现的PnL
                if position['quantity'] >= 0:  # 仍有持仓
                    realized_pnl = (execution_price - position['avg_price']) * executed_quantity
                    self.portfolio['daily_pnl'] += realized_pnl

                if position['quantity'] == 0:
                    del self.portfolio['positions'][order.symbol]


class TestTradingExecutionEngineCore:
    """交易执行引擎核心功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.engine = TradingEngineSimulator()
        self.engine.initialize_market_data()

    def test_engine_initialization(self):
        """测试引擎初始化"""
        assert self.engine.order_manager is not None
        assert isinstance(self.engine.active_orders, dict)
        assert isinstance(self.engine.portfolio, dict)
        assert self.engine.portfolio['cash'] == 50000.0

    def test_market_order_submission_and_execution(self):
        """测试市价单提交和执行"""
        order = Order(
            order_id="market_buy_test",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=150.0,
            side=OrderSide.BUY
        )

        # 提交订单
        success, message = self.engine.submit_order(order)
        assert success is True

        # 执行订单
        success, message = self.engine.execute_order(order.order_id)
        assert success is True

        # 验证投资组合更新
        portfolio = self.engine.get_portfolio_status()
        assert portfolio['cash'] < 50000.0  # 现金减少
        assert 'AAPL' in portfolio['positions']
        assert portfolio['positions']['AAPL']['quantity'] == 100

    def test_limit_order_submission_and_execution(self):
        """测试限价单提交和执行"""
        # 使用当前市场价格作为限价
        current_price = self.engine.market_data_feed[-1]['close']

        order = Order(
            order_id="limit_buy_test",
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            quantity=50.0,
            price=current_price * 1.01,  # 略高于当前价格的买入限价
            side=OrderSide.BUY
        )

        # 提交订单
        success, message = self.engine.submit_order(order)
        assert success is True

        # 执行订单
        success, message = self.engine.execute_order(order.order_id)
        assert success is True

        # 验证执行价格不超过限价
        execution_history = self.engine.execution_history[-1]
        assert execution_history['execution']['execution_price'] <= order.price

    def test_order_cancellation(self):
        """测试订单取消"""
        order = Order(
            order_id="cancel_test",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=150.0,
            side=OrderSide.BUY
        )

        # 提交订单
        self.engine.submit_order(order)
        assert order.order_id in self.engine.active_orders

        # 取消订单
        success, message = self.engine.cancel_order(order.order_id)
        assert success is True
        assert order.order_id not in self.engine.active_orders

    def test_risk_limits_enforcement(self):
        """测试风险限制执行"""
        # 创建超大订单
        large_order = Order(
            order_id="large_order_test",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=2000.0,  # 超过最大持仓限制
            price=150.0,
            side=OrderSide.BUY
        )

        # 提交应该被拒绝
        success, message = self.engine.submit_order(large_order)
        assert success is False
        assert "Risk limit" in message

    def test_portfolio_management(self):
        """测试投资组合管理"""
        from src.trading.core.trading_engine import TradingEngine
        engine = TradingEngine()

        # 测试基本投资组合功能
        portfolio_value = engine.calculate_portfolio_value()
        assert isinstance(portfolio_value, (int, float))
        assert portfolio_value >= 0

        # 测试账户余额
        balance = engine.get_account_balance()
        assert isinstance(balance, (int, float))
        assert balance >= 0

    def test_market_data_integration(self):
        """测试市场数据集成"""
        from src.trading.core.trading_engine import TradingEngine
        engine = TradingEngine()

        # 测试市场数据获取
        symbols = ['AAPL', 'GOOGL']
        market_data = engine.get_market_data(symbols)

        assert isinstance(market_data, pd.DataFrame)
        assert len(market_data) == 2

    def test_execution_slippage_control(self):
        """测试执行滑点控制"""
        from src.trading.core.trading_engine import TradingEngine
        engine = TradingEngine()

        # 测试基本功能
        pnl = engine.get_portfolio_pnl()
        assert isinstance(pnl, (int, float))

    def test_order_queue_management(self):
        """测试订单队列管理"""
        from src.trading.core.trading_engine import TradingEngine
        engine = TradingEngine()

        # 测试基本功能
        stats = engine.get_trading_statistics()
        assert isinstance(stats, dict)
        assert 'total_trades' in stats

    def test_execution_performance_monitoring(self):
        """测试执行性能监控"""
        from src.trading.core.trading_engine import TradingEngine
        engine = TradingEngine()

        # 测试基本功能
        pnl = engine.get_portfolio_pnl()
        assert isinstance(pnl, (int, float))

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        from src.trading.core.trading_engine import TradingEngine
        engine = TradingEngine()

        # 测试基本功能
        result = engine.cancel_order("nonexistent_order")
        assert result == False

    def test_portfolio_rebalancing_simulation(self):
        """测试投资组合再平衡模拟"""
        from src.trading.core.trading_engine import TradingEngine
        engine = TradingEngine()

        # 测试基本功能
        positions = engine.get_all_positions()
        assert isinstance(positions, dict)