# ---------------- src\core\trading\strategies\enhanced.py ----------------

from typing import Dict, Any, Type
import backtrader as bt
import pandas as pd
import numpy as np
from src.infrastructure.utils.tools import time_execution
from src.infrastructure.config.paths import path_config
from src.trading.backtester import BacktestEngine
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)  # 自动继承全局配置


class EnhancedTradingStrategy(bt.Strategy):
    """增强型交易策略类，集成多维度风险控制和自适应市场状态检测

    参数:
        volatility_window (int): 波动率计算窗口期，默认14天
        risk_multiplier (float): 风险乘数，默认2.0
        position_size_pct (float): 仓位占比，默认0.1
        trail_offset (float): 追踪止损偏移，默认0.05
        max_drawdown_pct (float): 最大回撤比例，默认0.2
        market_state_threshold (float): 市场状态阈值，默认0.3
        stress_test_period (int): 压力测试周期，默认30天
    """

    params = (
        ('volatility_window', 14),
        ('risk_multiplier', 2.0),
        ('position_size_pct', 0.1),
        ('trail_offset', 0.05),
        ('max_drawdown_pct', 0.2),
        ('market_state_threshold', 0.3),
        ('stress_test_period', 30),
        ('market_impact_model', None),  # 市场冲击成本模型
        ('slippage_model', None),  # 滑点模拟模型
    )

    def __init__(self):
        """初始化策略"""
        super().__init__()  # 显式调用父类初始化
        self.order = None
        self.trade_history = []
        self.market_state = 0  # 添加初始化
        self.market_state_changes = 0  # 添加初始化
        self.stress_test_counter = 0  # 添加初始化
        self.position_stop_loss = None  # 保留原有初始化
        self.position_entry_price = None  # 头寸入场价格
        self.position_entry_value = 0.0  # 头寸持仓数量
        # 交易成本模型
        self.market_impact_model = self.params.market_impact_model
        self.slippage_model = self.params.slippage_model

        # 初始化指标为空的LineBuffer
        self.volatility = bt.indicators.StandardDeviation(
            self.data.close,
            period=self.params.volatility_window
        )
        self.atr = bt.indicators.ATR(self.data, period=self.params.volatility_window)
        # 延迟计算，确保数据足够
        self.volatility.plotinfo.plot = False
        self.atr.plotinfo.plot = False

    def start(self):
        """策略启动时的初始化操作"""
        pass

    def notify_order(self, order):
        """订单状态通知处理"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self._handle_buy_order(order)
            elif order.issell():
                self._handle_sell_order(order)
            self.order = None

    def _handle_buy_order(self, order):
        if order.status != order.Completed:
            return  # 仅处理已完成的订单

        """处理买入订单"""
        self.position_entry_price = order.executed.price
        self.position_entry_date = self.data.datetime.date(0)  # 使用当前日期
        self.position_trail_price = order.executed.price

        # 添加非空检查
        if self.atr is not None and len(self.atr) > 0:
            self.position_stop_loss = order.executed.price - self.atr[0] * self.params.risk_multiplier
            self.position_take_profit = order.executed.price + self.atr[0] * self.params.risk_multiplier * 2
        else:
            logger.warning("ATR指标未初始化，使用默认止损/止盈值")
            self.position_stop_loss = order.executed.price * 0.95  # 默认止损值
            self.position_take_profit = order.executed.price * 1.10  # 默认止盈值

    def _handle_sell_order(self, order):
        """处理卖出订单"""
        if self.position_entry_price:
            self._record_trade(order)
            self._reset_position()

    def _record_trade(self, order):
        """记录交易历史"""
        trade_info = {
            'entry_date': self.position_entry_date,
            'exit_date': self.data.datetime.date(),
            'direction': 'Long',
            'entry_price': self.position_entry_price,
            'exit_price': order.executed.price,
            'pnl': order.executed.price - self.position_entry_price,
            'return_pct': (order.executed.price / self.position_entry_price - 1) * 100
        }
        self.trade_history.append(trade_info)

    def _reset_position(self):
        """重置仓位状态"""
        self.position_entry_price = None
        self.position_stop_loss = None
        self.position_take_profit = None
        self.position_trail_price = None

    def next(self):
        """执行下一个时间步的逻辑"""
        # 确保数据线存在
        if not all(hasattr(self.data, attr) for attr in ['high', 'low', 'close']):
            return

        # 延迟初始化指标
        if self.volatility is None:
            if len(self.data) >= self.params.volatility_window:
                self.atr = bt.indicators.ATR(self.data, period=self.params.volatility_window)
                self.volatility = bt.indicators.StandardDeviation(
                    self.data.close,
                    period=self.params.volatility_window
                )
            else:
                return

        # 等待指标完成预热（需要 volatility_window + 1 个数据点）
        required_bars = self.params.volatility_window + 1
        if len(self.data) < required_bars:
            return

        # 确保指标已初始化且有足够数据
        if (
                self.volatility is None
                or self.atr is None
                or len(self.volatility) < 1
                or len(self.atr) < 1
        ):
            return

        # 关键修复：检查索引是否存在
        if len(self.volatility) > 0 and len(self.atr) > 0:
            current_vol = self.volatility[0]
            prev_vol = self.volatility[-1] if len(self.volatility) > 1 else current_vol
            current_atr = self.atr[0]
            prev_atr = self.atr[-1] if len(self.atr) > 1 else current_atr
        else:
            return

        if self.order:
            return

        if not self.position:
            if not self._pass_volatility_filter():
                logger.debug("波动率过滤器未通过")
                return

            if not self._get_trading_signal():
                logger.debug("交易信号未触发")
                return

        """核心交易逻辑"""
        if self.position:
            self._manage_existing_position()
        else:
            self._evaluate_new_position()

    def _manage_existing_position(self):
        """管理已有仓位"""
        self._update_trailing_stop()
        self._check_stop_conditions()
        self._perform_stress_test()

    def _update_trailing_stop(self):
        """更新追踪止损"""
        if self.position_trail_price:
            new_trail = self.data.close[0] - self.atr[0] * self.params.trail_offset
            if new_trail > self.position_trail_price:
                self.position_trail_price = new_trail
                self.sell(exectype=bt.Order.StopTrail, trailamount=new_trail)  # 修正 trailamount

    def _check_stop_conditions(self):
        """检查止损止盈条件"""
        self._check_stop_loss()
        self._check_take_profit()
        self._check_max_drawdown()

    def _check_stop_loss(self):
        """检查止损条件"""
        if self.position_stop_loss is None:
            return  # 止损价未设置时直接返回
        if self.data.close[0] < self.position_stop_loss:
            self.sell()

    def _check_take_profit(self):
        """检查止盈条件"""
        if self.position_take_profit is not None and self.data.close[0] > self.position_take_profit:
            self.sell()

    def _check_max_drawdown(self):
        """检查最大回撤"""
        current_value = self.broker.getvalue()
        if current_value < self.position_entry_value * (1 - self.params.max_drawdown_pct):
            self.sell()

    def _perform_stress_test(self):
        """执行压力测试"""
        self.stress_test_counter += 1
        if self.stress_test_counter >= self.params.stress_test_period:
            self._adjust_risk_parameters()
            self.stress_test_counter = 0

    def _adjust_risk_parameters(self):
        """根据市场状态调整风险参数"""
        if self.market_state_changes > 3:
            logger.warning("市场波动加剧，进入保守模式")
            self.params.position_size_pct = 0.02
            self.params.risk_multiplier = 1.0
            self.market_state_changes = 0

    def _evaluate_new_position(self):
        """评估新仓位"""
        if self._pass_volatility_filter():
            self._determine_market_state()
            self._execute_new_trade()

    def _pass_volatility_filter(self) -> bool:
        if self.volatility is None or len(self.volatility) < 2:
            logger.debug("波动率指标未初始化或数据不足")
            return False

        # 获取当前和前一个值（避免直接索引）
        current_vol = self.volatility[0]
        prev_vol = self.volatility[-1] if len(self.volatility) > 1 else current_vol
        return current_vol <= prev_vol * 1.5  # 确保逻辑正确

    def _determine_market_state(self):
        if (self.volatility is None or self.atr is None or
                len(self.volatility) < 2 or len(self.atr) < 2):
            return

        # 计算波动率和ATR的变化率（绝对值）
        prev_vol = self.volatility[-1] if self.volatility[-1] != 0 else 1e-6
        current_vol = self.volatility[0]
        vol_change = abs((current_vol - prev_vol) / prev_vol)  # 使用绝对值

        prev_atr = self.atr[-1] if self.atr[-1] != 0 else 1e-6
        current_atr = self.atr[0]
        atr_change = abs((current_atr - prev_atr) / prev_atr)  # 使用绝对值

        # 调整条件：任一指标变化超过阈值即可触发状态变化
        if vol_change > 0.2 or atr_change > 0.2:
            self._update_market_state(1)  # 牛市
        elif vol_change < -0.2 or atr_change < -0.2:
            self._update_market_state(-1)  # 熊市
        else:
            self._update_market_state(0)

    def _update_market_state(self, new_state: int):
        """更新市场状态并跟踪变化"""
        if new_state != self.market_state:
            self.market_state_changes += 1
            self.prev_market_state = self.market_state
            self.market_state = new_state

    def _execute_new_trade(self):
        """执行新交易"""
        risk_unit = self.broker.getvalue() * self.params.position_size_pct
        position_size = risk_unit / (self.atr[0] * self.params.risk_multiplier)

        if self._get_trading_signal():
            # 提交订单并保存到 self.order
            self.order = self.buy(size=position_size)  # 关键修改：保存订单对象
            self.position_entry_value = self.broker.getvalue()
            self.position_entry_date = self.data.datetime.date()

    def _get_trading_signal(self) -> bool:
        """生成交易信号：当前价格 > 前一日价格"""
        if len(self.data.close) >= 2:
            # 明确使用close[0]和close[-1]比较
            return self.data.close[0] > self.data.close[-1]
        return False

    def stop(self):
        """策略终止处理"""
        if self.trade_history:
            pd.DataFrame(self.trade_history).to_csv("trading_log.csv", index=False)
            logger.info("交易记录已保存")

    def _execute_order(self, size, price, is_buy):
        """执行订单并考虑交易成本"""
        if is_buy:
            # 估计市场冲击成本
            impact_cost = self._estimate_market_impact(size, price)
            # 模拟滑点
            slippage = self._simulate_slippage(size, price)
            execution_price = price + impact_cost + slippage
        else:
            # 估计市场冲击成本（卖单冲击成本通常较低）
            impact_cost = self._estimate_market_impact(size, price) * 0.5
            # 模拟滑点
            slippage = self._simulate_slippage(size, price)
            execution_price = price - impact_cost - slippage

        return execution_price

    def _estimate_market_impact(self, size, price):
        """估计市场冲击成本"""
        if self.market_impact_model is None:
            # 简化的市场冲击模型：冲击成本与订单大小的平方根成正比
            return 0.01 * (abs(size) ** 0.5) / price
        else:
            return self.market_impact_model.estimate_impact(size, price)

    def _simulate_slippage(self, size, price):
        """模拟滑点"""
        if self.slippage_model is None:
            # 简化的滑点模型：滑点与订单大小的平方根成正比
            return 0.005 * (abs(size) ** 0.5) / price
        else:
            return self.slippage_model.simulate_slippage(size, price)

    def buy(self, size=None, price=None, **kwargs):
        execution_price = self._execute_order(size, price, is_buy=True)
        return super().buy(size=size, exectype=bt.Order.Market, price=execution_price, **kwargs)

    def sell(self, size=None, price=None, **kwargs):
        execution_price = self._execute_order(size, price, is_buy=False)
        return super().sell(size=size, exectype=bt.Order.Market, price=execution_price, **kwargs)


class StrategyExecutor:
    """策略执行器，用于运行和分析策略"""

    def __init__(self, strategy: Type[EnhancedTradingStrategy], data: pd.DataFrame):
        self.strategy = strategy
        self.data = data
        self.cerebro = bt.Cerebro()
        self._setup_cerebro()

    def _setup_cerebro(self):
        """设置Backtrader引擎"""
        self.cerebro.addstrategy(self.strategy)
        self.cerebro.adddata(bt.feeds.PandasData(dataname=self.data))
        self.cerebro.broker.setcash(100000.0)
        self.cerebro.broker.setcommission(commission=0.001)
        self.cerebro.addanalyzer(BacktestEngine, _name="backtest_analyzer")

    def run(self) -> Dict:
        """运行策略并返回分析结果"""
        results = self.cerebro.run()
        analyzer = results[0].analyzers.backtest_analyzer.get_analysis()
        return analyzer

    def plot(self):
        """可视化策略执行结果"""
        self.cerebro.plot(style='candlestick', volume=False)
