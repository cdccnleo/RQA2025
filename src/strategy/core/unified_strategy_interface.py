#!/usr/bin/env python3
"""
统一策略服务接口

定义策略服务层策略的统一接口，确保所有策略实现统一的API。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    POSITIONAL = "positional"
    DAY_TRADING = "day_trading"
    MACHINE_LEARNING = "machine_learning"
    QUANTITATIVE = "quantitative"
    HIGH_FREQUENCY = "high_frequency"


class StrategyStatus(Enum):
    """策略状态"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    BACKTESTING = "backtesting"
    LIVE_TRADING = "live_trading"


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_BUY = "close_buy"
    CLOSE_SELL = "close_sell"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class StrategySignal:
    """策略信号"""
    strategy_id: str
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: Optional[float] = None
    quantity: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StrategyPosition:
    """策略持仓"""
    strategy_id: str
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class StrategyOrder:
    """策略订单"""
    strategy_id: str
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: float
    timestamp: datetime
    price: Optional[float] = None
    status: str = "pending"


@dataclass
class StrategyMetrics:
    """策略指标"""
    strategy_id: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    calmar_ratio: float
    sortino_ratio: float
    alpha: Optional[float] = None
    beta: Optional[float] = None


class IStrategy(ABC):
    """
    策略统一接口

    所有策略实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        获取策略名称

        Returns:
            策略名称
        """

    @abstractmethod
    def get_strategy_type(self) -> StrategyType:
        """
        获取策略类型

        Returns:
            策略类型
        """

    @abstractmethod
    def get_strategy_description(self) -> str:
        """
        获取策略描述

        Returns:
            策略描述
        """

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        获取策略参数

        Returns:
            参数字典
        """

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        设置策略参数

        Args:
            parameters: 参数字典

        Returns:
            是否设置成功
        """

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证策略参数

        Args:
            parameters: 参数字典

        Returns:
            验证结果 {'valid': bool, 'errors': List[str], 'warnings': List[str]}
        """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化策略

        Args:
            config: 初始化配置

        Returns:
            是否初始化成功
        """

    @abstractmethod
    def on_market_data(self, data: MarketData) -> List[StrategySignal]:
        """
        处理市场数据

        Args:
            data: 市场数据

        Returns:
            策略信号列表
        """

    @abstractmethod
    def on_order_update(self, order: StrategyOrder) -> None:
        """
        处理订单更新

        Args:
            order: 订单信息
        """

    @abstractmethod
    def on_position_update(self, position: StrategyPosition) -> None:
        """
        处理持仓更新

        Args:
            position: 持仓信息
        """

    @abstractmethod
    def should_enter_position(self, symbol: str, data: pd.DataFrame) -> Optional[StrategySignal]:
        """
        判断是否应该开仓

        Args:
            symbol: 交易品种
            data: 历史数据

        Returns:
            开仓信号
        """

    @abstractmethod
    def should_exit_position(self, position: StrategyPosition, data: pd.DataFrame) -> Optional[StrategySignal]:
        """
        判断是否应该平仓

        Args:
            position: 当前持仓
            data: 历史数据

        Returns:
            平仓信号
        """

    @abstractmethod
    def calculate_position_size(self, capital: float, risk_per_trade: float, symbol: str) -> float:
        """
        计算仓位大小

        Args:
            capital: 总资本
            risk_per_trade: 每笔交易风险
            symbol: 交易品种

        Returns:
            仓位大小
        """

    @abstractmethod
    def get_risk_management_rules(self) -> Dict[str, Any]:
        """
        获取风险管理规则

        Returns:
            风险管理规则字典
        """

    @abstractmethod
    def get_strategy_status(self) -> StrategyStatus:
        """
        获取策略状态

        Returns:
            策略状态
        """

    @abstractmethod
    def get_current_positions(self) -> List[StrategyPosition]:
        """
        获取当前持仓

        Returns:
            持仓列表
        """

    @abstractmethod
    def get_pending_orders(self) -> List[StrategyOrder]:
        """
        获取待成交订单

        Returns:
            待成交订单列表
        """

    @abstractmethod
    def get_strategy_metrics(self) -> StrategyMetrics:
        """
        获取策略指标

        Returns:
            策略指标
        """

    @abstractmethod
    def start(self) -> bool:
        """
        启动策略

        Returns:
            是否启动成功
        """

    @abstractmethod
    def stop(self) -> bool:
        """
        停止策略

        Returns:
            是否停止成功
        """

    @abstractmethod
    def pause(self) -> bool:
        """
        暂停策略

        Returns:
            是否暂停成功
        """

    @abstractmethod
    def resume(self) -> bool:
        """
        恢复策略

        Returns:
            是否恢复成功
        """

    @abstractmethod
    def reset(self) -> bool:
        """
        重置策略状态

        Returns:
            是否重置成功
        """

    @abstractmethod
    def save_state(self, path: str) -> bool:
        """
        保存策略状态

        Args:
            path: 保存路径

        Returns:
            是否保存成功
        """

    @abstractmethod
    def load_state(self, path: str) -> bool:
        """
        加载策略状态

        Args:
            path: 加载路径

        Returns:
            是否加载成功
        """

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """
        获取支持的交易品种

        Returns:
            支持的交易品种列表
        """

    @abstractmethod
    def get_required_data_fields(self) -> List[str]:
        """
        获取所需的数据字段

        Returns:
            数据字段列表
        """

    @abstractmethod
    def validate_market_data(self, data: MarketData) -> bool:
        """
        验证市场数据

        Args:
            data: 市场数据

        Returns:
            是否有效
        """


class IStrategyManager(ABC):
    """
    策略管理器接口
    """

    @abstractmethod
    def register_strategy(self, strategy: IStrategy) -> bool:
        """
        注册策略

        Args:
            strategy: 策略实例

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister_strategy(self, strategy_name: str) -> bool:
        """
        注销策略

        Args:
            strategy_name: 策略名称

        Returns:
            是否注销成功
        """

    @abstractmethod
    def get_strategy(self, strategy_name: str) -> Optional[IStrategy]:
        """
        获取策略

        Args:
            strategy_name: 策略名称

        Returns:
            策略实例
        """

    @abstractmethod
    def list_strategies(self) -> List[str]:
        """
        列出所有策略

        Returns:
            策略名称列表
        """

    @abstractmethod
    def create_strategy(self, strategy_type: StrategyType, config: Dict[str, Any]) -> Optional[IStrategy]:
        """
        创建策略实例

        Args:
            strategy_type: 策略类型
            config: 策略配置

        Returns:
            策略实例
        """

    @abstractmethod
    def start_strategy(self, strategy_name: str) -> bool:
        """
        启动策略

        Args:
            strategy_name: 策略名称

        Returns:
            是否启动成功
        """

    @abstractmethod
    def stop_strategy(self, strategy_name: str) -> bool:
        """
        停止策略

        Args:
            strategy_name: 策略名称

        Returns:
            是否停止成功
        """

    @abstractmethod
    def get_strategy_status(self, strategy_name: str) -> StrategyStatus:
        """
        获取策略状态

        Args:
            strategy_name: 策略名称

        Returns:
            策略状态
        """

    @abstractmethod
    def get_all_strategy_status(self) -> Dict[str, StrategyStatus]:
        """
        获取所有策略状态

        Returns:
            策略状态字典
        """

    @abstractmethod
    def get_strategy_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """
        获取策略指标

        Args:
            strategy_name: 策略名称

        Returns:
            策略指标
        """

    @abstractmethod
    def compare_strategies(self, strategy_names: List[str], metrics: List[str]) -> Dict[str, Any]:
        """
        比较策略性能

        Args:
            strategy_names: 策略名称列表
            metrics: 比较指标列表

        Returns:
            比较结果字典
        """


class IBacktestEngine(ABC):
    """
    回测引擎接口
    """

    @abstractmethod
    def run_backtest(self, strategy: IStrategy, data: pd.DataFrame,
                     initial_capital: float = 100000) -> Dict[str, Any]:
        """
        运行回测

        Args:
            strategy: 策略实例
            data: 历史数据
            initial_capital: 初始资本

        Returns:
            回测结果字典
        """

    @abstractmethod
    def run_walk_forward_analysis(self, strategy: IStrategy, data: pd.DataFrame,
                                  train_window: int, test_window: int) -> Dict[str, Any]:
        """
        运行步进分析

        Args:
            strategy: 策略实例
            data: 历史数据
            train_window: 训练窗口
            test_window: 测试窗口

        Returns:
            步进分析结果
        """

    @abstractmethod
    def calculate_performance_metrics(self, returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> StrategyMetrics:
        """
        计算性能指标

        Args:
            returns: 策略收益率
            benchmark_returns: 基准收益率

        Returns:
            性能指标
        """

    @abstractmethod
    def generate_report(self, results: Dict[str, Any], format: str = "html") -> str:
        """
        生成报告

        Args:
            results: 回测结果
            format: 报告格式

        Returns:
            报告内容
        """

    @abstractmethod
    def plot_results(self, results: Dict[str, Any]) -> Any:
        """
        绘制结果图表

        Args:
            results: 回测结果

        Returns:
            图表对象
        """


class IStrategyOptimizer(ABC):
    """
    策略优化器接口
    """

    @abstractmethod
    def optimize_parameters(self, strategy: IStrategy, data: pd.DataFrame,
                            parameter_space: Dict[str, List[Any]],
                            optimization_metric: str = "sharpe_ratio",
                            max_evaluations: int = 100) -> Dict[str, Any]:
        """
        优化策略参数

        Args:
            strategy: 策略实例
            data: 历史数据
            parameter_space: 参数空间
            optimization_metric: 优化指标
            max_evaluations: 最大评估次数

        Returns:
            优化结果字典
        """

    @abstractmethod
    def optimize_portfolio(self, strategies: List[IStrategy], data: pd.DataFrame,
                           risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        优化策略组合

        Args:
            strategies: 策略列表
            data: 历史数据
            risk_free_rate: 无风险利率

        Returns:
            组合优化结果
        """

    @abstractmethod
    def risk_parity_optimization(self, strategies: List[IStrategy], data: pd.DataFrame) -> Dict[str, Any]:
        """
        风险平价优化

        Args:
            strategies: 策略列表
            data: 历史数据

        Returns:
            风险平价优化结果
        """

    @abstractmethod
    def monte_carlo_simulation(self, strategy: IStrategy, data: pd.DataFrame,
                               num_simulations: int = 1000) -> Dict[str, Any]:
        """
        蒙特卡洛模拟

        Args:
            strategy: 策略实例
            data: 历史数据
            num_simulations: 模拟次数

        Returns:
            模拟结果
        """


class IStrategyRiskManager(ABC):
    """
    策略风险管理器接口
    """

    @abstractmethod
    def calculate_var(self, strategy: IStrategy, confidence_level: float = 0.95) -> float:
        """
        计算VaR（风险价值）

        Args:
            strategy: 策略实例
            confidence_level: 置信水平

        Returns:
            VaR值
        """

    @abstractmethod
    def calculate_cvar(self, strategy: IStrategy, confidence_level: float = 0.95) -> float:
        """
        计算CVaR（条件风险价值）

        Args:
            strategy: 策略实例
            confidence_level: 置信水平

        Returns:
            CVaR值
        """

    @abstractmethod
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        计算最大回撤

        Args:
            returns: 收益率序列

        Returns:
            最大回撤
        """

    @abstractmethod
    def check_risk_limits(self, strategy: IStrategy, current_positions: List[StrategyPosition]) -> Dict[str, Any]:
        """
        检查风险限额

        Args:
            strategy: 策略实例
            current_positions: 当前持仓

        Returns:
            风险检查结果
        """

    @abstractmethod
    def apply_stop_loss(self, position: StrategyPosition, stop_loss_percentage: float) -> StrategyOrder:
        """
        应用止损

        Args:
            position: 持仓信息
            stop_loss_percentage: 止损百分比

        Returns:
            止损订单
        """

    @abstractmethod
    def apply_take_profit(self, position: StrategyPosition, take_profit_percentage: float) -> StrategyOrder:
        """
        应用止盈

        Args:
            position: 持仓信息
            take_profit_percentage: 止盈百分比

        Returns:
            止盈订单
        """
