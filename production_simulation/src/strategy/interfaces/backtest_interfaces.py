#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回测服务层接口定义
Backtest Service Layer Interfaces

定义统一的回测服务接口，支持多种回测模式和性能分析。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BacktestMode(Enum):

    """回测模式枚举"""
    SINGLE = "single"
    MULTI_STRATEGY = "multi_strategy"
    PARAMETER_SWEEP = "parameter_sweep"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    OPTIMIZATION = "optimization"


class BacktestStatus(Enum):

    """回测状态枚举"""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestConfig:

    """回测配置"""
    backtest_id: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission: float = 0.0003  # 交易佣金
    slippage: float = 0.0001    # 滑点
    benchmark_symbol: Optional[str] = None
    data_frequency: str = "1d"  # 数据频率
    mode: BacktestMode = BacktestMode.SINGLE
    parameters: Dict[str, Any] = None
    risk_limits: Dict[str, float] = None
    created_at: datetime = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()
        if self.parameters is None:
            self.parameters = {}
        if self.risk_limits is None:
            self.risk_limits = {}


@dataclass
class BacktestResult:

    """回测结果"""
    backtest_id: str
    strategy_id: str
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    status: BacktestStatus
    execution_time: float
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):

        if self.metadata is None:
            self.metadata = {}


@dataclass
class BacktestMetrics:

    """回测性能指标"""
    backtest_id: str
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    alpha: float
    beta: float
    information_ratio: float
    var_95: float
    expected_shortfall: float
    recovery_time: int  # 从最大回撤恢复所需的天数
    consecutive_wins: int
    consecutive_losses: int
    timestamp: datetime = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class BacktestTrade:

    """回测交易记录"""
    trade_id: str
    symbol: str
    side: str  # 'buy', 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    pnl: float
    pnl_pct: float
    strategy_id: str
    backtest_id: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):

        if self.metadata is None:
            self.metadata = {}


class IBacktestService(ABC):

    """
    回测服务接口
    Backtest Service Interface

    定义回测服务的核心功能接口。
    """

    @abstractmethod
    def create_backtest(self, config: BacktestConfig) -> str:
        """
        创建回测任务

        Args:
            config: 回测配置

        Returns:
            str: 回测任务ID
        """

    @abstractmethod
    def run_backtest(self, backtest_id: str) -> BacktestResult:
        """
        运行回测

        Args:
            backtest_id: 回测ID

        Returns:
            BacktestResult: 回测结果
        """

    @abstractmethod
    def get_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        获取回测结果

        Args:
            backtest_id: 回测ID

        Returns:
            Optional[BacktestResult]: 回测结果
        """

    @abstractmethod
    def cancel_backtest(self, backtest_id: str) -> bool:
        """
        取消回测

        Args:
            backtest_id: 回测ID

        Returns:
            bool: 取消是否成功
        """

    @abstractmethod
    def get_backtest_status(self, backtest_id: str) -> BacktestStatus:
        """
        获取回测状态

        Args:
            backtest_id: 回测ID

        Returns:
            BacktestStatus: 回测状态
        """

    @abstractmethod
    def list_backtests(self, strategy_id: Optional[str] = None,


                       status: Optional[BacktestStatus] = None) -> List[BacktestConfig]:
        """
        列出回测任务

        Args:
            strategy_id: 策略ID过滤器
            status: 状态过滤器

        Returns:
            List[BacktestConfig]: 回测配置列表
        """


class IBacktestEngine(ABC):

    """
    回测引擎接口
    Backtest Engine Interface

    定义回测引擎的核心功能。
    """

    @abstractmethod
    def initialize(self, config: BacktestConfig) -> bool:
        """
        初始化回测引擎

        Args:
            config: 回测配置

        Returns:
            bool: 初始化是否成功
        """

    @abstractmethod
    def load_data(self, symbols: List[str], start_date: datetime,


                  end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        加载历史数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict[str, pd.DataFrame]: 历史数据字典
        """

    @abstractmethod
    def execute_trades(self, signals: List[Dict[str, Any]],


                       market_data: pd.DataFrame) -> List[BacktestTrade]:
        """
        执行交易

        Args:
            signals: 交易信号列表
            market_data: 市场数据

        Returns:
            List[BacktestTrade]: 交易记录列表
        """

    @abstractmethod
    def calculate_metrics(self, returns: pd.Series, trades: List[BacktestTrade]) -> BacktestMetrics:
        """
        计算回测指标

        Args:
            returns: 收益率序列
            trades: 交易记录列表

        Returns:
            BacktestMetrics: 回测指标
        """

    @abstractmethod
    def generate_report(self, result: BacktestResult) -> Dict[str, Any]:
        """
        生成回测报告

        Args:
            result: 回测结果

        Returns:
            Dict[str, Any]: 回测报告
        """


class IBacktestPersistence(ABC):

    """
    回测持久化接口
    Backtest Persistence Interface

    处理回测结果和配置的持久化存储。
    """

    @abstractmethod
    def save_backtest_result(self, result: BacktestResult) -> bool:
        """
        保存回测结果

        Args:
            result: 回测结果

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def load_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        加载回测结果

        Args:
            backtest_id: 回测ID

        Returns:
            Optional[BacktestResult]: 回测结果
        """

    @abstractmethod
    def save_backtest_config(self, config: BacktestConfig) -> bool:
        """
        保存回测配置

        Args:
            config: 回测配置

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def load_backtest_config(self, backtest_id: str) -> Optional[BacktestConfig]:
        """
        加载回测配置

        Args:
            backtest_id: 回测ID

        Returns:
            Optional[BacktestConfig]: 回测配置
        """

    @abstractmethod
    def delete_backtest_data(self, backtest_id: str) -> bool:
        """
        删除回测数据

        Args:
            backtest_id: 回测ID

        Returns:
            bool: 删除是否成功
        """


class IBacktestVisualization(ABC):

    """
    回测可视化接口
    Backtest Visualization Interface

    提供回测结果的可视化功能。
    """

    @abstractmethod
    def plot_returns(self, result: BacktestResult, benchmark: Optional[pd.Series] = None) -> str:
        """
        绘制收益率曲线

        Args:
            result: 回测结果
            benchmark: 基准收益率序列

        Returns:
            str: 图表文件路径
        """

    @abstractmethod
    def plot_drawdown(self, result: BacktestResult) -> str:
        """
        绘制回撤曲线

        Args:
            result: 回测结果

        Returns:
            str: 图表文件路径
        """

    @abstractmethod
    def plot_monthly_returns(self, result: BacktestResult) -> str:
        """
        绘制月度收益率热力图

        Args:
            result: 回测结果

        Returns:
            str: 图表文件路径
        """

    @abstractmethod
    def generate_performance_report(self, result: BacktestResult) -> str:
        """
        生成性能报告

        Args:
            result: 回测结果

        Returns:
            str: 报告文件路径
        """


# 导出所有接口
__all__ = [
    'BacktestMode',
    'BacktestStatus',
    'BacktestConfig',
    'BacktestResult',
    'BacktestMetrics',
    'BacktestTrade',
    'IBacktestService',
    'IBacktestEngine',
    'IBacktestPersistence',
    'IBacktestVisualization'
]
