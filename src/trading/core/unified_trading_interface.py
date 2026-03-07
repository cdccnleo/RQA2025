#!/usr/bin/env python3
"""
统一交易订单处理接口

定义交易层订单处理的统一接口，确保所有交易组件实现统一的API。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"    # 限价单
    STOP = "stop"      # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单
    TRAILING_STOP = "trailing_stop"  # 追踪止损
    ICEBERG = "iceberg"  # 冰山订单
    TWAP = "twap"      # 时间加权平均价格
    VWAP = "vwap"      # 成交量加权平均价格
    ADAPTIVE = "adaptive"  # 自适应订单


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"     # 待处理
    SUBMITTED = "submitted"  # 已提交
    PARTIAL_FILLED = "partial_filled"  # 部分成交
    FILLED = "filled"       # 全部成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"   # 已拒绝
    EXPIRED = "expired"     # 已过期
    SUSPENDED = "suspended"  # 已暂停


class ExecutionVenue(Enum):
    """执行场所"""
    STOCK_EXCHANGE = "stock_exchange"  # 证券交易所
    FUTURES_EXCHANGE = "futures_exchange"  # 期货交易所
    OTC = "otc"  # 场外交易
    DARK_POOL = "dark_pool"  # 暗池
    ELECTRONIC_PLATFORM = "electronic_platform"  # 电子交易平台


class TimeInForce(Enum):
    """有效期类型"""
    DAY = "day"              # 当日有效
    GTC = "gtc"              # 撤销前有效
    GTD = "gtd"              # 指定日期有效
    IOC = "ioc"              # 立即成交或取消
    FOK = "fok"              # 全部成交或取消
    GTX = "gtx"              # 扩展交易时段


@dataclass
class Order:
    """
    订单数据类

    表示交易订单的所有信息。
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    venue: ExecutionVenue = ExecutionVenue.STOCK_EXCHANGE
    account_id: Optional[str] = None
    strategy_id: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    commission: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Trade:
    """
    成交数据类

    表示订单执行的成交记录。
    """
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    venue: ExecutionVenue
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Position:
    """
    持仓数据类

    表示当前的持仓状态。
    """
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    market_value: float = 0.0
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class Account:
    """
    账户数据类

    表示交易账户的信息。
    """
    account_id: str
    balance: float
    available_balance: float
    margin_used: float = 0.0
    margin_available: float = 0.0
    total_value: float = 0.0
    currency: str = "CNY"
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class ExecutionReport:
    """
    执行报告数据类

    表示订单执行的详细报告。
    """
    order: Order
    trades: List[Trade]
    execution_time: float
    total_commission: float
    slippage_cost: float
    market_impact: float
    benchmark_comparison: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class IOrderManager(ABC):
    """
    订单管理器统一接口

    所有订单管理器实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def submit_order(self, order: Order) -> bool:
        """
        提交订单

        Args:
            order: 订单对象

        Returns:
            是否提交成功
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否取消成功
        """

    @abstractmethod
    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> bool:
        """
        修改订单

        Args:
            order_id: 订单ID
            modifications: 修改内容字典

        Returns:
            是否修改成功
        """

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        获取订单信息

        Args:
            order_id: 订单ID

        Returns:
            订单对象
        """

    @abstractmethod
    def get_orders(self, status: Optional[OrderStatus] = None,
                   symbol: Optional[str] = None) -> List[Order]:
        """
        获取订单列表

        Args:
            status: 订单状态过滤
            symbol: 交易品种过滤

        Returns:
            订单列表
        """

    @abstractmethod
    def get_pending_orders(self) -> List[Order]:
        """
        获取待成交订单

        Returns:
            待成交订单列表
        """

    @abstractmethod
    def get_order_history(self, start_date: datetime, end_date: datetime,
                          symbol: Optional[str] = None) -> List[Order]:
        """
        获取订单历史

        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbol: 交易品种过滤

        Returns:
            订单历史列表
        """


class IExecutionEngine(ABC):
    """
    执行引擎统一接口
    """

    @abstractmethod
    def execute_order(self, order: Order) -> ExecutionReport:
        """
        执行订单

        Args:
            order: 订单对象

        Returns:
            执行报告
        """

    @abstractmethod
    def execute_batch_orders(self, orders: List[Order]) -> List[ExecutionReport]:
        """
        批量执行订单

        Args:
            orders: 订单列表

        Returns:
            执行报告列表
        """

    @abstractmethod
    def get_best_execution_price(self, symbol: str, side: OrderSide,
                                 quantity: float) -> Optional[float]:
        """
        获取最佳执行价格

        Args:
            symbol: 交易品种
            side: 买卖方向
            quantity: 数量

        Returns:
            最佳执行价格
        """

    @abstractmethod
    def estimate_execution_cost(self, order: Order) -> Dict[str, float]:
        """
        估算执行成本

        Args:
            order: 订单对象

        Returns:
            成本估算字典
        """

    @abstractmethod
    def optimize_execution(self, order: Order, constraints: Dict[str, Any]) -> Order:
        """
        优化订单执行

        Args:
            order: 原始订单
            constraints: 优化约束

        Returns:
            优化后的订单
        """

    @abstractmethod
    def get_execution_venues(self, symbol: str) -> List[ExecutionVenue]:
        """
        获取可用的执行场所

        Args:
            symbol: 交易品种

        Returns:
            执行场所列表
        """


class ITradingEngine(ABC):
    """
    交易引擎统一接口
    """

    @abstractmethod
    def start_trading(self) -> bool:
        """
        启动交易引擎

        Returns:
            是否启动成功
        """

    @abstractmethod
    def stop_trading(self) -> bool:
        """
        停止交易引擎

        Returns:
            是否停止成功
        """

    @abstractmethod
    def is_trading_active(self) -> bool:
        """
        检查交易是否激活

        Returns:
            是否激活
        """

    @abstractmethod
    def place_order(self, symbol: str, side: OrderSide, quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None) -> Optional[str]:
        """
        下单

        Args:
            symbol: 交易品种
            side: 买卖方向
            quantity: 数量
            order_type: 订单类型
            price: 价格（限价单时需要）

        Returns:
            订单ID
        """

    @abstractmethod
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        取消所有订单

        Args:
            symbol: 交易品种过滤

        Returns:
            取消的订单数量
        """

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """
        获取所有持仓

        Returns:
            持仓字典 {symbol: position}
        """

    @abstractmethod
    def get_account_info(self) -> Account:
        """
        获取账户信息

        Returns:
            账户信息
        """

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """
        获取投资组合价值

        Returns:
            投资组合总价值
        """

    @abstractmethod
    def calculate_portfolio_pnl(self) -> Dict[str, float]:
        """
        计算投资组合盈亏

        Returns:
            盈亏计算结果
        """


class IRiskManager(ABC):
    """
    风险管理器统一接口
    """

    @abstractmethod
    def check_order_risk(self, order: Order, account: Account,
                         positions: Dict[str, Position]) -> Dict[str, Any]:
        """
        检查订单风险

        Args:
            order: 订单对象
            account: 账户信息
            positions: 当前持仓

        Returns:
            风险检查结果
        """

    @abstractmethod
    def check_portfolio_risk(self, positions: Dict[str, Position],
                             account: Account) -> Dict[str, Any]:
        """
        检查投资组合风险

        Args:
            positions: 持仓信息
            account: 账户信息

        Returns:
            风险检查结果
        """

    @abstractmethod
    def calculate_position_limits(self, symbol: str, account: Account) -> Dict[str, float]:
        """
        计算仓位限制

        Args:
            symbol: 交易品种
            account: 账户信息

        Returns:
            仓位限制字典
        """

    @abstractmethod
    def apply_risk_limits(self, order: Order) -> Order:
        """
        应用风险限制

        Args:
            order: 原始订单

        Returns:
            应用风险限制后的订单
        """

    @abstractmethod
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        获取风险指标

        Returns:
            风险指标字典
        """

    @abstractmethod
    def set_risk_limits(self, limits: Dict[str, Any]) -> bool:
        """
        设置风险限额

        Args:
            limits: 风险限额字典

        Returns:
            是否设置成功
        """

    @abstractmethod
    def get_risk_limits(self) -> Dict[str, Any]:
        """
        获取风险限额

        Returns:
            风险限额字典
        """


class IPortfolioManager(ABC):
    """
    投资组合管理器统一接口
    """

    @abstractmethod
    def rebalance_portfolio(self, target_weights: Dict[str, float],
                            current_positions: Dict[str, Position]) -> List[Order]:
        """
        重新平衡投资组合

        Args:
            target_weights: 目标权重 {symbol: weight}
            current_positions: 当前持仓

        Returns:
            需要执行的订单列表
        """

    @abstractmethod
    def optimize_portfolio(self, assets: List[str], historical_data: pd.DataFrame,
                           constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化投资组合

        Args:
            assets: 资产列表
            historical_data: 历史数据
            constraints: 优化约束

        Returns:
            优化结果字典
        """

    @abstractmethod
    def calculate_portfolio_metrics(self, positions: Dict[str, Position],
                                    historical_returns: pd.DataFrame) -> Dict[str, Any]:
        """
        计算投资组合指标

        Args:
            positions: 持仓信息
            historical_returns: 历史收益率

        Returns:
            投资组合指标字典
        """

    @abstractmethod
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """
        获取投资组合分配

        Returns:
            资产分配字典 {symbol: weight}
        """

    @abstractmethod
    def simulate_portfolio_performance(self, positions: Dict[str, Position],
                                       scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        模拟投资组合表现

        Args:
            positions: 持仓信息
            scenarios: 情景分析列表

        Returns:
            模拟结果字典
        """


class IMarketDataProvider(ABC):
    """
    市场数据提供者统一接口
    """

    @abstractmethod
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        获取实时价格

        Args:
            symbol: 交易品种

        Returns:
            实时价格
        """

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: datetime,
                            end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """
        获取历史数据

        Args:
            symbol: 交易品种
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔

        Returns:
            历史数据DataFrame
        """

    @abstractmethod
    def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        获取订单簿

        Args:
            symbol: 交易品种
            depth: 深度

        Returns:
            订单簿字典
        """

    @abstractmethod
    def subscribe_market_data(self, symbols: List[str], callback: callable) -> bool:
        """
        订阅市场数据

        Args:
            symbols: 交易品种列表
            callback: 回调函数

        Returns:
            是否订阅成功
        """

    @abstractmethod
    def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """
        取消订阅市场数据

        Args:
            symbols: 交易品种列表

        Returns:
            是否取消成功
        """

    @abstractmethod
    def get_market_status(self, symbol: str) -> Dict[str, Any]:
        """
        获取市场状态

        Args:
            symbol: 交易品种

        Returns:
            市场状态字典
        """


class IBrokerAdapter(ABC):
    """
    经纪商适配器统一接口
    """

    @abstractmethod
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """
        连接经纪商

        Args:
            credentials: 连接凭据

        Returns:
            是否连接成功
        """

    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开连接

        Returns:
            是否断开成功
        """

    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查连接状态

        Returns:
            是否已连接
        """

    @abstractmethod
    def submit_order(self, order: Order) -> bool:
        """
        提交订单到经纪商

        Args:
            order: 订单对象

        Returns:
            是否提交成功
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        取消经纪商订单

        Args:
            order_id: 订单ID

        Returns:
            是否取消成功
        """

    @abstractmethod
    def get_account_balance(self) -> float:
        """
        获取账户余额

        Returns:
            账户余额
        """

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """
        获取经纪商持仓

        Returns:
            持仓字典
        """

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态
        """

    @abstractmethod
    def get_broker_info(self) -> Dict[str, Any]:
        """
        获取经纪商信息

        Returns:
            经纪商信息字典
        """
