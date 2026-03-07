"""
RQA2025 交易层接口定义

本模块定义交易层提供的业务接口，
这些接口描述交易层提供的核心交易服务能力。

交易层职责：
1. 执行引擎接口 - 订单执行和交易管理
2. 订单管理接口 - 订单生命周期管理
3. 投资组合管理接口 - 资产配置和风险管理
4. 交易监控接口 - 交易状态跟踪和异常检测
5. 市场连接接口 - 与交易所和券商的连接管理
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# 交易相关枚举
# =============================================================================

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"    # 限价单
    STOP = "stop"      # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"      # 待处理
    ACCEPTED = "accepted"    # 已接受
    PARTIAL = "partial"      # 部分成交
    FILLED = "filled"        # 完全成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"    # 已拒绝
    EXPIRED = "expired"      # 已过期


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"    # 买入
    SELL = "sell"  # 卖出


class ExecutionMode(Enum):
    """执行模式"""
    PAPER = "paper"        # 模拟交易
    LIVE = "live"          # 实盘交易
    BACKTEST = "backtest"  # 回测模式


# =============================================================================
# 交易数据结构
# =============================================================================

@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    client_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OrderResponse:
    """订单响应"""
    order_id: str
    status: OrderStatus
    executed_quantity: int = 0
    executed_price: Optional[float] = None
    remaining_quantity: int = 0
    message: Optional[str] = None
    timestamp: datetime = None
    transaction_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TradeExecution:
    """交易执行记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    executed_at: datetime
    commission: float = 0.0
    exchange: Optional[str] = None
    broker: Optional[str] = None


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    average_price: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class PortfolioSummary:
    """投资组合摘要"""
    total_value: float
    cash: float
    positions: Dict[str, Position]
    total_positions_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


# =============================================================================
# 执行引擎接口
# =============================================================================

class IExecutionEngine(Protocol):
    """执行引擎接口 - 订单执行和交易管理"""

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResponse:
        """提交订单"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""

    @abstractmethod
    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> OrderResponse:
        """修改订单"""

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        """获取订单状态"""

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """获取未完成订单"""

    @abstractmethod
    def get_trade_history(self, symbol: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[TradeExecution]:
        """获取交易历史"""


# =============================================================================
# 订单管理接口
# =============================================================================

class IOrderManager(Protocol):
    """订单管理器接口 - 订单生命周期管理"""

    @abstractmethod
    def create_order(self, order_request: OrderRequest) -> OrderResponse:
        """创建订单"""

    @abstractmethod
    def validate_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """验证订单有效性"""

    @abstractmethod
    def queue_order(self, order_response: OrderResponse) -> bool:
        """将订单加入执行队列"""

    @abstractmethod
    def get_order_queue(self) -> List[OrderResponse]:
        """获取订单队列"""

    @abstractmethod
    def prioritize_order(self, order_id: str, priority: int) -> bool:
        """设置订单优先级"""


# =============================================================================
# 投资组合管理接口
# =============================================================================

class IPortfolioManager(Protocol):
    """投资组合管理器接口 - 资产配置和风险管理"""

    @abstractmethod
    def get_portfolio_summary(self) -> PortfolioSummary:
        """获取投资组合摘要"""

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取特定持仓"""

    @abstractmethod
    def rebalance_portfolio(self, target_allocations: Dict[str, float]) -> Dict[str, Any]:
        """重新平衡投资组合"""

    @abstractmethod
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """计算投资组合指标"""

    @abstractmethod
    def get_portfolio_pnl(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取投资组合盈亏"""


# =============================================================================
# 交易监控接口
# =============================================================================

@dataclass
class TradingMetrics:
    """交易指标"""
    total_orders: int
    executed_orders: int
    cancelled_orders: int
    rejected_orders: int
    total_volume: float
    total_pnl: float
    win_rate: float
    average_slippage: float
    period_start: datetime
    period_end: datetime


class ITradingMonitor(Protocol):
    """交易监控器接口 - 交易状态跟踪和异常检测"""

    @abstractmethod
    def get_trading_metrics(self, period_days: int = 1) -> TradingMetrics:
        """获取交易指标"""

    @abstractmethod
    def detect_trading_anomalies(self) -> List[Dict[str, Any]]:
        """检测交易异常"""

    @abstractmethod
    def get_execution_quality_metrics(self) -> Dict[str, Any]:
        """获取执行质量指标"""

    @abstractmethod
    def monitor_market_impact(self) -> Dict[str, Any]:
        """监控市场冲击"""


# =============================================================================
# 市场连接接口
# =============================================================================

class IMarketConnector(Protocol):
    """市场连接器接口 - 与交易所和券商的连接管理"""

    @abstractmethod
    def connect(self) -> bool:
        """建立连接"""

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""

    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""

    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""

    @abstractmethod
    def test_connectivity(self) -> Dict[str, Any]:
        """测试连接质量"""


class IBrokerAdapter(Protocol):
    """券商适配器接口"""

    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """认证"""

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""

    @abstractmethod
    def get_available_balance(self) -> float:
        """获取可用余额"""

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResponse:
        """下单"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""


# =============================================================================
# 交易服务提供者接口
# =============================================================================

class ITradingServiceProvider(Protocol):
    """交易服务提供者接口 - 交易层的统一服务访问点"""

    @property
    def execution_engine(self) -> IExecutionEngine:
        """执行引擎"""

    @property
    def order_manager(self) -> IOrderManager:
        """订单管理器"""

    @property
    def portfolio_manager(self) -> IPortfolioManager:
        """投资组合管理器"""

    @property
    def trading_monitor(self) -> ITradingMonitor:
        """交易监控器"""

    @property
    def market_connector(self) -> IMarketConnector:
        """市场连接器"""

    @property
    def broker_adapter(self) -> IBrokerAdapter:
        """券商适配器"""

    @abstractmethod
    def get_service_status(self) -> str:
        """获取交易服务整体状态"""

    @abstractmethod
    def get_trading_summary(self) -> Dict[str, Any]:
        """获取交易统计摘要"""

    @abstractmethod
    def set_execution_mode(self, mode: ExecutionMode) -> bool:
        """设置执行模式"""
