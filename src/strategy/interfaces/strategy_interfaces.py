"""
RQA2025 策略层接口定义

本模块定义策略层提供的业务接口，
这些接口描述策略层提供的核心量化策略管理能力。

策略层职责：
1. 策略引擎接口 - 策略执行和信号生成
2. 策略管理接口 - 策略生命周期管理
3. 回测引擎接口 - 策略历史表现验证
4. 策略优化接口 - 策略参数调优和改进
5. 信号生成接口 - 交易信号计算和输出
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Type
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# 策略相关枚举
# =============================================================================

class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"      # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"        # 均值回归
    ARBITRAGE = "arbitrage"                  # 套利
    MOMENTUM = "momentum"                    # 动量
    VALUE = "value"                          # 价值投资


class StrategyStatus(Enum):
    """策略状态"""
    CREATED = "created"      # 已创建
    INITIALIZED = "initialized"  # 已初始化
    RUNNING = "running"      # 运行中
    PAUSED = "paused"        # 已暂停
    STOPPED = "stopped"      # 已停止
    ERROR = "error"          # 错误状态


# =============================================================================
# 策略相关数据结构
# =============================================================================

@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None


@dataclass
class StrategySignal:
    """策略信号"""
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    symbol: str
    price: float
    quantity: int
    confidence: float
    timestamp: datetime
    strategy_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StrategyOrder:
    """策略订单"""
    order_id: str
    symbol: str
    order_type: str  # 'MARKET', 'LIMIT'
    side: str  # 'BUY', 'SELL'
    quantity: int
    price: Optional[float]
    timestamp: datetime


@dataclass
class StrategyPosition:
    """策略持仓"""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime


@dataclass
class StrategyResult:
    """策略执行结果"""
    strategy_id: str
    signal: Optional[StrategySignal] = None
    success: bool = True
    message: str = ""
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StrategyConfig:
    """策略配置"""
    strategy_id: str
    strategy_name: str  # 策略名称（兼容工厂接口）
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    symbols: List[str]
    risk_limits: Dict[str, Any]  # 风险限制配置
    metadata: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        """向后兼容的name属性"""
        return self.strategy_name


# =============================================================================
# 策略相关枚举
# =============================================================================

class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"      # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"        # 均值回归
    ARBITRAGE = "arbitrage"                  # 套利
    MOMENTUM = "momentum"                    # 动量
    VALUE = "value"                          # 价值投资


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"              # 买入信号
    SELL = "sell"            # 卖出信号
    HOLD = "hold"            # 持有信号
    NEUTRAL = "neutral"      # 中性信号


class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"      # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"        # 均值回归
    ARBITRAGE = "arbitrage"                  # 套利
    MOMENTUM = "momentum"                    # 动量
    VALUE = "value"                          # 价值投资
    QUANTITATIVE = "quantitative"            # 量化
    HIGH_FREQUENCY = "high_frequency"        # 高频


# =============================================================================
# 策略数据结构
# =============================================================================

@dataclass
class StrategyConfig:
    """策略配置"""
    strategy_id: str
    strategy_name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    symbols: List[str]
    timeframe: str = "1d"
    risk_limits: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.risk_limits is None:
            self.risk_limits = {}


@dataclass
class Signal:
    """交易信号"""
    signal_id: str
    strategy_id: str
    symbol: str
    signal_type: SignalType
    strength: float  # 信号强度 (0-1)
    price: Optional[float] = None
    quantity: Optional[int] = None
    confidence: float = 0.5  # 置信度 (0-1)
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StrategyPerformance:
    """策略表现"""
    strategy_id: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    period_start: datetime
    period_end: datetime
    benchmark_return: Optional[float] = None


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_id: str
    performance: StrategyPerformance
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, float]]
    drawdown_curve: List[Dict[str, float]]
    monthly_returns: Dict[str, float]
    risk_metrics: Dict[str, Any]
    backtest_config: Dict[str, Any]


# =============================================================================
# 策略基础接口
# =============================================================================

class IStrategy(ABC):
    """策略接口 - 量化策略的核心抽象"""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称"""

    @abstractmethod
    def get_strategy_type(self) -> str:
        """获取策略类型"""

    @abstractmethod
    def get_strategy_description(self) -> str:
        """获取策略描述"""

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """设置策略参数"""

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证策略参数"""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化策略"""

    @abstractmethod
    def on_market_data(self, data: Any) -> List[Any]:
        """处理市场数据"""

    @abstractmethod
    def on_order_update(self, order: Any) -> None:
        """处理订单更新"""

    @abstractmethod
    def on_position_update(self, position: Any) -> None:
        """处理持仓更新"""

    @abstractmethod
    def should_enter_position(self, symbol: str, data: Any) -> Optional[Any]:
        """判断是否应该入场"""

    @abstractmethod
    def should_exit_position(self, position: Any, data: Any) -> Optional[Any]:
        """判断是否应该出场"""

    @abstractmethod
    def calculate_position_size(self, capital: float, risk_per_trade: float, symbol: str) -> float:
        """计算仓位大小"""

    @abstractmethod
    def get_risk_management_rules(self) -> Dict[str, Any]:
        """获取风险管理规则"""

    @abstractmethod
    def get_strategy_status(self) -> str:
        """获取策略状态"""

    @abstractmethod
    def get_current_positions(self) -> List[Any]:
        """获取当前持仓"""

    @abstractmethod
    def start(self) -> bool:
        """启动策略"""

    @abstractmethod
    def stop(self) -> bool:
        """停止策略"""

    @abstractmethod
    def pause(self) -> bool:
        """暂停策略"""

    @abstractmethod
    def resume(self) -> bool:
        """恢复策略"""


# =============================================================================
# 策略工厂接口
# =============================================================================

class IStrategyFactory(Protocol):
    """策略工厂接口 - 策略的统一创建和管理"""

    def register_strategy(self, strategy_type: StrategyType, strategy_class: Type[IStrategy]):
        """注册策略类"""

    def create_strategy(self, config: StrategyConfig) -> IStrategy:
        """创建策略实例"""

    def get_supported_types(self) -> List[StrategyType]:
        """获取支持的策略类型"""

    def get_strategy_info(self, strategy_type: StrategyType) -> Optional[Dict[str, Any]]:
        """获取策略信息"""


# =============================================================================
# 策略引擎接口
# =============================================================================

class IStrategyEngine(Protocol):
    """策略引擎接口 - 策略执行和信号生成"""

    @abstractmethod
    def initialize_strategy(self, config: StrategyConfig) -> bool:
        """初始化策略"""

    @abstractmethod
    def start_strategy(self, strategy_id: str) -> bool:
        """启动策略"""

    @abstractmethod
    def stop_strategy(self, strategy_id: str) -> bool:
        """停止策略"""

    @abstractmethod
    def pause_strategy(self, strategy_id: str) -> bool:
        """暂停策略"""

    @abstractmethod
    def resume_strategy(self, strategy_id: str) -> bool:
        """恢复策略"""

    @abstractmethod
    def process_market_data(self, strategy_id: str, market_data: Dict[str, Any]) -> List[Signal]:
        """处理市场数据，生成信号"""

    @abstractmethod
    def get_strategy_status(self, strategy_id: str) -> StrategyStatus:
        """获取策略状态"""

    @abstractmethod
    def update_strategy_config(self, strategy_id: str, updates: Dict[str, Any]) -> bool:
        """更新策略配置"""


# =============================================================================
# 策略管理接口
# =============================================================================

class IStrategyManager(Protocol):
    """策略管理器接口 - 策略生命周期管理"""

    @abstractmethod
    def create_strategy(self, config: StrategyConfig) -> str:
        """创建策略，返回策略ID"""

    @abstractmethod
    def load_strategy(self, strategy_id: str) -> Optional[StrategyConfig]:
        """加载策略配置"""

    @abstractmethod
    def save_strategy(self, config: StrategyConfig) -> bool:
        """保存策略配置"""

    @abstractmethod
    def delete_strategy(self, strategy_id: str) -> bool:
        """删除策略"""

    @abstractmethod
    def list_strategies(self, status_filter: Optional[StrategyStatus] = None) -> List[StrategyConfig]:
        """列出策略"""

    @abstractmethod
    def validate_strategy_config(self, config: StrategyConfig) -> Dict[str, Any]:
        """验证策略配置"""

    @abstractmethod
    def backup_strategy(self, strategy_id: str) -> str:
        """备份策略，返回备份ID"""

    @abstractmethod
    def restore_strategy(self, strategy_id: str, backup_id: str) -> bool:
        """恢复策略"""


# =============================================================================
# 回测引擎接口
# =============================================================================

class IBacktestEngine(Protocol):
    """回测引擎接口 - 策略历史表现验证"""

    @abstractmethod
    def run_backtest(self, strategy_config: StrategyConfig,
                     historical_data: Dict[str, List[Dict[str, Any]]],
                     start_date: str, end_date: str) -> BacktestResult:
        """运行回测"""

    @abstractmethod
    def run_walk_forward_analysis(self, strategy_config: StrategyConfig,
                                  historical_data: Dict[str, List[Dict[str, Any]]],
                                  window_size: int = 252) -> List[BacktestResult]:
        """运行滚动窗口分析"""

    @abstractmethod
    def calculate_performance_metrics(self, trades: List[Dict[str, Any]],
                                      equity_curve: List[Dict[str, float]]) -> StrategyPerformance:
        """计算性能指标"""

    @abstractmethod
    def generate_backtest_report(self, result: BacktestResult,
                                 output_format: str = "html") -> str:
        """生成回测报告"""

    @abstractmethod
    def validate_backtest_result(self, result: BacktestResult) -> Dict[str, Any]:
        """验证回测结果"""


# =============================================================================
# 策略优化接口
# =============================================================================

@dataclass
class OptimizationConfig:
    """优化配置"""
    strategy_id: str
    parameters_to_optimize: List[str]
    parameter_ranges: Dict[str, List[Any]]
    optimization_method: str = "grid_search"
    objective_function: str = "sharpe_ratio"
    max_iterations: int = 100
    cross_validation_folds: int = 5


@dataclass
class OptimizationResult:
    """优化结果"""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    parameter_importance: Dict[str, float]
    convergence_info: Dict[str, Any]


class IStrategyOptimizer(Protocol):
    """策略优化器接口 - 策略参数调优和改进"""

    @abstractmethod
    def optimize_strategy(self, config: OptimizationConfig,
                          historical_data: Dict[str, List[Dict[str, Any]]]) -> OptimizationResult:
        """优化策略参数"""

    @abstractmethod
    def perform_sensitivity_analysis(self, strategy_config: StrategyConfig,
                                     parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """执行敏感性分析"""

    @abstractmethod
    def find_optimal_allocation(self, strategy_configs: List[StrategyConfig],
                                historical_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """寻找最优策略组合配置"""

    @abstractmethod
    def validate_optimization_result(self, result: OptimizationResult) -> Dict[str, Any]:
        """验证优化结果"""


# =============================================================================
# 信号生成接口
# =============================================================================

class ISignalGenerator(Protocol):
    """信号生成器接口 - 交易信号计算和输出"""

    @abstractmethod
    def generate_signals(self, strategy_id: str, market_data: Dict[str, Any]) -> List[Signal]:
        """生成交易信号"""

    @abstractmethod
    def filter_signals(self, signals: List[Signal], filters: Dict[str, Any]) -> List[Signal]:
        """过滤信号"""

    @abstractmethod
    def rank_signals(self, signals: List[Signal], ranking_method: str = "strength") -> List[Signal]:
        """对信号进行排序"""

    @abstractmethod
    def aggregate_signals(self, signals: List[Signal], aggregation_method: str = "weighted") -> List[Signal]:
        """聚合信号"""

    @abstractmethod
    def validate_signals(self, signals: List[Signal]) -> Dict[str, Any]:
        """验证信号质量"""


# =============================================================================
# 策略持久化接口
# =============================================================================

class IStrategyPersistence(Protocol):
    """策略持久化接口 - 策略数据的存储和管理"""

    @abstractmethod
    def save_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> bool:
        """保存策略"""

    @abstractmethod
    def load_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """加载策略"""

    @abstractmethod
    def delete_strategy(self, strategy_id: str) -> bool:
        """删除策略"""

    @abstractmethod
    def list_strategies(self) -> List[str]:
        """列出所有策略"""

    @abstractmethod
    def save_strategy_config(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """保存策略配置"""

    @abstractmethod
    def load_strategy_config(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """加载策略配置"""


# =============================================================================
# 策略服务提供者接口
# =============================================================================

class IStrategyServiceProvider(Protocol):
    """策略服务提供者接口 - 策略层的统一服务访问点"""

    @property
    def strategy_engine(self) -> IStrategyEngine:
        """策略引擎"""

    @property
    def strategy_manager(self) -> IStrategyManager:
        """策略管理器"""

    @property
    def backtest_engine(self) -> IBacktestEngine:
        """回测引擎"""

    @property
    def strategy_optimizer(self) -> IStrategyOptimizer:
        """策略优化器"""

    @property
    def signal_generator(self) -> ISignalGenerator:
        """信号生成器"""

    @abstractmethod
    def get_service_status(self) -> str:
        """获取策略服务整体状态"""

# 别名：IStrategyService = IStrategyServiceProvider（向后兼容）
IStrategyService = IStrategyServiceProvider

