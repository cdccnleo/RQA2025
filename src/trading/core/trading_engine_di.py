"""
RQA2025 交易引擎 - 依赖注入版本

本模块展示如何通过依赖注入的方式使用基础设施服务，
避免硬编码的导入和直接依赖。

对比原版本 (trading_engine.py) 的硬编码依赖：
- 直接导入: from src.infrastructure.logging.core.unified_logger import get_unified_logger
- 直接导入: from src.infrastructure.integration import get_data_adapter
- 直接导入: from src.infrastructure import SystemMonitor

新版本通过构造函数注入基础设施服务依赖。
"""

from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ...infrastructure.interfaces.infrastructure_services import IInfrastructureServiceProvider
from .execution.trade_execution_engine import TradeExecutionEngine


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = 1      # 市价单
    LIMIT = 2       # 限价单
    STOP = 3        # 止损单


class OrderDirection(Enum):
    """订单方向枚举"""
    BUY = 1         # 买入
    SELL = -1       # 卖出


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = 1     # 待处理
    FILLED = 2      # 已成交
    CANCELLED = 3   # 已取消
    REJECTED = 4    # 已拒绝


class TradingEngine:
    """
    交易引擎 - 依赖注入版本

    通过构造函数注入基础设施服务，避免硬编码依赖。
    """

    def __init__(self,
                 infrastructure_provider: IInfrastructureServiceProvider,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化交易引擎

        Args:
            infrastructure_provider: 基础设施服务提供者
            config: 交易引擎配置
        """
        # 注入基础设施服务依赖
        self.infrastructure = infrastructure_provider
        self.logger = infrastructure_provider.logger
        self.monitor = infrastructure_provider.monitor
        self.cache = infrastructure_provider.cache_service
        self.config_manager = infrastructure_provider.config_manager

        # 初始化配置
        self.config = config or {}
        self._load_default_config()

        # 初始化执行引擎
        execution_config = self.config.get('execution', {})
        self.execution_engine = TradeExecutionEngine(execution_config)

        # 记录初始化完成
        self.logger.info("TradingEngine initialized with dependency injection",
                         extra={"config_keys": list(self.config.keys())})

        # 监控指标记录
        self.monitor.record_metric("trading_engine_initialized", 1)

    def _load_default_config(self):
        """加载默认配置"""
        defaults = {
            'max_orders_per_minute': 100,
            'default_order_timeout': 30,
            'risk_check_enabled': True,
            'market_data_cache_ttl': 300,
        }

        # 从配置管理器获取配置，如果失败则使用默认值
        for key, default_value in defaults.items():
            if key not in self.config:
                try:
                    config_value = self.config_manager.get(f"trading.{key}", default_value)
                    self.config[key] = config_value
                except Exception:
                    self.config[key] = default_value
                    self.logger.warning(
                        f"Failed to load config for {key}, using default: {default_value}")

    def place_order(self, symbol: str, order_type: OrderType, direction: OrderDirection,
                    quantity: int, price: Optional[float] = None) -> Dict[str, Any]:
        """
        下单

        Args:
            symbol: 交易标的
            order_type: 订单类型
            direction: 交易方向
            quantity: 数量
            price: 价格（限价单需要）

        Returns:
            订单结果字典
        """
        start_time = datetime.now()

        try:
            # 记录订单请求
            self.logger.info(f"Placing order: {symbol} {direction.name} {quantity}@{price}",
                             extra={"symbol": symbol, "quantity": quantity, "price": price})

            # 检查缓存中是否有市场数据
            market_key = f"market_data_{symbol}"
            market_data = self.cache.get(market_key)

            if market_data:
                self.logger.debug(f"Using cached market data for {symbol}")
            else:
                self.logger.debug(f"No cached market data for {symbol}")

            # 执行订单
            order_result = {
                "order_id": f"order_{int(datetime.now().timestamp())}",
                "status": "accepted",
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now()
            }

            # 记录成功指标
            execution_time = (datetime.now() - start_time).total_seconds()
            self.monitor.record_histogram("order_placement_time", execution_time)
            self.monitor.increment_counter("orders_placed_total")

            self.logger.info(f"Order placed successfully: {order_result['order_id']}")

            return order_result

        except Exception as e:
            # 记录错误
            execution_time = (datetime.now() - start_time).total_seconds()
            self.monitor.record_histogram("order_placement_error_time", execution_time)
            self.monitor.increment_counter("order_placement_errors_total")

            self.logger.error(f"Failed to place order: {str(e)}",
                              exc=e,
                              extra={"symbol": symbol, "quantity": quantity})

            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now()
            }

    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        获取投资组合状态

        Returns:
            投资组合状态字典
        """
        try:
            # 从缓存获取投资组合数据
            portfolio_key = "portfolio_status"
            cached_portfolio = self.cache.get(portfolio_key)

            if cached_portfolio:
                self.logger.debug("Using cached portfolio status")
                return cached_portfolio

            # 模拟投资组合数据
            portfolio = {
                "total_value": 1000000.0,
                "cash": 500000.0,
                "positions": {
                    "AAPL": {"quantity": 1000, "avg_price": 150.0, "current_price": 155.0},
                    "GOOGL": {"quantity": 500, "avg_price": 2000.0, "current_price": 2050.0}
                },
                "last_updated": datetime.now(),
                "status": "active"
            }

            # 缓存结果
            self.cache.set(portfolio_key, portfolio, ttl=60)  # 缓存1分钟

            # 记录监控指标
            self.monitor.record_metric("portfolio_value", portfolio["total_value"])
            self.monitor.record_metric("portfolio_positions", len(portfolio["positions"]))

            return portfolio

        except Exception as e:
            self.logger.error(f"Failed to get portfolio status: {str(e)}", exc=e)
            return {"status": "error", "error": str(e)}

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取市场数据

        Args:
            symbol: 交易标的

        Returns:
            市场数据字典或None
        """
        try:
            # 检查缓存
            cache_key = f"market_data_{symbol}"
            cached_data = self.cache.get(cache_key)

            if cached_data:
                self.logger.debug(f"Market data cache hit for {symbol}")
                return cached_data

            # 模拟市场数据获取
            market_data = {
                "symbol": symbol,
                "price": 150.0 + (hash(symbol) % 100),  # 模拟价格
                "volume": 1000000,
                "timestamp": datetime.now(),
                "bid": 149.5,
                "ask": 150.5
            }

            # 缓存数据
            cache_ttl = self.config.get('market_data_cache_ttl', 300)
            self.cache.set(cache_key, market_data, ttl=cache_ttl)

            # 记录监控指标
            self.monitor.increment_counter("market_data_requests_total")

            self.logger.debug(f"Retrieved market data for {symbol}")
            return market_data

        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {str(e)}", exc=e)
            return None

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取交易引擎健康状态

        Returns:
            健康状态字典
        """
        try:
            # 获取基础设施服务的健康状态
            infra_status = self.infrastructure.get_service_status()
            infra_health = self.infrastructure.get_service_health_report()

            # 获取本地组件状态
            local_status = {
                "execution_engine": "healthy" if self.execution_engine else "unhealthy",
                "config_loaded": bool(self.config),
                "cache_available": self.cache.is_healthy() if hasattr(self.cache, 'is_healthy') else True,
                "monitor_available": self.monitor.is_healthy() if hasattr(self.monitor, 'is_healthy') else True,
            }

            health_status = {
                "overall_status": "healthy" if all(s == "healthy" for s in local_status.values()) else "degraded",
                "infrastructure_status": infra_status.value,
                "local_components": local_status,
                "infrastructure_health": {k: v.status for k, v in infra_health.items()},
                "timestamp": datetime.now()
            }

            # 记录健康检查指标
            self.monitor.record_metric("trading_engine_health_check", 1)

            return health_status

        except Exception as e:
            self.logger.error(f"Failed to get health status: {str(e)}", exc=e)
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now()
            }


# =============================================================================
# 工厂函数 - 创建依赖注入的交易引擎实例
# =============================================================================

def create_trading_engine(config: Optional[Dict[str, Any]] = None) -> TradingEngine:
    """
    创建交易引擎实例（依赖注入版本）

    Args:
        config: 交易引擎配置

    Returns:
        配置好的交易引擎实例
    """
    from ...infrastructure.core.infrastructure_service_provider import get_infrastructure_provider

    # 获取基础设施服务提供者
    infrastructure_provider = get_infrastructure_provider()

    # 初始化基础设施服务（如果尚未初始化）
    infrastructure_provider.initialize_all_services()

    # 创建交易引擎实例
    trading_engine = TradingEngine(
        infrastructure_provider=infrastructure_provider,
        config=config
    )

    return trading_engine


# =============================================================================
# 向后兼容性函数
# =============================================================================

def get_default_trading_engine() -> TradingEngine:
    """
    获取默认交易引擎实例（向后兼容）

    Returns:
        默认配置的交易引擎实例
    """
    return create_trading_engine()
