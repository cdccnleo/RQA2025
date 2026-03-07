import logging
"""
智能订单路由组件

from src.engine.logging.unified_logger import get_unified_logger
实现智能订单路由算法，支持多市场订单路由，优化订单执行效率。
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from src.infrastructure.config.config_center import ConfigCenterManager
from src.infrastructure.logging.distributed_monitoring import DistributedMonitoringManager

logger = logging.getLogger(__name__)


class MarketInfo:

    """市场信息"""
    market_id: str
    market_name: str
    market_type: str
    trading_hours: Dict[str, str]
    tick_size: float
    lot_size: int
    max_order_size: int
    min_order_size: int
    commission_rate: float
    settlement_days: int
    is_active: bool = True

    def __init__(self, market_id: str, market_name: str, market_type: str,


                 trading_hours: Dict[str, str], tick_size: float, lot_size: int,
                 max_order_size: int, min_order_size: int, commission_rate: float,
                 settlement_days: int, is_active: bool = True):
        self.market_id = market_id
        self.market_name = market_name
        self.market_type = market_type
        self.trading_hours = trading_hours
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_order_size = max_order_size
        self.min_order_size = min_order_size
        self.commission_rate = commission_rate
        self.settlement_days = settlement_days
        self.is_active = is_active


@dataclass
class RoutingDecision:

    """路由决策"""
    target_market: str
    routing_reason: str
    expected_cost: float
    expected_slippage: float
    confidence_score: float
    alternative_markets: List[str]
    routing_time: datetime


class IntelligentOrderRouter:

    """智能订单路由组件"""

    def __init__(self, config: Dict[str, Any]):

        self.config = config
        self._init_distributed_components()
        self.markets: Dict[str, MarketInfo] = {}
        self._load_market_info()
        logger.info("智能订单路由组件初始化完成")

    def _init_distributed_components(self):
        """初始化分布式组件"""
        try:
            config_center_config = self.config.get('config_center', {})
            self.config_manager = ConfigCenterManager(config_center_config)

            monitoring_config = self.config.get('distributed_monitoring', {})
            self.monitoring_manager = DistributedMonitoringManager(monitoring_config)

            logger.info("分布式组件初始化成功")
        except Exception as e:
            logger.error(f"分布式组件初始化失败: {e}")
            raise

    def _load_market_info(self):
        """加载市场信息"""
        try:
            # 加载默认市场信息

            default_markets = {
                'SSE': MarketInfo(
                    market_id='SSE',
                    market_name='上海证券交易所',
                    market_type='equity',
                    trading_hours={'open': '09:30', 'close': '15:00'},
                    tick_size=0.01,
                    lot_size=100,
                    max_order_size=1000000,
                    min_order_size=100,
                    commission_rate=0.0003,
                    settlement_days=1
                ),
                'SZSE': MarketInfo(
                    market_id='SZSE',
                    market_name='深圳证券交易所',
                    market_type='equity',
                    trading_hours={'open': '09:30', 'close': '15:00'},
                    tick_size=0.01,
                    lot_size=100,
                    max_order_size=1000000,
                    min_order_size=100,
                    commission_rate=0.0003,
                    settlement_days=1
                )
            }

            self.markets.update(default_markets)
            logger.info(f"加载了 {len(self.markets)} 个市场信息")

        except Exception as e:
            logger.error(f"加载市场信息失败: {e}")

    def route_order(self, order: Dict[str, Any],


                    strategy: str = 'hybrid_optimization') -> RoutingDecision:
        """路由订单到合适的市场"""
        try:
            available_markets = self._get_available_markets(order)

            if not available_markets:
                raise ValueError("没有可用的市场")

            # 简单的路由策略：选择第一个可用市场
            target_market = available_markets[0]

            decision = RoutingDecision(
                target_market=target_market,
                routing_reason="默认路由",
                expected_cost=0.001,
                expected_slippage=0.001,
                confidence_score=0.8,
                alternative_markets=available_markets[1:],
                routing_time=datetime.now()
            )

            logger.info(f"订单路由完成: {target_market}")
            return decision

        except Exception as e:
            logger.error(f"订单路由失败: {e}")
            raise

    def _get_available_markets(self, order: Dict[str, Any]) -> List[str]:
        """获取可用的市场列表"""
        try:
            available_markets = []

            for market_id, market_info in self.markets.items():
                if not market_info.is_active:
                    continue

                order_size = order.get('quantity', 0)
                if order_size < market_info.min_order_size or order_size > market_info.max_order_size:
                    continue

                available_markets.append(market_id)

            return available_markets

        except Exception as e:
            logger.error(f"获取可用市场失败: {e}")
            return []


def create_intelligent_order_router(config: Dict[str, Any]) -> IntelligentOrderRouter:
    """创建智能订单路由组件"""
    return IntelligentOrderRouter(config)
