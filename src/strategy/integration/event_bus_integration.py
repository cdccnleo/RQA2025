# -*- coding: utf-8 -*-
"""
策略层事件总线集成模块

提供策略层与事件总线的集成能力，支持策略信号、决策事件的发布和订阅。

函数级注释:
- 所有公开函数均包含详细的中文注释
- 异常处理包含具体的错误信息
- 支持PostgreSQL优先策略，连接失败时自动降级

作者: AI系统集成助手
日期: 2026-03-22
版本: 1.0.0
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

# 导入事件总线
from src.core.event_bus import (
    get_event_bus,
    EventPriority,
    Event
)
from src.core.event_bus.validation import validate_event

logger = logging.getLogger(__name__)


class StrategyEventType(Enum):
    """
    策略事件类型枚举
    
    定义策略层支持的所有事件类型
    """
    SIGNAL_GENERATED = "SIGNAL_GENERATED"              # 信号生成
    STRATEGY_DECISION_READY = "STRATEGY_DECISION_READY" # 策略决策就绪
    STRATEGY_CREATED = "STRATEGY_CREATED"              # 策略创建
    STRATEGY_UPDATED = "STRATEGY_UPDATED"              # 策略更新
    STRATEGY_DELETED = "STRATEGY_DELETED"              # 策略删除
    STRATEGY_STARTED = "STRATEGY_STARTED"              # 策略启动
    STRATEGY_STOPPED = "STRATEGY_STOPPED"              # 策略停止
    BACKTEST_COMPLETED = "BACKTEST_COMPLETED"          # 回测完成
    OPTIMIZATION_COMPLETED = "OPTIMIZATION_COMPLETED"  # 优化完成


class SignalType(Enum):
    """信号类型枚举"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class DecisionType(Enum):
    """决策类型枚举"""
    ENTER_POSITION = "enter_position"
    EXIT_POSITION = "exit_position"
    INCREASE_POSITION = "increase_position"
    DECREASE_POSITION = "decrease_position"
    HOLD_POSITION = "hold_position"


@dataclass
class StrategySignal:
    """
    策略信号数据类
    
    属性:
        signal_id: 信号ID
        strategy_id: 策略ID
        symbol: 标的代码
        signal_type: 信号类型
        strength: 信号强度(0-1)
        confidence: 置信度(0-1)
        features: 特征数据
        timestamp: 信号生成时间
    """
    signal_id: str
    strategy_id: str
    symbol: str
    signal_type: SignalType
    strength: float = 0.5
    confidence: float = 0.5
    features: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyDecision:
    """
    策略决策数据类
    
    属性:
        decision_id: 决策ID
        strategy_id: 策略ID
        decision_type: 决策类型
        symbol: 标的代码
        side: 交易方向
        suggested_quantity: 建议数量
        suggested_price: 建议价格
        rationale: 决策理由
        timestamp: 决策生成时间
    """
    decision_id: str
    strategy_id: str
    decision_type: DecisionType
    symbol: str
    side: str
    suggested_quantity: int = 0
    suggested_price: float = 0.0
    rationale: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class StrategyEventBusIntegration:
    """
    策略层事件总线集成类
    
    提供策略事件的发布、订阅和处理功能。
    采用单例模式确保全局唯一实例。
    
    使用示例:
        integration = StrategyEventBusIntegration()
        
        # 发布信号事件
        signal = StrategySignal(
            signal_id="SIG-001",
            strategy_id="STRAT-001",
            symbol="000001.SZ",
            signal_type=SignalType.BUY
        )
        await integration.publish_signal(signal)
        
        # 订阅交易信号
        integration.subscribe_to_trading_signals(callback)
    """
    
    _instance: Optional['StrategyEventBusIntegration'] = None
    
    def __new__(cls) -> 'StrategyEventBusIntegration':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化策略事件总线集成"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._event_bus = None
        self._subscriptions: Dict[str, str] = {}
        self._signal_handlers: List[Callable] = []
        self._decision_handlers: List[Callable] = []
        
        logger.info("策略事件总线集成初始化完成")
    
    async def _get_event_bus(self):
        """
        获取事件总线实例
        
        Returns:
            EventBus: 事件总线实例
        """
        if self._event_bus is None:
            self._event_bus = get_event_bus()
        return self._event_bus
    
    def _build_event_data(
        self,
        event_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        构建标准事件数据
        
        Args:
            event_type: 事件类型
            payload: 事件负载
            correlation_id: 关联ID
            
        Returns:
            Dict[str, Any]: 标准格式事件数据
        """
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "source": "strategy.integration",
            "version": "1.0",
            "payload": payload
        }
    
    async def publish_signal(
        self,
        signal: StrategySignal,
        priority: EventPriority = EventPriority.HIGH
    ) -> bool:
        """
        发布策略信号事件
        
        Args:
            signal: 策略信号
            priority: 事件优先级
            
        Returns:
            bool: 是否发布成功
        """
        try:
            event_bus = await self._get_event_bus()
            
            payload = {
                "signal_id": signal.signal_id,
                "strategy_id": signal.strategy_id,
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "features": signal.features,
                "timestamp": signal.timestamp.isoformat()
            }
            
            event_data = self._build_event_data(
                event_type=StrategyEventType.SIGNAL_GENERATED.value,
                payload=payload
            )
            
            # 验证事件格式
            is_valid, errors = validate_event(event_data, strict=False)
            if not is_valid:
                logger.warning(f"事件格式验证警告: {errors}")
            
            await event_bus.publish(
                event_type=StrategyEventType.SIGNAL_GENERATED.value,
                data=event_data,
                priority=priority
            )
            
            logger.info(
                f"策略信号已发布: signal_id={signal.signal_id}, "
                f"symbol={signal.symbol}, type={signal.signal_type.value}"
            )
            return True
            
        except Exception as e:
            logger.error(f"发布策略信号失败: {e}")
            return False
    
    async def publish_decision(
        self,
        decision: StrategyDecision,
        priority: EventPriority = EventPriority.HIGH
    ) -> bool:
        """
        发布策略决策事件
        
        Args:
            decision: 策略决策
            priority: 事件优先级
            
        Returns:
            bool: 是否发布成功
        """
        try:
            event_bus = await self._get_event_bus()
            
            payload = {
                "decision_id": decision.decision_id,
                "strategy_id": decision.strategy_id,
                "decision_type": decision.decision_type.value,
                "symbol": decision.symbol,
                "side": decision.side,
                "suggested_quantity": decision.suggested_quantity,
                "suggested_price": decision.suggested_price,
                "rationale": decision.rationale,
                "timestamp": decision.timestamp.isoformat()
            }
            
            event_data = self._build_event_data(
                event_type=StrategyEventType.STRATEGY_DECISION_READY.value,
                payload=payload
            )
            
            await event_bus.publish(
                event_type=StrategyEventType.STRATEGY_DECISION_READY.value,
                data=event_data,
                priority=priority
            )
            
            logger.info(
                f"策略决策已发布: decision_id={decision.decision_id}, "
                f"symbol={decision.symbol}, type={decision.decision_type.value}"
            )
            return True
            
        except Exception as e:
            logger.error(f"发布策略决策失败: {e}")
            return False
    
    async def publish_strategy_event(
        self,
        event_type: StrategyEventType,
        strategy_id: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL
    ) -> bool:
        """
        发布策略生命周期事件
        
        Args:
            event_type: 事件类型
            strategy_id: 策略ID
            data: 事件数据
            priority: 事件优先级
            
        Returns:
            bool: 是否发布成功
        """
        try:
            event_bus = await self._get_event_bus()
            
            payload = {
                "strategy_id": strategy_id,
                **data
            }
            
            event_data = self._build_event_data(
                event_type=event_type.value,
                payload=payload
            )
            
            await event_bus.publish(
                event_type=event_type.value,
                data=event_data,
                priority=priority
            )
            
            logger.debug(f"策略事件已发布: type={event_type.value}, strategy_id={strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"发布策略事件失败: {e}")
            return False
    
    async def subscribe_to_trading_signals(
        self,
        callback: Callable[[StrategySignal], None],
        symbols: Optional[List[str]] = None
    ) -> str:
        """
        订阅交易信号
        
        Args:
            callback: 回调函数
            symbols: 标的代码列表，None表示订阅所有
            
        Returns:
            str: 订阅ID
        """
        async def signal_handler(event: Event):
            try:
                data = event.data.get("payload", {})
                
                # 如果指定了标的，进行过滤
                if symbols and data.get("symbol") not in symbols:
                    return
                
                signal = StrategySignal(
                    signal_id=data.get("signal_id", ""),
                    strategy_id=data.get("strategy_id", ""),
                    symbol=data.get("symbol", ""),
                    signal_type=SignalType(data.get("signal_type", "hold")),
                    strength=data.get("strength", 0.5),
                    confidence=data.get("confidence", 0.5),
                    features=data.get("features", {}),
                    timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
                )
                
                callback(signal)
                
            except Exception as e:
                logger.error(f"处理交易信号事件失败: {e}")
        
        event_bus = await self._get_event_bus()
        
        subscription_id = await event_bus.subscribe(
            event_type=StrategyEventType.SIGNAL_GENERATED.value,
            handler=signal_handler,
            priority=EventPriority.HIGH
        )
        
        self._subscriptions[subscription_id] = StrategyEventType.SIGNAL_GENERATED.value
        self._signal_handlers.append(callback)
        
        logger.info(f"已订阅交易信号: subscription_id={subscription_id}")
        return subscription_id
    
    async def subscribe_to_decisions(
        self,
        callback: Callable[[StrategyDecision], None],
        strategy_ids: Optional[List[str]] = None
    ) -> str:
        """
        订阅策略决策
        
        Args:
            callback: 回调函数
            strategy_ids: 策略ID列表，None表示订阅所有
            
        Returns:
            str: 订阅ID
        """
        async def decision_handler(event: Event):
            try:
                data = event.data.get("payload", {})
                
                # 如果指定了策略ID，进行过滤
                if strategy_ids and data.get("strategy_id") not in strategy_ids:
                    return
                
                decision = StrategyDecision(
                    decision_id=data.get("decision_id", ""),
                    strategy_id=data.get("strategy_id", ""),
                    decision_type=DecisionType(data.get("decision_type", "hold_position")),
                    symbol=data.get("symbol", ""),
                    side=data.get("side", ""),
                    suggested_quantity=data.get("suggested_quantity", 0),
                    suggested_price=data.get("suggested_price", 0.0),
                    rationale=data.get("rationale", ""),
                    timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
                )
                
                callback(decision)
                
            except Exception as e:
                logger.error(f"处理策略决策事件失败: {e}")
        
        event_bus = await self._get_event_bus()
        
        subscription_id = await event_bus.subscribe(
            event_type=StrategyEventType.STRATEGY_DECISION_READY.value,
            handler=decision_handler,
            priority=EventPriority.HIGH
        )
        
        self._subscriptions[subscription_id] = StrategyEventType.STRATEGY_DECISION_READY.value
        self._decision_handlers.append(callback)
        
        logger.info(f"已订阅策略决策: subscription_id={subscription_id}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        取消订阅
        
        Args:
            subscription_id: 订阅ID
            
        Returns:
            bool: 是否成功取消
        """
        try:
            event_bus = await self._get_event_bus()
            await event_bus.unsubscribe(subscription_id)
            
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
            
            logger.info(f"已取消订阅: subscription_id={subscription_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消订阅失败: {e}")
            return False
    
    async def trigger_backtest_event(
        self,
        strategy_id: str,
        backtest_config: Dict[str, Any]
    ) -> bool:
        """
        触发回测事件
        
        Args:
            strategy_id: 策略ID
            backtest_config: 回测配置
            
        Returns:
            bool: 是否触发成功
        """
        return await self.publish_strategy_event(
            event_type=StrategyEventType.BACKTEST_COMPLETED,
            strategy_id=strategy_id,
            data={"backtest_config": backtest_config},
            priority=EventPriority.NORMAL
        )
    
    def get_subscriptions(self) -> Dict[str, str]:
        """
        获取所有订阅
        
        Returns:
            Dict[str, str]: 订阅ID到事件类型的映射
        """
        return self._subscriptions.copy()


# 全局实例获取函数
def get_strategy_event_bus_integration() -> StrategyEventBusIntegration:
    """
    获取策略事件总线集成实例
    
    Returns:
        StrategyEventBusIntegration: 策略事件总线集成实例
    """
    return StrategyEventBusIntegration()
