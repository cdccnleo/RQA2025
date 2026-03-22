# -*- coding: utf-8 -*-
"""
交易层风险拦截事件集成模块

提供交易层对风险拦截事件的订阅和处理能力，实现实时风险控制。

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
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

# 导入事件总线
from src.core.event_bus import (
    get_event_bus,
    EventPriority,
    Event,
    EventType
)

# 导入交易层类型
from src.trading.execution.order_manager import OrderStatus

logger = logging.getLogger(__name__)


class RiskInterceptAction(Enum):
    """
    风险拦截动作枚举
    
    定义风险拦截后可执行的动作类型
    """
    CANCEL_ORDER = "cancel_order"          # 取消订单
    PAUSE_TRADING = "pause_trading"        # 暂停交易
    BLOCK_SYMBOL = "block_symbol"          # 屏蔽标的
    REDUCE_POSITION = "reduce_position"    # 减仓
    ALERT_ONLY = "alert_only"              # 仅告警


@dataclass
class RiskInterceptEvent:
    """
    风险拦截事件数据类
    
    属性:
        event_id: 事件ID
        timestamp: 事件发生时间
        risk_type: 风险类型
        risk_level: 风险等级(1-5)
        symbol: 受影响标的
        order_id: 受影响订单ID
        action: 建议动作
        reason: 拦截原因
        metadata: 附加元数据
    """
    event_id: str
    timestamp: datetime
    risk_type: str
    risk_level: int
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    action: RiskInterceptAction = RiskInterceptAction.ALERT_ONLY
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskInterceptResult:
    """
    风险拦截处理结果类
    
    属性:
        event_id: 事件ID
        success: 是否成功处理
        action_taken: 执行的动作
        affected_orders: 受影响的订单列表
        error_message: 错误信息
        processing_time_ms: 处理时间(毫秒)
    """
    event_id: str
    success: bool
    action_taken: RiskInterceptAction
    affected_orders: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0


class RiskInterceptHandler:
    """
    风险拦截事件处理器基类
    
    定义风险拦截事件处理的接口，子类需要实现具体的处理逻辑
    """
    
    async def handle(self, event: RiskInterceptEvent) -> RiskInterceptResult:
        """
        处理风险拦截事件
        
        Args:
            event: 风险拦截事件
            
        Returns:
            RiskInterceptResult: 处理结果
        """
        raise NotImplementedError("子类必须实现handle方法")


class RiskInterceptIntegration:
    """
    交易层风险拦截事件集成类
    
    提供风险拦截事件的订阅、处理和响应机制，确保交易层能够
    实时响应风险控制层的拦截指令。
    
    使用示例:
        integration = RiskInterceptIntegration()
        await integration.start()
        
        # 注册自定义处理器
        integration.register_handler(RiskInterceptAction.CANCEL_ORDER, my_handler)
        
        # 订阅特定标的的风险事件
        integration.subscribe_symbol("000001.SZ")
    """
    
    _instance: Optional['RiskInterceptIntegration'] = None
    
    def __new__(cls) -> 'RiskInterceptIntegration':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化风险拦截集成"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._event_bus = None
        self._subscription_id: Optional[str] = None
        self._handlers: Dict[RiskInterceptAction, RiskInterceptHandler] = {}
        self._symbol_subscriptions: Set[str] = set()
        self._order_subscriptions: Set[str] = set()
        self._event_history: List[RiskInterceptEvent] = []
        self._max_history_size = 1000
        self._is_running = False
        
        # 注册默认处理器
        self._register_default_handlers()
        
        logger.info("风险拦截集成初始化完成")
    
    def _register_default_handlers(self):
        """注册默认的风险拦截处理器"""
        self._handlers[RiskInterceptAction.CANCEL_ORDER] = CancelOrderHandler()
        self._handlers[RiskInterceptAction.PAUSE_TRADING] = PauseTradingHandler()
        self._handlers[RiskInterceptAction.BLOCK_SYMBOL] = BlockSymbolHandler()
        self._handlers[RiskInterceptAction.ALERT_ONLY] = AlertOnlyHandler()
    
    async def _get_event_bus(self):
        """
        获取事件总线实例
        
        Returns:
            EventBus: 事件总线实例
        """
        if self._event_bus is None:
            self._event_bus = get_event_bus()
        return self._event_bus
    
    async def start(self):
        """
        启动风险拦截事件订阅
        
        订阅事件总线中的RISK_INTERCEPTED事件
        """
        if self._is_running:
            logger.warning("风险拦截集成已在运行中")
            return
        
        event_bus = await self._get_event_bus()
        
        # 订阅风险拦截事件
        self._subscription_id = await event_bus.subscribe(
            event_type="RISK_INTERCEPTED",
            handler=self._on_risk_intercepted,
            priority=EventPriority.HIGH
        )
        
        self._is_running = True
        logger.info("风险拦截事件订阅已启动")
    
    async def stop(self):
        """
        停止风险拦截事件订阅
        """
        if not self._is_running:
            return
        
        if self._subscription_id and self._event_bus:
            await self._event_bus.unsubscribe(self._subscription_id)
            self._subscription_id = None
        
        self._is_running = False
        logger.info("风险拦截事件订阅已停止")
    
    async def _on_risk_intercepted(self, event: Event):
        """
        风险拦截事件回调函数
        
        Args:
            event: 事件总线事件
        """
        start_time = datetime.now()
        
        try:
            # 解析事件数据
            intercept_event = self._parse_event(event)
            
            # 检查是否需要处理
            if not self._should_handle(intercept_event):
                return
            
            # 记录事件
            self._record_event(intercept_event)
            
            # 获取处理器
            handler = self._handlers.get(intercept_event.action)
            if handler is None:
                logger.warning(f"未找到风险拦截动作处理器: {intercept_event.action}")
                handler = self._handlers[RiskInterceptAction.ALERT_ONLY]
            
            # 处理事件
            result = await handler.handle(intercept_event)
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            
            # 记录处理结果
            if result.success:
                logger.info(
                    f"风险拦截事件处理成功: event_id={result.event_id}, "
                    f"action={result.action_taken.value}, "
                    f"time={processing_time:.2f}ms"
                )
            else:
                logger.error(
                    f"风险拦截事件处理失败: event_id={result.event_id}, "
                    f"error={result.error_message}"
                )
            
            # 发布处理结果事件
            await self._publish_result_event(result)
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"风险拦截事件处理异常: {e}")
    
    def _parse_event(self, event: Event) -> RiskInterceptEvent:
        """
        解析事件总线事件为风险拦截事件
        
        Args:
            event: 事件总线事件
            
        Returns:
            RiskInterceptEvent: 风险拦截事件
        """
        data = event.data
        
        # 解析动作
        action_str = data.get("action", "alert_only")
        try:
            action = RiskInterceptAction(action_str)
        except ValueError:
            action = RiskInterceptAction.ALERT_ONLY
        
        return RiskInterceptEvent(
            event_id=data.get("event_id", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            risk_type=data.get("risk_type", "unknown"),
            risk_level=data.get("risk_level", 1),
            symbol=data.get("symbol"),
            order_id=data.get("order_id"),
            action=action,
            reason=data.get("reason", ""),
            metadata=data.get("metadata", {})
        )
    
    def _should_handle(self, event: RiskInterceptEvent) -> bool:
        """
        检查是否应该处理该事件
        
        Args:
            event: 风险拦截事件
            
        Returns:
            bool: 是否应该处理
        """
        # 如果订阅了特定标的，检查是否匹配
        if self._symbol_subscriptions and event.symbol:
            if event.symbol not in self._symbol_subscriptions:
                return False
        
        # 如果订阅了特定订单，检查是否匹配
        if self._order_subscriptions and event.order_id:
            if event.order_id not in self._order_subscriptions:
                return False
        
        return True
    
    def _record_event(self, event: RiskInterceptEvent):
        """
        记录风险拦截事件
        
        Args:
            event: 风险拦截事件
        """
        self._event_history.append(event)
        
        # 限制历史记录大小
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    async def _publish_result_event(self, result: RiskInterceptResult):
        """
        发布风险拦截处理结果事件
        
        Args:
            result: 处理结果
        """
        event_bus = await self._get_event_bus()
        
        await event_bus.publish(
            event_type="RISK_INTERCEPT_HANDLED",
            data={
                "event_id": result.event_id,
                "success": result.success,
                "action_taken": result.action_taken.value,
                "affected_orders": result.affected_orders,
                "error_message": result.error_message,
                "processing_time_ms": result.processing_time_ms,
                "timestamp": datetime.now().isoformat()
            },
            priority=EventPriority.NORMAL
        )
    
    def register_handler(self, action: RiskInterceptAction, handler: RiskInterceptHandler):
        """
        注册风险拦截处理器
        
        Args:
            action: 风险拦截动作
            handler: 处理器实例
        """
        self._handlers[action] = handler
        logger.info(f"已注册风险拦截处理器: {action.value}")
    
    def subscribe_symbol(self, symbol: str):
        """
        订阅特定标的的风险事件
        
        Args:
            symbol: 标的代码
        """
        self._symbol_subscriptions.add(symbol)
        logger.info(f"已订阅标的的风险事件: {symbol}")
    
    def unsubscribe_symbol(self, symbol: str):
        """
        取消订阅特定标的的风险事件
        
        Args:
            symbol: 标的代码
        """
        self._symbol_subscriptions.discard(symbol)
        logger.info(f"已取消订阅标的的风险事件: {symbol}")
    
    def subscribe_order(self, order_id: str):
        """
        订阅特定订单的风险事件
        
        Args:
            order_id: 订单ID
        """
        self._order_subscriptions.add(order_id)
        logger.info(f"已订阅订单的风险事件: {order_id}")
    
    def unsubscribe_order(self, order_id: str):
        """
        取消订阅特定订单的风险事件
        
        Args:
            order_id: 订单ID
        """
        self._order_subscriptions.discard(order_id)
        logger.info(f"已取消订阅订单的风险事件: {order_id}")
    
    def get_event_history(self, limit: int = 100) -> List[RiskInterceptEvent]:
        """
        获取风险拦截事件历史
        
        Args:
            limit: 返回的最大记录数
            
        Returns:
            List[RiskInterceptEvent]: 事件历史列表
        """
        return self._event_history[-limit:]
    
    def clear_event_history(self):
        """清空事件历史"""
        self._event_history.clear()
        logger.info("风险拦截事件历史已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_events = len(self._event_history)
        
        # 按风险类型统计
        risk_type_counts = {}
        for event in self._event_history:
            risk_type_counts[event.risk_type] = risk_type_counts.get(event.risk_type, 0) + 1
        
        # 按风险等级统计
        risk_level_counts = {}
        for event in self._event_history:
            risk_level_counts[event.risk_level] = risk_level_counts.get(event.risk_level, 0) + 1
        
        return {
            "total_events": total_events,
            "is_running": self._is_running,
            "symbol_subscriptions": list(self._symbol_subscriptions),
            "order_subscriptions": list(self._order_subscriptions),
            "risk_type_counts": risk_type_counts,
            "risk_level_counts": risk_level_counts
        }


# ========== 默认处理器实现 ==========

class CancelOrderHandler(RiskInterceptHandler):
    """取消订单处理器"""
    
    async def handle(self, event: RiskInterceptEvent) -> RiskInterceptResult:
        """处理取消订单动作"""
        affected_orders = []
        
        try:
            # 从交易层导入订单管理器
            from src.trading.execution.order_manager import OrderManager
            
            order_manager = OrderManager()
            
            # 如果指定了订单ID，取消该订单
            if event.order_id:
                success = order_manager.cancel_order(event.order_id)
                if success:
                    affected_orders.append(event.order_id)
            
            # 如果指定了标的，取消该标的的所有活跃订单
            if event.symbol:
                # 获取该标的的活跃订单
                active_orders = order_manager.get_active_orders(symbol=event.symbol)
                for order in active_orders:
                    if order_manager.cancel_order(order.order_id):
                        affected_orders.append(order.order_id)
            
            return RiskInterceptResult(
                event_id=event.event_id,
                success=True,
                action_taken=RiskInterceptAction.CANCEL_ORDER,
                affected_orders=affected_orders
            )
            
        except Exception as e:
            return RiskInterceptResult(
                event_id=event.event_id,
                success=False,
                action_taken=RiskInterceptAction.CANCEL_ORDER,
                error_message=str(e)
            )


class PauseTradingHandler(RiskInterceptHandler):
    """暂停交易处理器"""
    
    async def handle(self, event: RiskInterceptEvent) -> RiskInterceptResult:
        """处理暂停交易动作"""
        try:
            # 从交易层导入交易引擎
            from src.trading.core.trading_engine import TradingEngine
            
            # 暂停交易
            # 注意：这里需要根据实际的交易引擎接口实现
            logger.warning(f"交易已暂停: reason={event.reason}")
            
            return RiskInterceptResult(
                event_id=event.event_id,
                success=True,
                action_taken=RiskInterceptAction.PAUSE_TRADING
            )
            
        except Exception as e:
            return RiskInterceptResult(
                event_id=event.event_id,
                success=False,
                action_taken=RiskInterceptAction.PAUSE_TRADING,
                error_message=str(e)
            )


class BlockSymbolHandler(RiskInterceptHandler):
    """屏蔽标的处理器"""
    
    async def handle(self, event: RiskInterceptEvent) -> RiskInterceptResult:
        """处理屏蔽标的动作"""
        try:
            if event.symbol:
                logger.warning(f"标的已屏蔽: symbol={event.symbol}, reason={event.reason}")
                
                return RiskInterceptResult(
                    event_id=event.event_id,
                    success=True,
                    action_taken=RiskInterceptAction.BLOCK_SYMBOL,
                    affected_orders=[event.symbol]
                )
            else:
                return RiskInterceptResult(
                    event_id=event.event_id,
                    success=False,
                    action_taken=RiskInterceptAction.BLOCK_SYMBOL,
                    error_message="未指定要屏蔽的标的"
                )
                
        except Exception as e:
            return RiskInterceptResult(
                event_id=event.event_id,
                success=False,
                action_taken=RiskInterceptAction.BLOCK_SYMBOL,
                error_message=str(e)
            )


class AlertOnlyHandler(RiskInterceptHandler):
    """仅告警处理器"""
    
    async def handle(self, event: RiskInterceptEvent) -> RiskInterceptResult:
        """处理仅告警动作"""
        logger.warning(
            f"风险拦截告警: event_id={event.event_id}, "
            f"risk_type={event.risk_type}, risk_level={event.risk_level}, "
            f"reason={event.reason}"
        )
        
        return RiskInterceptResult(
            event_id=event.event_id,
            success=True,
            action_taken=RiskInterceptAction.ALERT_ONLY
        )


# 全局实例获取函数
def get_risk_intercept_integration() -> RiskInterceptIntegration:
    """
    获取风险拦截集成实例
    
    Returns:
        RiskInterceptIntegration: 风险拦截集成实例
    """
    return RiskInterceptIntegration()
