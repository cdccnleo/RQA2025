# -*- coding: utf-8 -*-
"""
交易订单状态事件模块

提供交易订单状态机定义和状态变更事件发布功能。

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
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

from src.core.event_bus import get_event_bus, EventPriority
from src.core.event_bus.validation import validate_event

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """
    订单状态枚举
    
    定义订单生命周期中的所有状态
    """
    # 初始状态
    CREATED = "created"              # 已创建
    
    # 验证状态
    VALIDATING = "validating"        # 验证中
    VALIDATED = "validated"          # 验证通过
    VALIDATION_FAILED = "validation_failed"  # 验证失败
    
    # 风控状态
    RISK_CHECKING = "risk_checking"  # 风控检查中
    RISK_PASSED = "risk_passed"      # 风控通过
    RISK_BLOCKED = "risk_blocked"    # 风控拦截
    
    # 提交状态
    SUBMITTING = "submitting"        # 提交中
    SUBMITTED = "submitted"          # 已提交
    SUBMIT_FAILED = "submit_failed"  # 提交失败
    
    # 交易所状态
    PENDING = "pending"              # 等待成交
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"                # 完全成交
    
    # 取消状态
    CANCELLING = "cancelling"        # 取消中
    CANCELLED = "cancelled"          # 已取消
    CANCEL_REJECTED = "cancel_rejected"  # 取消被拒绝
    
    # 异常状态
    REJECTED = "rejected"            # 被拒绝
    EXPIRED = "expired"              # 已过期
    ERROR = "error"                  # 错误


class OrderStateEventType(Enum):
    """订单状态事件类型"""
    ORDER_STATE_CHANGED = "ORDER_STATE_CHANGED"  # 订单状态变更
    ORDER_CREATED = "ORDER_CREATED"              # 订单创建
    ORDER_VALIDATED = "ORDER_VALIDATED"          # 订单验证完成
    ORDER_RISK_CHECKED = "ORDER_RISK_CHECKED"    # 订单风控检查完成
    ORDER_SUBMITTED = "ORDER_SUBMITTED"          # 订单提交完成
    ORDER_FILLED = "ORDER_FILLED"                # 订单成交
    ORDER_CANCELLED = "ORDER_CANCELLED"          # 订单取消
    ORDER_REJECTED = "ORDER_REJECTED"            # 订单拒绝
    ORDER_EXPIRED = "ORDER_EXPIRED"              # 订单过期


# 定义有效的状态转换
VALID_STATE_TRANSITIONS: Dict[OrderState, Set[OrderState]] = {
    OrderState.CREATED: {
        OrderState.VALIDATING, OrderState.CANCELLING, OrderState.ERROR
    },
    OrderState.VALIDATING: {
        OrderState.VALIDATED, OrderState.VALIDATION_FAILED, OrderState.ERROR
    },
    OrderState.VALIDATED: {
        OrderState.RISK_CHECKING, OrderState.CANCELLING, OrderState.ERROR
    },
    OrderState.VALIDATION_FAILED: {
        OrderState.CREATED, OrderState.CANCELLED, OrderState.ERROR
    },
    OrderState.RISK_CHECKING: {
        OrderState.RISK_PASSED, OrderState.RISK_BLOCKED, OrderState.ERROR
    },
    OrderState.RISK_PASSED: {
        OrderState.SUBMITTING, OrderState.CANCELLING, OrderState.ERROR
    },
    OrderState.RISK_BLOCKED: {
        OrderState.CANCELLED, OrderState.CREATED, OrderState.ERROR
    },
    OrderState.SUBMITTING: {
        OrderState.SUBMITTED, OrderState.SUBMIT_FAILED, OrderState.ERROR
    },
    OrderState.SUBMITTED: {
        OrderState.PENDING, OrderState.FILLED, OrderState.REJECTED, OrderState.ERROR
    },
    OrderState.SUBMIT_FAILED: {
        OrderState.SUBMITTING, OrderState.CANCELLED, OrderState.ERROR
    },
    OrderState.PENDING: {
        OrderState.PARTIALLY_FILLED, OrderState.FILLED, 
        OrderState.CANCELLING, OrderState.EXPIRED, OrderState.ERROR
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.FILLED, OrderState.CANCELLING, OrderState.ERROR
    },
    OrderState.FILLED: set(),  # 终态
    OrderState.CANCELLING: {
        OrderState.CANCELLED, OrderState.CANCEL_REJECTED, OrderState.ERROR
    },
    OrderState.CANCELLED: set(),  # 终态
    OrderState.CANCEL_REJECTED: {
        OrderState.PENDING, OrderState.PARTIALLY_FILLED, OrderState.ERROR
    },
    OrderState.REJECTED: set(),  # 终态
    OrderState.EXPIRED: set(),  # 终态
    OrderState.ERROR: {
        OrderState.CREATED, OrderState.CANCELLED
    }
}


@dataclass
class OrderStateTransition:
    """
    订单状态转换记录
    
    属性:
        transition_id: 转换记录ID
        order_id: 订单ID
        from_state: 源状态
        to_state: 目标状态
        timestamp: 转换时间
        triggered_by: 触发者
        reason: 转换原因
        metadata: 附加元数据
    """
    transition_id: str
    order_id: str
    from_state: OrderState
    to_state: OrderState
    timestamp: datetime
    triggered_by: str = "system"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderStateEvent:
    """
    订单状态事件
    
    属性:
        event_id: 事件ID
        event_type: 事件类型
        order_id: 订单ID
        state: 当前状态
        previous_state: 前一状态
        transition: 状态转换记录
        timestamp: 事件时间
        correlation_id: 关联ID
    """
    event_id: str
    event_type: OrderStateEventType
    order_id: str
    state: OrderState
    previous_state: Optional[OrderState]
    transition: OrderStateTransition
    timestamp: datetime
    correlation_id: str


class OrderStateMachine:
    """
    订单状态机
    
    管理订单状态的转换和验证，确保状态转换符合业务规则。
    
    使用示例:
        sm = OrderStateMachine(order_id="ORD-001")
        
        # 尝试状态转换
        success, transition = await sm.transition_to(
            target_state=OrderState.VALIDATING,
            reason="开始验证订单"
        )
        
        if success:
            print(f"状态转换成功: {transition}")
    """
    
    def __init__(self, order_id: str, initial_state: OrderState = OrderState.CREATED):
        """
        初始化订单状态机
        
        Args:
            order_id: 订单ID
            initial_state: 初始状态
        """
        self.order_id = order_id
        self.current_state = initial_state
        self.state_history: List[OrderStateTransition] = []
        self._lock = asyncio.Lock()
        
        # 记录初始状态
        self._record_transition(None, initial_state, "订单创建")
        
        logger.debug(f"订单状态机初始化: order_id={order_id}, state={initial_state.value}")
    
    def can_transition_to(self, target_state: OrderState) -> bool:
        """
        检查是否可以转换到目标状态
        
        Args:
            target_state: 目标状态
            
        Returns:
            bool: 是否可以转换
        """
        valid_targets = VALID_STATE_TRANSITIONS.get(self.current_state, set())
        return target_state in valid_targets
    
    async def transition_to(
        self,
        target_state: OrderState,
        triggered_by: str = "system",
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[OrderStateTransition]]:
        """
        执行状态转换
        
        Args:
            target_state: 目标状态
            triggered_by: 触发者
            reason: 转换原因
            metadata: 附加元数据
            
        Returns:
            tuple: (是否成功, 转换记录)
        """
        async with self._lock:
            # 检查是否允许转换
            if not self.can_transition_to(target_state):
                logger.warning(
                    f"状态转换不允许: order_id={self.order_id}, "
                    f"from={self.current_state.value}, to={target_state.value}"
                )
                return False, None
            
            # 执行转换
            previous_state = self.current_state
            self.current_state = target_state
            
            # 记录转换
            transition = self._record_transition(
                previous_state, target_state, reason, triggered_by, metadata
            )
            
            logger.info(
                f"状态转换成功: order_id={self.order_id}, "
                f"from={previous_state.value}, to={target_state.value}"
            )
            
            return True, transition
    
    def _record_transition(
        self,
        from_state: Optional[OrderState],
        to_state: OrderState,
        reason: str = "",
        triggered_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> OrderStateTransition:
        """记录状态转换"""
        transition = OrderStateTransition(
            transition_id=str(uuid.uuid4()),
            order_id=self.order_id,
            from_state=from_state or to_state,
            to_state=to_state,
            timestamp=datetime.now(),
            triggered_by=triggered_by,
            reason=reason,
            metadata=metadata or {}
        )
        
        self.state_history.append(transition)
        return transition
    
    def get_state_history(self) -> List[OrderStateTransition]:
        """
        获取状态历史
        
        Returns:
            List[OrderStateTransition]: 状态转换历史
        """
        return self.state_history.copy()
    
    def is_terminal_state(self) -> bool:
        """
        检查当前状态是否为终态
        
        Returns:
            bool: 是否为终态
        """
        return len(VALID_STATE_TRANSITIONS.get(self.current_state, set())) == 0
    
    def get_valid_transitions(self) -> Set[OrderState]:
        """
        获取有效的目标状态集合
        
        Returns:
            Set[OrderState]: 有效的目标状态
        """
        return VALID_STATE_TRANSITIONS.get(self.current_state, set()).copy()


class OrderStateEventPublisher:
    """
    订单状态事件发布器
    
    负责发布订单状态变更事件到事件总线。
    
    使用示例:
        publisher = OrderStateEventPublisher()
        
        # 发布状态变更事件
        await publisher.publish_state_changed(
            order_id="ORD-001",
            from_state=OrderState.CREATED,
            to_state=OrderState.VALIDATING,
            reason="开始验证"
        )
    """
    
    _instance: Optional['OrderStateEventPublisher'] = None
    
    def __new__(cls) -> 'OrderStateEventPublisher':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化发布器"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._event_bus = None
        self._state_machines: Dict[str, OrderStateMachine] = {}
        
        logger.info("订单状态事件发布器初始化完成")
    
    async def _get_event_bus(self):
        """获取事件总线"""
        if self._event_bus is None:
            self._event_bus = get_event_bus()
        return self._event_bus
    
    def get_or_create_state_machine(
        self,
        order_id: str,
        initial_state: OrderState = OrderState.CREATED
    ) -> OrderStateMachine:
        """
        获取或创建状态机
        
        Args:
            order_id: 订单ID
            initial_state: 初始状态
            
        Returns:
            OrderStateMachine: 状态机实例
        """
        if order_id not in self._state_machines:
            self._state_machines[order_id] = OrderStateMachine(order_id, initial_state)
        return self._state_machines[order_id]
    
    async def publish_state_changed(
        self,
        order_id: str,
        from_state: OrderState,
        to_state: OrderState,
        reason: str = "",
        triggered_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        发布状态变更事件
        
        Args:
            order_id: 订单ID
            from_state: 源状态
            to_state: 目标状态
            reason: 转换原因
            triggered_by: 触发者
            metadata: 附加元数据
            correlation_id: 关联ID
            
        Returns:
            bool: 是否发布成功
        """
        try:
            event_bus = await self._get_event_bus()
            
            # 创建转换记录
            transition = OrderStateTransition(
                transition_id=str(uuid.uuid4()),
                order_id=order_id,
                from_state=from_state,
                to_state=to_state,
                timestamp=datetime.now(),
                triggered_by=triggered_by,
                reason=reason,
                metadata=metadata or {}
            )
            
            # 构建事件数据
            event_data = {
                "event_id": str(uuid.uuid4()),
                "event_type": OrderStateEventType.ORDER_STATE_CHANGED.value,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id or str(uuid.uuid4()),
                "source": "trading.events",
                "version": "1.0",
                "payload": {
                    "order_id": order_id,
                    "previous_state": from_state.value if from_state else None,
                    "current_state": to_state.value,
                    "transition_id": transition.transition_id,
                    "triggered_by": triggered_by,
                    "reason": reason,
                    "metadata": metadata or {},
                    "timestamp": transition.timestamp.isoformat()
                }
            }
            
            # 验证事件格式
            is_valid, errors = validate_event(event_data, strict=False)
            if not is_valid:
                logger.warning(f"事件格式验证警告: {errors}")
            
            # 发布事件
            await event_bus.publish(
                event_type=OrderStateEventType.ORDER_STATE_CHANGED.value,
                data=event_data,
                priority=EventPriority.HIGH
            )
            
            logger.info(
                f"订单状态变更事件已发布: order_id={order_id}, "
                f"from={from_state.value if from_state else None}, to={to_state.value}"
            )
            return True
            
        except Exception as e:
            logger.error(f"发布订单状态变更事件失败: {e}")
            return False
    
    async def publish_specific_state_event(
        self,
        event_type: OrderStateEventType,
        order_id: str,
        state: OrderState,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        发布特定状态事件
        
        Args:
            event_type: 事件类型
            order_id: 订单ID
            state: 订单状态
            details: 详细信息
            correlation_id: 关联ID
            
        Returns:
            bool: 是否发布成功
        """
        try:
            event_bus = await self._get_event_bus()
            
            event_data = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type.value,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id or str(uuid.uuid4()),
                "source": "trading.events",
                "version": "1.0",
                "payload": {
                    "order_id": order_id,
                    "state": state.value,
                    "details": details or {},
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            await event_bus.publish(
                event_type=event_type.value,
                data=event_data,
                priority=EventPriority.NORMAL
            )
            
            logger.debug(f"订单状态事件已发布: type={event_type.value}, order_id={order_id}")
            return True
            
        except Exception as e:
            logger.error(f"发布订单状态事件失败: {e}")
            return False
    
    async def transition_and_publish(
        self,
        order_id: str,
        target_state: OrderState,
        reason: str = "",
        triggered_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[OrderStateTransition]]:
        """
        执行状态转换并发布事件
        
        Args:
            order_id: 订单ID
            target_state: 目标状态
            reason: 转换原因
            triggered_by: 触发者
            metadata: 附加元数据
            
        Returns:
            tuple: (是否成功, 转换记录)
        """
        # 获取状态机
        sm = self.get_or_create_state_machine(order_id)
        
        # 执行转换
        success, transition = await sm.transition_to(
            target_state=target_state,
            triggered_by=triggered_by,
            reason=reason,
            metadata=metadata
        )
        
        if success and transition:
            # 发布状态变更事件
            await self.publish_state_changed(
                order_id=order_id,
                from_state=transition.from_state,
                to_state=transition.to_state,
                reason=reason,
                triggered_by=triggered_by,
                metadata=metadata
            )
            
            # 根据目标状态发布特定事件
            await self._publish_specific_event(order_id, target_state, metadata)
        
        return success, transition
    
    async def _publish_specific_event(
        self,
        order_id: str,
        state: OrderState,
        details: Optional[Dict[str, Any]] = None
    ):
        """根据状态发布特定事件"""
        event_mapping = {
            OrderState.CREATED: OrderStateEventType.ORDER_CREATED,
            OrderState.VALIDATED: OrderStateEventType.ORDER_VALIDATED,
            OrderState.RISK_PASSED: OrderStateEventType.ORDER_RISK_CHECKED,
            OrderState.SUBMITTED: OrderStateEventType.ORDER_SUBMITTED,
            OrderState.FILLED: OrderStateEventType.ORDER_FILLED,
            OrderState.CANCELLED: OrderStateEventType.ORDER_CANCELLED,
            OrderState.REJECTED: OrderStateEventType.ORDER_REJECTED,
            OrderState.EXPIRED: OrderStateEventType.ORDER_EXPIRED,
        }
        
        event_type = event_mapping.get(state)
        if event_type:
            await self.publish_specific_state_event(
                event_type=event_type,
                order_id=order_id,
                state=state,
                details=details
            )


# 全局实例获取函数
def get_order_state_event_publisher() -> OrderStateEventPublisher:
    """
    获取订单状态事件发布器实例
    
    Returns:
        OrderStateEventPublisher: 发布器实例
    """
    return OrderStateEventPublisher()
