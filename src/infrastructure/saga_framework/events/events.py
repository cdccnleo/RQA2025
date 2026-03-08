"""
Saga Events Module

定义Saga框架使用的事件类型和事件处理接口。
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


@dataclass
class DomainEvent:
    """
    领域事件基类
    
    用于在Saga执行过程中传递事件信息。
    
    Attributes:
        type: 事件类型
        saga_id: Saga实例ID
        data: 事件数据
        timestamp: 事件时间戳
        metadata: 事件元数据
    """
    
    type: str
    saga_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """
        转换为JSON字符串
        
        Returns:
            JSON格式的字符串
        """
        return json.dumps({
            "type": self.type,
            "saga_id": self.saga_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }, ensure_ascii=False)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            包含事件数据的字典
        """
        return {
            "type": self.type,
            "saga_id": self.saga_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainEvent":
        """
        从字典创建事件实例
        
        Args:
            data: 包含事件数据的字典
            
        Returns:
            DomainEvent实例
        """
        return cls(
            type=data["type"],
            saga_id=data["saga_id"],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class CompensationEvent(DomainEvent):
    """
    补偿事件
    
    用于触发补偿事务。
    
    Attributes:
        failed_event_type: 失败的事件类型
        failed_event_data: 失败的事件数据
    """
    
    failed_event_type: str = ""
    failed_event_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.type:
            self.type = "CompensationEvent"


class SagaEvents:
    """
    Saga相关事件类型定义
    
    定义Saga执行过程中使用的标准事件类型。
    """
    
    # Saga生命周期事件
    SAGA_STARTED = "SagaStarted"
    SAGA_COMPLETED = "SagaCompleted"
    SAGA_FAILED = "SagaFailed"
    SAGA_COMPENSATING = "SagaCompensating"
    SAGA_COMPENSATED = "SagaCompensated"
    
    # 步骤执行事件
    STEP_STARTED = "StepStarted"
    STEP_COMPLETED = "StepCompleted"
    STEP_FAILED = "StepFailed"
    
    # 补偿事件
    COMPENSATION_STARTED = "CompensationStarted"
    COMPENSATION_COMPLETED = "CompensationCompleted"
    COMPENSATION_FAILED = "CompensationFailed"


@dataclass
class HandlerResult:
    """
    事件处理器结果
    
    用于返回事件处理的结果。
    
    Attributes:
        success: 是否成功
        next_event: 下一个事件（可选）
        error: 错误信息（可选）
    """
    
    success: bool
    next_event: Optional[DomainEvent] = None
    error: Optional[str] = None


class EventHandler(ABC):
    """
    事件处理器接口
    
    定义事件处理器的标准接口。
    """
    
    @abstractmethod
    async def process(self, event: DomainEvent) -> HandlerResult:
        """
        处理事件
        
        Args:
            event: 要处理的事件
            
        Returns:
            处理结果
        """
        pass


class CompensationHandler(ABC):
    """
    补偿处理器接口
    
    定义补偿处理器的标准接口。
    """
    
    @abstractmethod
    async def compensate(self, event: CompensationEvent) -> bool:
        """
        执行补偿
        
        Args:
            event: 补偿事件
            
        Returns:
            补偿是否成功
        """
        pass
