"""
Saga Context Module

提供Saga执行过程中的上下文管理功能。
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class SagaContext:
    """
    Saga执行上下文
    
    用于在Saga执行过程中传递数据和共享状态。
    支持数据存储、服务获取和元数据管理。
    
    Attributes:
        saga_id: Saga实例唯一标识
        data: 上下文数据存储字典
        metadata: 元数据存储字典
        services: 服务实例字典
    """
    
    saga_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    services: Dict[str, Any] = field(default_factory=dict)
    
    def set(self, key: str, value: Any) -> None:
        """
        设置上下文数据
        
        Args:
            key: 数据键名
            value: 数据值
        """
        self.data[key] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取上下文数据
        
        Args:
            key: 数据键名
            default: 默认值
            
        Returns:
            数据值或默认值
        """
        return self.data.get(key, default)
        
    def set_metadata(self, key: str, value: Any) -> None:
        """
        设置元数据
        
        Args:
            key: 元数据键名
            value: 元数据值
        """
        self.metadata[key] = value
        
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        获取元数据
        
        Args:
            key: 元数据键名
            default: 默认值
            
        Returns:
            元数据值或默认值
        """
        return self.metadata.get(key, default)
        
    def register_service(self, name: str, service: Any) -> None:
        """
        注册服务实例
        
        Args:
            name: 服务名称
            service: 服务实例
        """
        self.services[name] = service
        
    def get_service(self, name: str) -> Optional[Any]:
        """
        获取服务实例
        
        Args:
            name: 服务名称
            
        Returns:
            服务实例或None
        """
        return self.services.get(name)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含上下文数据的字典
        """
        return {
            "saga_id": self.saga_id,
            "data": self.data,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SagaContext":
        """
        从字典创建上下文实例
        
        Args:
            data: 包含上下文数据的字典
            
        Returns:
            SagaContext实例
        """
        return cls(
            saga_id=data["saga_id"],
            data=data.get("data", {}),
            metadata=data.get("metadata", {})
        )
