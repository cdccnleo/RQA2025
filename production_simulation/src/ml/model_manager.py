"""
模型管理器模块（别名模块）
提供向后兼容的导入路径

实际实现在 models/model_manager.py 中
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


class ModelStatus(Enum):
    """模型状态"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    TRAINING = "training"
    READY = "ready"
    ERROR = "error"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    model_name: str
    model_type: str
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    author: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


try:
    from .models.model_manager import ModelManager, ModelType
except ImportError:
    # 提供基础实现
    class ModelManager:
        pass
    
    class ModelType:
        pass

__all__ = ['ModelManager', 'ModelType', 'ModelStatus', 'ModelMetadata']

