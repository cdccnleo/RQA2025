
from .core.gpu_manager import GPUManager
from .core.resource_manager import CoreResourceManager as ResourceManager
from pathlib import Path
"""
基础设施层 - 资源管理组件

资源管理相关的文件
"""

__version__ = "1.0.0"

__all__ = [
    'ResourceManager',
    'GPUManager'
]
