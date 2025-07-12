"""
资源管理核心模块
包含以下组件：
- ResourceManager: 统一资源管理器
- GPUManager: GPU资源管理
- QuotaManager: 配额管理
"""
from .resource_manager import ResourceManager, ResourceAllocationError
from .gpu_manager import GPUManager
from .quota_manager import QuotaManager

__all__ = [
    'ResourceManager',
    'ResourceAllocationError',
    'GPUManager',
    'QuotaManager'
]
