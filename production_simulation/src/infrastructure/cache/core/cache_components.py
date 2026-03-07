
# 导入新的接口体系和基础组件

from .base import BaseCacheComponent
from .mixins import CRUDOperationsMixin  # 新增Mixin导入
from datetime import datetime
from typing import Dict, Any, Optional
"""
缓存组件实现

提供标准的缓存组件实现，遵循新的接口体系。
基于BaseCacheComponent提供统一的基础功能。
"""


class CacheComponent(BaseCacheComponent, CRUDOperationsMixin):
    """统一CacheComponent实现 - 支持ICacheComponent协议"""

    def __init__(self, component_id: int, component_type: str = "memory", **kwargs):
        """
        初始化缓存组件

        Args:
            component_id: 组件ID
            component_type: 组件类型
            **kwargs: 其他配置参数
        """
        # 初始化基础组件
        super().__init__(component_id=component_id, component_type=component_type,
                         config=kwargs.get('config', {}))

        # 初始化CRUD操作Mixin
        CRUDOperationsMixin.__init__(self, storage_backend={})

        # 组件特定属性
        self.metrics = {}

        # 初始化通用组件属性
        self._init_component_attributes()

        # 初始化组件
        self.initialize_component(self.config)

    @property
    def component_name(self) -> str:
        """组件唯一标识符"""
        return f"CacheComponent_{self.component_id}"

    @property
    def component_type(self) -> str:
        """组件类型"""
        return self._component_type

    def initialize_component(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        try:
            self._initialized = True
            self.status = "healthy"
            self._last_check = datetime.now()
            return True
        except Exception as e:
            self._error_count += 1
            self.status = "error"
            return False

    def get_component_status(self) -> Dict[str, Any]:
        """获取组件状态信息"""
        self._last_check = datetime.now()
        return {
            'status': self.status,
            'initialized': self._initialized,
            'last_check': self._last_check,
            'error_count': self._error_count,
            'cache_size': self.size()  # 使用Mixin方法
        }

    def shutdown_component(self) -> None:
        """关闭组件"""
        self.status = "stopped"
        self.clear()  # 使用Mixin方法

    def health_check(self) -> bool:
        """健康检查"""
        try:
            self._last_check = datetime.now()
            return self._initialized and self.status in ["healthy", "initialized"]
        except Exception:
            return False

    def get_cache_item(self, key: str) -> Any:
        """获取缓存项 (协议实现)"""
        return self.get(key)  # 使用Mixin方法

    def set_cache_item(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项 (协议实现)"""
        return self.set(key, value, ttl)  # 使用Mixin方法

    def delete_cache_item(self, key: str) -> bool:
        """删除缓存项 (协议实现)"""
        return self.delete(key)  # 使用Mixin方法

    def has_cache_item(self, key: str) -> bool:
        """检查缓存项是否存在 (协议实现)"""
        return self.exists(key)  # 使用Mixin方法

    def clear_all_cache(self) -> bool:
        """清空所有缓存 (协议实现)"""
        return self.clear()  # 使用Mixin方法

    def get_cache_size(self) -> int:
        """获取缓存大小 (协议实现)"""
        return self.size()  # 使用Mixin方法

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'size': len(self._storage),
            'status': self.status,
            'created_at': self.creation_time.isoformat(),
            'component_type': self.component_type
        }

    # ==================== 兼容性方法 ====================

    def get_component_id(self) -> int:
        """获取组件ID (兼容旧接口)"""
        return self.component_id if self.component_id is not None else 0

    def get_component_status_string(self) -> str:
        """获取组件状态字符串 (兼容旧接口) - 返回字符串状态"""
        return self.status

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息 (兼容旧接口)"""
        return {
            "component_id": self.component_id if self.component_id is not None else 0,
            "component_type": self.component_type,
            "component_name": self.component_name,
            "status": self.status,
            "created_at": self.creation_time.isoformat(),
            "metrics": self.metrics,
            "config": self.config,
            "description": "重构后的统一缓存组件实现",
            "version": "3.0.0",
            "type": "refactored_cache_component"
        }
