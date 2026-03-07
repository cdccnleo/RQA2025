
#!/usr/bin/env python3
import threading

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
"""
基础设施层基础类

提供基础设施层组件的通用基类
"""


class BaseInfrastructureComponent(ABC):

    """基础设施组件基类"""

    def __init__(self, component_name: str):
        """初始化组件"""
        self.component_name = component_name
        self.start_time = datetime.now()
        self._lock = threading.Lock()
        self._initialized = False

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "component": self.component_name,
            "status": "running" if self._initialized else "stopped",
            "uptime": str(datetime.now() - self.start_time),
            "timestamp": datetime.now().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = self._perform_health_check()
            return {
                "component": self.component_name,
                "status": "healthy" if status else "unhealthy",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "component": self.component_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    @abstractmethod
    def _perform_health_check(self) -> bool:
        """执行具体的健康检查"""

    def initialize(self) -> bool:
        """初始化组件"""
        with self._lock:
            if not self._initialized:
                try:
                    self._initialize_component()
                    self._initialized = True
                    return True
                except Exception as e:
                    print(f"❌ 组件 {self.component_name} 初始化失败: {e}")
                    return False
            return True

    def _initialize_component(self):
        """初始化具体组件"""

    def shutdown(self) -> bool:
        """关闭组件"""
        with self._lock:
            if self._initialized:
                try:
                    self._shutdown_component()
                    self._initialized = False
                    return True
                except Exception as e:
                    print(f"❌ 组件 {self.component_name} 关闭失败: {e}")
                    return False
            return True

    def _shutdown_component(self):
        """关闭具体组件"""


class BaseServiceComponent(BaseInfrastructureComponent):

    """服务组件基类"""

    def __init__(self, service_name: str, host: str = "localhost", port: int = 0):
        """初始化服务组件"""
        super().__init__(service_name)
        self.host = host
        self.port = port
        self.is_running = False

    def _perform_health_check(self) -> bool:
        """服务健康检查"""
        return self.is_running and self._check_service_health()

    def _check_service_health(self) -> bool:
        """检查服务具体健康状态"""
        return True

    def _initialize_component(self):
        """初始化服务"""
        self._start_service()

    def _shutdown_component(self):
        """关闭服务"""
        self._stop_service()

    @abstractmethod
    def _start_service(self):
        """启动服务"""

    @abstractmethod
    def _stop_service(self):
        """停止服务"""


class BaseManagerComponent(BaseInfrastructureComponent):

    """管理器组件基类"""

    def __init__(self, manager_name: str, max_items: int = 1000):
        """初始化管理器组件"""
        super().__init__(manager_name)
        self.max_items = max_items
        self._items = {}
        self._item_lock = threading.Lock()

    def _perform_health_check(self) -> bool:
        """管理器健康检查"""
        return len(self._items) <= self.max_items

    def add_item(self, key: str, item: Any) -> bool:
        """添加项目"""
        with self._item_lock:
            if len(self._items) >= self.max_items:
                return False
            self._items[key] = item
            return True

    def get_item(self, key: str) -> Optional[Any]:
        """获取项目"""
        with self._item_lock:
            return self._items.get(key)

    def remove_item(self, key: str) -> bool:
        """移除项目"""
        with self._item_lock:
            if key in self._items:
                del self._items[key]
                return True
            return False

    def list_items(self) -> List[str]:
        """列出所有项目"""
        with self._item_lock:
            return list(self._items.keys())

    def clear_items(self):
        """清空所有项目"""
        with self._item_lock:
            self._items.clear()

    def _initialize_component(self):
        """初始化管理器"""
        self.clear_items()

    def _shutdown_component(self):
        """关闭管理器"""
        self.clear_items()


__all__ = [
    'BaseInfrastructureComponent',
    'BaseServiceComponent',
    'BaseManagerComponent'
]
