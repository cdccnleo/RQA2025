"""
数据适配器注册模块
管理所有数据适配器的注册和发现
"""

from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
from enum import Enum
import logging

from .base import BaseAdapter, AdapterConfig


logger = logging.getLogger(__name__)


class AdapterStatus(Enum):

    """适配器状态枚举"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class AdapterInfo:

    """适配器信息类"""
    name: str
    adapter_type: str
    status: AdapterStatus
    version: str = "1.0.0"
    description: str = ""
    capabilities: List[str] = None
    config_schema: Dict[str, Any] = None

    def __post_init__(self):

        if self.capabilities is None:
            self.capabilities = []
        if self.config_schema is None:
            self.config_schema = {}


class AdapterRegistry:

    """适配器注册管理器"""

    def __init__(self):

        self._adapters: Dict[str, Type[BaseAdapter]] = {}
        self._adapter_infos: Dict[str, AdapterInfo] = {}
        self._active_adapters: Dict[str, BaseAdapter] = {}

    def register_adapter(self, name: str, adapter_class: Type[BaseAdapter],


                         info: AdapterInfo) -> bool:
        """注册适配器"""
        try:
            if name in self._adapters:
                logger.warning(f"适配器 {name} 已被注册，将被覆盖")

            self._adapters[name] = adapter_class
            self._adapter_infos[name] = info

            logger.info(f"适配器 {name} 注册成功")
            return True

        except Exception as e:
            logger.error(f"注册适配器 {name} 失败: {str(e)}")
            return False

    def unregister_adapter(self, name: str) -> bool:
        """注销适配器"""
        try:
            if name in self._active_adapters:
                # 先停止活跃适配器
                self.deactivate_adapter(name)

            if name in self._adapters:
                del self._adapters[name]
                del self._adapter_infos[name]
                logger.info(f"适配器 {name} 注销成功")
                return True
            else:
                logger.warning(f"适配器 {name} 未注册")
                return False

        except Exception as e:
            logger.error(f"注销适配器 {name} 失败: {str(e)}")
            return False

    def get_adapter_class(self, name: str) -> Optional[Type[BaseAdapter]]:
        """获取适配器类"""
        return self._adapters.get(name)

    def get_adapter_info(self, name: str) -> Optional[AdapterInfo]:
        """获取适配器信息"""
        return self._adapter_infos.get(name)

    def list_adapters(self) -> List[str]:
        """列出所有注册的适配器"""
        return list(self._adapters.keys())

    def list_adapters_by_type(self, adapter_type: str) -> List[str]:
        """按类型列出适配器"""
        return [
            name for name, info in self._adapter_infos.items()
            if info.adapter_type == adapter_type
        ]

    def create_adapter(self, name: str, config: AdapterConfig) -> Optional[BaseAdapter]:
        """创建适配器实例"""
        try:
            adapter_class = self.get_adapter_class(name)
            if adapter_class:
                adapter = adapter_class(config)
                self._active_adapters[name] = adapter
                logger.info(f"适配器 {name} 创建成功")
                return adapter
            else:
                logger.error(f"适配器 {name} 未注册")
                return None

        except Exception as e:
            logger.error(f"创建适配器 {name} 失败: {str(e)}")
            return None

    def activate_adapter(self, name: str, config: AdapterConfig) -> bool:
        """激活适配器"""
        try:
            if name in self._active_adapters:
                logger.info(f"适配器 {name} 已激活")
                return True

            adapter = self.create_adapter(name, config)
            if adapter and adapter.connect():
                logger.info(f"适配器 {name} 激活成功")
                return True
            else:
                logger.error(f"适配器 {name} 激活失败")
                return False

        except Exception as e:
            logger.error(f"激活适配器 {name} 失败: {str(e)}")
            return False

    def deactivate_adapter(self, name: str) -> bool:
        """停用适配器"""
        try:
            if name in self._active_adapters:
                adapter = self._active_adapters[name]
                if adapter.disconnect():
                    del self._active_adapters[name]
                    logger.info(f"适配器 {name} 停用成功")
                    return True
                else:
                    logger.error(f"适配器 {name} 断开连接失败")
                    return False
            else:
                logger.warning(f"适配器 {name} 未激活")
                return False

        except Exception as e:
            logger.error(f"停用适配器 {name} 失败: {str(e)}")
            return False

    def get_active_adapters(self) -> List[str]:
        """获取活跃适配器列表"""
        return list(self._active_adapters.keys())

    def get_adapter_status(self, name: str) -> AdapterStatus:
        """获取适配器状态"""
        if name in self._active_adapters:
            adapter = self._active_adapters[name]
            if adapter.is_connected():
                return AdapterStatus.AVAILABLE
            else:
                return AdapterStatus.ERROR
        elif name in self._adapters:
            return AdapterStatus.UNAVAILABLE
        else:
            return AdapterStatus.ERROR

    def get_registry_info(self) -> Dict[str, Any]:
        """获取注册表信息"""
        return {
            'total_adapters': len(self._adapters),
            'active_adapters': len(self._active_adapters),
            'adapter_types': list(set(
                info.adapter_type for info in self._adapter_infos.values()
            )),
            'adapters': {
                name: {
                    'type': info.adapter_type,
                    'status': self.get_adapter_status(name).value,
                    'version': info.version
                }
                for name, info in self._adapter_infos.items()
            }
        }


# 全局适配器注册实例
_adapter_registry = AdapterRegistry()


def get_adapter_registry() -> AdapterRegistry:
    """获取全局适配器注册实例"""
    return _adapter_registry
