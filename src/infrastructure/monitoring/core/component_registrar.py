#!/usr/bin/env python3
"""
RQA2025 基础设施层组件注册器

负责组件的注册和发现功能。
这是从ComponentRegistry中拆分出来的注册管理组件。
"""

import logging
import inspect
from typing import Dict, Any, Optional, List, Type
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class ComponentRegistrar:
    """
    组件注册器

    负责组件的注册、注销和发现功能。
    """

    def __init__(self):
        """初始化组件注册器"""
        self._components: Dict[str, Type] = {}  # 注册的组件类
        self._metadata: Dict[str, Dict[str, Any]] = {}  # 组件元数据
        self._lock = threading.RLock()

        logger.info("组件注册器初始化完成")

    def register_component(self, name: str, component_class: Type,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        注册一个组件类

        Args:
            name: 组件的唯一名称
            component_class: 组件的类
            metadata: 组件元数据

        Returns:
            bool: 是否注册成功
        """
        with self._lock:
            try:
                if name in self._components:
                    logger.warning(f"组件 {name} 已经注册，将覆盖现有注册")
                    return False

                self._components[name] = component_class

                # 存储元数据
                if metadata:
                    self._metadata[name] = metadata
                else:
                    # 尝试从类中提取元数据
                    self._metadata[name] = {
                        'name': name,
                        'component_type': getattr(component_class, '__name__', 'unknown'),
                        'version': getattr(component_class, 'VERSION', '0.1.0'),
                        'description': getattr(component_class, '__doc__', 'No description'),
                        'capabilities': getattr(component_class, 'CAPABILITIES', []),
                        'dependencies': getattr(component_class, 'DEPENDENCIES', [])
                    }

                logger.info(f"组件 {name} ({component_class.__name__}) 已注册")
                return True

            except Exception as e:
                logger.error(f"注册组件 {name} 失败: {e}")
                return False

    def unregister_component(self, name: str) -> bool:
        """
        注销一个组件

        Args:
            name: 组件名称

        Returns:
            bool: 是否注销成功
        """
        with self._lock:
            try:
                if name in self._components:
                    del self._components[name]
                    if name in self._metadata:
                        del self._metadata[name]
                    logger.info(f"组件 {name} 已注销")
                    return True
                else:
                    logger.warning(f"组件 {name} 未找到，无法注销")
                    return False

            except Exception as e:
                logger.error(f"注销组件 {name} 失败: {e}")
                return False

    def get_component(self, name: str) -> Optional[Type]:
        """
        获取已注册的组件类

        Args:
            name: 组件名称

        Returns:
            Optional[Type]: 组件类，如果未找到则返回None
        """
        with self._lock:
            return self._components.get(name)

    def list_components(self) -> List[Dict[str, Any]]:
        """
        列出所有注册组件的元数据

        Returns:
            List[Dict[str, Any]]: 组件元数据列表
        """
        with self._lock:
            return list(self._metadata.values())

    def find_components_by_capability(self, capability: str) -> List[str]:
        """
        根据能力查找组件

        Args:
            capability: 能力名称

        Returns:
            List[str]: 具有该能力的组件名称列表
        """
        with self._lock:
            matching_components = []
            for name, metadata in self._metadata.items():
                if capability in metadata.get('capabilities', []):
                    matching_components.append(name)
            return matching_components

    def find_components_by_type(self, component_type: str) -> List[str]:
        """
        根据类型查找组件

        Args:
            component_type: 组件类型

        Returns:
            List[str]: 该类型的组件名称列表
        """
        with self._lock:
            matching_components = []
            for name, metadata in self._metadata.items():
                if metadata.get('component_type') == component_type:
                    matching_components.append(name)
            return matching_components

    def get_component_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取组件元数据

        Args:
            name: 组件名称

        Returns:
            Optional[Dict[str, Any]]: 组件元数据
        """
        with self._lock:
            return self._metadata.get(name)

    def update_component_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """
        更新组件元数据

        Args:
            name: 组件名称
            metadata: 新的元数据

        Returns:
            bool: 是否更新成功
        """
        with self._lock:
            try:
                if name not in self._metadata:
                    logger.warning(f"组件 {name} 未找到，无法更新元数据")
                    return False

                self._metadata[name].update(metadata)
                logger.info(f"组件 {name} 元数据已更新")
                return True

            except Exception as e:
                logger.error(f"更新组件 {name} 元数据失败: {e}")
                return False

    def get_registered_count(self) -> int:
        """
        获取已注册组件数量

        Returns:
            int: 注册组件数量
        """
        with self._lock:
            return len(self._components)

    def is_registered(self, name: str) -> bool:
        """
        检查组件是否已注册

        Args:
            name: 组件名称

        Returns:
            bool: 是否已注册
        """
        with self._lock:
            return name in self._components

    def get_registration_summary(self) -> Dict[str, Any]:
        """
        获取注册摘要

        Returns:
            Dict[str, Any]: 注册统计信息
        """
        with self._lock:
            # 按类型统计
            type_counts = {}
            capability_counts = {}

            for metadata in self._metadata.values():
                comp_type = metadata.get('component_type', 'unknown')
                type_counts[comp_type] = type_counts.get(comp_type, 0) + 1

                for capability in metadata.get('capabilities', []):
                    capability_counts[capability] = capability_counts.get(capability, 0) + 1

            return {
                'total_registered': len(self._components),
                'by_type': type_counts,
                'by_capability': capability_counts,
                'component_names': list(self._components.keys())
            }

    def clear_all_registrations(self) -> int:
        """
        清空所有注册

        Returns:
            int: 清空的注册数量
        """
        with self._lock:
            cleared_count = len(self._components)
            self._components.clear()
            self._metadata.clear()
            logger.info(f"已清空所有注册，共 {cleared_count} 个组件")
            return cleared_count

    def validate_registration(self, name: str, component_class: Type) -> List[str]:
        """
        验证组件注册

        Args:
            name: 组件名称
            component_class: 组件类

        Returns:
            List[str]: 验证错误列表
        """
        errors = []

        # 检查名称
        if not name or not isinstance(name, str):
            errors.append("组件名称必须是非空字符串")

        # 检查类
        if not inspect.isclass(component_class):
            errors.append("必须提供有效的类对象")

        # 检查是否已注册
        if self.is_registered(name):
            errors.append(f"组件 '{name}' 已被注册")

        # 检查类是否有必需的方法
        required_methods = ['__init__']
        for method in required_methods:
            if not hasattr(component_class, method):
                errors.append(f"组件类缺少必需方法: {method}")

        return errors

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取注册器健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            summary = self.get_registration_summary()

            issues = []

            if summary['total_registered'] == 0:
                issues.append("没有已注册的组件")

            # 检查是否有重复的能力声明（可能表示设计问题）
            capability_counts = summary['by_capability']
            duplicate_capabilities = [
                cap for cap, count in capability_counts.items() if count > 3
            ]
            if duplicate_capabilities:
                issues.append(f"能力重复声明过多: {duplicate_capabilities}")

            return {
                'status': 'healthy' if not issues else 'warning',
                'total_registered': summary['total_registered'],
                'issues': issues,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局组件注册器实例
global_component_registrar = ComponentRegistrar()
