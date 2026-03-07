#!/usr/bin/env python3
"""
RQA2025 基础设施层组件实例管理器

负责组件实例的创建、启动、停止和生命周期管理。
这是从ComponentRegistry中拆分出来的实例管理组件。
"""

import logging
from typing import Dict, Any, Optional, List, Type
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class ComponentInstance:
    """组件实例信息"""

    def __init__(self, metadata: Dict[str, Any], instance: Optional[Any] = None):
        """
        初始化组件实例

        Args:
            metadata: 组件元数据
            instance: 组件实例对象
        """
        self.metadata = metadata
        self.instance = instance
        self.config: Dict[str, Any] = {}
        self.is_active = False
        self.startup_time: Optional[datetime] = None
        self.shutdown_time: Optional[datetime] = None
        self.error_count = 0
        self.last_error: Optional[str] = None

    def create_instance(self, component_class: Type, config: Dict[str, Any]) -> Any:
        """
        创建组件实例

        Args:
            component_class: 组件类
            config: 实例配置

        Returns:
            Any: 创建的实例

        Raises:
            Exception: 创建失败时抛出异常
        """
        try:
            self.instance = component_class(**config)
            self.config = config
            self.is_active = False
            logger.info(f"组件实例 {self.metadata['name']} 创建成功")
            return self.instance
        except Exception as e:
            error_msg = f"创建组件实例 {self.metadata['name']} 失败: {e}"
            self.last_error = error_msg
            self.error_count += 1
            logger.error(error_msg)
            raise

    def start(self) -> bool:
        """
        启动组件实例

        Returns:
            bool: 是否启动成功
        """
        try:
            if hasattr(self.instance, 'start') and callable(self.instance.start):
                self.instance.start()
            self.is_active = True
            self.startup_time = datetime.now()
            logger.info(f"组件实例 {self.metadata['name']} 已启动")
            return True
        except Exception as e:
            error_msg = f"启动组件实例 {self.metadata['name']} 失败: {e}"
            self.last_error = error_msg
            self.error_count += 1
            logger.error(error_msg)
            return False

    def stop(self) -> bool:
        """
        停止组件实例

        Returns:
            bool: 是否停止成功
        """
        try:
            if hasattr(self.instance, 'stop') and callable(self.instance.stop):
                self.instance.stop()
            self.is_active = False
            self.shutdown_time = datetime.now()
            logger.info(f"组件实例 {self.metadata['name']} 已停止")
            return True
        except Exception as e:
            error_msg = f"停止组件实例 {self.metadata['name']} 失败: {e}"
            self.last_error = error_msg
            self.error_count += 1
            logger.error(error_msg)
            return False

    def restart(self) -> bool:
        """
        重启组件实例

        Returns:
            bool: 是否重启成功
        """
        try:
            if self.stop():
                return self.start()
            return False
        except Exception as e:
            error_msg = f"重启组件实例 {self.metadata['name']} 失败: {e}"
            self.last_error = error_msg
            self.error_count += 1
            logger.error(error_msg)
            return False

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新组件配置

        Args:
            new_config: 新配置

        Returns:
            bool: 是否更新成功
        """
        try:
            self.config.update(new_config)
            if hasattr(self.instance, 'update_config') and callable(self.instance.update_config):
                self.instance.update_config(new_config)
            logger.info(f"组件实例 {self.metadata['name']} 配置已更新")
            return True
        except Exception as e:
            error_msg = f"更新组件实例 {self.metadata['name']} 配置失败: {e}"
            self.last_error = error_msg
            self.error_count += 1
            logger.error(error_msg)
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        获取组件实例状态

        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            'name': self.metadata['name'],
            'is_active': self.is_active,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'shutdown_time': self.shutdown_time.isoformat() if self.shutdown_time else None,
            'current_config': self.config,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'uptime_seconds': (
                datetime.now() - self.startup_time
            ).total_seconds() if self.startup_time and self.is_active else 0
        }

    def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            # 检查实例是否存在
            if not self.instance:
                return {
                    'status': 'error',
                    'message': '实例未创建',
                    'timestamp': datetime.now().isoformat()
                }

            # 检查实例是否活跃
            if not self.is_active:
                return {
                    'status': 'inactive',
                    'message': '实例未启动',
                    'timestamp': datetime.now().isoformat()
                }

            # 调用实例的健康检查方法（如果有）
            if hasattr(self.instance, 'health_check') and callable(self.instance.health_check):
                return self.instance.health_check()

            # 默认健康检查
            return {
                'status': 'healthy',
                'message': '实例运行正常',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"健康检查失败: {e}"
            self.last_error = error_msg
            self.error_count += 1
            return {
                'status': 'error',
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            }


class ComponentInstanceManager:
    """
    组件实例管理器

    负责组件实例的创建、启动、停止、配置更新和健康监控。
    """

    def __init__(self):
        """初始化组件实例管理器"""
        self._instances: Dict[str, ComponentInstance] = {}  # 运行中的组件实例
        self._lock = threading.RLock()

        logger.info("组件实例管理器初始化完成")

    def create_instance(self, name: str, component_class: Type,
                       metadata: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Optional[ComponentInstance]:
        """
        创建组件实例

        Args:
            name: 组件名称
            component_class: 组件类
            metadata: 组件元数据
            config: 实例配置

        Returns:
            Optional[ComponentInstance]: 创建的实例
        """
        with self._lock:
            try:
                if name in self._instances:
                    logger.warning(f"组件实例 {name} 已存在，将重新创建")
                    self.stop_instance(name)

                instance = ComponentInstance(metadata)
                instance.create_instance(component_class, config or {})
                self._instances[name] = instance

                logger.info(f"组件实例 {name} 创建完成")
                return instance

            except Exception as e:
                logger.error(f"创建组件实例 {name} 失败: {e}")
                return None

    def start_instance(self, name: str) -> bool:
        """
        启动组件实例

        Args:
            name: 组件名称

        Returns:
            bool: 是否启动成功
        """
        with self._lock:
            instance = self._instances.get(name)
            if not instance:
                logger.error(f"组件实例 {name} 未找到")
                return False

            return instance.start()

    def stop_instance(self, name: str) -> bool:
        """
        停止组件实例

        Args:
            name: 组件名称

        Returns:
            bool: 是否停止成功
        """
        with self._lock:
            instance = self._instances.get(name)
            if not instance:
                logger.warning(f"组件实例 {name} 未找到")
                return False

            success = instance.stop()
            if success:
                del self._instances[name]
            return success

    def restart_instance(self, name: str) -> bool:
        """
        重启组件实例

        Args:
            name: 组件名称

        Returns:
            bool: 是否重启成功
        """
        with self._lock:
            instance = self._instances.get(name)
            if not instance:
                logger.error(f"组件实例 {name} 未找到")
                return False

            return instance.restart()

    def update_instance_config(self, name: str, new_config: Dict[str, Any]) -> bool:
        """
        更新组件实例配置

        Args:
            name: 组件名称
            new_config: 新配置

        Returns:
            bool: 是否更新成功
        """
        with self._lock:
            instance = self._instances.get(name)
            if not instance:
                logger.error(f"组件实例 {name} 未找到")
                return False

            return instance.update_config(new_config)

    def get_instance(self, name: str) -> Optional[ComponentInstance]:
        """
        获取组件实例

        Args:
            name: 组件名称

        Returns:
            Optional[ComponentInstance]: 组件实例
        """
        with self._lock:
            return self._instances.get(name)

    def get_instance_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取组件实例状态

        Args:
            name: 组件名称

        Returns:
            Optional[Dict[str, Any]]: 状态信息
        """
        with self._lock:
            instance = self._instances.get(name)
            return instance.get_status() if instance else None

    def list_instances(self) -> List[Dict[str, Any]]:
        """
        列出所有实例状态

        Returns:
            List[Dict[str, Any]]: 实例状态列表
        """
        with self._lock:
            return [instance.get_status() for instance in self._instances.values()]

    def get_active_instances(self) -> List[str]:
        """
        获取活跃实例名称列表

        Returns:
            List[str]: 活跃实例名称
        """
        with self._lock:
            return [name for name, instance in self._instances.items() if instance.is_active]

    def get_instance_count(self) -> Dict[str, int]:
        """
        获取实例数量统计

        Returns:
            Dict[str, int]: 统计信息
        """
        with self._lock:
            total = len(self._instances)
            active = sum(1 for instance in self._instances.values() if instance.is_active)
            inactive = total - active
            error_count = sum(instance.error_count for instance in self._instances.values())

            return {
                'total': total,
                'active': active,
                'inactive': inactive,
                'error_instances': sum(1 for instance in self._instances.values() if instance.error_count > 0),
                'total_errors': error_count
            }

    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        检查所有实例健康状态

        Returns:
            Dict[str, Dict[str, Any]]: 健康检查结果
        """
        with self._lock:
            results = {}
            for name, instance in self._instances.items():
                results[name] = instance.health_check()
            return results

    def stop_all_instances(self) -> Dict[str, bool]:
        """
        停止所有实例

        Returns:
            Dict[str, bool]: 停止结果
        """
        with self._lock:
            results = {}
            for name in list(self._instances.keys()):
                results[name] = self.stop_instance(name)
            return results

    def cleanup_failed_instances(self, max_errors: int = 5) -> int:
        """
        清理失败的实例

        Args:
            max_errors: 最大错误次数阈值

        Returns:
            int: 清理的实例数量
        """
        with self._lock:
            failed_instances = [
                name for name, instance in self._instances.items()
                if instance.error_count >= max_errors
            ]

            for name in failed_instances:
                logger.warning(f"清理失败实例: {name} (错误次数: {self._instances[name].error_count})")
                self.stop_instance(name)

            return len(failed_instances)

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取管理器健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            counts = self.get_instance_count()
            health_checks = self.health_check_all()

            issues = []

            if counts['total'] == 0:
                issues.append("没有运行中的实例")

            if counts['active'] == 0 and counts['total'] > 0:
                issues.append("所有实例都未激活")

            error_instances = [name for name, health in health_checks.items() if health['status'] == 'error']
            if error_instances:
                issues.append(f"存在错误实例: {error_instances}")

            # 计算健康评分
            if counts['total'] == 0:
                health_score = 100
            else:
                healthy_instances = sum(1 for health in health_checks.values() if health['status'] == 'healthy')
                health_score = int((healthy_instances / counts['total']) * 100)

            return {
                'status': 'healthy' if health_score >= 80 else 'warning' if health_score >= 50 else 'error',
                'health_score': health_score,
                'instance_counts': counts,
                'issues': issues,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局组件实例管理器实例
global_instance_manager = ComponentInstanceManager()
