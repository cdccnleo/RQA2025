#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略服务注册和发现
Strategy Service Registry and Discovery

提供策略服务的动态注册、发现和管理功能。
"""

import asyncio
from typing import Dict, List, Any, Optional, Type, Callable
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import importlib
import pkgutil

logger = logging.getLogger(__name__)


class ServiceMetadata:

    """
    服务元数据
    Service Metadata

    描述服务的元信息。
    """

    def __init__(self, service_id: str, service_type: str,


                 version: str = "1.0.0", description: str = "",
                 dependencies: List[str] = None, tags: List[str] = None):
        """
        初始化服务元数据

        Args:
            service_id: 服务ID
            service_type: 服务类型
            version: 版本号
            description: 描述
            dependencies: 依赖服务列表
            tags: 标签列表
        """
        self.service_id = service_id
        self.service_type = service_type
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        self.tags = tags or []
        self.registered_at = datetime.now()
        self.last_heartbeat = datetime.now()
        self.status = "registered"
        self.instance = None
        self.health_check: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            Dict[str, Any]: 字典格式的元数据
        """
        return {
            'service_id': self.service_id,
            'service_type': self.service_type,
            'version': self.version,
            'description': self.description,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'registered_at': self.registered_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'status': self.status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceMetadata':
        """
        从字典格式创建

        Args:
            data: 字典格式数据

        Returns:
            ServiceMetadata: 服务元数据实例
        """
        metadata = cls(
            service_id=data['service_id'],
            service_type=data['service_type'],
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            dependencies=data.get('dependencies', []),
            tags=data.get('tags', [])
        )

        metadata.registered_at = datetime.fromisoformat(data['registered_at'])
        metadata.last_heartbeat = datetime.fromisoformat(data['last_heartbeat'])
        metadata.status = data.get('status', 'registered')

        return metadata

    def update_heartbeat(self):
        """更新心跳时间"""
        self.last_heartbeat = datetime.now()

    def is_healthy(self) -> bool:
        """
        检查服务是否健康

        Returns:
            bool: 是否健康
        """
        # 检查心跳是否超时（30秒）
        heartbeat_timeout = timedelta(seconds=30)
        if datetime.now() - self.last_heartbeat > heartbeat_timeout:
            self.status = "unhealthy"
            return False

        # 执行健康检查
        if self.health_check:
            try:
                return self.health_check()
            except Exception as e:
                logger.error(f"健康检查失败 {self.service_id}: {e}")
                self.status = "unhealthy"
                return False

        self.status = "healthy"
        return True


class ServiceRegistry:

    """
    服务注册表
    Service Registry

    管理所有已注册的服务实例。
    """

    def __init__(self, registry_path: str = "./data / service_registry"):
        """
        初始化服务注册表

        Args:
            registry_path: 注册表存储路径
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # 服务存储
        self.services: Dict[str, ServiceMetadata] = {}

        # 服务类型索引
        self.type_index: Dict[str, List[str]] = {}

        # 标签索引
        self.tag_index: Dict[str, List[str]] = {}

        # 监控任务
        self.monitoring_task: Optional[asyncio.Task] = None

        # 自动发现配置
        self.auto_discovery_enabled = True
        self.discovery_interval = 60  # 60秒

        logger.info(f"服务注册表初始化完成，存储路径: {self.registry_path}")

    async def register_service(self, service_id: str, service_type: str,
                               service_instance: Any, metadata: Dict[str, Any] = None) -> bool:
        """
        注册服务

        Args:
            service_id: 服务ID
            service_type: 服务类型
            service_instance: 服务实例
            metadata: 服务元数据

        Returns:
            bool: 注册是否成功
        """
        try:
            if service_id in self.services:
                logger.warning(f"服务 {service_id} 已存在，将更新")
                await self.unregister_service(service_id)

            # 创建服务元数据
            service_metadata = ServiceMetadata(
                service_id=service_id,
                service_type=service_type,
                version=metadata.get('version', '1.0.0') if metadata else '1.0.0',
                description=metadata.get('description', '') if metadata else '',
                dependencies=metadata.get('dependencies', []) if metadata else [],
                tags=metadata.get('tags', []) if metadata else []
            )

            service_metadata.instance = service_instance

            # 设置健康检查函数
            if hasattr(service_instance, 'health_check'):
                service_metadata.health_check = service_instance.health_check

            # 存储服务
            self.services[service_id] = service_metadata

            # 更新索引
            self._update_indices(service_metadata)

            # 保存到持久化存储
            await self._persist_service(service_metadata)

            # 发布注册事件
            await self._publish_event("service_registered", {
                "service_id": service_id,
                "service_type": service_type,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"服务注册成功: {service_id} ({service_type})")
            return True

        except Exception as e:
            logger.error(f"服务注册失败: {e}")
            return False

    async def unregister_service(self, service_id: str) -> bool:
        """
        注销服务

        Args:
            service_id: 服务ID

        Returns:
            bool: 注销是否成功
        """
        try:
            if service_id not in self.services:
                logger.warning(f"服务 {service_id} 不存在")
                return False

            service_metadata = self.services[service_id]

            # 从索引中移除
            self._remove_from_indices(service_metadata)

            # 删除持久化存储
            await self._delete_persisted_service(service_id)

            # 删除服务
            del self.services[service_id]

            # 发布注销事件
            await self._publish_event("service_unregistered", {
                "service_id": service_id,
                "service_type": service_metadata.service_type,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"服务注销成功: {service_id}")
            return True

        except Exception as e:
            logger.error(f"服务注销失败: {e}")
            return False

    def get_service(self, service_id: str) -> Optional[Any]:
        """
        获取服务实例

        Args:
            service_id: 服务ID

        Returns:
            Optional[Any]: 服务实例
        """
        if service_id in self.services:
            return self.services[service_id].instance
        return None

    def get_service_metadata(self, service_id: str) -> Optional[ServiceMetadata]:
        """
        获取服务元数据

        Args:
            service_id: 服务ID

        Returns:
            Optional[ServiceMetadata]: 服务元数据
        """
        return self.services.get(service_id)

    def discover_services(self, service_type: str = None,


                          tags: List[str] = None) -> List[ServiceMetadata]:
        """
        发现服务

        Args:
            service_type: 服务类型过滤器
            tags: 标签过滤器

        Returns:
            List[ServiceMetadata]: 服务元数据列表
        """
        candidates = []

        if service_type:
            # 按类型过滤
            if service_type in self.type_index:
                service_ids = self.type_index[service_type]
                candidates = [self.services[sid] for sid in service_ids if sid in self.services]

        if tags:
            # 按标签过滤
            tag_candidates = []
            for tag in tags:
                if tag in self.tag_index:
                    tag_candidates.extend(self.tag_index[tag])

            if candidates:
                # 取交集
                tag_service_ids = set(tag_candidates)
                candidates = [s for s in candidates if s.service_id in tag_service_ids]
            else:
                # 只有标签过滤
                tag_service_ids = set(tag_candidates)
                candidates = [self.services[sid] for sid in tag_service_ids if sid in self.services]

        if not service_type and not tags:
            # 返回所有服务
            candidates = list(self.services.values())

        # 过滤健康的服务
        healthy_services = [s for s in candidates if s.is_healthy()]

        return healthy_services

    def get_service_types(self) -> List[str]:
        """
        获取所有服务类型

        Returns:
            List[str]: 服务类型列表
        """
        return list(self.type_index.keys())

    def get_service_tags(self) -> List[str]:
        """
        获取所有标签

        Returns:
            List[str]: 标签列表
        """
        return list(self.tag_index.keys())

    def _update_indices(self, service_metadata: ServiceMetadata):
        """
        更新索引

        Args:
            service_metadata: 服务元数据
        """
        service_id = service_metadata.service_id
        service_type = service_metadata.service_type
        tags = service_metadata.tags

        # 更新类型索引
        if service_type not in self.type_index:
            self.type_index[service_type] = []
        if service_id not in self.type_index[service_type]:
            self.type_index[service_type].append(service_id)

        # 更新标签索引
        for tag in tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
        if service_id not in self.tag_index[tag]:
            self.tag_index[tag].append(service_id)

    def _remove_from_indices(self, service_metadata: ServiceMetadata):
        """
        从索引中移除

        Args:
            service_metadata: 服务元数据
        """
        service_id = service_metadata.service_id
        service_type = service_metadata.service_type
        tags = service_metadata.tags

        # 从类型索引中移除
        if service_type in self.type_index:
            if service_id in self.type_index[service_type]:
                self.type_index[service_type].remove(service_id)
        if not self.type_index[service_type]:
            del self.type_index[service_type]

        # 从标签索引中移除
        for tag in tags:
            if tag in self.tag_index:
                if service_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(service_id)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]

    async def _persist_service(self, service_metadata: ServiceMetadata):
        """
        持久化服务信息

        Args:
            service_metadata: 服务元数据
        """
        try:
            file_path = self.registry_path / f"{service_metadata.service_id}.json"
            data = service_metadata.to_dict()

            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"持久化服务失败: {e}")

    async def _delete_persisted_service(self, service_id: str):
        """
        删除持久化服务信息

        Args:
            service_id: 服务ID
        """
        try:
            file_path = self.registry_path / f"{service_id}.json"
            if file_path.exists():
                file_path.unlink()

        except Exception as e:
            logger.error(f"删除持久化服务失败: {e}")

    async def load_persisted_services(self):
        """
        加载持久化服务信息
        """
        try:
            for file_path in self.registry_path.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf - 8') as f:
                        data = json.load(f)

                    service_metadata = ServiceMetadata.from_dict(data)
                    service_metadata.status = "restored"  # 标记为已恢复

                    self.services[service_metadata.service_id] = service_metadata
                    self._update_indices(service_metadata)

                    logger.info(f"恢复服务: {service_metadata.service_id}")

                except Exception as e:
                    logger.error(f"加载服务文件失败 {file_path}: {e}")

        except Exception as e:
            logger.error(f"加载持久化服务失败: {e}")

    async def start_monitoring(self):
        """
        启动服务监控
        """
        if self.monitoring_task and not self.monitoring_task.done():
            return

        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("服务监控已启动")

    async def stop_monitoring(self):
        """
        停止服务监控
        """
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("服务监控已停止")

    async def _monitoring_loop(self):
        """
        监控循环
        """
        while True:
            try:
                # 检查所有服务的健康状态
                unhealthy_services = []

                for service_id, service_metadata in self.services.items():
                    if not service_metadata.is_healthy():
                        unhealthy_services.append(service_id)

                        # 发布健康检查失败事件
                        await self._publish_event("service_unhealthy", {
                            "service_id": service_id,
                            "service_type": service_metadata.service_type,
                            "timestamp": datetime.now().isoformat()
                        })

                if unhealthy_services:
                    logger.warning(f"发现不健康服务: {unhealthy_services}")

                # 等待下一次检查
                await asyncio.sleep(self.discovery_interval)

            except asyncio.CancelledError:
                logger.info("服务监控循环被取消")
                break
            except Exception as e:
                logger.error(f"服务监控循环异常: {e}")
                await asyncio.sleep(5)

    async def _publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        发布事件

        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        try:
            # 这里可以集成事件总线
            # 暂时只记录日志
            logger.info(f"服务事件: {event_type} - {event_data}")

        except Exception as e:
            logger.error(f"事件发布异常: {e}")

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        获取注册表统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        total_services = len(self.services)
        service_types = len(self.type_index)
        tags = len(self.tag_index)

        healthy_count = sum(1 for s in self.services.values() if s.is_healthy())
        unhealthy_count = total_services - healthy_count

        return {
            'total_services': total_services,
            'service_types': service_types,
            'tags': tags,
            'healthy_services': healthy_count,
            'unhealthy_services': unhealthy_count,
            'registry_path': str(self.registry_path)
        }


# 策略服务发现器

class StrategyServiceDiscovery:

    """
    策略服务发现器
    Strategy Service Discovery

    自动发现和注册策略服务组件。
    """

    def __init__(self, registry: ServiceRegistry):
        """
        初始化服务发现器

        Args:
            registry: 服务注册表
        """
        self.registry = registry
        self.discovered_modules = set()

        logger.info("策略服务发现器初始化完成")

    async def discover_and_register_services(self, base_package: str = "src.strategy"):
        """
        发现并注册服务

        Args:
            base_package: 基础包名
        """
        try:
            logger.info(f"开始发现服务包: {base_package}")

            # 发现所有策略服务模块
            await self._discover_strategy_modules(base_package)

            # 注册发现的服务
            await self._register_discovered_services()

            logger.info("服务发现和注册完成")

        except Exception as e:
            logger.error(f"服务发现失败: {e}")

    async def _discover_strategy_modules(self, base_package: str):
        """
        发现策略服务模块

        Args:
            base_package: 基础包名
        """
        try:
            # 动态导入策略服务模块
            package = importlib.import_module(base_package)

            # 遍历所有子模块
            for importer, modname, ispkg in pkgutil.walk_packages(
                package.__path__, package.__name__ + "."
            ):
                if ispkg:
                    continue

                try:
                    module = importlib.import_module(modname)
                    self.discovered_modules.add(module)

                    logger.debug(f"发现模块: {modname}")

                except Exception as e:
                    logger.error(f"导入模块失败 {modname}: {e}")

        except Exception as e:
            logger.error(f"发现策略模块失败: {e}")

    async def _register_discovered_services(self):
        """
        注册发现的服务
        """
        for module in self.discovered_modules:
            try:
                await self._register_module_services(module)

            except Exception as e:
                logger.error(f"注册模块服务失败 {module.__name__}: {e}")

    async def _register_module_services(self, module):
        """
        注册模块中的服务

        Args:
            module: 模块对象
        """
        # 检查模块中是否有服务类
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)

                # 检查是否是服务类
                if (isinstance(attr, type)
                    and hasattr(attr, '__bases__')
                    and any(base.__name__ in ['IStrategyService', 'IBacktestService',
                                              'IOptimizationService', 'IMonitoringService']
                            for base in attr.__bases__)):

                    # 创建服务ID
                    service_id = f"{module.__name__}.{attr_name}"

                    # 确定服务类型
                    service_type = self._determine_service_type(attr)

                    if service_type:
                        # 创建服务实例
                        try:
                            service_instance = attr()

                            # 注册服务
                            metadata = {
                                'version': getattr(module, '__version__', '1.0.0'),
                                'description': getattr(attr, '__doc__', '').strip() if attr.__doc__ else '',
                                'tags': ['auto_discovered']
                            }

                            await self.registry.register_service(
                                service_id, service_type, service_instance, metadata
                            )

                            logger.info(f"自动注册服务: {service_id}")

                        except Exception as e:
                            logger.error(f"创建服务实例失败 {service_id}: {e}")

            except Exception as e:
                logger.error(f"检查模块属性失败 {attr_name}: {e}")

    def _determine_service_type(self, service_class: Type) -> Optional[str]:
        """
        确定服务类型

        Args:
            service_class: 服务类

        Returns:
            Optional[str]: 服务类型
        """
        # 检查基类来确定服务类型
        for base in service_class.__bases__:
            base_name = base.__name__

            if base_name == 'IStrategyService':
                return 'strategy_service'
            elif base_name == 'IBacktestService':
                return 'backtest_service'
            elif base_name == 'IOptimizationService':
                return 'optimization_service'
            elif base_name == 'IMonitoringService':
                return 'monitoring_service'

        return None


# 便捷函数
async def create_service_registry() -> ServiceRegistry:
    """
    创建服务注册表

    Returns:
        ServiceRegistry: 服务注册表实例
    """
    registry = ServiceRegistry()
    await registry.load_persisted_services()
    await registry.start_monitoring()
    return registry


async def create_service_discovery(registry: ServiceRegistry) -> StrategyServiceDiscovery:
    """
    创建服务发现器

    Args:
        registry: 服务注册表

    Returns:
        StrategyServiceDiscovery: 服务发现器实例
    """
    discovery = StrategyServiceDiscovery(registry)
    await discovery.discover_and_register_services()
    return discovery


# 导出
__all__ = [
    'ServiceMetadata',
    'ServiceRegistry',
    'StrategyServiceDiscovery',
    'create_service_registry',
    'create_service_discovery'
]
