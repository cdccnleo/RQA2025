#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据层服务发现管理器

提供数据层服务的自动注册、发现和管理功能，
实现完整的服务治理体系。

设计模式：服务发现模式 + 注册表模式
职责：统一管理数据层服务的注册、发现和生命周期
"""

from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from dataclasses import dataclass

from src.core.integration import get_data_adapter
from ..interfaces.standard_interfaces import DataSourceType


@dataclass
class ServiceRegistration:

    """服务注册信息"""
    service_type: Type
    implementation: Type
    data_type: Optional[DataSourceType]
    singleton: bool
    config: Dict[str, Any]
    registered_at: datetime
    health_check_registered: bool = False


@dataclass
class ServiceDiscoveryResult:

    """服务发现结果"""
    service_name: str
    service_instance: Any
    data_type: Optional[DataSourceType]
    found: bool
    source: str  # 'local', 'infrastructure', 'not_found'
    timestamp: datetime


class DataServiceDiscoveryManager:

    """
    数据层服务发现管理器

    提供数据层服务的自动注册、发现和管理功能：
    - 自动服务注册和发现
    - 健康检查集成
    - 服务依赖管理
    - 事件驱动的服务管理
    - 服务生命周期管理
    """

    def __init__(self):
        """
        初始化服务发现管理器 - 使用统一基础设施集成层
        """
        # 初始化统一基础设施集成层
        try:
            data_adapter = get_data_adapter()
            self.cache_manager = data_adapter.get_cache_manager()
            self.config_manager = data_adapter.get_config_manager()
            self.logger = data_adapter.get_logger()
            self.event_bus = data_adapter.get_event_bus()
            self.health_checker = data_adapter.get_health_checker()
        except Exception as e:
            # 降级处理
            import logging
            self.cache_manager = None
            self.config_manager = None
            self.logger = logging.getLogger(__name__)
            self.event_bus = None
            self.health_checker = None

        # 服务注册记录
        self._service_registrations: Dict[str, ServiceRegistration] = {}
        self._service_discovery_cache: Dict[str, ServiceDiscoveryResult] = {}
        self._service_instances: Dict[str, Any] = {}  # 简化的服务容器

        # 服务依赖图
        self._service_dependencies: Dict[str, List[str]] = {}
        self._reverse_dependencies: Dict[str, List[str]] = {}

        # 初始化标准服务配置
        self._standard_service_configs = self._initialize_standard_configs()

        # 注册事件处理器
        self._register_event_handlers()

        self._log_operation('initialized', 'DataServiceDiscoveryManager', 'success')

    def _initialize_standard_configs(self) -> Dict[str, Dict[str, Any]]:
        """初始化标准服务配置"""
        return {
            'DataManager': {
                'class': 'src.data.data_manager.DataManager',
                'singleton': True,
                'config': {
                    'max_concurrent': 10,
                    'enable_cache': True,
                    'enable_validation': True,
                    'enable_quality_monitor': True
                },
                'dependencies': ['DataCache', 'DataValidator', 'DataQualityMonitor']
            },
            'DataCache': {
                'class': 'src.data.cache.data_cache.DataCache',
                'singleton': True,
                'config': {
                    'max_size': 10000,
                    'ttl': 3600,
                    'enable_compression': True
                },
                'dependencies': []
            },
            'DataValidator': {
                'class': 'src.data.validation.validator_components.ValidatorComponent',
                'singleton': True,
                'config': {
                    'strict_mode': False,
                    'enable_schema_validation': True
                },
                'dependencies': []
            },
            'DataQualityMonitor': {
                'class': 'src.data.quality.unified_quality_monitor.UnifiedQualityMonitor',
                'singleton': True,
                'config': {
                    'enable_realtime_monitoring': True,
                    'alert_threshold': 0.8
                },
                'dependencies': []
            },
            'AsyncDataProcessor': {
                'class': 'src.data.parallel.async_data_processor.AsyncDataProcessor',
                'singleton': True,
                'config': {
                    'max_concurrent': 20,
                    'enable_metrics': True
                },
                'dependencies': ['DataManager']
            }
        }

    def _register_event_handlers(self):
        """注册事件处理器"""
        if self.event_bus:
            # 注册服务状态变更事件处理器
            self.event_bus.subscribe('service_status_changed', self._handle_service_status_change)

            # 注册健康检查事件处理器
            self.event_bus.subscribe('health_check_completed', self._handle_health_check_event)

    # =========================================================================
    # 核心服务管理功能
    # =========================================================================

    def register_standard_services(self) -> Dict[str, Any]:
        """
        注册所有标准数据层服务

        Returns:
            注册结果
        """
        results = {
            'registered': [],
            'failed': [],
            'timestamp': datetime.now().isoformat()
        }

        try:
            for service_name, config in self._standard_service_configs.items():
                try:
                    # 动态导入服务类
                    module_path, class_name = config['class'].rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    service_class = getattr(module, class_name)

                    # 注册服务到简化的服务容器
                    try:
                        self._service_instances[service_name] = service_class
                        success = True
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"服务注册失败 {service_name}: {e}")
                        success = False

                    if success:
                        # 记录注册信息
                        self._service_registrations[service_name] = ServiceRegistration(
                            service_type=type(service_name, (), {}),  # 创建类型占位符
                            implementation=service_class,
                            data_type=None,
                            singleton=config['singleton'],
                            config=config['config'],
                            registered_at=datetime.now()
                        )

                        # 注册健康检查
                        self._register_service_health_check(service_name, service_class)

                        results['registered'].append(service_name)
                        self._log_operation('register_standard', service_name, 'success')
                    else:
                        results['failed'].append({'name': service_name, 'error': '注册失败'})
                        self._log_operation('register_standard', service_name, 'failed')

                except Exception as e:
                    results['failed'].append({'name': service_name, 'error': str(e)})
                    self._log_operation('register_standard', service_name, f'failed: {e}')

        except Exception as e:
            results['error'] = str(e)
            self._log_operation('register_standard_services', 'batch', f'failed: {e}')

        # 发布服务注册完成事件
        if self.event_bridge:
            self.event_bridge.publish_data_event(
                'services_registered',
                {'service_count': len(results['registered'])}
            )

        return results

    def register_data_adapter_services(self, adapters_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        注册数据适配器服务

        Args:
            adapters_config: 适配器配置字典

        Returns:
            注册结果
        """
        results = {
            'registered': [],
            'failed': [],
            'timestamp': datetime.now().isoformat()
        }

        try:
            for adapter_name, config in adapters_config.items():
                try:
                    # 动态导入适配器类
                    module_path, class_name = config['class'].rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    adapter_class = getattr(module, class_name)

                    # 获取数据类型
                    data_type_str = config.get('data_type', adapter_name.lower())
                    data_type = DataSourceType(data_type_str) if data_type_str in [
                        dt.value for dt in DataSourceType] else None

                    # 注册适配器服务到简化的服务容器
                    try:
                        self._service_instances[f"{adapter_name}Adapter"] = adapter_class
                        success = True
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"适配器服务注册失败 {adapter_name}: {e}")
                        success = False

                    if success:
                        results['registered'].append(f"{adapter_name}Adapter")
                        self._log_operation('register_adapter', adapter_name, 'success')
                    else:
                        results['failed'].append({'name': adapter_name, 'error': '注册失败'})
                        self._log_operation('register_adapter', adapter_name, 'failed')

                except Exception as e:
                    results['failed'].append({'name': adapter_name, 'error': str(e)})
                    self._log_operation('register_adapter', adapter_name, f'failed: {e}')

        except Exception as e:
            results['error'] = str(e)

        return results

    def discover_service(self, service_name: str, data_type: Optional[DataSourceType] = None,


                         use_cache: bool = True) -> ServiceDiscoveryResult:
        """
        发现服务

        Args:
            service_name: 服务名称
            data_type: 数据类型
            use_cache: 是否使用缓存

        Returns:
            服务发现结果
        """
        cache_key = f"{service_name}:{data_type.value if data_type else 'general'}"

        # 检查缓存
        if use_cache and cache_key in self._service_discovery_cache:
            cached_result = self._service_discovery_cache[cache_key]
            # 检查缓存是否过期（5分钟）
            if (datetime.now() - cached_result.timestamp).total_seconds() < 300:
                return cached_result

        try:
            # 首先尝试从简化的服务容器获取
            service_instance = self._service_instances.get(service_name)

            if service_instance is not None:
                result = ServiceDiscoveryResult(
                    service_name=service_name,
                    service_instance=service_instance,
                    data_type=data_type,
                    found=True,
                    source='local',
                    timestamp=datetime.now()
                )
            else:
                # 如果本地没有找到，返回未找到结果
                result = ServiceDiscoveryResult(
                    service_name=service_name,
                    service_instance=None,
                    data_type=data_type,
                    found=False,
                    source='not_found',
                    timestamp=datetime.now()
                )

            # 缓存结果
            if use_cache:
                self._service_discovery_cache[cache_key] = result

            self._log_operation('discover', service_name,
                                'success' if result.found else 'not_found')
            return result

        except Exception as e:
            self._log_operation('discover', service_name, f'failed: {e}')

            result = ServiceDiscoveryResult(
                service_name=service_name,
                service_instance=None,
                data_type=data_type,
                found=False,
                source='error',
                timestamp=datetime.now()
            )

            return result

    def get_service_status(self, service_name: str, data_type: Optional[DataSourceType] = None) -> Dict[str, Any]:
        """
        获取服务状态

        Args:
            service_name: 服务名称
            data_type: 数据类型

        Returns:
            服务状态信息
        """
        try:
            discovery_result = self.discover_service(service_name, data_type, use_cache=False)

            if not discovery_result.found:
                return {
                    'service_name': service_name,
                    'status': 'not_found',
                    'data_type': data_type.value if data_type else None,
                    'timestamp': datetime.now().isoformat()
                }

            service_instance = discovery_result.service_instance

            # 获取健康状态
            health_status = 'unknown'
            if hasattr(service_instance, 'is_healthy'):
                try:
                    health_status = 'healthy' if service_instance.is_healthy() else 'unhealthy'
                except Exception:
                    health_status = 'error'

            # 获取服务信息（简化实现）
            service_info = {
                'service_name': service_name,
                'data_type': data_type,
                'available': service_name in self._service_instances
            }

            return {
                'service_name': service_name,
                'data_type': data_type.value if data_type else None,
                'status': health_status,
                'source': discovery_result.source,
                'service_info': service_info,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'service_name': service_name,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # =========================================================================
    # 健康检查和监控
    # =========================================================================

    def _register_service_health_check(self, service_name: str, service_class: Type) -> None:
        """为服务注册健康检查"""
        try:

            def health_check_func():

                try:
                    discovery_result = self.discover_service(service_name, use_cache=False)
                    if discovery_result.found and hasattr(discovery_result.service_instance, 'health_check'):
                        return discovery_result.service_instance.health_check()
                    elif discovery_result.found:
                        return {'status': 'healthy', 'message': 'Service available but no health check implemented'}
                    else:
                        return {'status': 'unhealthy', 'message': 'Service not available'}
                except Exception as e:
                    return {'status': 'unhealthy', 'message': str(e)}

            # 注册健康检查（简化实现）
            if self.health_checker:
                try:
                    self.health_checker.register_check(
                        f"service_{service_name}",
                        health_check_func
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"健康检查注册失败 {service_name}: {e}")

            # 标记已注册健康检查
            if service_name in self._service_registrations:
                self._service_registrations[service_name].health_check_registered = True

            self._log_operation('health_check_register', service_name, 'success')

        except Exception as e:
            self._log_operation('health_check_register', service_name, f'failed: {e}')

    def get_services_health_status(self) -> Dict[str, Any]:
        """
        获取所有服务的健康状态

        Returns:
            服务健康状态汇总
        """
        health_status = {
            'services': {},
            'overall_status': 'healthy',
            'total_services': len(self._service_registrations),
            'healthy_services': 0,
            'unhealthy_services': 0,
            'timestamp': datetime.now().isoformat()
        }

        unhealthy_count = 0

        for service_name in self._service_registrations.keys():
            service_health = self.get_service_status(service_name)

            health_status['services'][service_name] = service_health

            if service_health['status'] == 'unhealthy':
                unhealthy_count += 1
                health_status['overall_status'] = 'unhealthy'
            elif service_health['status'] == 'healthy':
                health_status['healthy_services'] += 1
            elif health_status['overall_status'] == 'healthy':
                health_status['overall_status'] = 'warning'

        health_status['unhealthy_services'] = unhealthy_count

        return health_status

    # =========================================================================
    # 事件处理器
    # =========================================================================

    def _handle_service_status_change(self, event_data: Dict[str, Any]):
        """处理服务状态变更事件"""
        service_name = event_data.get('service_name')
        new_status = event_data.get('status')

        self._log_operation('status_change', service_name, new_status)

        # 清除相关缓存
        if service_name:
            cache_keys_to_remove = [
                key for key in self._service_discovery_cache.keys()
                if key.startswith(service_name)
            ]
            for key in cache_keys_to_remove:
                del self._service_discovery_cache[key]

    def _handle_health_check_event(self, event_data: Dict[str, Any]):
        """处理健康检查事件"""
        service_name = event_data.get('component', '').replace('service_', '')
        status = event_data.get('status')

        # 更新服务注册信息
        if service_name in self._service_registrations:
            # 这里可以添加更复杂的健康状态跟踪逻辑
            pass

        self._log_operation('health_check_event', service_name, status)

    # =========================================================================
    # 服务依赖管理
    # =========================================================================

    def get_service_dependencies(self, service_name: str) -> List[str]:
        """
        获取服务依赖

        Args:
            service_name: 服务名称

        Returns:
            依赖服务列表
        """
        return self._service_dependencies.get(service_name, [])

    def validate_service_dependencies(self) -> Dict[str, Any]:
        """
        验证所有服务依赖关系

        Returns:
            依赖验证结果
        """
        validation_result = {
            'valid': True,
            'issues': [],
            'checked_services': len(self._service_registrations),
            'timestamp': datetime.now().isoformat()
        }

        for service_name, dependencies in self._service_dependencies.items():
            for dep in dependencies:
                if dep not in self._service_registrations:
                    # 尝试发现依赖服务
                    discovery_result = self.discover_service(dep, use_cache=False)
                    if not discovery_result.found:
                        validation_result['valid'] = False
                        validation_result['issues'].append({
                            'service': service_name,
                            'missing_dependency': dep,
                            'type': 'undiscovered'
                        })

        return validation_result

    def get_service_dependency_graph(self) -> Dict[str, Any]:
        """
        获取服务依赖图

        Returns:
            服务依赖图
        """
        return {
            'services': list(self._service_registrations.keys()),
            'dependencies': self._service_dependencies.copy(),
            'reverse_dependencies': self._reverse_dependencies.copy(),
            'timestamp': datetime.now().isoformat()
        }

    # =========================================================================
    # 工具方法
    # =========================================================================

    def _log_operation(self, operation: str, service_name: str, status: str) -> None:
        """
        记录操作日志

        Args:
            operation: 操作类型
            service_name: 服务名称
            status: 操作状态
        """
        try:
            message = f"服务发现管理器 - {operation}: {service_name}, 状态: {status}"
            print(f"[DataServiceDiscoveryManager] {message}")

            # 可以通过事件总线发布日志事件
            if self.event_bus:
                event_data = {
                    'operation': operation,
                    'service_name': service_name,
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                }
                self.event_bus.publish('service_operation_log', event_data)

        except Exception:
            # 最后的降级方案
            print(f"[DataServiceDiscoveryManager] {operation}: {service_name} - {status}")

    def get_discovery_stats(self) -> Dict[str, Any]:
        """
        获取发现统计信息

        Returns:
            发现统计信息
        """
        return {
            'registered_services': len(self._service_registrations),
            'cached_discoveries': len(self._service_discovery_cache),
            'service_dependencies': len(self._service_dependencies),
            'timestamp': datetime.now().isoformat()
        }

    def clear_discovery_cache(self) -> None:
        """清除发现缓存"""
        self._service_discovery_cache.clear()
        self._log_operation('cache_clear', 'discovery_cache', 'success')

    def shutdown(self) -> Dict[str, Any]:
        """
        关闭服务发现管理器

        Returns:
            关闭结果
        """
        try:
            # 清除所有缓存
            self.clear_discovery_cache()
            self._service_registrations.clear()
            self._service_dependencies.clear()
            self._reverse_dependencies.clear()

            self._log_operation('shutdown', 'DataServiceDiscoveryManager', 'success')

            return {
                'status': 'success',
                'message': 'Service discovery manager shut down successfully',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Shutdown failed: {e}',
                'timestamp': datetime.now().isoformat()
            }
