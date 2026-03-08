"""
服务发现层

提供服务注册、服务发现和负载均衡功能。
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ServiceRegistry:

    """服务注册中心，负责服务的注册和发现"""

    def __init__(self):

        self.services: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, Dict[str, Any]] = {}

    def register_service(self, service_name: str, service_obj: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """注册服务"""
        try:
            service_info = {
                'service_obj': service_obj,
                'config': config,
                'status': 'registered',
                'registration_time': '2025 - 01 - 01T00:00:00Z'
            }

            self.services[service_name] = service_info

            # 设置健康检查
            if 'health_check' in config:
                self.health_checks[service_name] = {
                    'path': config['health_check'],
                    'interval': config.get('health_check_interval', 30),
                    'timeout': config.get('health_check_timeout', 10)
                }

            logger.info(f"注册服务成功: {service_name}")
            return {'success': True, 'message': f'服务 {service_name} 注册成功'}

        except Exception as e:
            logger.error(f"注册服务失败: {service_name}, 错误: {str(e)}")
            return {'success': False, 'error': str(e)}

    def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """发现服务"""
        if service_name in self.services:
            service_info = self.services[service_name].copy()
            # 移除敏感信息
            service_info.pop('service_obj', None)
            return service_info
        return None

    def get_all_services(self) -> Dict[str, Dict[str, Any]]:
        """获取所有服务"""
        all_services = {}
        for service_name, service_info in self.services.items():
            all_services[service_name] = {
                'config': service_info['config'],
                'status': service_info['status'],
                'registration_time': service_info['registration_time']
            }
        return all_services

    def health_check_all(self) -> Dict[str, Any]:
        """健康检查所有服务"""
        health_results = {
            'overall_healthy': True,
            'service_results': {},
            'unhealthy_services': []
        }

        for service_name in self.services:
            health_result = self._check_service_health(service_name)
            health_results['service_results'][service_name] = health_result

            if not health_result['healthy']:
                health_results['overall_healthy'] = False
                health_results['unhealthy_services'].append(service_name)

        return health_results

    def _check_service_health(self, service_name: str) -> Dict[str, Any]:
        """检查单个服务健康状态"""
        if service_name not in self.health_checks:
            return {'healthy': True, 'message': 'No health check configured'}

        try:
            # 这里应该实现实际的健康检查逻辑
            # 目前返回模拟结果
            return {
                'healthy': True,
                'response_time': 0.1,
                'status_code': 200,
                'message': 'Service is healthy'
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'message': 'Health check failed'
            }


class LoadBalancer:

    """负载均衡器，负责服务的负载均衡"""

    def __init__(self):

        self.instances: Dict[str, List[Dict[str, Any]]] = {}
        self.strategies = {
            'round_robin': self._round_robin,
            'least_connections': self._least_connections,
            'weighted': self._weighted
        }
        self.current_strategy = 'round_robin'
        self.current_indices: Dict[str, int] = {}

    def set_strategy(self, strategy: str) -> Dict[str, Any]:
        """设置负载均衡策略"""
        if strategy not in self.strategies:
            return {
                'success': False,
                'error': f'不支持的策略: {strategy}'
            }

        self.current_strategy = strategy
        logger.info(f"设置负载均衡策略: {strategy}")

        return {
            'success': True,
            'message': f'负载均衡策略设置成功: {strategy}'
        }

    def add_instance(self, service_name: str, instance_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """添加服务实例"""
        try:
            if service_name not in self.instances:
                self.instances[service_name] = []

            instance_info = {
                'name': instance_name,
                'config': config,
                'status': 'active',
                'connections': 0,
                'weight': config.get('weight', 1)
            }

            self.instances[service_name].append(instance_info)
            logger.info(f"添加服务实例: {service_name}/{instance_name}")

            return {
                'success': True,
                'message': f'服务实例添加成功: {service_name}/{instance_name}'
            }

        except Exception as e:
            logger.error(f"添加服务实例失败: {service_name}/{instance_name}, 错误: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_instance(self, service_name: str) -> Optional[Dict[str, Any]]:
        """获取服务实例"""
        if service_name not in self.instances or not self.instances[service_name]:
            return None

        # 使用当前策略选择实例
        strategy_func = self.strategies.get(self.current_strategy)
        if strategy_func:
            return strategy_func(service_name)
        else:
            # 默认使用轮询
            return self._round_robin(service_name)

    def _round_robin(self, service_name: str) -> Optional[Dict[str, Any]]:
        """轮询策略"""
        instances = self.instances.get(service_name, [])
        if not instances:
            return None

        if service_name not in self.current_indices:
            self.current_indices[service_name] = 0

        instance = instances[self.current_indices[service_name]]
        self.current_indices[service_name] = (
            self.current_indices[service_name] + 1) % len(instances)

        return instance

    def _least_connections(self, service_name: str) -> Optional[Dict[str, Any]]:
        """最少连接策略"""
        instances = self.instances.get(service_name, [])
        if not instances:
            return None

        # 选择连接数最少的实例
        min_connections = float('inf')
        selected_instance = None

        for instance in instances:
            if instance['connections'] < min_connections:
                min_connections = instance['connections']
                selected_instance = instance

        return selected_instance

    def _weighted(self, service_name: str) -> Optional[Dict[str, Any]]:
        """加权策略"""
        instances = self.instances.get(service_name, [])
        if not instances:
            return None

        # 根据权重选择实例
        total_weight = sum(instance['weight'] for instance in instances)
        if total_weight == 0:
            return instances[0] if instances else None

        # 简单的加权选择（实际应该使用更复杂的算法）
        import secrets
        rand_val = secrets.uniform(0, total_weight)
        current_weight = 0

        for instance in instances:
            current_weight += instance['weight']
            if rand_val <= current_weight:
                return instance

        return instances[-1] if instances else None

    def remove_instance(self, service_name: str, instance_name: str) -> Dict[str, Any]:
        """移除服务实例"""
        try:
            if service_name not in self.instances:
                return {
                    'success': False,
                    'error': f'服务不存在: {service_name}'
                }

            instances = self.instances[service_name]
            for i, instance in enumerate(instances):
                if instance['name'] == instance_name:
                    del instances[i]
                    logger.info(f"移除服务实例: {service_name}/{instance_name}")

                    return {
                        'success': True,
                        'message': f'服务实例移除成功: {service_name}/{instance_name}'
                    }

            return {
                'success': False,
                'error': f'实例不存在: {service_name}/{instance_name}'
            }

        except Exception as e:
            logger.error(f"移除服务实例失败: {service_name}/{instance_name}, 错误: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """获取负载均衡统计"""
        stats = {
            'strategy': self.current_strategy,
            'services': {},
            'total_instances': 0
        }

        for service_name, instances in self.instances.items():
            service_stats = {
                'instance_count': len(instances),
                'active_instances': len([i for i in instances if i['status'] == 'active']),
                'total_connections': sum(i['connections'] for i in instances),
                'average_connections': sum(i['connections'] for i in instances) / len(instances) if instances else 0
            }
            stats['services'][service_name] = service_stats
            stats['total_instances'] += len(instances)

        return stats
