"""
Resource Optimizer Module
资源优化器模块

This module provides resource optimization capabilities for trading engine components
此模块为交易引擎组件提供资源优化能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime
import threading
import time
import psutil
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ResourceOptimizer:

    """
    Resource Optimizer Class
    资源优化器类

    Manages and optimizes resource allocation across trading engine components
    管理和优化跨交易引擎组件的资源分配
    """

    def __init__(self, max_memory_gb: float = 8.0, max_cpu_percent: float = 80.0):
        """
        Initialize resource optimizer
        初始化资源优化器

        Args:
            max_memory_gb: Maximum memory usage in GB
                          最大内存使用量（GB）
            max_cpu_percent: Maximum CPU usage percentage
                            最大CPU使用百分比
        """
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent

        # Resource tracking
        self.resource_usage = {
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'disk_usage_percent': 0.0,
            'network_connections': 0
        }

        # Component resource allocation
        self.component_resources = defaultdict(lambda: {
            'allocated_memory_mb': 0.0,
            'allocated_cpu_percent': 0.0,
            'priority': 1,
            'resource_limits': {},
            'usage_history': deque(maxlen=100)
        })

        # Resource pools
        self.memory_pool = []
        self.cpu_pool = []

        # Optimization settings
        self.enable_resource_pooling = True
        self.enable_dynamic_allocation = True
        self.resource_check_interval = 10.0  # seconds
        self.is_monitoring = False
        self.monitoring_thread = None

        # Resource limits
        self.resource_limits = {
            'max_memory_per_component': 1024,  # MB
            'max_cpu_per_component': 50,  # %
            'min_memory_per_component': 10,  # MB
            'min_cpu_per_component': 1  # %
        }

        logger.info("Resource optimizer initialized")

    def start_resource_monitoring(self) -> bool:
        """
        Start resource monitoring
        开始资源监控

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_monitoring:
            logger.warning("Resource monitoring already running")
            return False

        try:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Resource monitoring started")
            return True
        except Exception as e:
            logger.error(f"Failed to start resource monitoring: {str(e)}")
            self.is_monitoring = False
            return False

    def stop_resource_monitoring(self) -> bool:
        """
        Stop resource monitoring
        停止资源监控

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_monitoring:
            logger.warning("Resource monitoring not running")
            return False

        try:
            self.is_monitoring = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Resource monitoring stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop resource monitoring: {str(e)}")
            return False

    def allocate_resources(self,


                           component_name: str,
                           memory_mb: float = 0.0,
                           cpu_percent: float = 0.0,
                           priority: int = 1) -> Dict[str, Any]:
        """
        Allocate resources to a component
        为组件分配资源

        Args:
            component_name: Name of the component
                          组件名称
            memory_mb: Memory to allocate (MB)
                      要分配的内存（MB）
            cpu_percent: CPU percentage to allocate
                        要分配的CPU百分比
            priority: Component priority (higher = more resources)
                     组件优先级（越高=更多资源）

        Returns:
            dict: Resource allocation result
                  资源分配结果
        """
        try:
            # Check if allocation is within limits
            if not self._validate_allocation(memory_mb, cpu_percent):
                return {
                    'success': False,
                    'error': 'Resource allocation exceeds system limits',
                    'requested_memory': memory_mb,
                    'requested_cpu': cpu_percent
                }

            # Update component resource allocation
            component_data = self.component_resources[component_name]
            component_data['allocated_memory_mb'] = memory_mb
            component_data['allocated_cpu_percent'] = cpu_percent
            component_data['priority'] = priority

            # Set resource limits for the component
            component_data['resource_limits'] = {
                'memory_limit_mb': min(memory_mb * 1.2, self.resource_limits['max_memory_per_component']),
                'cpu_limit_percent': min(cpu_percent * 1.2, self.resource_limits['max_cpu_per_component'])
            }

            logger.info(
                f"Allocated {memory_mb}MB memory and {cpu_percent}% CPU to {component_name}")

            return {
                'success': True,
                'component': component_name,
                'allocated_memory_mb': memory_mb,
                'allocated_cpu_percent': cpu_percent,
                'priority': priority,
                'limits': component_data['resource_limits']
            }

        except Exception as e:
            logger.error(f"Resource allocation failed for {component_name}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def deallocate_resources(self, component_name: str) -> Dict[str, Any]:
        """
        Deallocate resources from a component
        从组件释放资源

        Args:
            component_name: Name of the component
                          组件名称

        Returns:
            dict: Resource deallocation result
                  资源释放结果
        """
        try:
            if component_name in self.component_resources:
                component_data = self.component_resources[component_name]
                deallocated_memory = component_data['allocated_memory_mb']
                deallocated_cpu = component_data['allocated_cpu_percent']

                # Reset component resources
                component_data['allocated_memory_mb'] = 0.0
                component_data['allocated_cpu_percent'] = 0.0

                logger.info(
                    f"Deallocated {deallocated_memory}MB memory and {deallocated_cpu}% CPU from {component_name}")

                return {
                    'success': True,
                    'component': component_name,
                    'deallocated_memory_mb': deallocated_memory,
                    'deallocated_cpu_percent': deallocated_cpu
                }
            else:
                return {'success': False, 'error': f'Component {component_name} not found'}

        except Exception as e:
            logger.error(f"Resource deallocation failed for {component_name}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage
        获取当前资源使用情况

        Returns:
            dict: Resource usage information
                  资源使用信息
        """
        try:
            # Get system resource usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')

            current_usage = {
                'memory_mb': memory.used / (1024 * 1024),
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'disk_usage_percent': disk.percent,
                'timestamp': datetime.now()
            }

            # Update tracking
            self.resource_usage.update(current_usage)

            return current_usage

        except Exception as e:
            logger.error(f"Failed to get resource usage: {str(e)}")
            return {'error': str(e)}

    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """
        Optimize resource allocation across components
        优化跨组件的资源分配

        Returns:
            dict: Resource optimization results
                  资源优化结果
        """
        try:
            current_usage = self.get_resource_usage()
            results = {
                'timestamp': datetime.now(),
                'current_usage': current_usage,
                'optimizations': [],
                'recommendations': []
            }

            # Check memory usage
            memory_percent = current_usage.get('memory_percent', 0)
            if memory_percent > 85:
                results['optimizations'].append('memory_optimization')
                results['recommendations'].append(
                    'High memory usage detected. Consider reducing cache sizes or increasing memory allocation.')

            # Check CPU usage
            cpu_percent = current_usage.get('cpu_percent', 0)
            if cpu_percent > self.max_cpu_percent:
                results['optimizations'].append('cpu_optimization')
                results['recommendations'].append(
                    'High CPU usage detected. Consider workload redistribution.')

            # Check disk usage
            disk_percent = current_usage.get('disk_usage_percent', 0)
            if disk_percent > 90:
                results['optimizations'].append('disk_cleanup')
                results['recommendations'].append(
                    'High disk usage detected. Consider cleaning up temporary files.')

            # Optimize component allocations based on priority
            self._optimize_component_allocations(results)

            return results

        except Exception as e:
            logger.error(f"Resource optimization failed: {str(e)}")
            return {'error': str(e)}

    def _optimize_component_allocations(self, results: Dict[str, Any]) -> None:
        """
        Optimize component resource allocations
        优化组件资源分配

        Args:
            results: Results dictionary to update
                    要更新的结果字典
        """
        try:
            # Sort components by priority
            sorted_components = sorted(
                self.component_resources.items(),
                key=lambda x: x[1]['priority'],
                reverse=True
            )

            total_memory_available = self.max_memory_gb * 1024  # Convert to MB
            total_cpu_available = self.max_cpu_percent

            memory_allocated = 0
            cpu_allocated = 0

            for component_name, component_data in sorted_components:
                # Allocate memory based on priority
                if memory_allocated < total_memory_available:
                    memory_allocation = min(
                        component_data['allocated_memory_mb'],
                        total_memory_available * (component_data['priority'] / 10)
                    )
                    component_data['allocated_memory_mb'] = memory_allocation
                    memory_allocated += memory_allocation

                # Allocate CPU based on priority
                if cpu_allocated < total_cpu_available:
                    cpu_allocation = min(
                        component_data['allocated_cpu_percent'],
                        total_cpu_available * (component_data['priority'] / 10)
                    )
                    component_data['allocated_cpu_percent'] = cpu_allocation
                    cpu_allocated += cpu_allocation

            results['component_optimizations'] = {
                'total_memory_allocated_mb': memory_allocated,
                'total_cpu_allocated_percent': cpu_allocated,
                'optimized_components': len(sorted_components)
            }

        except Exception as e:
            logger.error(f"Component allocation optimization failed: {str(e)}")

    def monitor_component_resources(self,

                                    component_name: str,
                                    memory_usage: float = 0.0,
                                    cpu_usage: float = 0.0) -> Dict[str, Any]:
        """
        Monitor resource usage of a specific component
        监控特定组件的资源使用情况

        Args:
            component_name: Name of the component
                          组件名称
            memory_usage: Current memory usage (MB)
                         当前内存使用量（MB）
            cpu_usage: Current CPU usage (%)
                      当前CPU使用量（%）

        Returns:
            dict: Component monitoring result
                  组件监控结果
        """
        try:
            component_data = self.component_resources[component_name]

            # Record usage in history
            usage_record = {
                'timestamp': datetime.now(),
                'memory_mb': memory_usage,
                'cpu_percent': cpu_usage
            }
            component_data['usage_history'].append(usage_record)

            # Check against limits
            memory_limit = component_data['resource_limits'].get('memory_limit_mb', float('inf'))
            cpu_limit = component_data['resource_limits'].get('cpu_limit_percent', float('inf'))

            alerts = []
            if memory_usage > memory_limit * 0.9:  # 90% of limit
                alerts.append(
                    f"Memory usage ({memory_usage}MB) approaching limit ({memory_limit}MB)")
            if cpu_usage > cpu_limit * 0.9:  # 90% of limit
                alerts.append(f"CPU usage ({cpu_usage}%) approaching limit ({cpu_limit}%)")

            return {
                'component': component_name,
                'current_memory_mb': memory_usage,
                'current_cpu_percent': cpu_usage,
                'memory_limit_mb': memory_limit,
                'cpu_limit_percent': cpu_limit,
                'alerts': alerts,
                'usage_history_size': len(component_data['usage_history'])
            }

        except Exception as e:
            logger.error(f"Component resource monitoring failed for {component_name}: {str(e)}")
            return {'error': str(e)}

    def get_resource_limits(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get resource limits for components
        获取组件的资源限制

        Args:
            component_name: Specific component name (None for all)
                           特定组件名称（None表示全部）

        Returns:
            dict: Resource limits information
                  资源限制信息
        """
        try:
            if component_name:
                if component_name in self.component_resources:
                    return {
                        'component': component_name,
                        'limits': self.component_resources[component_name]['resource_limits'],
                        'allocation': {
                            'memory_mb': self.component_resources[component_name]['allocated_memory_mb'],
                            'cpu_percent': self.component_resources[component_name]['allocated_cpu_percent']
                        }
                    }
                else:
                    return {'error': f'Component {component_name} not found'}
            else:
                # Return all components' limits
                all_limits = {}
                for name, data in self.component_resources.items():
                    all_limits[name] = {
                        'limits': data['resource_limits'],
                        'allocation': {
                            'memory_mb': data['allocated_memory_mb'],
                            'cpu_percent': data['allocated_cpu_percent']
                        }
                    }
                return all_limits

        except Exception as e:
            logger.error(f"Failed to get resource limits: {str(e)}")
            return {'error': str(e)}

    def _validate_allocation(self, memory_mb: float, cpu_percent: float) -> bool:
        """
        Validate resource allocation request
        验证资源分配请求

        Args:
            memory_mb: Requested memory (MB)
                      请求的内存（MB）
            cpu_percent: Requested CPU percentage
                        请求的CPU百分比

        Returns:
            bool: True if allocation is valid
                  如果分配有效则返回True
        """
        try:
            # Check memory limits
            if memory_mb > self.resource_limits['max_memory_per_component']:
                return False
            if memory_mb < self.resource_limits['min_memory_per_component']:
                return False

            # Check CPU limits
            if cpu_percent > self.resource_limits['max_cpu_per_component']:
                return False
            if cpu_percent < self.resource_limits['min_cpu_per_component']:
                return False

            # Check total allocation
            total_allocated_memory = sum(
                data['allocated_memory_mb'] for data in self.component_resources.values()
            )
            total_allocated_cpu = sum(
                data['allocated_cpu_percent'] for data in self.component_resources.values()
            )

            if (total_allocated_memory + memory_mb) > (self.max_memory_gb * 1024):
                return False
            if (total_allocated_cpu + cpu_percent) > self.max_cpu_percent:
                return False

            return True

        except Exception as e:
            logger.error(f"Resource allocation validation failed: {str(e)}")
            return False

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop
        主要的监控循环
        """
        logger.info("Resource monitoring loop started")

        while self.is_monitoring:
            try:
                # Update resource usage
                self.get_resource_usage()

                # Optimize resource allocation periodically
                if len(self.component_resources) > 0:
                    self.optimize_resource_allocation()

                # Sleep before next monitoring cycle
                time.sleep(self.resource_check_interval)

            except Exception as e:
                logger.error(f"Resource monitoring loop error: {str(e)}")
                time.sleep(self.resource_check_interval)

        logger.info("Resource monitoring loop stopped")

    def get_resource_report(self) -> Dict[str, Any]:
        """
        Get comprehensive resource report
        获取全面的资源报告

        Returns:
            dict: Resource report
                  资源报告
        """
        try:
            report = {
                'timestamp': datetime.now(),
                'system_resources': self.get_resource_usage(),
                'component_resources': {},
                'resource_limits': self.resource_limits,
                'optimization_settings': {
                    'enable_resource_pooling': self.enable_resource_pooling,
                    'enable_dynamic_allocation': self.enable_dynamic_allocation,
                    'max_memory_gb': self.max_memory_gb,
                    'max_cpu_percent': self.max_cpu_percent
                }
            }

            # Add component resource information
            for name, data in self.component_resources.items():
                report['component_resources'][name] = {
                    'allocated_memory_mb': data['allocated_memory_mb'],
                    'allocated_cpu_percent': data['allocated_cpu_percent'],
                    'priority': data['priority'],
                    'limits': data['resource_limits'],
                    'usage_history_size': len(data['usage_history'])
                }

            return report

        except Exception as e:
            logger.error(f"Failed to generate resource report: {str(e)}")
            return {'error': str(e)}

    def reset_resource_tracking(self) -> None:
        """
        Reset resource tracking data
        重置资源跟踪数据

        Returns:
            None
        """
        self.component_resources.clear()
        self.resource_usage = {
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'disk_usage_percent': 0.0,
            'network_connections': 0
        }
        logger.info("Resource tracking data reset")


# Global resource optimizer instance
# 全局资源优化器实例
resource_optimizer = ResourceOptimizer()

__all__ = ['ResourceOptimizer', 'resource_optimizer']
