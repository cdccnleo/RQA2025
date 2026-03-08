"""
组件健康监控器

Component Health Monitor for system integration testing.

Extracted from system_integration_tester.py to improve code organization.

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .enums import ComponentStatus
from .models import ComponentHealth

logger = logging.getLogger(__name__)

class ComponentHealthMonitor:

    """组件健康监控器"""

    def __init__(self):

        self.components: Dict[str, ComponentHealth] = {}
        self.monitoring_threads = {}
        self.is_monitoring = False

        # 组件配置
        self.component_configs = {
            'deep_learning_manager': {
                'check_interval': 30,
                'timeout': 10,
                'health_thresholds': {
                    'response_time': 5.0,
                    'error_rate': 0.05,
                    'memory_usage': 0.8
                }
            },
            'data_pipeline': {
                'check_interval': 60,
                'timeout': 15,
                'health_thresholds': {
                    'response_time': 10.0,
                    'error_rate': 0.02,
                    'throughput': 100
                }
            },
            'risk_monitoring': {
                'check_interval': 30,
                'timeout': 8,
                'health_thresholds': {
                    'response_time': 3.0,
                    'error_rate': 0.01,
                    'cpu_usage': 0.7
                }
            },
            'automation_engine': {
                'check_interval': 20,
                'timeout': 5,
                'health_thresholds': {
                    'response_time': 2.0,
                    'error_rate': 0.03,
                    'memory_usage': 0.6
                }
            }
        }

    def start_monitoring(self):
        """启动健康监控"""
        if self.is_monitoring:
            logger.warning("健康监控已启动")
            return

        self.is_monitoring = True

        for component_name, config in self.component_configs.items():
            thread = threading.Thread(
                target=self._monitor_component,
                args=(component_name, config),
                daemon=True
            )
            self.monitoring_threads[component_name] = thread
            thread.start()

        logger.info("组件健康监控已启动")

    def stop_monitoring(self):
        """停止健康监控"""
        self.is_monitoring = False

        # 等待所有监控线程结束
        for thread in self.monitoring_threads.values():
            thread.join(timeout=5)

        logger.info("组件健康监控已停止")

    def _monitor_component(self, component_name: str, config: Dict[str, Any]):
        """监控单个组件"""
        check_interval = config['check_interval']
        timeout = config['timeout']
        thresholds = config['health_thresholds']

        while self.is_monitoring:
            try:
                # 模拟健康检查
                health_status = self._check_component_health(component_name, timeout, thresholds)

                # 更新组件状态
                self.components[component_name] = health_status

                # 根据状态采取行动
                if health_status.status == ComponentStatus.UNHEALTHY:
                    logger.warning(f"组件 {component_name} 状态不健康")
                elif health_status.status == ComponentStatus.OFFLINE:
                    logger.error(f"组件 {component_name} 离线")

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"监控组件 {component_name} 出错: {e}")
                time.sleep(check_interval)

    def _check_component_health(self, component_name: str, timeout: float,


                                thresholds: Dict[str, float]) -> ComponentHealth:
        """检查组件健康状态"""
        start_time = time.time()

        try:
            # 模拟组件健康检查
            response_time = np.random.uniform(0.1, 8.0)
            error_count = np.random.randint(0, 10)
            memory_usage = np.random.uniform(0.1, 0.9)
            cpu_usage = np.random.uniform(0.1, 0.8)
            throughput = np.random.uniform(50, 200)

            # 计算健康得分
            health_score = self._calculate_health_score(
                response_time, error_count, memory_usage, cpu_usage, throughput, thresholds
            )

            # 确定状态
            if health_score >= 0.8:
                status = ComponentStatus.HEALTHY
            elif health_score >= 0.6:
                status = ComponentStatus.DEGRADED
            elif health_score >= 0.3:
                status = ComponentStatus.UNHEALTHY
            else:
                status = ComponentStatus.OFFLINE

            custom_metrics = {
                'health_score': health_score,
                'error_rate': error_count / 100,  # 假设100个请求
                'uptime': 0.99  # 99% uptime
            }

            return ComponentHealth(
                component_name=component_name,
                status=status,
                last_check=datetime.now(),
                response_time=response_time,
                error_count=error_count,
                throughput=throughput,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                custom_metrics=custom_metrics
            )

        except Exception as e:
            logger.error(f"健康检查失败 {component_name}: {e}")

            return ComponentHealth(
                component_name=component_name,
                status=ComponentStatus.OFFLINE,
                last_check=datetime.now(),
                response_time=time.time() - start_time,
                error_count=1,
                throughput=None,
                memory_usage=None,
                cpu_usage=None,
                custom_metrics={'error': str(e)}
            )

    def _calculate_health_score(self, response_time: float, error_count: int,


                                memory_usage: float, cpu_usage: float, throughput: float,
                                thresholds: Dict[str, float]) -> float:
        """计算健康得分"""
import numpy as np
import threading
        scores = []

        # 响应时间得分
        if 'response_time' in thresholds:
            rt_threshold = thresholds['response_time']
            rt_score = max(0, 1 - (response_time / rt_threshold))
            scores.append(rt_score)

        # 错误率得分
        if 'error_rate' in thresholds:
            error_rate = error_count / 100  # 假设100个请求
            er_threshold = thresholds['error_rate']
            er_score = max(0, 1 - (error_rate / er_threshold))
            scores.append(er_score)

        # 内存使用得分
        if 'memory_usage' in thresholds:
            mem_threshold = thresholds['memory_usage']
            mem_score = max(0, 1 - (memory_usage / mem_threshold))
            scores.append(mem_score)

        # CPU使用得分
        if 'cpu_usage' in thresholds:
            cpu_threshold = thresholds['cpu_usage']
            cpu_score = max(0, 1 - (cpu_usage / cpu_threshold))
            scores.append(cpu_score)

        # 吞吐量得分
        if 'throughput' in thresholds:
            tp_threshold = thresholds['throughput']
            tp_score = min(1, throughput / tp_threshold)
            scores.append(tp_score)

        return np.mean(scores) if scores else 0.5

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要"""
        total_components = len(self.components)
        healthy_count = sum(1 for comp in self.components.values()
                            if comp.status == ComponentStatus.HEALTHY)
        degraded_count = sum(1 for comp in self.components.values()
                             if comp.status == ComponentStatus.DEGRADED)
        unhealthy_count = sum(1 for comp in self.components.values()
                              if comp.status == ComponentStatus.UNHEALTHY)
        offline_count = sum(1 for comp in self.components.values()
                            if comp.status == ComponentStatus.OFFLINE)

        overall_health = healthy_count / total_components if total_components > 0 else 0

        return {
            'total_components': total_components,
            'healthy_count': healthy_count,
            'degraded_count': degraded_count,
            'unhealthy_count': unhealthy_count,
            'offline_count': offline_count,
            'overall_health_score': overall_health,
            'component_details': {
                name: comp.to_dict() for name, comp in self.components.items()
            }
        }


__all__ = ['ComponentHealthMonitor']
