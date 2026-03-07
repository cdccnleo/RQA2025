"""
Scaling Automation Module
扩容自动化模块

This module provides automated scaling capabilities for quantitative trading systems
此模块为量化交易系统提供自动化扩容能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import psutil
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ScalingType(Enum):

    """Scaling types"""
    HORIZONTAL = "horizontal"  # Scale out / in (add / remove instances)
    VERTICAL = "vertical"      # Scale up / down (increase / decrease resources)
    AUTO = "auto"             # Automatic scaling based on metrics


class ScalingDirection(Enum):

    """Scaling directions"""
    UP = "up"       # Scale up / increase resources
    DOWN = "down"   # Scale down / decrease resources


class ScalingStatus(Enum):

    """Scaling status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ScalingPolicy:

    """
    Scaling policy data class
    扩容策略数据类
    """
    policy_id: str
    name: str
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period: int  # seconds
    max_instances: int
    min_instances: int
    scaling_type: str
    enabled: bool = True
    last_scaled: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ScalingEvent:

    """
    Scaling event data class
    扩容事件数据类
    """
    event_id: str
    policy_id: str
    scaling_type: str
    direction: str
    old_instances: int
    new_instances: int
    metric_value: float
    threshold: float
    timestamp: datetime
    status: str
    execution_time: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ResourceMonitor:

    """
    Resource Monitor Class
    资源监控器类

    Monitors system resources for scaling decisions
    监控系统资源以进行扩容决策
    """

    def __init__(self):
        """
        Initialize resource monitor
        初始化资源监控器
        """
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))
        self.monitoring_interval = 30  # seconds

    def collect_metrics(self) -> Dict[str, float]:
        """
        Collect current resource metrics
        收集当前资源指标

        Returns:
            dict: Resource metrics
                  资源指标
        """
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'network_connections': len(psutil.net_connections()),
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }

            # Store in history
            timestamp = datetime.now()
            for metric_name, value in metrics.items():
                self.metrics_history[metric_name].append((timestamp, value))

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {str(e)}")
            return {}

    def get_metric_average(self, metric_name: str, window_minutes: int = 5) -> float:
        """
        Get average metric value over a time window
        获取时间窗口内的平均指标值

        Args:
            metric_name: Name of the metric
                        指标名称
            window_minutes: Time window in minutes
                          时间窗口（分钟）

        Returns:
            float: Average metric value
                   平均指标值
        """
        if metric_name not in self.metrics_history:
            return 0.0

        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_values = [
            value for timestamp, value in self.metrics_history[metric_name]
            if timestamp >= cutoff_time
        ]

        return sum(recent_values) / len(recent_values) if recent_values else 0.0

    def get_metric_trend(self, metric_name: str, window_minutes: int = 10) -> str:
        """
        Get metric trend over time
        获取指标的时间趋势

        Args:
            metric_name: Name of the metric
                        指标名称
            window_minutes: Time window in minutes
                          时间窗口（分钟）

        Returns:
            str: Trend direction ('increasing', 'decreasing', 'stable')
                 趋势方向
        """
        if metric_name not in self.metrics_history:
            return 'stable'

        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_values = [
            value for timestamp, value in self.metrics_history[metric_name]
            if timestamp >= cutoff_time
        ]

        if len(recent_values) < 5:
            return 'stable'

        # Simple linear trend
        first_half = recent_values[:len(recent_values) // 2]
        second_half = recent_values[len(recent_values) // 2:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if second_avg > first_avg * 1.05:  # 5% increase
            return 'increasing'
        elif second_avg < first_avg * 0.95:  # 5% decrease
            return 'decreasing'
        else:
            return 'stable'


class ScalingExecutor:

    """
    Scaling Executor Class
    扩容执行器类

    Executes scaling operations
    执行扩容操作
    """

    def __init__(self):
        """
        Initialize scaling executor
        初始化扩容执行器
        """
        self.active_scalings = {}
        self.scaling_history = deque(maxlen=100)

    def execute_scaling(self,


                        scaling_type: ScalingType,
                        direction: ScalingDirection,
                        current_instances: int,
                        target_instances: int,
                        resource_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a scaling operation
        执行扩容操作

        Args:
            scaling_type: Type of scaling
                         扩容类型
            direction: Scaling direction
                      扩容方向
            current_instances: Current number of instances
                             当前实例数量
            target_instances: Target number of instances
                            目标实例数量
            resource_config: Resource configuration
                           资源配置

        Returns:
            dict: Scaling result
                  扩容结果
        """
        scaling_id = f"scaling_{scaling_type.value}_{direction.value}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

        result = {
            'scaling_id': scaling_id,
            'success': False,
            'start_time': datetime.now(),
            'execution_time': 0.0
        }

        start_time = time.time()

        try:
            if scaling_type == ScalingType.HORIZONTAL:
                result.update(self._execute_horizontal_scaling(
                    direction, current_instances, target_instances, resource_config
                ))
            elif scaling_type == ScalingType.VERTICAL:
                result.update(self._execute_vertical_scaling(
                    direction, resource_config
                ))
            elif scaling_type == ScalingType.AUTO:
                result.update(self._execute_auto_scaling(
                    direction, current_instances, target_instances, resource_config
                ))

            result['success'] = True
            result['execution_time'] = time.time() - start_time

            # Store in history
            self.scaling_history.append({
                'scaling_id': scaling_id,
                'type': scaling_type.value,
                'direction': direction.value,
                'result': result,
                'timestamp': datetime.now()
            })

        except Exception as e:
            result['execution_time'] = time.time() - start_time
            result['error'] = str(e)
            logger.error(f"Scaling execution failed: {str(e)}")

        return result

    def _execute_horizontal_scaling(self,


                                    direction: ScalingDirection,
                                    current_instances: int,
                                    target_instances: int,
                                    resource_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute horizontal scaling
        执行水平扩容

        Args:
            direction: Scaling direction
                      扩容方向
            current_instances: Current instance count
                             当前实例数量
            target_instances: Target instance count
                            目标实例数量
            resource_config: Resource configuration
                           资源配置

        Returns:
            dict: Scaling result
                  扩容结果
        """
        if direction == ScalingDirection.UP:
            instances_to_add = target_instances - current_instances
            return self._add_instances(instances_to_add, resource_config)
        else:
            instances_to_remove = current_instances - target_instances
            return self._remove_instances(instances_to_remove, resource_config)

    def _execute_vertical_scaling(self,


                                  direction: ScalingDirection,
                                  resource_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute vertical scaling
        执行垂直扩容

        Args:
            direction: Scaling direction
                      扩容方向
            resource_config: Resource configuration
                           资源配置

        Returns:
            dict: Scaling result
                  扩容结果
        """
        if direction == ScalingDirection.UP:
            return self._scale_up_resources(resource_config)
        else:
            return self._scale_down_resources(resource_config)

    def _execute_auto_scaling(self,


                              direction: ScalingDirection,
                              current_instances: int,
                              target_instances: int,
                              resource_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute auto scaling
        执行自动扩容

        Args:
            direction: Scaling direction
                      扩容方向
            current_instances: Current instance count
                             当前实例数量
            target_instances: Target instance count
                            目标实例数量
            resource_config: Resource configuration
                           资源配置

        Returns:
            dict: Scaling result
                  扩容结果
        """
        # Auto scaling combines horizontal and vertical scaling
        horizontal_result = self._execute_horizontal_scaling(
            direction, current_instances, target_instances, resource_config
        )

        vertical_result = self._execute_vertical_scaling(
            direction, resource_config
        )

        return {
            'horizontal_scaling': horizontal_result,
            'vertical_scaling': vertical_result,
            'combined_success': horizontal_result.get('success', False) and vertical_result.get('success', False)
        }

    def _add_instances(self, count: int, resource_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add instances (placeholder)"""
        logger.info(f"Adding {count} instances")
        return {
            'instances_added': count,
            'resource_type': resource_config.get('instance_type', 't3.micro')
        }

    def _remove_instances(self, count: int, resource_config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove instances (placeholder)"""
        logger.info(f"Removing {count} instances")
        return {
            'instances_removed': count,
            'graceful_shutdown': True
        }

    def _scale_up_resources(self, resource_config: Dict[str, Any]) -> Dict[str, Any]:
        """Scale up resources (placeholder)"""
        logger.info("Scaling up resources")
        return {
            'cpu_increase': resource_config.get('cpu_increment', 1),
            'memory_increase': resource_config.get('memory_increment', 1024)
        }

    def _scale_down_resources(self, resource_config: Dict[str, Any]) -> Dict[str, Any]:
        """Scale down resources (placeholder)"""
        logger.info("Scaling down resources")
        return {
            'cpu_decrease': resource_config.get('cpu_decrement', 1),
            'memory_decrease': resource_config.get('memory_decrement', 512)
        }


class ScalingAutomationEngine:

    """
    Scaling Automation Engine Class
    扩容自动化引擎类

    Core engine for automated scaling operations
    自动化扩容操作的核心引擎
    """

    def __init__(self, engine_name: str = "default_scaling_engine"):
        """
        Initialize scaling automation engine
        初始化扩容自动化引擎

        Args:
            engine_name: Name of the engine
                        引擎名称
        """
        self.engine_name = engine_name
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # Sub - components
        self.resource_monitor = ResourceMonitor()
        self.scaling_executor = ScalingExecutor()

        # Scaling policies
        self.policies: Dict[str, ScalingPolicy] = {}

        # Current state
        self.current_instances = 3  # Default
        self.scaling_events: List[ScalingEvent] = []

        # Configuration
        self.monitoring_interval = 60  # seconds
        self.scaling_cooldown = 300    # 5 minutes between scaling operations

        # Statistics
        self.stats = {
            'total_scalings': 0,
            'successful_scalings': 0,
            'failed_scalings': 0,
            'auto_scalings': 0,
            'manual_scalings': 0
        }

        # Setup default policies
        self._setup_default_policies()

        logger.info(f"Scaling automation engine {engine_name} initialized")

    def _setup_default_policies(self) -> None:
        """Setup default scaling policies"""
        # CPU - based scaling
        self.add_scaling_policy(
            'cpu_policy',
            'CPU Usage Scaling',
            'cpu_percent',
            80.0,  # Scale up at 80%
            30.0,  # Scale down at 30%
            300,   # 5 minute cooldown
            10,    # Max instances
            1,     # Min instances
            ScalingType.HORIZONTAL
        )

        # Memory - based scaling
        self.add_scaling_policy(
            'memory_policy',
            'Memory Usage Scaling',
            'memory_percent',
            85.0,  # Scale up at 85%
            40.0,  # Scale down at 40%
            300,
            10,
            1,
            ScalingType.HORIZONTAL
        )

    def add_scaling_policy(self,


                           policy_id: str,
                           name: str,
                           metric_name: str,
                           scale_up_threshold: float,
                           scale_down_threshold: float,
                           cooldown_period: int,
                           max_instances: int,
                           min_instances: int,
                           scaling_type: ScalingType) -> None:
        """
        Add a scaling policy
        添加扩容策略

        Args:
            policy_id: Unique policy identifier
                      唯一策略标识符
            name: Policy name
                 策略名称
            metric_name: Metric to monitor
                        要监控的指标
            scale_up_threshold: Threshold for scaling up
                              扩容阈值
            scale_down_threshold: Threshold for scaling down
                                缩容阈值
            cooldown_period: Cooldown period in seconds
                           冷却期（秒）
            max_instances: Maximum number of instances
                          最大实例数量
            min_instances: Minimum number of instances
                          最小实例数量
            scaling_type: Type of scaling
                         扩容类型
        """
        policy = ScalingPolicy(
            policy_id=policy_id,
            name=name,
            metric_name=metric_name,
            scale_up_threshold=scale_up_threshold,
            scale_down_threshold=scale_down_threshold,
            cooldown_period=cooldown_period,
            max_instances=max_instances,
            min_instances=min_instances,
            scaling_type=scaling_type.value
        )

        self.policies[policy_id] = policy
        logger.info(f"Added scaling policy: {name} ({policy_id})")

    def start_auto_scaling(self) -> bool:
        """
        Start automatic scaling
        启动自动扩容

        Returns:
            bool: True if started successfully
                  启动成功返回True
        """
        if self.is_running:
            logger.warning("Scaling engine is already running")
            return False

        try:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._auto_scaling_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Auto scaling started")
            return True
        except Exception as e:
            logger.error(f"Failed to start auto scaling: {str(e)}")
            self.is_running = False
            return False

    def stop_auto_scaling(self) -> bool:
        """
        Stop automatic scaling
        停止自动扩容

        Returns:
            bool: True if stopped successfully
                  停止成功返回True
        """
        if not self.is_running:
            logger.warning("Scaling engine is not running")
            return False

        try:
            self.is_running = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Auto scaling stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop auto scaling: {str(e)}")
            return False

    def _auto_scaling_loop(self) -> None:
        """
        Main auto scaling loop
        主要的自动扩容循环
        """
        logger.info("Auto scaling loop started")

        while self.is_running:
            try:
                # Collect metrics
                metrics = self.resource_monitor.collect_metrics()

                # Evaluate policies
                for policy in self.policies.values():
                    if not policy.enabled:
                        continue

                    self._evaluate_policy(policy, metrics)

                # Sleep before next evaluation
                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Auto scaling loop error: {str(e)}")
                time.sleep(self.monitoring_interval)

        logger.info("Auto scaling loop stopped")

    def _evaluate_policy(self, policy: ScalingPolicy, metrics: Dict[str, float]) -> None:
        """
        Evaluate a scaling policy
        评估扩容策略

        Args:
            policy: Scaling policy to evaluate
                   要评估的扩容策略
            metrics: Current metrics
                    当前指标
        """
        metric_name = policy.metric_name
        if metric_name not in metrics:
            return

        current_value = metrics[metric_name]

        # Check cooldown period
        if policy.last_scaled:
            time_since_last_scale = (datetime.now() - policy.last_scaled).total_seconds()
            if time_since_last_scale < policy.cooldown_period:
                return

        # Determine scaling direction
        scaling_needed = None
        threshold = None

        if current_value >= policy.scale_up_threshold:
            if self.current_instances < policy.max_instances:
                scaling_needed = ScalingDirection.UP
                threshold = policy.scale_up_threshold
        elif current_value <= policy.scale_down_threshold:
            if self.current_instances > policy.min_instances:
                scaling_needed = ScalingDirection.DOWN
                threshold = policy.scale_down_threshold

        if scaling_needed:
            self._execute_scaling(policy, scaling_needed, current_value, threshold)

    def _execute_scaling(self,


                         policy: ScalingPolicy,
                         direction: ScalingDirection,
                         metric_value: float,
                         threshold: float) -> None:
        """
        Execute scaling based on policy
        根据策略执行扩容

        Args:
            policy: Scaling policy
                   扩容策略
            direction: Scaling direction
                      扩容方向
            metric_value: Current metric value
                         当前指标值
            threshold: Threshold that triggered scaling
                      触发扩容的阈值
        """
        try:
            # Calculate target instances
            if direction == ScalingDirection.UP:
                target_instances = min(self.current_instances + 1, policy.max_instances)
            else:
                target_instances = max(self.current_instances - 1, policy.min_instances)

            if target_instances == self.current_instances:
                return

            # Execute scaling
            result = self.scaling_executor.execute_scaling(
                ScalingType(policy.scaling_type),
                direction,
                self.current_instances,
                target_instances,
                {}  # Resource config placeholder
            )

            if result['success']:
                old_instances = self.current_instances
                self.current_instances = target_instances

                # Update policy
                policy.last_scaled = datetime.now()

                # Record event
                event = ScalingEvent(
                    event_id=f"event_{policy.policy_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                    policy_id=policy.policy_id,
                    scaling_type=policy.scaling_type,
                    direction=direction.value,
                    old_instances=old_instances,
                    new_instances=target_instances,
                    metric_value=metric_value,
                    threshold=threshold,
                    timestamp=datetime.now(),
                    status=ScalingStatus.COMPLETED.value,
                    execution_time=result['execution_time']
                )

                self.scaling_events.append(event)
                self.stats['total_scalings'] += 1
                self.stats['successful_scalings'] += 1
                self.stats['auto_scalings'] += 1

                logger.info(
                    f"Auto scaling executed: {old_instances} -> {target_instances} instances")

            else:
                # Record failed scaling
                event = ScalingEvent(
                    event_id=f"event_{policy.policy_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                    policy_id=policy.policy_id,
                    scaling_type=policy.scaling_type,
                    direction=direction.value,
                    old_instances=self.current_instances,
                    new_instances=target_instances,
                    metric_value=metric_value,
                    threshold=threshold,
                    timestamp=datetime.now(),
                    status=ScalingStatus.FAILED.value,
                    error_message=result.get('error', 'Unknown error')
                )

                self.scaling_events.append(event)
                self.stats['total_scalings'] += 1
                self.stats['failed_scalings'] += 1

        except Exception as e:
            logger.error(f"Scaling execution failed: {str(e)}")

    def manual_scaling(self,


                       target_instances: int,
                       scaling_type: ScalingType = ScalingType.HORIZONTAL,
                       reason: str = "") -> Dict[str, Any]:
        """
        Execute manual scaling
        执行手动扩容

        Args:
            target_instances: Target number of instances
                            目标实例数量
            scaling_type: Type of scaling
                         扩容类型
            reason: Reason for manual scaling
                   手动扩容原因

        Returns:
            dict: Scaling result
                  扩容结果
        """
        if target_instances < 1:
            return {'success': False, 'error': 'Invalid target instance count'}

        direction = ScalingDirection.UP if target_instances > self.current_instances else ScalingDirection.DOWN

        result = self.scaling_executor.execute_scaling(
            scaling_type,
            direction,
            self.current_instances,
            target_instances,
            {}
        )

        if result['success']:
            old_instances = self.current_instances
            self.current_instances = target_instances

            # Record event
            event = ScalingEvent(
                event_id=f"manual_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                policy_id='manual',
                scaling_type=scaling_type.value,
                direction=direction.value,
                old_instances=old_instances,
                new_instances=target_instances,
                metric_value=0.0,
                threshold=0.0,
                timestamp=datetime.now(),
                status=ScalingStatus.COMPLETED.value,
                execution_time=result['execution_time']
            )

            self.scaling_events.append(event)
            self.stats['total_scalings'] += 1
            self.stats['successful_scalings'] += 1
            self.stats['manual_scalings'] += 1

        return result

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current scaling state
        获取当前扩容状态

        Returns:
            dict: Current state
                  当前状态
        """
        return {
            'current_instances': self.current_instances,
            'is_auto_scaling': self.is_running,
            'active_policies': sum(1 for p in self.policies.values() if p.enabled),
            'last_scaling_event': self.scaling_events[-1].to_dict() if self.scaling_events else None,
            'resource_metrics': self.resource_monitor.collect_metrics()
        }

    def get_scaling_history(self,


                            limit: int = 10,
                            policy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get scaling history
        获取扩容历史

        Args:
            limit: Maximum number of records
                  最大记录数
            policy_id: Filter by policy ID
                      按策略ID过滤

        Returns:
            list: Scaling events
                  扩容事件
        """
        events = self.scaling_events

        if policy_id:
            events = [e for e in events if e.policy_id == policy_id]

        return [event.to_dict() for event in events[-limit:]]

    def enable_policy(self, policy_id: str) -> bool:
        """
        Enable a scaling policy
        启用扩容策略

        Args:
            policy_id: Policy identifier
                      策略标识符

        Returns:
            bool: True if enabled
                  启用成功返回True
        """
        if policy_id in self.policies:
            self.policies[policy_id].enabled = True
            return True
        return False

    def disable_policy(self, policy_id: str) -> bool:
        """
        Disable a scaling policy
        禁用扩容策略

        Args:
            policy_id: Policy identifier
                      策略标识符

        Returns:
            bool: True if disabled
                  禁用成功返回True
        """
        if policy_id in self.policies:
            self.policies[policy_id].enabled = False
            return True
        return False

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get scaling engine statistics
        获取扩容引擎统计信息

        Returns:
            dict: Engine statistics
                  引擎统计信息
        """
        return {
            'engine_name': self.engine_name,
            'is_running': self.is_running,
            'current_instances': self.current_instances,
            'total_policies': len(self.policies),
            'active_policies': sum(1 for p in self.policies.values() if p.enabled),
            'total_events': len(self.scaling_events),
            'stats': self.stats
        }


# Global scaling automation engine instance
# 全局扩容自动化引擎实例
scaling_engine = ScalingAutomationEngine()

__all__ = [
    'ScalingType',
    'ScalingDirection',
    'ScalingStatus',
    'ScalingPolicy',
    'ScalingEvent',
    'ResourceMonitor',
    'ScalingExecutor',
    'ScalingAutomationEngine',
    'scaling_engine'
]
