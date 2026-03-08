"""
特征层监控集成管理器

提供统一的监控集成接口，将监控体系集成到特征层的各个组件中，
包括性能监控、指标收集、告警管理等功能的自动化集成。
"""

import time
import threading
from typing import Dict, List, Optional, Any
from functools import wraps
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

# 使用统一基础设施集成层
try:
    from src.infrastructure.integration import get_features_adapter
    _features_adapter = get_features_adapter()
    _unified_monitoring = _features_adapter.get_monitoring() if _features_adapter else None
except ImportError:
    _features_adapter = None
    _unified_monitoring = None

from .features_monitor import MetricType, get_monitor


class IntegrationLevel(Enum):

    """集成级别枚举"""
    BASIC = "basic"           # 基础监控（仅关键指标）
    STANDARD = "standard"     # 标准监控（性能指标 + 警）
    ADVANCED = "advanced"     # 高级监控（全功能监控）


@dataclass
class ComponentIntegrationConfig:

    """组件集成配置"""
    component_name: str
    integration_level: IntegrationLevel
    auto_monitor: bool = True
    collect_metrics: bool = True
    enable_alerts: bool = True
    custom_metrics: List[str] = field(default_factory=list)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)


class MonitoringIntegrationManager:

    """
    特征层监控集成管理器

    提供统一的监控集成接口，自动将监控功能集成到特征层组件中。
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化监控集成管理器

        Args:
            config: 集成配置
        """
        self.config = config or {}
        self.monitor = get_monitor(self.config.get('monitor_config', {}))
        self.integrated_components: Dict[str, ComponentIntegrationConfig] = {}
        self.integration_locks: Dict[str, threading.Lock] = {}

        # 默认集成配置
        self.default_configs = {
            'FeatureEngineer': ComponentIntegrationConfig(
                component_name='FeatureEngineer',
                integration_level=IntegrationLevel.STANDARD,
                custom_metrics=['feature_generation_time',
                                'cache_hit_rate', 'data_validation_errors'],
                performance_thresholds={'feature_generation_time': 5.0, 'cache_hit_rate': 0.8}
            ),
            'TechnicalProcessor': ComponentIntegrationConfig(
                component_name='TechnicalProcessor',
                integration_level=IntegrationLevel.STANDARD,
                custom_metrics=['indicator_calculation_time',
                                'memory_usage', 'calculation_accuracy'],
                performance_thresholds={'indicator_calculation_time': 2.0, 'memory_usage': 100.0}
            ),
            'FeatureSelector': ComponentIntegrationConfig(
                component_name='FeatureSelector',
                integration_level=IntegrationLevel.BASIC,
                custom_metrics=['selection_time', 'selected_features_count'],
                performance_thresholds={'selection_time': 1.0}
            ),
            'FeatureStandardizer': ComponentIntegrationConfig(
                component_name='FeatureStandardizer',
                integration_level=IntegrationLevel.BASIC,
                custom_metrics=['standardization_time', 'data_quality_score'],
                performance_thresholds={'standardization_time': 1.0}
            ),
            'FeaturesMonitor': ComponentIntegrationConfig(
                component_name='FeaturesMonitor',
                integration_level=IntegrationLevel.ADVANCED,
                custom_metrics=['monitoring_overhead', 'alert_count', 'metric_collection_rate'],
                performance_thresholds={'monitoring_overhead': 0.1}
            )
        }

    def integrate_component(self, component: Any, component_type: str,


                            config: Optional[ComponentIntegrationConfig] = None) -> None:
        """
        集成组件到监控体系

        Args:
            component: 要集成的组件实例
            component_type: 组件类型
            config: 集成配置，如果为None则使用默认配置
        """
        if config is None:
            config = self.default_configs.get(component_type, ComponentIntegrationConfig(
                component_name=component_type,
                integration_level=IntegrationLevel.BASIC
            ))

        component_name = f"{component_type}_{id(component)}"

        # 注册组件到监控器
        self.monitor.register_component(component_name, component_type, component)

        # 存储集成配置
        self.integrated_components[component_name] = config
        self.integration_locks[component_name] = threading.Lock()

        # 根据集成级别应用监控功能
        self._apply_monitoring_integration(component, component_name, config)

        print(f"✅ 已集成组件: {component_name} (级别: {config.integration_level.value})")

    def _apply_monitoring_integration(self, component: Any, component_name: str,


                                      config: ComponentIntegrationConfig) -> None:
        """
        应用监控集成到组件

        Args:
            component: 组件实例
            component_name: 组件名称
            config: 集成配置
        """
        # 基础监控：添加性能监控装饰器
        if config.integration_level in [IntegrationLevel.BASIC, IntegrationLevel.STANDARD, IntegrationLevel.ADVANCED]:
            self._add_performance_monitoring(component, component_name, config)

        # 标准监控：添加指标收集
        if config.integration_level in [IntegrationLevel.STANDARD, IntegrationLevel.ADVANCED]:
            self._add_metrics_collection(component, component_name, config)

        # 高级监控：添加告警功能
        if config.integration_level == IntegrationLevel.ADVANCED:
            self._add_alert_functionality(component, component_name, config)

    def _add_performance_monitoring(self, component: Any, component_name: str,


                                    config: ComponentIntegrationConfig) -> None:
        """
        添加性能监控功能

        Args:
            component: 组件实例
            component_name: 组件名称
            config: 集成配置
        """
        # 为关键方法添加性能监控
        key_methods = self._get_key_methods(component)

        for method_name in key_methods:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)

                @wraps(original_method)
                def monitored_method(*args, **kwargs):

                    start_time = time.time()
                    try:
                        result = original_method(*args, **kwargs)
                        execution_time = time.time() - start_time

                        # 记录性能指标
                        self.monitor.collect_metrics(
                            component_name,
                            f"{method_name}_execution_time",
                            execution_time,
                            MetricType.HISTOGRAM
                        )

                        # 检查性能阈值
                        threshold_key = f"{method_name}_execution_time"
                        if threshold_key in config.performance_thresholds:
                            threshold = config.performance_thresholds[threshold_key]
                            if execution_time > threshold:
                                self.monitor.alert_manager.send_alert(
                                    f"{component_name}.{method_name}",
                                    "PERFORMANCE_THRESHOLD_EXCEEDED",
                                    f"执行时间 {execution_time:.2f}s 超过阈值 {threshold}s"
                                )

                        return result
                    except Exception as e:
                        execution_time = time.time() - start_time

                        # 记录错误指标
                        self.monitor.collect_metrics(
                            component_name,
                            f"{method_name}_error_count",
                            1,
                            MetricType.COUNTER
                        )

                        # 发送错误告警
                        if config.enable_alerts:
                            self.monitor.alert_manager.send_alert(
                                f"{component_name}.{method_name}",
                                "METHOD_EXECUTION_ERROR",
                                f"方法执行错误: {str(e)}"
                            )

                        raise

                setattr(component, method_name, monitored_method)

    def _add_metrics_collection(self, component: Any, component_name: str,


                                config: ComponentIntegrationConfig) -> None:
        """
        添加指标收集功能

        Args:
            component: 组件实例
            component_name: 组件名称
            config: 集成配置
        """
        # 添加指标收集方法

        def collect_custom_metric(metric_name: str, value: float,


                                  metric_type: MetricType = MetricType.GAUGE) -> None:
            """收集自定义指标"""
            self.monitor.collect_metrics(component_name, metric_name, value, metric_type)

        # 添加指标收集方法到组件
        component.collect_metric = collect_custom_metric

        # 为特定组件添加专门的指标收集
        if component_name.startswith('FeatureEngineer'):
            self._add_feature_engineer_metrics(component, component_name)
        elif component_name.startswith('TechnicalProcessor'):
            self._add_technical_processor_metrics(component, component_name)

    def _add_alert_functionality(self, component: Any, component_name: str,


                                 config: ComponentIntegrationConfig) -> None:
        """
        添加告警功能

        Args:
            component: 组件实例
            component_name: 组件名称
            config: 集成配置
        """
        # 添加告警方法

        def send_component_alert(alert_type: str, message: str,


                                 severity: str = "WARNING") -> None:
            """发送组件告警"""
            self.monitor.alert_manager.send_alert(
                severity, message, component_name
            )

        # 添加告警方法到组件
        component.send_alert = send_component_alert

    def _get_key_methods(self, component: Any) -> List[str]:
        """
        获取组件的关键方法列表

        Args:
            component: 组件实例

        Returns:
            关键方法名列表
        """
        # 根据组件类型返回关键方法
        component_type = type(component).__name__

        if component_type == 'FeatureEngineer':
            return ['generate_technical_features', 'generate_sentiment_features',
                    'merge_features', '_validate_stock_data']
        elif component_type == 'TechnicalProcessor':
            return ['calculate_rsi', 'calculate_macd', 'calculate_bollinger',
                    'calculate_ma', 'calculate_indicators']
        elif component_type == 'FeatureSelector':
            return ['select_features', 'fit', 'transform']
        elif component_type == 'FeatureStandardizer':
            return ['fit', 'transform', 'fit_transform']
        else:
            # 默认返回所有公共方法
            return [method for method in dir(component)
                    if not method.startswith('_') and callable(getattr(component, method))]

    def _add_feature_engineer_metrics(self, component: Any, component_name: str) -> None:
        """
        为FeatureEngineer添加专门的指标收集

        Args:
            component: 组件实例
            component_name: 组件名称
        """
        # 重写关键方法以添加指标收集
        original_generate_technical = component.generate_technical_features

        def monitored_generate_technical(*args, **kwargs):

            start_time = time.time()
            try:
                result = original_generate_technical(*args, **kwargs)

                # 收集特征生成指标
                execution_time = time.time() - start_time
                component.collect_metric('feature_generation_time', execution_time)

                if isinstance(result, pd.DataFrame):
                    component.collect_metric('generated_features_count', len(result.columns))
                    component.collect_metric('data_rows_count', len(result))

                return result
            except Exception as e:
                component.collect_metric('feature_generation_errors', 1, MetricType.COUNTER)
                raise

        component.generate_technical_features = monitored_generate_technical

    def _add_technical_processor_metrics(self, component: Any, component_name: str) -> None:
        """
        为TechnicalProcessor添加专门的指标收集

        Args:
            component: 组件实例
            component_name: 组件名称
        """
        # 重写关键方法以添加指标收集
        original_calculate_rsi = component.calculate_rsi

        def monitored_calculate_rsi(*args, **kwargs):

            start_time = time.time()
            try:
                result = original_calculate_rsi(*args, **kwargs)

                # 收集指标计算指标
                execution_time = time.time() - start_time
                component.collect_metric('indicator_calculation_time', execution_time)
                component.collect_metric('rsi_calculations', 1, MetricType.COUNTER)

                return result
            except Exception as e:
                component.collect_metric('indicator_calculation_errors', 1, MetricType.COUNTER)
                raise

        component.calculate_rsi = monitored_calculate_rsi

    def get_integration_status(self) -> Dict[str, Any]:
        """
        获取集成状态

        Returns:
            集成状态信息
        """
        status = {
            'integrated_components': len(self.integrated_components),
            'monitor_status': self.monitor.is_monitoring,
            'components': {}
        }

        for component_name, config in self.integrated_components.items():
            status['components'][component_name] = {
                'integration_level': config.integration_level.value,
                'auto_monitor': config.auto_monitor,
                'collect_metrics': config.collect_metrics,
                'enable_alerts': config.enable_alerts,
                'custom_metrics': config.custom_metrics
            }

        return status

    def export_integration_report(self, file_path: str) -> None:
        """
        导出集成报告

        Args:
            file_path: 报告文件路径
        """
        report = {
            'integration_status': self.get_integration_status(),
            'monitor_metrics': self.monitor.get_all_metrics(),
            'component_status': self.monitor.get_all_status(),
            'generated_at': datetime.now().isoformat()
        }

        with open(file_path, 'w', encoding='utf - 8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


def integrate_feature_layer_components(config: Optional[Dict] = None) -> MonitoringIntegrationManager:
    """
    集成特征层所有组件的便捷函数

    Args:
        config: 集成配置

    Returns:
        监控集成管理器实例
    """
    integration_manager = MonitoringIntegrationManager(config)

    # 这里可以自动发现和集成特征层的组件
    # 目前需要手动调用 integrate_component 方法

    return integration_manager


# 便捷的监控装饰器

def monitor_operation(component_name: str, operation_name: str):
    """
    监控操作装饰器

    Args:
        component_name: 组件名称
        operation_name: 操作名称
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            monitor = get_monitor()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # 记录性能指标
                monitor.collect_metrics(
                    component_name,
                    f"{operation_name}_execution_time",
                    execution_time,
                    MetricType.HISTOGRAM
                )

                return result
            except Exception as e:
                execution_time = time.time() - start_time

                # 记录错误指标
                monitor.collect_metrics(
                    component_name,
                    f"{operation_name}_error_count",
                    1,
                    MetricType.COUNTER
                )

                raise

        return wrapper
    return decorator
