#!/usr/bin/env python3
"""
RQA2025 基础设施层性能评估器

负责评估性能条件和执行适应动作。
这是从AdaptiveConfigurator中拆分出来的职责单一的组件。
"""

import logging
from typing import Dict, Any, Optional, List
import re

from .adaptive_configurator import ConfigurationRule
from ..core.performance_monitor import global_performance_monitor

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    性能评估器

    负责评估性能条件、执行配置动作，并提供性能优化建议。
    """

    def __init__(self):
        """初始化性能评估器"""
        logger.info("性能评估器初始化完成")

    def evaluate_condition(self, condition: str) -> bool:
        """
        评估条件表达式

        Args:
            condition: 条件表达式，如 "cpu_usage > 80"

        Returns:
            bool: 条件是否满足
        """
        try:
            # 获取性能指标
            metrics = global_performance_monitor.get_recent_metrics()

            # 解析条件表达式
            result = self._parse_condition(condition, metrics)

            logger.debug(f"条件评估: {condition} -> {result}")
            return result

        except Exception as e:
            logger.warning(f"条件评估失败 '{condition}': {e}")
            return False

    def execute_action(self, action: str, parameter_path: str) -> bool:
        """
        执行配置动作

        Args:
            action: 动作名称
            parameter_path: 参数路径

        Returns:
            bool: 是否成功执行
        """
        try:
            # 解析参数路径
            parts = parameter_path.split('.')
            if len(parts) < 2:
                return False

            component_name = parts[0]
            config_key = '.'.join(parts[1:])

            # 获取当前配置
            current_config = self._get_current_config(component_name, config_key)
            if current_config is None:
                logger.warning(f"无法获取当前配置: {parameter_path}")
                return False

            # 计算新配置值
            new_value = self._calculate_new_value(action, current_config, parameter_path)

            # 应用配置
            return self._apply_config_change(component_name, config_key, current_config, new_value, action)

        except Exception as e:
            logger.error(f"执行动作失败 '{action}' for {parameter_path}: {e}")
            return False

    def get_performance_insights(self) -> Dict[str, Any]:
        """
        获取性能洞察信息

        Returns:
            Dict[str, Any]: 性能洞察
        """
        try:
            metrics = global_performance_monitor.get_recent_metrics()

            insights = {
                'bottlenecks': self._identify_bottlenecks(metrics),
                'trends': self._analyze_trends(),
                'recommendations': self._generate_recommendations(metrics),
                'anomalies': self._detect_anomalies(metrics)
            }

            return insights

        except Exception as e:
            logger.error(f"获取性能洞察失败: {e}")
            return {}

    def validate_rule_condition(self, rule: ConfigurationRule) -> List[str]:
        """
        验证规则条件

        Args:
            rule: 配置规则

        Returns:
            List[str]: 验证错误列表
        """
        errors = []

        # 检查条件语法
        if not self._is_valid_condition_syntax(rule.condition):
            errors.append(f"条件语法无效: {rule.condition}")

        # 检查指标是否存在
        metrics = global_performance_monitor.get_recent_metrics()
        metric_name = self._extract_metric_name(rule.condition)
        if metric_name and metric_name not in metrics:
            errors.append(f"指标不存在: {metric_name}")

        # 检查动作是否支持
        if not self._is_supported_action(rule.action):
            errors.append(f"不支持的动作: {rule.action}")

        return errors

    def _parse_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """
        解析条件表达式

        Args:
            condition: 条件表达式
            metrics: 性能指标

        Returns:
            bool: 条件结果
        """
        # 简单的条件解析器
        parts = condition.split()
        if len(parts) != 3:
            return False

        metric_name, operator, threshold_str = parts
        threshold = float(threshold_str)

        if metric_name not in metrics:
            return False

        value = metrics[metric_name]

        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001
        elif operator == "!=":
            return abs(value - threshold) >= 0.001

        return False

    def _extract_metric_name(self, condition: str) -> Optional[str]:
        """
        从条件中提取指标名称

        Args:
            condition: 条件表达式

        Returns:
            Optional[str]: 指标名称
        """
        parts = condition.split()
        if len(parts) >= 1:
            return parts[0]
        return None

    def _is_valid_condition_syntax(self, condition: str) -> bool:
        """
        检查条件语法是否有效

        Args:
            condition: 条件表达式

        Returns:
            bool: 是否有效
        """
        # 简单的语法检查
        pattern = r'^\w+\s*[><=!]+\s*\d+(\.\d+)?$'
        return bool(re.match(pattern, condition.strip()))

    def _is_supported_action(self, action: str) -> bool:
        """
        检查动作是否支持

        Args:
            action: 动作名称

        Returns:
            bool: 是否支持
        """
        supported_actions = {
            'increase_interval', 'decrease_interval', 'optimize_memory',
            'scale_up', 'scale_down', 'enable_cache', 'disable_cache'
        }
        return action in supported_actions

    def _calculate_new_value(self, action: str, current_value: Any, parameter_path: str) -> Any:
        """
        计算新配置值

        Args:
            action: 动作
            current_value: 当前值
            parameter_path: 参数路径

        Returns:
            Any: 新值
        """
        from ..core.constants import ADAPTATION_FACTOR_BALANCED

        if not isinstance(current_value, (int, float)):
            return current_value

        factor = ADAPTATION_FACTOR_BALANCED

        if action.startswith("increase"):
            return current_value * factor
        elif action.startswith("decrease"):
            return current_value / factor
        elif action.startswith("optimize"):
            # 基于历史数据优化
            return self._optimize_value(parameter_path, current_value)

        return current_value

    def _optimize_value(self, parameter_path: str, current_value: Any) -> Any:
        """
        基于历史数据优化值

        Args:
            parameter_path: 参数路径
            current_value: 当前值

        Returns:
            Any: 优化后的值
        """
        # 这里可以实现更复杂的优化逻辑
        # 暂时返回当前值
        return current_value

    def _get_current_config(self, component_name: str, config_key: str) -> Any:
        """
        获取当前配置值

        Args:
            component_name: 组件名称
            config_key: 配置键

        Returns:
            Any: 配置值
        """
        # 这里需要访问组件的配置
        # 暂时返回默认值
        return getattr(self, f'_get_{component_name}_config', lambda k: None)(config_key)

    def _apply_config_change(self, component_name: str, config_key: str,
                           old_value: Any, new_value: Any, reason: str) -> bool:
        """
        应用配置变更

        Args:
            component_name: 组件名称
            config_key: 配置键
            old_value: 旧值
            new_value: 新值
            reason: 变更原因

        Returns:
            bool: 是否成功
        """
        try:
            # 发布配置变更事件
            from ..core.component_bus import publish_event
            publish_event("component.config.updated", {
                'component': component_name,
                'key': config_key,
                'old_value': old_value,
                'new_value': new_value,
                'reason': reason
            })

            logger.info(f"配置变更: {component_name}.{config_key} {old_value} -> {new_value}")
            return True

        except Exception as e:
            logger.error(f"应用配置变更失败: {e}")
            return False

    def _identify_bottlenecks(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        识别性能瓶颈

        Args:
            metrics: 性能指标

        Returns:
            List[Dict[str, Any]]: 瓶颈列表
        """
        bottlenecks = []

        # CPU瓶颈检测
        if metrics.get('cpu_usage', 0) > 80:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'high' if metrics['cpu_usage'] > 90 else 'medium',
                'current_value': metrics['cpu_usage'],
                'threshold': 80,
                'recommendation': '考虑增加CPU资源或优化CPU密集型任务'
            })

        # 内存瓶颈检测
        if metrics.get('memory_usage', 0) > 85:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'high' if metrics['memory_usage'] > 95 else 'medium',
                'current_value': metrics['memory_usage'],
                'threshold': 85,
                'recommendation': '考虑增加内存或优化内存使用'
            })

        return bottlenecks

    def _analyze_trends(self) -> Dict[str, Any]:
        """
        分析性能趋势

        Returns:
            Dict[str, Any]: 趋势分析结果
        """
        # 这里可以实现趋势分析逻辑
        return {
            'cpu_trend': 'stable',
            'memory_trend': 'increasing',
            'analysis_period': '1h'
        }

    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """
        生成优化建议

        Args:
            metrics: 性能指标

        Returns:
            List[str]: 建议列表
        """
        recommendations = []

        if metrics.get('cpu_usage', 0) > 70:
            recommendations.append("CPU使用率较高，建议优化计算密集型任务")

        if metrics.get('memory_usage', 0) > 80:
            recommendations.append("内存使用率较高，建议检查内存泄漏")

        if len(recommendations) == 0:
            recommendations.append("系统性能表现良好，继续监控")

        return recommendations

    def _detect_anomalies(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        检测性能异常

        Args:
            metrics: 性能指标

        Returns:
            List[Dict[str, Any]]: 异常列表
        """
        anomalies = []

        # 简单的异常检测
        for metric_name, value in metrics.items():
            if value > 95:  # 极端值检测
                anomalies.append({
                    'metric': metric_name,
                    'value': value,
                    'type': 'extreme_value',
                    'severity': 'critical'
                })

        return anomalies

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            insights = self.get_performance_insights()

            return {
                'status': 'healthy',
                'bottlenecks_count': len(insights.get('bottlenecks', [])),
                'anomalies_count': len(insights.get('anomalies', [])),
                'recommendations_count': len(insights.get('recommendations', [])),
                'last_evaluation': 'now'
            }

        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局性能评估器实例
global_performance_evaluator = PerformanceEvaluator()
