#!/usr/bin/env python3
"""
RQA2025 基础设施层自适应配置器

实现运行时配置的动态调整和优化。
基于系统负载、性能指标和历史数据智能调整配置参数。
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, List
import threading
from datetime import datetime, timedelta

from ..core.performance_monitor import global_performance_monitor, monitor_performance
from ..core.component_bus import global_component_bus, Message, MessageType
from ..core.constants import (
    ADAPTATION_HISTORY_RETENTION_DAYS,
    ADAPTATION_COOLDOWN_HIGH, ADAPTATION_COOLDOWN_LOW, ADAPTATION_COOLDOWN_DEFAULT
)

# 导入规则类型
from .rule_types import AdaptationStrategy, ConfigurationRule, AdaptationHistory

# 导入拆分的组件
from .configuration_rule_manager import global_rule_manager, ConfigurationRuleManager
from .baseline_manager import global_baseline_manager, BaselineManager

logger = logging.getLogger(__name__)


class AdaptiveConfigurator:
    """
    自适应配置器

    基于实时监控数据和性能指标，动态调整系统配置。
    支持多种适应策略和配置规则。
    """

    def __init__(
        self,
        strategy: AdaptationStrategy = AdaptationStrategy.BALANCED,
        monitoring_interval: int = 60,
        *,
        load_default_rules: bool = True,
    ):
        """
        初始化自适应配置器

        Args:
            strategy: 适应策略
            monitoring_interval: 监控间隔（秒）
        """
        self.strategy = strategy
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        self.monitor_thread = None
        self.stop_event = threading.Event()

        # 使用拆分的组件
        self.rule_manager = ConfigurationRuleManager()
        self.rule_lock = threading.RLock()
        from .performance_evaluator import PerformanceEvaluator  # 避免循环导入
        self.performance_evaluator = PerformanceEvaluator()
        self.baseline_manager = BaselineManager()

        # 适应历史
        self.adaptation_history: List[AdaptationHistory] = []
        self.history_lock = threading.RLock()

        # 配置快照
        self.config_snapshots = {}
        self.snapshot_lock = threading.RLock()
        
        # ⭐ 兼容性属性：为测试提供向后兼容的访问方式
        self.baseline_lock = self.baseline_manager.baseline_lock if hasattr(self.baseline_manager, 'baseline_lock') else threading.RLock()
        self.baseline_data: Dict[str, List[float]] = {}
        self._legacy_rules_cache: List[ConfigurationRule] = []

        # 添加默认规则
        if load_default_rules:
            self._init_default_rules()
        self._sync_legacy_rules_cache()

        # 事件订阅
        self._setup_event_subscriptions()

        logger.info(f"自适应配置器初始化完成，策略: {strategy.value}")

    def _init_default_rules(self):
        """初始化默认规则"""
        default_rules = self.rule_manager.create_default_rules(self.strategy)
        for rule in default_rules:
            self.rule_manager.add_rule(rule)
        self._sync_legacy_rules_cache()

    def _setup_event_subscriptions(self):
        """设置事件订阅"""
        global_component_bus.subscribe("AdaptiveConfigurator", "performance.metrics.updated", self._handle_performance_update)
        global_component_bus.subscribe("AdaptiveConfigurator", "system.load.changed", self._handle_load_change)
        global_component_bus.subscribe("AdaptiveConfigurator", "component.config.updated", self._handle_config_change)

    @property
    def rules(self) -> List[ConfigurationRule]:
        """获取所有规则（兼容性属性）"""
        return list(self._legacy_rules_cache)

    def _sync_legacy_rules_cache(self) -> None:
        """同步兼容性规则缓存"""
        if hasattr(self.rule_manager, "get_rules"):
            rules = self.rule_manager.get_rules()
        else:
            rules = getattr(self.rule_manager, "rules", [])
        self._legacy_rules_cache = list(rules)
    
    def _evaluate_condition(self, condition: str, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        评估条件（兼容性方法）
        
        Args:
            condition: 条件表达式
            metrics: 性能指标
            
        Returns:
            条件是否满足
        """
        metrics = metrics or {}
        if not metrics:
            try:
                metrics = global_performance_monitor.get_recent_metrics()
            except Exception:
                metrics = {}
        try:
            return bool(eval(condition, {"__builtins__": {}}, metrics))
        except Exception:
            pass
        if hasattr(self.performance_evaluator, 'evaluate_condition'):
            try:
                return self.performance_evaluator.evaluate_condition(condition, metrics)
            except TypeError:
                return self.performance_evaluator.evaluate_condition(condition)
        return False
    
    def _calculate_new_value(
        self,
        action_or_rule: Any,
        current_value: Any,
        parameter_path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        计算新配置值（兼容性方法）
        
        Args:
            action_or_rule: 动作字符串或配置规则
            current_value: 当前值
            parameter_path: 参数路径（旧版调用约定）
            metrics: 性能指标
            
        Returns:
            新的配置值
        """
        metrics = metrics or {}
        rule: Optional[ConfigurationRule] = None
        action: str = ""

        if isinstance(action_or_rule, ConfigurationRule):
            rule = action_or_rule
            action = (rule.action or "").lower()
            parameter_path = rule.parameter_path

            if rule.adjustment_value is not None:
                return rule.adjustment_value

            if hasattr(self.performance_evaluator, "calculate_new_value"):
                try:
                    return self.performance_evaluator.calculate_new_value(current_value, rule, metrics)
                except TypeError:
                    # 兼容旧版签名
                    pass
        else:
            action = str(action_or_rule or "").lower()

        multipliers = {
            AdaptationStrategy.BALANCED: 1.25,
            AdaptationStrategy.CONSERVATIVE: 1.1,
            AdaptationStrategy.AGGRESSIVE: 1.5,
        }
        multiplier = multipliers.get(self.strategy, 1.25)

        if action == "increase_interval":
            return current_value * multiplier
        if action == "decrease_interval":
            return current_value / multiplier if multiplier else current_value
        if action in {"set_value", "assign"} and rule and rule.adjustment_value is not None:
            return rule.adjustment_value

        return current_value
    
    def add_rule(self, rule: ConfigurationRule):
        """
        添加配置规则

        Args:
            rule: 配置规则
        """
        success = self.rule_manager.add_rule(rule)
        if success:
            self._sync_legacy_rules_cache()
        return success

    def remove_rule(self, parameter_path: str):
        """
        移除配置规则

        Args:
            parameter_path: 参数路径
        """
        removed = self.rule_manager.remove_rule(parameter_path)
        if removed:
            self._sync_legacy_rules_cache()
        return removed

    def start(self):
        """启动自适应配置器"""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="AdaptiveConfigurator-Monitor",
            daemon=True
        )
        self.monitor_thread.start()

        logger.info("自适应配置器已启动")

    def stop(self):
        """停止自适应配置器"""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        logger.info("自适应配置器已停止")

    @monitor_performance("adaptive_configurator.monitoring_loop")
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                self._evaluate_rules()
                self._update_baselines()
                self._cleanup_history()

                # 使用停止事件进行可中断的睡眠
                if self.stop_event.wait(self.monitoring_interval):
                    break  # 如果收到停止信号，退出循环

            except Exception as e:
                logger.error(f"自适应配置器监控循环异常: {e}")
                # 错误时等待更长时间，但也可被中断
                if self.stop_event.wait(ADAPTATION_COOLDOWN_HIGH):
                    break

    def _evaluate_rules(self):
        """评估配置规则"""
        active_rules = self.rule_manager.get_active_rules()

        for rule in active_rules:
            # 评估条件
            if self.performance_evaluator.evaluate_condition(rule.condition):
                # 执行动作
                if self.performance_evaluator.execute_action(rule.action, rule.parameter_path):
                    self.rule_manager.update_rule_last_applied(rule.parameter_path, rule.condition)
                    logger.info(f"应用配置规则: {rule.parameter_path}")




    def _optimize_value(self, parameter_path: str, current_value: Any) -> Any:
        """
        基于历史数据优化值

        Args:
            parameter_path: 参数路径
            current_value: 当前值

        Returns:
            Any: 优化后的值
        """
        # 获取历史适应数据
        with self.history_lock:
            relevant_history = [
                h for h in self.adaptation_history
                if h.parameter_path == parameter_path and h.performance_impact
            ]

        if len(relevant_history) < 3:
            return current_value

        # 计算性能影响
        impacts = [h.performance_impact.get('overall_score', 0) for h in relevant_history]
        values = [h.new_value for h in relevant_history]

        if not impacts or not values:
            return current_value

        # 找到性能最好的值
        best_idx = impacts.index(max(impacts))
        return values[best_idx]

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
        # 可以通过组件总线或其他方式获取
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
            # 记录适应历史
            history_entry = AdaptationHistory(
                timestamp=datetime.now(),
                parameter_path=f"{component_name}.{config_key}",
                old_value=old_value,
                new_value=new_value,
                reason=reason
            )

            with self.history_lock:
                self.adaptation_history.append(history_entry)

            # 发布配置变更事件
            global_component_bus.publish(Message(
                type=MessageType.CONFIG_UPDATED,
                topic="config.adaptive_change",
                payload={
                    'component': component_name,
                    'key': config_key,
                    'old_value': old_value,
                    'new_value': new_value,
                    'reason': reason
                }
            ))

            logger.info(f"自适应配置变更: {component_name}.{config_key} {old_value} -> {new_value} ({reason})")
            return True

        except Exception as e:
            logger.error(f"应用配置变更失败: {e}")
            return False

    def _update_baselines(self):
        """更新性能基线"""
        try:
            metrics = global_performance_monitor.get_recent_metrics()

            for metric_name, value in metrics.items():
                self.baseline_manager.update_baseline(metric_name, value)
                with self.baseline_lock:
                    values = self.baseline_data.setdefault(metric_name, [])
                    values.append(value)
                    if len(values) > 100:
                        del values[:-100]

        except Exception as e:
            logger.warning(f"更新性能基线失败: {e}")

    def _cleanup_history(self):
        """清理历史记录"""
        cutoff_date = datetime.now() - timedelta(days=ADAPTATION_HISTORY_RETENTION_DAYS)

        with self.history_lock:
            original_count = len(self.adaptation_history)
            self.adaptation_history = [
                h for h in self.adaptation_history
                if h.timestamp > cutoff_date
            ]

            removed_count = original_count - len(self.adaptation_history)
            if removed_count > 0:
                logger.debug(f"清理 {removed_count} 条过期适应历史记录")

    def get_baseline_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        获取基线统计信息

        Args:
            metric_name: 指标名称

        Returns:
            Optional[Dict[str, float]]: 统计信息
        """
        stats = self.baseline_manager.get_baseline_stats(metric_name)
        if stats:
            return stats

        with self.baseline_lock:
            values = self.baseline_data.get(metric_name)
            if not values:
                return None

            count = len(values)
            if count == 0:
                return None

            return {
                'mean': sum(values) / count,
                'min': min(values),
                'max': max(values),
                'count': count,
                'latest': values[-1],
            }

    def get_adaptation_history(self, parameter_path: Optional[str] = None,
                              limit: int = 50) -> List[AdaptationHistory]:
        """
        获取适应历史记录

        Args:
            parameter_path: 参数路径过滤
            limit: 限制返回数量

        Returns:
            List[AdaptationHistory]: 历史记录
        """
        with self.history_lock:
            history = list(self.adaptation_history)

            if parameter_path:
                history = [h for h in history if h.parameter_path == parameter_path]

            # 按时间倒序排列
            history.sort(key=lambda h: h.timestamp, reverse=True)

            return history[:limit]

    def _handle_performance_update(self, message: Message):
        """处理性能指标更新事件"""
        # 可以在这里添加额外的适应逻辑
        pass

    def _handle_load_change(self, message: Message):
        """处理系统负载变化事件"""
        load_data = message.payload
        logger.info(f"检测到负载变化: {load_data}")

        # 可以触发紧急适应措施
        if load_data.get('cpu_usage', 0) > 90:
            self._emergency_adaptation("high_cpu_load")

    def _handle_config_change(self, message: Message):
        """处理配置变更事件"""
        # 记录配置快照
        config_data = message.payload
        component = config_data.get('component')
        key = config_data.get('key')
        value = config_data.get('new_value')

        with self.snapshot_lock:
            if component not in self.config_snapshots:
                self.config_snapshots[component] = {}
            self.config_snapshots[component][key] = {
                'value': value,
                'timestamp': datetime.now()
            }

    def _emergency_adaptation(self, reason: str):
        """紧急适应措施"""
        logger.warning(f"触发紧急适应: {reason}")

        # 激进的配置调整
        emergency_rules = [
            ConfigurationRule(
                parameter_path="monitoring.interval",
                metric_name="cpu_usage",
                condition="cpu_usage > 85",
                action="increase_interval",
                priority=10,
                cooldown_minutes=ADAPTATION_COOLDOWN_HIGH
            )
        ]

        # 临时添加紧急规则
        for rule in emergency_rules:
            self.add_rule(rule)

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        with self.rule_lock:
            with self.history_lock:
                return {
                    'status': 'healthy' if self.is_running else 'stopped',
                    'rules_count': len(self.rules),
                    'history_count': len(self.adaptation_history),
                    'baseline_metrics': len(self.baseline_data),
                    'last_adaptation': max([h.timestamp for h in self.adaptation_history], default=None),
                    'strategy': self.strategy.value
                }


# 默认配置规则
DEFAULT_ADAPTATION_RULES = [
    ConfigurationRule(
        parameter_path="monitoring.collection_interval",
        metric_name="cpu_usage",
        condition="cpu_usage > 80",
        action="increase_interval",
        priority=5,
        cooldown_minutes=ADAPTATION_COOLDOWN_HIGH
    ),
    ConfigurationRule(
        parameter_path="monitoring.collection_interval",
        metric_name="cpu_usage",
        condition="cpu_usage < 30",
        action="decrease_interval",
        priority=3,
        cooldown_minutes=ADAPTATION_COOLDOWN_LOW
    ),
    ConfigurationRule(
        parameter_path="alert.check_interval",
        metric_name="memory_usage",
        condition="memory_usage > 85",
        action="increase_interval",
        priority=4,
        cooldown_minutes=ADAPTATION_COOLDOWN_DEFAULT
    )
]


def create_adaptive_configurator(strategy: AdaptationStrategy = AdaptationStrategy.BALANCED,
                                rules: Optional[List[ConfigurationRule]] = None) -> AdaptiveConfigurator:
    """
    创建自适应配置器

    Args:
        strategy: 适应策略
        rules: 配置规则列表

    Returns:
        AdaptiveConfigurator: 配置器实例
    """
    configurator = AdaptiveConfigurator(strategy=strategy, load_default_rules=False)

    # 添加默认规则
    for rule in DEFAULT_ADAPTATION_RULES:
        configurator.add_rule(rule)

    # 添加自定义规则
    if rules:
        for rule in rules:
            configurator.add_rule(rule)

    return configurator
