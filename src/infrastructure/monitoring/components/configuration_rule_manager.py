#!/usr/bin/env python3
"""
RQA2025 基础设施层配置规则管理器

负责管理自适应配置的规则，包括添加、移除、查找和验证规则。
这是从AdaptiveConfigurator中拆分出来的职责单一的组件。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import threading

from .rule_types import ConfigurationRule, AdaptationStrategy

logger = logging.getLogger(__name__)


class ConfigurationRuleManager:
    """
    配置规则管理器

    负责配置规则的生命周期管理，包括规则的添加、移除、查找和验证。
    """

    def __init__(self):
        """初始化配置规则管理器"""
        self.rules: List[ConfigurationRule] = []
        self.rule_lock = threading.RLock()
        self.last_validation_errors: List[str] = []

        logger.info("配置规则管理器初始化完成")

    def add_rule(self, rule: ConfigurationRule) -> bool:
        """
        添加配置规则

        Args:
            rule: 配置规则

        Returns:
            bool: 是否成功添加
        """
        with self.rule_lock:
            try:
                # 检查规则是否已存在
                for existing_rule in self.rules:
                    if existing_rule.parameter_path == rule.parameter_path and \
                       existing_rule.condition == rule.condition:
                        logger.warning(f"规则已存在: {rule.parameter_path}")
                        return False

                self.rules.append(rule)
                self._sort_rules_by_priority()
                logger.info(f"配置规则添加成功: {rule.parameter_path}")
                return True

            except Exception as e:
                logger.error(f"添加配置规则失败: {e}")
                return False

    def remove_rule(self, parameter_path: str, condition: Optional[str] = None) -> int:
        """
        移除配置规则

        Args:
            parameter_path: 参数路径
            condition: 条件（可选，用于精确匹配）

        Returns:
            int: 移除的规则数量
        """
        with self.rule_lock:
            original_count = len(self.rules)

            if condition:
                # 精确匹配
                self.rules = [
                    r for r in self.rules
                    if not (r.parameter_path == parameter_path and r.condition == condition)
                ]
            else:
                # 移除所有匹配参数路径的规则
                self.rules = [
                    r for r in self.rules
                    if r.parameter_path != parameter_path
                ]

            removed_count = original_count - len(self.rules)

            if removed_count > 0:
                logger.info(f"移除配置规则: {parameter_path} ({removed_count}个)")
            else:
                logger.warning(f"未找到要移除的规则: {parameter_path}")

            return removed_count

    def get_rules(self, parameter_path: Optional[str] = None) -> List[ConfigurationRule]:
        """
        获取配置规则

        Args:
            parameter_path: 参数路径过滤（可选）

        Returns:
            List[ConfigurationRule]: 配置规则列表
        """
        with self.rule_lock:
            if parameter_path:
                return [r for r in self.rules if r.parameter_path == parameter_path]
            return self.rules.copy()

    def get_active_rules(self) -> List[ConfigurationRule]:
        """
        获取活跃的配置规则（未在冷却期内）

        Returns:
            List[ConfigurationRule]: 活跃规则列表
        """
        with self.rule_lock:
            current_time = datetime.now()
            active_rules = []

            for rule in self.rules:
                # 检查是否在冷却期内
                if rule.last_applied and \
                   (current_time - rule.last_applied) < timedelta(minutes=rule.cooldown_minutes):
                    continue
                active_rules.append(rule)

            return active_rules

    def update_rule_last_applied(self, parameter_path: str, condition: str):
        """
        更新规则的最后应用时间

        Args:
            parameter_path: 参数路径
            condition: 条件
        """
        with self.rule_lock:
            for rule in self.rules:
                if rule.parameter_path == parameter_path and rule.condition == condition:
                    rule.last_applied = datetime.now()
                    logger.debug(f"更新规则最后应用时间: {parameter_path}")
                    break

    def validate_rule(self, rule: ConfigurationRule, *, return_errors: bool = False):
        """
        验证配置规则

        Args:
            rule: 配置规则
            return_errors: 是否返回错误列表

        Returns:
            当 return_errors=True 时返回错误列表，否则返回 bool
        """
        errors = []

        # 检查必需字段
        if not rule.parameter_path:
            errors.append("参数路径不能为空")

        if not rule.condition:
            errors.append("条件表达式不能为空")

        if rule.action is None and rule.adjustment_value is None:
            errors.append("必须指定调整值或执行动作")

        if rule.action and not rule.metric_name:
            errors.append("指标名称不能为空")

        # 验证优先级
        if not (1 <= rule.priority <= 10):
            errors.append("优先级必须在1-10之间")

        # 验证冷却时间
        if rule.cooldown_minutes < 0:
            errors.append("冷却时间不能为负数")

        self.last_validation_errors = errors
        if return_errors:
            return errors
        return len(errors) == 0

    def clear_all_rules(self) -> int:
        """
        清空所有规则

        Returns:
            int: 清空的规则数量
        """
        with self.rule_lock:
            cleared_count = len(self.rules)
            self.rules.clear()
            logger.info(f"清空所有配置规则: {cleared_count}个")
            return cleared_count

    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        获取规则统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.rule_lock:
            total_rules = len(self.rules)
            active_rules = len(self.get_active_rules())

            # 按优先级统计
            priorities = {}
            for rule in self.rules:
                priorities[rule.priority] = priorities.get(rule.priority, 0) + 1

            # 按参数路径统计
            param_paths = {}
            for rule in self.rules:
                param_paths[rule.parameter_path] = param_paths.get(rule.parameter_path, 0) + 1

            # 计算平均冷却时间
            avg_cooldown = sum(r.cooldown_minutes for r in self.rules) / total_rules if total_rules > 0 else 0

            return {
                'total_rules': total_rules,
                'active_rules': active_rules,
                'inactive_rules': total_rules - active_rules,
                'priorities_distribution': priorities,
                'parameters_distribution': param_paths,
                'avg_cooldown_minutes': round(avg_cooldown, 2),
                'last_updated': datetime.now().isoformat()
            }

    def _sort_rules_by_priority(self):
        """按优先级排序规则"""
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def create_default_rules(self, strategy: AdaptationStrategy) -> List[ConfigurationRule]:
        """
        创建默认配置规则

        Args:
            strategy: 适应策略

        Returns:
            List[ConfigurationRule]: 默认规则列表
        """
        # 基础规则配置
        base_rules = self._get_base_rules_config()

        # 根据策略调整规则
        adjusted_rules = self._adjust_rules_for_strategy(base_rules, strategy)

        # 转换为ConfigurationRule对象
        return [ConfigurationRule(**rule_config) for rule_config in adjusted_rules]

    def _get_base_rules_config(self) -> List[Dict[str, Any]]:
        """
        获取基础规则配置

        Returns:
            List[Dict[str, Any]]: 基础规则配置列表
        """
        from ..core.constants import ADAPTATION_COOLDOWN_HIGH, ADAPTATION_COOLDOWN_LOW, ADAPTATION_COOLDOWN_DEFAULT

        return [
            {
                "parameter_path": "monitoring.collection_interval",
                "metric_name": "cpu_usage",
                "condition": "cpu_usage > 80",
                "action": "increase_interval",
                "priority": 5,
                "cooldown_minutes": ADAPTATION_COOLDOWN_HIGH
            },
            {
                "parameter_path": "monitoring.collection_interval",
                "metric_name": "cpu_usage",
                "condition": "cpu_usage < 30",
                "action": "decrease_interval",
                "priority": 3,
                "cooldown_minutes": ADAPTATION_COOLDOWN_LOW
            },
            {
                "parameter_path": "alert.check_interval",
                "metric_name": "memory_usage",
                "condition": "memory_usage > 85",
                "action": "increase_interval",
                "priority": 4,
                "cooldown_minutes": ADAPTATION_COOLDOWN_DEFAULT
            }
        ]

    def _adjust_rules_for_strategy(self, rules: List[Dict[str, Any]],
                                  strategy: AdaptationStrategy) -> List[Dict[str, Any]]:
        """
        根据策略调整规则

        Args:
            rules: 基础规则列表
            strategy: 适应策略

        Returns:
            List[Dict[str, Any]]: 调整后的规则列表
        """
        adjusted_rules = []

        for rule in rules:
            rule_copy = rule.copy()

            if strategy == AdaptationStrategy.CONSERVATIVE:
                # 保守策略：降低优先级，增加冷却时间
                rule_copy["priority"] = max(1, rule_copy["priority"] - 1)
                rule_copy["cooldown_minutes"] += 5
            elif strategy == AdaptationStrategy.AGGRESSIVE:
                # 激进策略：提高优先级，减少冷却时间
                rule_copy["priority"] = min(10, rule_copy["priority"] + 1)
                rule_copy["cooldown_minutes"] = max(1, rule_copy["cooldown_minutes"] - 2)

            adjusted_rules.append(rule_copy)

        return adjusted_rules

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        with self.rule_lock:
            stats = self.get_rule_statistics()

            # 检查规则健康度
            issues = []

            if stats['total_rules'] == 0:
                issues.append("没有配置规则")
            elif stats['active_rules'] == 0:
                issues.append("所有规则都在冷却期内")

            # 检查规则分布是否均衡
            if len(stats['priorities_distribution']) < 3 and stats['total_rules'] > 5:
                issues.append("规则优先级分布不均衡")

            return {
                'status': 'healthy' if not issues else 'warning',
                'total_rules': stats['total_rules'],
                'active_rules': stats['active_rules'],
                'issues': issues,
                'last_check': datetime.now().isoformat()
            }


# 全局配置规则管理器实例
global_rule_manager = ConfigurationRuleManager()
