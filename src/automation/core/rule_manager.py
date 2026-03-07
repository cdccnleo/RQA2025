#!/usr/bin/env python3
"""
RQA2025 自动化层规则管理器
Automation Layer Rule Manager

实现规则的创建、管理和评估。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .automation_models import AutomationRule, AutomationType, TaskPriority, AutomationMetrics
from .rule_executor import RuleExecutor

# 获取统一基础设施集成层的日志适配器
try:
    from src.core.integration import get_models_adapter
    models_adapter = get_models_adapter()
except Exception as e:
    models_adapter = None

# 日志记录
try:
    from src.infrastructure.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    try:
        from src.infrastructure.logging.core.interfaces import get_logger
        logger = get_logger(__name__)
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)


class RuleManager:

    """
    规则管理器
    负责自动化规则的生命周期管理和评估
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 规则存储
        self.rules: Dict[str, AutomationRule] = {}

        # 规则执行器
        self.executor = RuleExecutor(self.config.get('executor_config', {}))

        # 规则模板
        self.rule_templates = self._load_rule_templates()

        # 指标收集
        self.metrics = AutomationMetrics()

        logger.info("规则管理器已初始化")

    def create_rule(self, rule_data: Dict[str, Any]) -> AutomationRule:
        """创建规则"""
        rule_id = rule_data.get(
            'rule_id') or f"rule_{datetime.now().strftime('%Y % m % d % H % M % S % f')}"

        rule = AutomationRule(
            rule_id=rule_id,
            name=rule_data['name'],
            description=rule_data.get('description', ''),
            automation_type=AutomationType(rule_data.get('automation_type', 'rule_based')),
            conditions=rule_data['conditions'],
            actions=rule_data['actions'],
            priority=TaskPriority(rule_data.get('priority', 'medium')),
            enabled=rule_data.get('enabled', True)
        )

        self.rules[rule_id] = rule
        logger.info(f"创建规则: {rule_id} - {rule.name}")

        return rule

    def create_rule_from_template(self, template_name: str, customizations: Dict[str, Any]) -> AutomationRule:
        """从模板创建规则"""
        if template_name not in self.rule_templates:
            raise ValueError(f"规则模板不存在: {template_name}")

        template = self.rule_templates[template_name]
        rule_data = template.copy()
        rule_data.update(customizations)

        return self.create_rule(rule_data)

    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新规则"""
        if rule_id not in self.rules:
            return False

        rule = self.rules[rule_id]

        # 更新字段
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        rule.updated_at = datetime.now()
        logger.info(f"更新规则: {rule_id}")

        return True

    async def delete_rule(self, rule_id: str) -> bool:
        """删除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"删除规则: {rule_id}")
            return True
        return False

    async def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        return await self.update_rule(rule_id, {'enabled': True})

    async def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        return await self.update_rule(rule_id, {'enabled': False})

    async def evaluate_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估所有规则"""
        triggered_rules = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            if await self._evaluate_rule_conditions(rule, context):
                triggered_rules.append({
                    'rule': rule,
                    'context': context,
                    'evaluation_time': datetime.now().isoformat()
                })

                self.metrics.rule_execution_count[rule.rule_id] = \
                    self.metrics.rule_execution_count.get(rule.rule_id, 0) + 1

        return triggered_rules

    async def get_rule(self, rule_id: str) -> Optional[AutomationRule]:
        """获取规则"""
        return self.rules.get(rule_id)

    def list_rules(self, filters: Optional[Dict[str, Any]] = None) -> List[AutomationRule]:
        """列出规则"""
        rules = list(self.rules.values())

        if filters:
            filtered_rules = []
            for rule in rules:
                match = True
                for key, value in filters.items():
                    if hasattr(rule, key):
                        rule_value = getattr(rule, key)
                        if isinstance(rule_value, AutomationType):
                            rule_value = rule_value.value
                        elif isinstance(rule_value, TaskPriority):
                            rule_value = rule_value.value

                        if rule_value != value:
                            match = False
                            break

                if match:
                    filtered_rules.append(rule)

            rules = filtered_rules

        return rules

    def get_rule_templates(self) -> Dict[str, Dict[str, Any]]:
        """获取规则模板"""
        return self.rule_templates.copy()

    async def _evaluate_rule_conditions(self, rule: AutomationRule, context: Dict[str, Any]) -> bool:
        """评估规则条件"""
        conditions = rule.conditions

        for condition_key, condition_value in conditions.items():
            if not await self._evaluate_single_condition(condition_key, condition_value, context):
                return False

        return True

    async def _evaluate_single_condition(self, condition_key: str, condition_value: Any, context: Dict[str, Any]) -> bool:
        """评估单个条件"""
        if condition_key == 'metric_threshold':
            return await self._evaluate_metric_threshold(condition_value, context)
        elif condition_key == 'time_window':
            return await self._evaluate_time_window(condition_value, context)
        elif condition_key == 'event_type':
            return await self._evaluate_event_type(condition_value, context)
        elif condition_key == 'service_status':
            return await self._evaluate_service_status(condition_value, context)
        elif condition_key == 'performance_metric':
            return await self._evaluate_performance_metric(condition_value, context)
        else:
            # 简单值匹配
            context_value = context.get(condition_key)
            return context_value == condition_value

    async def _evaluate_metric_threshold(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估指标阈值条件"""
        metric_name = condition.get('metric')
        threshold = condition.get('threshold')
        operator = condition.get('operator', '>')

        context_value = context.get(metric_name)
        if context_value is None:
            return False

        if operator == '>' and not (context_value > threshold):
            return False
        elif operator == '<' and not (context_value < threshold):
            return False
        elif operator == '>=' and not (context_value >= threshold):
            return False
        elif operator == '<=' and not (context_value <= threshold):
            return False
        elif operator == '==' and not (context_value == threshold):
            return False

        return True

    async def _evaluate_time_window(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估时间窗口条件"""
        start_time = condition.get('start_time')
        end_time = condition.get('end_time')
        current_time = datetime.now().time()

        if start_time and end_time:
            return start_time <= current_time <= end_time

        return True

    async def _evaluate_event_type(self, condition: str, context: Dict[str, Any]) -> bool:
        """评估事件类型条件"""
        event_type = context.get('event_type')
        return event_type == condition

    async def _evaluate_service_status(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估服务状态条件"""
        service_name = condition.get('service_name')
        expected_status = condition.get('status')

        # 这里应该从服务发现或监控系统获取实际状态
        # 暂时从context获取
        service_status = context.get(f'service_{service_name}_status')
        return service_status == expected_status

    async def _evaluate_performance_metric(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估性能指标条件"""
        metric_name = condition.get('metric')
        threshold = condition.get('threshold')
        operator = condition.get('operator', '>')

        # 这里应该从监控系统获取实际性能指标
        # 暂时从context获取
        metric_value = context.get(metric_name)
        if metric_value is None:
            return False

        return await self._evaluate_metric_threshold({
            'metric': metric_name,
            'threshold': threshold,
            'operator': operator
        }, context)

    def _load_rule_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载规则模板"""
        return {
            'cpu_high_scaling': {
                'name': 'CPU使用率高时自动扩展',
                'description': '当CPU使用率超过阈值时自动扩展服务实例',
                'automation_type': 'rule_based',
                'conditions': {
                    'metric_threshold': {
                        'metric': 'cpu_usage_percent',
                        'threshold': 80.0,
                        'operator': '>'
                    },
                    'time_window': {
                        'start_time': '09:00:00',
                        'end_time': '18:00:00'
                    }
                },
                'actions': [
                    {
                        'type': 'scaling',
                        'service_name': 'trading_engine',
                        'scaling_type': 'scale_up',
                        'target_instances': 2
                    },
                    {
                        'type': 'notification',
                        'message': 'CPU使用率过高，已触发自动扩展',
                        'channels': ['slack', 'email'],
                        'priority': 'high'
                    }
                ],
                'priority': 'high'
            },

            'memory_low_restart': {
                'name': '内存不足时重启服务',
                'description': '当内存使用率过低（可能存在内存泄漏）时重启服务',
                'automation_type': 'rule_based',
                'conditions': {
                    'metric_threshold': {
                        'metric': 'memory_usage_percent',
                        'threshold': 5.0,
                        'operator': '<'
                    }
                },
                'actions': [
                    {
                        'type': 'restart_service',
                        'service_name': 'data_processor',
                        'restart_type': 'graceful',
                        'timeout_seconds': 30
                    },
                    {
                        'type': 'notification',
                        'message': '检测到内存使用异常，已重启服务',
                        'channels': ['email'],
                        'priority': 'medium'
                    }
                ],
                'priority': 'medium'
            },

            'error_rate_high_alert': {
                'name': '错误率高时告警',
                'description': '当服务错误率超过阈值时发送告警',
                'automation_type': 'rule_based',
                'conditions': {
                    'metric_threshold': {
                        'metric': 'error_rate_percent',
                        'threshold': 5.0,
                        'operator': '>'
                    }
                },
                'actions': [
                    {
                        'type': 'notification',
                        'message': '服务错误率异常升高，请立即检查',
                        'channels': ['slack', 'email', 'sms'],
                        'priority': 'critical'
                    }
                ],
                'priority': 'critical'
            },

            'daily_backup': {
                'name': '每日自动备份',
                'description': '每天特定时间执行数据备份',
                'automation_type': 'scheduled',
                'conditions': {
                    'time_window': {
                        'start_time': '02:00:00',
                        'end_time': '03:00:00'
                    }
                },
                'actions': [
                    {
                        'type': 'run_script',
                        'script_path': '/opt / rqa2025 / scripts / backup.sh',
                        'script_args': ['--full - backup'],
                        'timeout_seconds': 3600
                    },
                    {
                        'type': 'notification',
                        'message': '每日备份已完成',
                        'channels': ['email'],
                        'priority': 'low'
                    }
                ],
                'priority': 'medium'
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        return {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules.values() if r.enabled]),
            'rule_execution_count': self.metrics.rule_execution_count,
            'templates_available': len(self.rule_templates)
        }


__all__ = [
    'RuleManager'
]
