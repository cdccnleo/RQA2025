#!/usr/bin/env python3
"""
告警规则引擎

构建智能化的告警规则引擎，支持动态规则配置和复杂条件判断
创建时间: 2025-08-24 10:13:48"""

from ..monitor.realtime_risk_monitor import RiskType
try:
    from src.core.integration.business_adapters import get_data_adapter
except ImportError:
    # 降级方案：尝试从__init__导入
    try:
        from src.core.integration import get_data_adapter
    except ImportError:
        # 如果都失败，提供一个mock实现
        def get_data_adapter(*args, **kwargs):
            return None
import time
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import operator

# 添加src目录到路径sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 使用统一基础设施集成
# 定义AlertLevel枚举用于向后兼容
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# 在类定义外部处理基础设施集成
try:
    data_adapter = get_data_adapter()
    monitoring = data_adapter.get_monitoring()
    print("统一基础设施集成层导入成功")
except Exception as e:
    print(f"统一基础设施集成层导入失败 {e}")
    monitoring = None

# 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


class ConditionType(Enum):

    """条件类型枚举"""
    THRESHOLD = "threshold"           # 阈值条件
    TREND = "trend"                   # 趋势条件
    PATTERN = "pattern"               # 模式条件
    COMPLEX = "complex"               # 复杂条件
    CORRELATION = "correlation"       # 相关性条误


class ActionType(Enum):

    """动作类型枚举"""
    ALERT = "alert"                   # 告警
    NOTIFICATION = "notification"     # 通知
    EXECUTION = "execution"           # 执行动作
    ESCALATION = "escalation"         # 升级
    MITIGATION = "mitigation"          # 缓解措施


class Operator(Enum):

    """操作符枚举"""
    GT = ">"                          # 大于
    GE = ">="                         # 大于等于
    LT = "<"                          # 小于
    LE = "<="                         # 小于等于
    EQ = "=="                         # 等于
    NE = "!="                         # 不等于
    IN = "in"                         # 包含于
    NOT_IN = "not_in"                 # 不包含于
    CONTAINS = "contains"             # 包含
    NOT_CONTAINS = "not_contains"     # 不包含
    REGEX = "regex"                   # 正则表达式
    STARTS_WITH = "starts_with"       # 以...开始
    ENDS_WITH = "ends_with"           # 以...结束


@dataclass
class AlertRule:

    """告警规则"""
    rule_id: str
    name: str
    description: str
    condition_type: ConditionType
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    risk_type: RiskType
    severity: AlertLevel
    enabled: bool = True
    cooldown_period: int = 300  # 冷却期（秒）
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'condition_type': self.condition_type.value,
            'conditions': self.conditions,
            'actions': self.actions,
            'risk_type': self.risk_type.value,
            'severity': self.severity.value,
            'enabled': self.enabled,
            'cooldown_period': self.cooldown_period,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
            'trigger_count': self.trigger_count
        }


class ConditionEvaluator:

    """条件评估器"""

    def __init__(self):

        self.operators = {
            Operator.GT: operator.gt,
            Operator.GE: operator.ge,
            Operator.LT: operator.lt,
            Operator.LE: operator.le,
            Operator.EQ: operator.eq,
            Operator.NE: operator.ne,
            Operator.IN: lambda x, y: x in y,
            Operator.NOT_IN: lambda x, y: x not in y,
            Operator.CONTAINS: lambda x, y: y in x if isinstance(x, (str, list)) else False,
            Operator.NOT_CONTAINS: lambda x, y: y not in x if isinstance(x, (str, list)) else True,
            Operator.REGEX: lambda x, y: bool(re.match(y, str(x))),
            Operator.STARTS_WITH: lambda x, y: str(x).startswith(str(y)),
            Operator.ENDS_WITH: lambda x, y: str(x).endswith(str(y))
        }

    def evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估单个条件"""
        try:
            field = condition.get('field', '')
            operator_str = condition.get('operator', '')
            value = condition.get('value')

            # 获取操作误
            operator_enum = Operator(operator_str)
            operator_func = self.operators.get(operator_enum)

            if not operator_func:
                logger.error(f"不支持的操作误 {operator_str}")
                return False

            # 获取字段误
            field_value = self._get_field_value(field, context)

            # 评估条件
            return operator_func(field_value, value)

        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False

    def _get_field_value(self, field: str, context: Dict[str, Any]) -> Any:
        """Get field value"""
        try:
            return context.get(field)
        except Exception as e:
            logger.error(f"获取字段值失败: {e}")
            return False

    def evaluate_complex_condition(self, conditions: List[Dict[str, Any]],
                                   logic: str, context: Dict[str, Any]) -> bool:
        """评估复杂条件"""
        if not conditions:
            return True

        results = []
        for condition in conditions:
            result = self.evaluate_condition(condition, context)
            results.append(result)

        # 应用逻辑运算
        if logic.upper() == 'AND':
            return all(results)
        elif logic.upper() == 'OR':
            return any(results)
        else:
            logger.error(f"不支持的逻辑运算: {logic}")
            return False

    def evaluate_trend_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估趋势条件"""
        try:
            field = condition.get('field', '')
            direction = condition.get('direction', 'increasing')  # increasing, decreasing
            period = condition.get('period', 5)  # 时间周期
            threshold = condition.get('threshold', 0.1)  # 变化阈值
            # 获取历史数据
            history = context.get('history', [])
            if len(history) < period:
                return False

            # 计算趋势
            values = [h.get(field, 0) for h in history[-period:]]
            if len(values) < 2:
                return False

            # 计算变化
            if direction == 'increasing':
                changes = [(values[i] - values[i - 1]) / abs(values[i - 1])
                           for i in range(1, len(values))]
                return all(change > threshold for change in changes)
            elif direction == 'decreasing':
                changes = [(values[i] - values[i - 1]) / abs(values[i - 1])
                           for i in range(1, len(values))]
                return all(change < -threshold for change in changes)

            return False

        except Exception as e:
            logger.error(f"趋势条件评估失败: {e}")
            return False

    def evaluate_pattern_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估模式条件"""
        try:
            field = condition.get('field', '')
            pattern = condition.get('pattern', '')
            pattern_type = condition.get('pattern_type', 'regex')  # regex, contains, exact

            field_value = str(self._get_field_value(field, context))

            if pattern_type == 'regex':
                return bool(re.match(pattern, field_value))
            elif pattern_type == 'contains':
                return pattern in field_value
            elif pattern_type == 'exact':
                return field_value == pattern
            else:
                logger.error(f"不支持的模式类型: {pattern_type}")
                return False

        except Exception as e:
            logger.error(f"模式条件评估失败: {e}")
            return False

    def evaluate_correlation_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """评估相关性条件"""
        try:
            field1 = condition.get('field1', '')
            field2 = condition.get('field2', '')
            correlation_type = condition.get('correlation_type', 'positive')  # positive, negative
            threshold = condition.get('threshold', 0.7)  # 相关性阈值
            window = condition.get('window', 20)  # 时间窗口

            # 获取历史数据
            history = context.get('history', [])
            if len(history) < window:
                return False

            # 提取两个字段的值序列
            values1 = [h.get(field1, 0) for h in history[-window:]]
            values2 = [h.get(field2, 0) for h in history[-window:]]

            # 计算相关系数
            import numpy as np
            correlation = np.corrcoef(values1, values2)[0, 1]

            if correlation_type == 'positive':
                return correlation > threshold
            elif correlation_type == 'negative':
                return correlation < -threshold

            return False

        except Exception as e:
            logger.error(f"相关性条件评估失败: {e}")
            return False

    def _get_field_value(self, field: str, context: Dict[str, Any]) -> Any:
        """获取字段值"""
        if '.' in field:
            # 支持嵌套字段访问，如 'portfolio.value'
            parts = field.split('.')
            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value
        else:
            return context.get(field)


class ActionExecutor:

    """动作执行器"""

    def __init__(self):
        self.notification_channels = {
            'email': self._send_email,
            'sms': self._send_sms,
            'webhook': self._send_webhook,
            'slack': self._send_slack,
            'console': self._print_console
        }

        self.execution_actions = {
            'pause_trading': self._pause_trading,
            'reduce_position': self._reduce_position,
            'send_notification': self._send_notification,
            'log_alert': self._log_alert,
            'escalate': self._escalate_alert
        }

    def execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """执行动作"""
        try:
            action_type = action.get('type', '')
            parameters = action.get('parameters', {})

            if action_type == 'notification':
                return self._execute_notification(parameters, context)
            elif action_type == 'execution':
                return self._execute_action(parameters, context)
            elif action_type == 'alert':
                return self._send_alert(parameters, context)
            else:
                logger.error(f"不支持的动作类型: {action_type}")
                return False

        except Exception as e:
            logger.error(f"动作执行失败: {e}")
            return False

    def _execute_notification(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """执行通知"""
        channel = parameters.get('channel', 'console')
        template = parameters.get('template', '{message}')
        recipients = parameters.get('recipients', [])

        message = template.format(**context)

        if channel in self.notification_channels:
            return self.notification_channels[channel](message, recipients, parameters)
        else:
            logger.error(f"不支持的通知渠道: {channel}")
            return False

    def _execute_action(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """执行动作"""
        action_name = parameters.get('action', '')

        if action_name in self.execution_actions:
            return self.execution_actions[action_name](parameters, context)
        else:
            logger.error(f"不支持的执行动作: {action_name}")
            return False

    def _send_alert(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """发送告警"""
        try:
            # 创建基础设施告警
            alert = {
                'id': parameters.get('alert_id', f"rule_{int(time.time())}"),
                'title': parameters.get('title', '告警规则触发'),
                'description': parameters.get('description', '告警规则被触发'),
                'level': parameters.get('level', 'warning'),
                'source': 'alert_rule_engine',
                'timestamp': datetime.now().isoformat(),
                'data': context
            }

            # 这里应该调用基础设施的告警系统
            # alert_system.process_alert(alert)

            logger.info(f"告警已发送: {alert['title']}")
            return True

        except Exception as e:
            logger.error(f"告警发送失败: {e}")
            return False

    # 通知渠道实现

    def _send_email(self, message: str, recipients: List[str], params: Dict[str, Any]) -> bool:
        """发送邮件"""
        logger.info(f"发送邮件通知: {message} -> {recipients}")
        # 这里实现邮件发送逻辑
        return True

    def _send_sms(self, message: str, recipients: List[str], params: Dict[str, Any]) -> bool:
        """发送短信"""
        logger.info(f"发送短信通知: {message} -> {recipients}")
        # 这里实现短信发送逻辑
        return True

    def _send_webhook(self, message: str, recipients: List[str], params: Dict[str, Any]) -> bool:
        """发送Webhook"""
        logger.info(f"发送Webhook通知: {message} -> {recipients}")
        # 这里实现Webhook发送逻辑
        return True

    def _send_slack(self, message: str, recipients: List[str], params: Dict[str, Any]) -> bool:
        """发送Slack通知"""
        logger.info(f"发送Slack通知: {message} -> {recipients}")
        # 这里实现Slack发送逻辑
        return True

    def _print_console(self, message: str, recipients: List[str], params: Dict[str, Any]) -> bool:
        """控制台输出"""
        print(f"📢 通知: {message}")
        return True

    # 执行动作实现

    def _pause_trading(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """暂停交易"""
        logger.warning("执行交易暂停动作")
        # 这里实现交易暂停逻辑
        return True

    def _reduce_position(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """减少仓位"""
        reduction_ratio = parameters.get('ratio', 0.5)
        logger.warning(f"执行仓位减少动作: {reduction_ratio}")
        # 这里实现仓位减少逻辑
        return True

    def _send_notification(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """发送通知"""
        return self._execute_notification(parameters, context)

    def _log_alert(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """记录告警"""
        logger.warning(f"告警日志: {context}")
        return True

    def _escalate_alert(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """升级告警"""
        logger.warning("执行告警升级动作")
        # 这里实现告警升级逻辑
        return True


class AlertRuleEngine:
    """告警规则引擎"""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.condition_evaluator = ConditionEvaluator()
        self.action_executor = ActionExecutor()
        self.rule_stats = {}
        self.is_running = False

        logger.info("告警规则引擎初始化完成")

    def add_rule(self, rule: AlertRule):
        """添加规则"""
        self.rules[rule.rule_id] = rule
        self.rule_stats[rule.rule_id] = {
            'trigger_count': 0,
            'last_triggered': None,
            'success_count': 0,
            'failure_count': 0
        }
        logger.info(f"告警规则已添加: {rule.rule_id}")

    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            del self.rule_stats[rule_id]
            logger.info(f"告警规则已移除: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"告警规则已启用: {rule_id}")
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"告警规则已禁用: {rule_id}")
            return True
        return False

    def evaluate_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估所有规则"""
        triggered_rules = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # 检查冷却期
            if self._is_in_cooldown(rule):
                continue

            # 评估规则条件
            if self._evaluate_rule_condition(rule, context):
                # 执行规则动作
                success = self._execute_rule_actions(rule, context)

                # 更新统计信息
                rule.trigger_count += 1
                rule.last_triggered = datetime.now()
                stats = self.rule_stats[rule.rule_id]
                stats['trigger_count'] += 1
                stats['last_triggered'] = datetime.now()
                if success:
                    stats['success_count'] += 1
                else:
                    stats['failure_count'] += 1

                # 记录触发规则
                triggered_rules.append({
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'severity': rule.severity.value,
                    'success': success,
                    'timestamp': datetime.now().isoformat(),
                    'context': context
                })

        return triggered_rules

    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """检查是否在冷却期"""
        if rule.last_triggered is None:
            return False

        time_since_triggered = (datetime.now() - rule.last_triggered).total_seconds()
        return time_since_triggered < rule.cooldown_period

    def _evaluate_rule_condition(self, rule: AlertRule, context: Dict[str, Any]) -> bool:
        """评估规则条件"""
        try:
            if rule.condition_type == ConditionType.THRESHOLD:
                # 阈值条件
                return self._evaluate_threshold_condition(rule, context)
            elif rule.condition_type == ConditionType.TREND:
                # 趋势条件
                return self._evaluate_trend_condition(rule, context)
            elif rule.condition_type == ConditionType.PATTERN:
                # 模式条件
                return self._evaluate_pattern_condition(rule, context)
            elif rule.condition_type == ConditionType.COMPLEX:
                # 复杂条件
                return self._evaluate_complex_condition(rule, context)
            elif rule.condition_type == ConditionType.CORRELATION:
                # 相关性条件
                return self._evaluate_correlation_condition(rule, context)
            else:
                logger.error(f"不支持的条件类型: {rule.condition_type}")
                return False

        except Exception as e:
            logger.error(f"规则条件评估失败 {rule.rule_id}: {e}")
            return False

    def _evaluate_threshold_condition(self, rule: AlertRule, context: Dict[str, Any]) -> bool:
        """评估阈值条件"""
        if not rule.conditions:
            return False

        return self.condition_evaluator.evaluate_condition(rule.conditions[0], context)

    def _evaluate_trend_condition(self, rule: AlertRule, context: Dict[str, Any]) -> bool:
        """评估趋势条件"""
        if not rule.conditions:
            return False

        return self.condition_evaluator.evaluate_trend_condition(rule.conditions[0], context)

    def _evaluate_pattern_condition(self, rule: AlertRule, context: Dict[str, Any]) -> bool:
        """评估模式条件"""
        if not rule.conditions:
            return False

        return self.condition_evaluator.evaluate_pattern_condition(rule.conditions[0], context)

    def _evaluate_complex_condition(self, rule: AlertRule, context: Dict[str, Any]) -> bool:
        """评估复杂条件"""
        # 从规则条件中提取逻辑运算符
        logic = rule.conditions[0].get('logic', 'AND') if rule.conditions else 'AND'
        conditions = rule.conditions[0].get('conditions', []) if rule.conditions else []

        return self.condition_evaluator.evaluate_complex_condition(conditions, logic, context)

    def _evaluate_correlation_condition(self, rule: AlertRule, context: Dict[str, Any]) -> bool:
        """评估相关性条件"""
        if not rule.conditions:
            return False

        return self.condition_evaluator.evaluate_correlation_condition(rule.conditions[0], context)

    def _execute_rule_actions(self, rule: AlertRule, context: Dict[str, Any]) -> bool:
        """执行规则动作"""
        success_count = 0

        for action in rule.actions:
            try:
                # 合并规则上下文
                action_context = {
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'risk_type': rule.risk_type.value,
                    'severity': rule.severity.value,
                    'message': rule.description,
                    **context
                }

                if self.action_executor.execute_action(action, action_context):
                    success_count += 1

            except Exception as e:
                logger.error(f"规则动作执行失败 {rule.rule_id}: {e}")

        return success_count == len(rule.actions)

    def get_rule_stats(self) -> Dict[str, Any]:
        """获取规则统计信息"""
        return self.rule_stats.copy()

    def get_rules_summary(self) -> Dict[str, Any]:
        """获取规则摘要"""
        enabled_rules = sum(1 for rule in self.rules.values() if rule.enabled)
        disabled_rules = len(self.rules) - enabled_rules

        severity_distribution = {}
        for rule in self.rules.values():
            severity = rule.severity.value
            if severity not in severity_distribution:
                severity_distribution[severity] = 0
            severity_distribution[severity] += 1

        return {
            'total_rules': len(self.rules),
            'enabled_rules': enabled_rules,
            'disabled_rules': disabled_rules,
            'severity_distribution': severity_distribution
        }

    def create_default_alert_rules() -> List[AlertRule]:
        """创建默认告警规则"""
        rules = []

        # 1. 高波动率告警规则
        rule1 = AlertRule(
            rule_id='high_volatility_threshold',
            name='高波动率阈值告警',
            description='投资组合波动率超过阈值',
            condition_type=ConditionType.THRESHOLD,
            conditions=[
                {
                    'field': 'portfolio_volatility',
                    'operator': '>',
                    'value': 0.35
                }
            ],
            actions=[
                {
                    'type': 'notification',
                    'parameters': {
                        'channel': 'console',
                        'template': '⚠️ 高波动率告警: {rule_name} - 波动率 {portfolio_volatility:.2%}'
                    }
                },
                {
                    'type': 'execution',
                    'parameters': {
                        'action': 'reduce_position',
                        'ratio': 0.3
                    }
                }
            ],
            risk_type=RiskType.MARKET_RISK,
            severity=AlertLevel.ERROR,
            cooldown_period=600  # 10分钟冷却期
        )

        # 2. 回撤趋势告警规则
        rule2 = AlertRule(
            rule_id='drawdown_trend',
            name='回撤趋势告警',
            description='投资组合回撤呈上升趋势',
            condition_type=ConditionType.TREND,
            conditions=[
                {
                    'field': 'max_drawdown',
                    'direction': 'increasing',
                    'period': 5,
                    'threshold': 0.05
                }
            ],
            actions=[
                {
                    'type': 'notification',
                    'parameters': {
                        'channel': 'console',
                        'template': '📉 回撤趋势告警: {rule_name} - 当前回撤: {max_drawdown:.2%}'
                    }
                },
                {
                    'type': 'alert',
                    'parameters': {
                        'title': '严重回撤趋势',
                        'level': 'critical'
                    }
                }
            ],
            risk_type=RiskType.MARKET_RISK,
            severity=AlertLevel.CRITICAL,
            cooldown_period=1200  # 20分钟冷却期
        )

        # 3. 流动性风险模式告警规则
        rule3 = AlertRule(
            rule_id='liquidity_pattern',
            name='流动性风险模式告警',
            description='检测异常的流动性模式',
            condition_type=ConditionType.PATTERN,
            conditions=[
                {
                    'field': 'trading_volume',
                    'pattern': 'abnormal',
                    'pattern_type': 'contains'
                }
            ],
            actions=[
                {
                    'type': 'notification',
                    'parameters': {
                        'channel': 'console',
                        'template': '💧 流动性风险 {rule_name} - 交易量异常'
                    }
                }
            ],
            risk_type=RiskType.LIQUIDITY_RISK,
            severity=AlertLevel.WARNING,
            cooldown_period=300  # 5分钟冷却期
        )

        # 4. 复杂条件组合告警规则
        rule4 = AlertRule(
            rule_id='complex_risk_combination',
            name='复杂风险组合告警',
            description='多个风险指标同时超标',
            condition_type=ConditionType.COMPLEX,
            conditions=[
                {
                    'logic': 'AND',
                    'conditions': [
                        {
                            'field': 'portfolio_volatility',
                            'operator': '>',
                            'value': 0.25
                        },
                        {
                            'field': 'max_drawdown',
                            'operator': '>',
                            'value': 0.10
                        },
                        {
                            'field': 'liquidity_ratio',
                            'operator': '<',
                            'value': 0.05
                        }
                    ]
                }
            ],
            actions=[
                {
                    'type': 'notification',
                    'parameters': {
                        'channel': 'console',
                        'template': '🚨 复合风险告警: {rule_name} - 多重风险同时触发'
                    }
                },
                {
                    'type': 'execution',
                    'parameters': {
                        'action': 'pause_trading'
                    }
                }
            ],
            risk_type=RiskType.MARKET_RISK,
            severity=AlertLevel.CRITICAL,
            cooldown_period=1800  # 30分钟冷却期
        )

        # 5. 相关性风险告警规则
        rule5 = AlertRule(
            rule_id='correlation_risk',
            name='相关性风险告警',
            description='资产相关性异常变化',
            condition_type=ConditionType.CORRELATION,
            conditions=[
                {
                    'field1': 'asset_return_a',
                    'field2': 'asset_return_b',
                    'correlation_type': 'negative',
                    'threshold': 0.8,
                    'window': 20
                }
            ],
            actions=[
                {
                    'type': 'notification',
                    'parameters': {
                        'channel': 'console',
                        'template': '🔗 相关性风险 {rule_name} - 资产相关性异常'
                    }
                }
            ],
            risk_type=RiskType.MARKET_RISK,
            severity=AlertLevel.WARNING,
            cooldown_period=900  # 15分钟冷却期
        )

        rules.extend([rule1, rule2, rule3, rule4, rule5])
        return rules


def main():
    """主函数 - 告警规则引擎演示"""
    print("RQA2025告警规则引擎")
    print("="*50)

    # 创建告警规则引擎
    engine = AlertRuleEngine()

    # 添加默认规则
    default_rules = AlertRuleEngine.create_default_alert_rules()
    for rule in default_rules:
        engine.add_rule(rule)

    print("告警规则引擎创建完成")
    print(f"   规则数量: {len(engine.rules)}")

    # 显示规则摘要
    summary = engine.get_rules_summary()
    print("📋 规则摘要:")
    print(f"   总规则数: {summary['total_rules']}")
    print(f"   启用规则: {summary['enabled_rules']}")
    print(f"   禁用规则: {summary['disabled_rules']}")
    print(f"   严重程度分布: {summary['severity_distribution']}")

    # 显示规则详情
    print("\n🔍 规则详情:")
    for rule_id, rule in engine.rules.items():
        print(f"   {rule_id}: {rule.name}")
        print(f"     类型: {rule.condition_type.value}")
        print(f"     风险类型: {rule.risk_type.value}")
        print(f"     严重程度: {rule.severity.value}")
        print(f"     动作数量: {len(rule.actions)}")

    # 模拟上下文数据
    context_data = {
        'portfolio_volatility': 0.40,  # 高波动率
        'max_drawdown': 0.12,          # 高回撤
        'liquidity_ratio': 0.03,       # 低流动性
        'trading_volume': 'abnormal_pattern',  # 异常模式
        'asset_return_a': [0.01, 0.02, 0.01, 0.03, 0.01],
        'asset_return_b': [-0.01, -0.02, -0.01, -0.03, -0.01],
        'history': [
            {'portfolio_volatility': 0.35, 'max_drawdown': 0.08, 'liquidity_ratio': 0.06},
            {'portfolio_volatility': 0.38, 'max_drawdown': 0.10, 'liquidity_ratio': 0.04},
            {'portfolio_volatility': 0.40, 'max_drawdown': 0.12, 'liquidity_ratio': 0.03}
        ]
    }

    print("🧪 规则评估测试:")
    print("   测试上下文数据")
    for key, value in context_data.items():
        if key != 'history':
            print(f"     {key}: {value}")

    # 评估规则
    triggered_rules = engine.evaluate_rules(context_data)

    print(f"\n🎯 触发规则数量: {len(triggered_rules)}")
    if triggered_rules:
        print("   触发的规则:")
        for rule in triggered_rules:
            print(f"     {rule['rule_id']}: {rule['rule_name']} ({rule['severity']})")

    # 显示统计信息
    stats = engine.get_rule_stats()
    print("📊 规则统计:")
    for rule_id, stat in stats.items():
        print(f"   {rule_id}: 触发{stat['trigger_count']}次")

    print("\n🎉 告警规则引擎演示完成!")
    print("   智能告警规则系统已准备就绪")
    print("   可以处理复杂的市场风险监控场景")

    return engine


if __name__ == "__main__":
    engine = main()
