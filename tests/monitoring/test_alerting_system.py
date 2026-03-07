#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
告警系统配置测试
Alerting System Configuration Tests

测试告警系统的完整性和正确性，包括：
1. 告警规则配置验证
2. 告警触发机制测试
3. 告警通知配置测试
4. 告警升级策略测试
5. 告警抑制和分组测试
6. 告警历史和趋势分析测试
7. 告警响应时间和SLA测试
8. 告警系统可用性测试
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
from pathlib import Path
import smtplib
import requests

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestAlertRuleConfiguration:
    """测试告警规则配置"""

    def setup_method(self):
        """测试前准备"""
        self.alert_manager = Mock()
        self.rule_validator = Mock()

    def test_metric_threshold_alert_rules(self):
        """测试指标阈值告警规则"""
        # 定义告警规则配置
        alert_rules = {
            'high_cpu_usage': {
                'metric': 'cpu_percent',
                'condition': '>',
                'threshold': 80.0,
                'duration': '5m',
                'severity': 'warning',
                'description': 'CPU使用率过高',
                'enabled': True
            },
            'critical_memory_usage': {
                'metric': 'memory_percent',
                'condition': '>=',
                'threshold': 90.0,
                'duration': '2m',
                'severity': 'critical',
                'description': '内存使用率严重过高',
                'enabled': True
            },
            'low_disk_space': {
                'metric': 'disk_usage_percent',
                'condition': '>',
                'threshold': 85.0,
                'duration': '10m',
                'severity': 'warning',
                'description': '磁盘空间不足',
                'enabled': True
            },
            'high_error_rate': {
                'metric': 'error_rate',
                'condition': '>',
                'threshold': 5.0,
                'duration': '1m',
                'severity': 'critical',
                'description': '错误率过高',
                'enabled': True
            }
        }

        def validate_alert_rules(rules: Dict) -> List[str]:
            """验证告警规则配置"""
            errors = []
            valid_conditions = ['>', '>=', '<', '<=', '==', '!=']
            valid_severities = ['info', 'warning', 'error', 'critical']

            for rule_name, rule_config in rules.items():
                # 检查必需字段
                required_fields = ['metric', 'condition', 'threshold', 'severity', 'description']
                for field in required_fields:
                    if field not in rule_config:
                        errors.append(f"告警规则 {rule_name} 缺少字段: {field}")

                # 验证条件
                if 'condition' in rule_config:
                    if rule_config['condition'] not in valid_conditions:
                        errors.append(f"告警规则 {rule_name} 无效条件: {rule_config['condition']}")

                # 验证严重程度
                if 'severity' in rule_config:
                    if rule_config['severity'] not in valid_severities:
                        errors.append(f"告警规则 {rule_name} 无效严重程度: {rule_config['severity']}")

                # 验证阈值
                if 'threshold' in rule_config:
                    threshold = rule_config['threshold']
                    if not isinstance(threshold, (int, float)):
                        errors.append(f"告警规则 {rule_name} 阈值必须是数字: {threshold}")

                # 验证持续时间
                if 'duration' in rule_config:
                    duration = rule_config['duration']
                    if not isinstance(duration, str) or not duration.endswith(('s', 'm', 'h')):
                        errors.append(f"告警规则 {rule_name} 无效持续时间格式: {duration}")

            return errors

        # 验证告警规则配置
        errors = validate_alert_rules(alert_rules)

        # 应该没有配置错误
        assert len(errors) == 0, f"告警规则配置存在错误: {errors}"

        # 验证规则逻辑
        cpu_rule = alert_rules['high_cpu_usage']
        assert cpu_rule['condition'] == '>', "CPU规则应该使用大于条件"
        assert cpu_rule['threshold'] == 80.0, "CPU阈值应该是80%"
        assert cpu_rule['severity'] == 'warning', "CPU告警应该是警告级别"

        memory_rule = alert_rules['critical_memory_usage']
        assert memory_rule['condition'] == '>=', "内存规则应该使用大于等于条件"
        assert memory_rule['threshold'] == 90.0, "内存阈值应该是90%"
        assert memory_rule['severity'] == 'critical', "内存告警应该是严重级别"

    def test_service_health_alert_rules(self):
        """测试服务健康告警规则"""
        # 定义服务健康告警规则
        service_alert_rules = {
            'api_service_down': {
                'service': 'api_gateway',
                'check_type': 'health_check',
                'condition': 'status != "healthy"',
                'duration': '30s',
                'severity': 'critical',
                'description': 'API服务不可用',
                'auto_recovery': True,
                'enabled': True
            },
            'database_connection_pool_exhausted': {
                'service': 'database',
                'check_type': 'connection_pool',
                'condition': 'active_connections / max_connections > 0.9',
                'duration': '1m',
                'severity': 'warning',
                'description': '数据库连接池即将耗尽',
                'enabled': True
            },
            'queue_depth_high': {
                'service': 'message_queue',
                'check_type': 'queue_depth',
                'condition': 'queue_length > 1000',
                'duration': '5m',
                'severity': 'warning',
                'description': '队列深度过高',
                'enabled': True
            },
            'external_service_timeout': {
                'service': 'payment_gateway',
                'check_type': 'external_dependency',
                'condition': 'response_time > 5000',
                'duration': '2m',
                'severity': 'error',
                'description': '外部服务响应超时',
                'enabled': True
            }
        }

        def validate_service_alert_rules(rules: Dict) -> List[str]:
            """验证服务告警规则配置"""
            errors = []
            valid_check_types = ['health_check', 'connection_pool', 'queue_depth', 'external_dependency', 'custom']
            valid_severities = ['info', 'warning', 'error', 'critical']

            for rule_name, rule_config in rules.items():
                # 检查必需字段
                required_fields = ['service', 'check_type', 'condition', 'severity', 'description']
                for field in required_fields:
                    if field not in rule_config:
                        errors.append(f"服务告警规则 {rule_name} 缺少字段: {field}")

                # 验证检查类型
                if 'check_type' in rule_config:
                    if rule_config['check_type'] not in valid_check_types:
                        errors.append(f"服务告警规则 {rule_name} 无效检查类型: {rule_config['check_type']}")

                # 验证严重程度
                if 'severity' in rule_config:
                    if rule_config['severity'] not in valid_severities:
                        errors.append(f"服务告警规则 {rule_name} 无效严重程度: {rule_config['severity']}")

                # 验证条件表达式（基础验证）
                if 'condition' in rule_config:
                    condition = rule_config['condition']
                    # 检查是否包含基本的比较操作符
                    has_operator = any(op in condition for op in ['>', '<', '>=', '<=', '==', '!='])
                    if not has_operator and '!=' not in condition and '=' not in condition:
                        errors.append(f"服务告警规则 {rule_name} 条件表达式无效: {condition}")

                # 验证持续时间
                if 'duration' in rule_config:
                    duration = rule_config['duration']
                    if not isinstance(duration, str) or not duration.endswith(('s', 'm', 'h')):
                        errors.append(f"服务告警规则 {rule_name} 无效持续时间格式: {duration}")

            return errors

        # 验证服务告警规则配置
        errors = validate_service_alert_rules(service_alert_rules)

        # 应该没有配置错误
        assert len(errors) == 0, f"服务告警规则配置存在错误: {errors}"

        # 验证规则配置
        api_rule = service_alert_rules['api_service_down']
        assert api_rule['service'] == 'api_gateway', "API规则应该针对正确的服务"
        assert api_rule['check_type'] == 'health_check', "API规则应该使用健康检查"
        assert api_rule['severity'] == 'critical', "API宕机应该是严重级别"
        assert api_rule['auto_recovery'] is True, "API服务应该启用自动恢复"

        db_rule = service_alert_rules['database_connection_pool_exhausted']
        assert db_rule['service'] == 'database', "数据库规则应该针对正确的服务"
        assert 'active_connections' in db_rule['condition'], "数据库规则应该检查连接数"

    def test_business_logic_alert_rules(self):
        """测试业务逻辑告警规则"""
        # 定义业务逻辑告警规则
        business_alert_rules = {
            'low_transaction_volume': {
                'metric': 'transaction_count_per_minute',
                'condition': '< 10',
                'duration': '5m',
                'severity': 'warning',
                'description': '交易量异常偏低',
                'business_impact': 'high',
                'enabled': True
            },
            'high_failed_payment_rate': {
                'metric': 'payment_failure_rate',
                'condition': '> 0.05',
                'duration': '3m',
                'severity': 'critical',
                'description': '支付失败率过高',
                'business_impact': 'critical',
                'enabled': True
            },
            'user_login_failures': {
                'metric': 'failed_login_attempts_per_hour',
                'condition': '> 100',
                'duration': '10m',
                'severity': 'error',
                'description': '用户登录失败次数过多',
                'business_impact': 'medium',
                'enabled': True
            },
            'inventory_low_stock': {
                'metric': 'low_stock_items_count',
                'condition': '> 50',
                'duration': '15m',
                'severity': 'warning',
                'description': '库存不足商品数量过多',
                'business_impact': 'medium',
                'enabled': True
            }
        }

        def validate_business_alert_rules(rules: Dict) -> List[str]:
            """验证业务告警规则配置"""
            errors = []
            valid_severities = ['info', 'warning', 'error', 'critical']
            valid_business_impacts = ['low', 'medium', 'high', 'critical']

            for rule_name, rule_config in rules.items():
                # 检查必需字段
                required_fields = ['metric', 'condition', 'severity', 'description', 'business_impact']
                for field in required_fields:
                    if field not in rule_config:
                        errors.append(f"业务告警规则 {rule_name} 缺少字段: {field}")

                # 验证严重程度
                if 'severity' in rule_config:
                    if rule_config['severity'] not in valid_severities:
                        errors.append(f"业务告警规则 {rule_name} 无效严重程度: {rule_config['severity']}")

                # 验证业务影响
                if 'business_impact' in rule_config:
                    if rule_config['business_impact'] not in valid_business_impacts:
                        errors.append(f"业务告警规则 {rule_name} 无效业务影响级别: {rule_config['business_impact']}")

                # 验证条件格式
                if 'condition' in rule_config:
                    condition = rule_config['condition']
                    # 检查是否以比较操作符开头
                    valid_starts = ['>', '<', '>=', '<=', '==', '!=']
                    if not any(condition.strip().startswith(op) for op in valid_starts):
                        errors.append(f"业务告警规则 {rule_name} 条件格式无效: {condition}")

                # 验证持续时间
                if 'duration' in rule_config:
                    duration = rule_config['duration']
                    if not isinstance(duration, str) or not duration.endswith(('s', 'm', 'h')):
                        errors.append(f"业务告警规则 {rule_name} 无效持续时间格式: {duration}")

            return errors

        # 验证业务告警规则配置
        errors = validate_business_alert_rules(business_alert_rules)

        # 应该没有配置错误
        assert len(errors) == 0, f"业务告警规则配置存在错误: {errors}"

        # 验证规则配置
        payment_rule = business_alert_rules['high_failed_payment_rate']
        assert payment_rule['metric'] == 'payment_failure_rate', "支付规则应该监控正确的指标"
        assert '> 0.05' in payment_rule['condition'], "支付规则应该检查5%阈值"
        assert payment_rule['severity'] == 'critical', "支付失败应该是严重级别"
        assert payment_rule['business_impact'] == 'critical', "支付失败应该有严重业务影响"

        volume_rule = business_alert_rules['low_transaction_volume']
        assert volume_rule['metric'] == 'transaction_count_per_minute', "交易量规则应该监控正确的指标"
        assert volume_rule['business_impact'] == 'high', "交易量影响应该是高影响"


class TestAlertTriggeringMechanism:
    """测试告警触发机制"""

    def setup_method(self):
        """测试前准备"""
        self.alert_engine = Mock()
        self.trigger_tester = Mock()

    def test_threshold_based_alert_triggering(self):
        """测试基于阈值的告警触发"""
        # 定义告警规则和测试指标数据
        alert_rules = {
            'cpu_high': {'metric': 'cpu_percent', 'threshold': 80.0, 'condition': '>', 'severity': 'warning'},
            'memory_critical': {'metric': 'memory_percent', 'threshold': 90.0, 'condition': '>=', 'severity': 'critical'}
        }

        # 模拟指标数据流
        metrics_stream = [
            {'timestamp': datetime.now(), 'cpu_percent': 65.0, 'memory_percent': 70.0},
            {'timestamp': datetime.now(), 'cpu_percent': 82.0, 'memory_percent': 75.0},  # CPU超阈值
            {'timestamp': datetime.now(), 'cpu_percent': 78.0, 'memory_percent': 92.0},  # 内存超阈值
            {'timestamp': datetime.now(), 'cpu_percent': 85.0, 'memory_percent': 95.0},  # 两者都超阈值
        ]

        def evaluate_alerts(rules: Dict, metrics_data: List[Dict]) -> List[Dict]:
            """评估告警触发"""
            alerts_triggered = []

            for metric_point in metrics_data:
                for rule_name, rule in rules.items():
                    metric_name = rule['metric']
                    threshold = rule['threshold']
                    condition = rule['condition']
                    severity = rule['severity']

                    if metric_name in metric_point:
                        value = metric_point[metric_name]

                        # 评估条件
                        triggered = False
                        if condition == '>' and value > threshold:
                            triggered = True
                        elif condition == '>=' and value >= threshold:
                            triggered = True
                        elif condition == '<' and value < threshold:
                            triggered = True
                        elif condition == '<=' and value <= threshold:
                            triggered = True

                        if triggered:
                            alert = {
                                'alert_id': f"alert_{rule_name}_{int(time.time() * 1000)}",
                                'rule_name': rule_name,
                                'severity': severity,
                                'metric': metric_name,
                                'value': value,
                                'threshold': threshold,
                                'condition': condition,
                                'timestamp': metric_point['timestamp'],
                                'status': 'firing'
                            }
                            alerts_triggered.append(alert)

            return alerts_triggered

        # 评估告警触发
        triggered_alerts = evaluate_alerts(alert_rules, metrics_stream)

        # 验证告警触发
        assert len(triggered_alerts) >= 3, f"应该触发至少3个告警，实际: {len(triggered_alerts)}"

        # 检查具体的告警
        cpu_alerts = [a for a in triggered_alerts if a['rule_name'] == 'cpu_high']
        memory_alerts = [a for a in triggered_alerts if a['rule_name'] == 'memory_critical']

        assert len(cpu_alerts) >= 2, "应该有至少2个CPU告警（第2和第4个数据点）"
        assert len(memory_alerts) >= 2, "应该有至少2个内存告警（第3和第4个数据点）"

        # 验证告警详情
        for alert in triggered_alerts:
            assert 'alert_id' in alert, "告警应该有ID"
            assert 'rule_name' in alert, "告警应该有规则名称"
            assert 'severity' in alert, "告警应该有严重程度"
            assert 'metric' in alert, "告警应该有指标名称"
            assert 'value' in alert, "告警应该有触发值"
            assert 'threshold' in alert, "告警应该有阈值"
            assert 'timestamp' in alert, "告警应该有时间戳"
            assert alert['status'] == 'firing', "告警状态应该是firing"

        # 验证告警准确性
        for alert in cpu_alerts:
            assert alert['value'] > 80.0, f"CPU告警值应该大于80: {alert['value']}"
            assert alert['threshold'] == 80.0, "CPU告警阈值应该是80"

        for alert in memory_alerts:
            assert alert['value'] >= 90.0, f"内存告警值应该大于等于90: {alert['value']}"
            assert alert['threshold'] == 90.0, "内存告警阈值应该是90"

    def test_duration_based_alert_triggering(self):
        """测试基于持续时间的告警触发"""
        # 定义需要持续时间验证的告警规则
        duration_rules = {
            'persistent_high_cpu': {
                'metric': 'cpu_percent',
                'threshold': 75.0,
                'condition': '>',
                'duration_required': 3,  # 需要连续3个周期
                'severity': 'warning'
            },
            'sustained_memory_pressure': {
                'metric': 'memory_percent',
                'threshold': 85.0,
                'condition': '>=',
                'duration_required': 2,  # 需要连续2个周期
                'severity': 'error'
            }
        }

        # 模拟指标数据流（带有持续性）
        metrics_stream = [
            # CPU: 逐渐升高然后降低
            {'timestamp': datetime.now(), 'cpu_percent': 70.0, 'memory_percent': 80.0},
            {'timestamp': datetime.now(), 'cpu_percent': 78.0, 'memory_percent': 82.0},
            {'timestamp': datetime.now(), 'cpu_percent': 82.0, 'memory_percent': 87.0},  # CPU持续高，第3周期
            {'timestamp': datetime.now(), 'cpu_percent': 85.0, 'memory_percent': 89.0},  # CPU持续高，内存也开始持续高
            {'timestamp': datetime.now(), 'cpu_percent': 72.0, 'memory_percent': 91.0},  # CPU降低但内存仍高
            {'timestamp': datetime.now(), 'cpu_percent': 68.0, 'memory_percent': 88.0},  # 两者都降低
        ]

        def evaluate_duration_alerts(rules: Dict, metrics_data: List[Dict]) -> List[Dict]:
            """评估持续时间告警"""
            alerts_triggered = []
            violation_counts = {rule_name: 0 for rule_name in rules}

            for i, metric_point in enumerate(metrics_data):
                for rule_name, rule in rules.items():
                    metric_name = rule['metric']
                    threshold = rule['threshold']
                    condition = rule['condition']
                    duration_required = rule['duration_required']

                    if metric_name in metric_point:
                        value = metric_point[metric_name]

                        # 检查是否违反阈值
                        violated = False
                        if condition == '>' and value > threshold:
                            violated = True
                        elif condition == '>=' and value >= threshold:
                            violated = True

                        if violated:
                            violation_counts[rule_name] += 1

                            # 检查是否达到持续时间要求
                            if violation_counts[rule_name] >= duration_required:
                                # 检查是否是新的告警（避免重复触发）
                                existing_alerts = [a for a in alerts_triggered
                                                 if a['rule_name'] == rule_name and a['status'] == 'firing']

                                if not existing_alerts:
                                    alert = {
                                        'alert_id': f"alert_{rule_name}_{int(time.time() * 1000)}",
                                        'rule_name': rule_name,
                                        'severity': rule['severity'],
                                        'metric': metric_name,
                                        'value': value,
                                        'threshold': threshold,
                                        'duration_violated': violation_counts[rule_name],
                                        'timestamp': metric_point['timestamp'],
                                        'status': 'firing'
                                    }
                                    alerts_triggered.append(alert)
                        else:
                            # 重置违规计数
                            violation_counts[rule_name] = 0

            return alerts_triggered

        # 评估持续时间告警
        duration_alerts = evaluate_duration_alerts(duration_rules, metrics_stream)

        # 验证持续时间告警触发
        # 应该触发CPU持续高告警（第3个数据点开始满足3周期要求）
        cpu_alerts = [a for a in duration_alerts if a['rule_name'] == 'persistent_high_cpu']
        memory_alerts = [a for a in duration_alerts if a['rule_name'] == 'sustained_memory_pressure']

        assert len(cpu_alerts) == 1, f"应该触发1个CPU持续告警，实际: {len(cpu_alerts)}"
        assert len(memory_alerts) == 1, f"应该触发1个内存持续告警，实际: {len(memory_alerts)}"

        # 验证CPU告警在第3个数据点触发（索引2）
        cpu_alert = cpu_alerts[0]
        assert cpu_alert['duration_violated'] >= 3, f"CPU告警应该在至少3个周期后触发: {cpu_alert['duration_violated']}"

        # 验证内存告警在第4个数据点触发（索引3）
        memory_alert = memory_alerts[0]
        assert memory_alert['duration_violated'] >= 2, f"内存告警应该在至少2个周期后触发: {memory_alert['duration_violated']}"

    def test_alert_deduplication_and_grouping(self):
        """测试告警去重和分组"""
        # 模拟重复告警场景
        raw_alerts = [
            {'rule': 'cpu_high', 'severity': 'warning', 'value': 85.0, 'timestamp': datetime.now()},
            {'rule': 'cpu_high', 'severity': 'warning', 'value': 87.0, 'timestamp': datetime.now()},  # 重复
            {'rule': 'memory_high', 'severity': 'error', 'value': 92.0, 'timestamp': datetime.now()},
            {'rule': 'cpu_high', 'severity': 'warning', 'value': 89.0, 'timestamp': datetime.now()},  # 重复
            {'rule': 'disk_full', 'severity': 'critical', 'value': 95.0, 'timestamp': datetime.now()},
            {'rule': 'memory_high', 'severity': 'error', 'value': 94.0, 'timestamp': datetime.now()}, # 重复但更严重
        ]

        def deduplicate_and_group_alerts(alerts: List[Dict], dedup_window_seconds: int = 300) -> Dict:
            """去重和分组告警"""
            grouped_alerts = {
                'active_alerts': {},
                'suppressed_alerts': [],
                'grouped_by_rule': {},
                'grouped_by_severity': {'info': [], 'warning': [], 'error': [], 'critical': []}
            }

            current_time = datetime.now()

            for alert in alerts:
                rule_name = alert['rule']
                severity = alert['severity']

                # 规则分组
                if rule_name not in grouped_alerts['grouped_by_rule']:
                    grouped_alerts['grouped_by_rule'][rule_name] = []
                grouped_alerts['grouped_by_rule'][rule_name].append(alert)

                # 严重程度分组
                grouped_alerts['grouped_by_severity'][severity].append(alert)

                # 去重逻辑：相同规则的告警在时间窗口内只保留最新的
                alert_key = rule_name

                if alert_key in grouped_alerts['active_alerts']:
                    existing_alert = grouped_alerts['active_alerts'][alert_key]

                    # 如果新告警更严重或值更高，替换现有告警
                    severity_levels = {'info': 1, 'warning': 2, 'error': 3, 'critical': 4}
                    new_severity_level = severity_levels[severity]
                    existing_severity_level = severity_levels[existing_alert['severity']]

                    should_replace = (
                        new_severity_level > existing_severity_level or
                        (new_severity_level == existing_severity_level and alert['value'] > existing_alert['value'])
                    )

                    if should_replace:
                        grouped_alerts['suppressed_alerts'].append(existing_alert)
                        grouped_alerts['active_alerts'][alert_key] = alert
                    else:
                        grouped_alerts['suppressed_alerts'].append(alert)
                else:
                    grouped_alerts['active_alerts'][alert_key] = alert

            return grouped_alerts

        # 处理告警去重和分组
        processed_alerts = deduplicate_and_group_alerts(raw_alerts)

        # 验证去重结果
        active_alerts = processed_alerts['active_alerts']
        suppressed_alerts = processed_alerts['suppressed_alerts']

        # 应该有3个活跃告警（cpu_high, memory_high, disk_full）
        assert len(active_alerts) == 3, f"应该有3个活跃告警，实际: {len(active_alerts)}"

        # 应该有3个被抑制的告警（2个cpu_high重复，1个memory_high被更严重的替换）
        assert len(suppressed_alerts) == 3, f"应该有3个被抑制告警，实际: {len(suppressed_alerts)}"

        # 验证活跃告警
        assert 'cpu_high' in active_alerts, "应该有活跃的CPU告警"
        assert 'memory_high' in active_alerts, "应该有活跃的内存告警"
        assert 'disk_full' in active_alerts, "应该有活跃的磁盘告警"

        # 验证CPU告警保留了最高值
        cpu_alert = active_alerts['cpu_high']
        assert cpu_alert['value'] == 89.0, f"CPU告警应该保留最高值89.0，实际: {cpu_alert['value']}"

        # 验证分组
        rule_groups = processed_alerts['grouped_by_rule']
        assert len(rule_groups['cpu_high']) == 3, "CPU规则应该有3个告警"
        assert len(rule_groups['memory_high']) == 2, "内存规则应该有2个告警"

        severity_groups = processed_alerts['grouped_by_severity']
        assert len(severity_groups['warning']) == 3, "应该有3个警告级别告警"
        assert len(severity_groups['error']) == 2, "应该有2个错误级别告警"
        assert len(severity_groups['critical']) == 1, "应该有1个严重级别告警"


class TestAlertNotificationConfiguration:
    """测试告警通知配置"""

    def setup_method(self):
        """测试前准备"""
        self.notification_manager = Mock()
        self.channel_tester = Mock()

    @patch('smtplib.SMTP')
    def test_email_notification_channel(self, mock_smtp):
        """测试邮件通知渠道"""
        # 模拟SMTP连接
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # 定义邮件通知配置
        email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'use_tls': True,
            'username': 'alerts@rqa2025.com',
            'password': 'secure_password',
            'from_address': 'alerts@rqa2025.com',
            'recipients': {
                'critical': ['ops@rqa2025.com', 'manager@rqa2025.com'],
                'warning': ['devops@rqa2025.com'],
                'error': ['developers@rqa2025.com']
            },
            'templates': {
                'critical': 'Critical Alert: {alert_title}\n\n{alert_description}\n\nTime: {timestamp}\n\nPlease take immediate action.',
                'warning': 'Warning Alert: {alert_title}\n\n{alert_description}\n\nTime: {timestamp}',
                'error': 'Error Alert: {alert_title}\n\n{alert_description}\n\nTime: {timestamp}'
            }
        }

        def send_alert_email(config: Dict, alert: Dict) -> bool:
            """发送告警邮件"""
            try:
                severity = alert.get('severity', 'warning')
                recipients = config['recipients'].get(severity, [])

                if not recipients:
                    return False

                # 获取邮件模板
                template = config['templates'].get(severity, config['templates']['warning'])

                # 渲染邮件内容
                email_body = template.format(
                    alert_title=alert.get('title', 'System Alert'),
                    alert_description=alert.get('description', 'No description provided'),
                    timestamp=alert.get('timestamp', datetime.now()).isoformat()
                )

                # 构造邮件
                subject = f"RQA2025 Alert - {severity.upper()}: {alert.get('title', 'System Alert')}"
                message = f"Subject: {subject}\n\n{email_body}"

                # 发送邮件（模拟）
                with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                    if config.get('use_tls'):
                        server.starttls()
                    server.login(config['username'], config['password'])

                    for recipient in recipients:
                        server.sendmail(config['from_address'], recipient, message)

                return True

            except Exception as e:
                print(f"邮件发送失败: {e}")
                return False

        # 测试不同严重程度的告警邮件
        test_alerts = [
            {
                'title': 'High CPU Usage',
                'description': 'CPU usage is at 85% for the last 5 minutes',
                'severity': 'warning',
                'timestamp': datetime.now()
            },
            {
                'title': 'Database Connection Failed',
                'description': 'Unable to connect to primary database',
                'severity': 'critical',
                'timestamp': datetime.now()
            }
        ]

        # 发送测试告警
        for alert in test_alerts:
            success = send_alert_email(email_config, alert)
            assert success, f"告警邮件发送失败: {alert['title']}"

        # 验证SMTP调用
        assert mock_smtp.called, "应该调用SMTP连接"

        # 验证邮件发送参数
        smtp_calls = mock_smtp.call_args_list
        assert len(smtp_calls) >= 2, "应该至少调用SMTP两次（每个告警一次）"

    @patch('requests.post')
    def test_slack_notification_channel(self, mock_post):
        """测试Slack通知渠道"""
        # 模拟Slack API响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'ok': True, 'timestamp': '1234567890.123456'}
        mock_post.return_value = mock_response

        # 定义Slack通知配置
        slack_config = {
            'webhook_url': 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX',
            'channels': {
                'critical': '#alerts-critical',
                'warning': '#alerts-warning',
                'error': '#alerts-dev'
            },
            'username': 'RQA2025 Alert Bot',
            'icon_emoji': ':warning:',
            'templates': {
                'critical': {
                    'color': 'danger',
                    'title': '🚨 Critical Alert',
                    'text': '{description}'
                },
                'warning': {
                    'color': 'warning',
                    'title': '⚠️ Warning Alert',
                    'text': '{description}'
                },
                'error': {
                    'color': 'danger',
                    'title': '❌ Error Alert',
                    'text': '{description}'
                }
            }
        }

        def send_slack_alert(config: Dict, alert: Dict) -> bool:
            """发送Slack告警"""
            try:
                severity = alert.get('severity', 'warning')
                channel = config['channels'].get(severity)

                if not channel:
                    return False

                # 获取消息模板
                template = config['templates'].get(severity, config['templates']['warning'])

                # 构造Slack消息
                message = {
                    'channel': channel,
                    'username': config.get('username', 'Alert Bot'),
                    'icon_emoji': config.get('icon_emoji', ':warning:'),
                    'attachments': [{
                        'color': template['color'],
                        'title': template['title'].format(**alert),
                        'text': template['text'].format(**alert),
                        'fields': [
                            {
                                'title': 'Severity',
                                'value': severity.upper(),
                                'short': True
                            },
                            {
                                'title': 'Time',
                                'value': alert.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                                'short': True
                            }
                        ],
                        'footer': 'RQA2025 Monitoring System',
                        'ts': int(time.time())
                    }]
                }

                # 发送到Slack
                response = requests.post(config['webhook_url'], json=message)
                return response.status_code == 200 and response.json().get('ok', False)

            except Exception as e:
                print(f"Slack通知发送失败: {e}")
                return False

        # 测试不同严重程度的告警Slack通知
        test_alerts = [
            {
                'description': 'CPU usage is at 85% for the last 5 minutes',
                'severity': 'warning',
                'timestamp': datetime.now()
            },
            {
                'description': 'Database connection pool exhausted',
                'severity': 'critical',
                'timestamp': datetime.now()
            }
        ]

        # 发送测试告警
        for alert in test_alerts:
            success = send_slack_alert(slack_config, alert)
            assert success, f"Slack告警发送失败: {alert['description'][:50]}..."

        # 验证HTTP请求
        assert mock_post.called, "应该调用HTTP POST请求"

        # 验证请求参数
        call_args = mock_post.call_args
        request_data = call_args[1]['json']  # json参数

        assert 'channel' in request_data, "Slack消息应该包含频道"
        assert 'username' in request_data, "Slack消息应该包含用户名"
        assert 'attachments' in request_data, "Slack消息应该包含附件"

        # 验证附件内容
        attachment = request_data['attachments'][0]
        assert 'color' in attachment, "附件应该包含颜色"
        assert 'title' in attachment, "附件应该包含标题"
        assert 'fields' in attachment, "附件应该包含字段"

    def test_pagerduty_integration(self):
        """测试PagerDuty集成"""
        # 模拟PagerDuty集成配置
        pagerduty_config = {
            'routing_key': 'your-pagerduty-routing-key',
            'api_url': 'https://events.pagerduty.com/v2/enqueue',
            'services': {
                'critical': {
                    'service_id': 'PXXXXXX',
                    'escalation_policy_id': 'PYYYYYY'
                },
                'warning': {
                    'service_id': 'PAAAAA',
                    'escalation_policy_id': 'PBBBBB'
                }
            }
        }

        def create_pagerduty_event(config: Dict, alert: Dict) -> Dict:
            """创建PagerDuty事件"""
            severity = alert.get('severity', 'warning')
            service_config = config['services'].get(severity)

            if not service_config:
                return {'success': False, 'error': f'No service configured for severity: {severity}'}

            # 构造PagerDuty事件
            event = {
                'routing_key': config['routing_key'],
                'event_action': 'trigger',
                'payload': {
                    'summary': alert.get('title', 'System Alert'),
                    'source': 'rqa2025-monitoring',
                    'severity': severity,
                    'component': alert.get('component', 'system'),
                    'group': alert.get('group', 'infrastructure'),
                    'class': alert.get('class', 'performance'),
                    'custom_details': {
                        'metric': alert.get('metric'),
                        'value': alert.get('value'),
                        'threshold': alert.get('threshold'),
                        'description': alert.get('description')
                    }
                }
            }

            # 只有critical级别的告警才路由到PagerDuty
            if severity == 'critical':
                return {
                    'success': True,
                    'event': event,
                    'service_id': service_config['service_id'],
                    'escalation_policy': service_config['escalation_policy_id']
                }
            else:
                return {
                    'success': False,
                    'reason': f'Severity {severity} does not trigger PagerDuty alert'
                }

        # 测试critical级别告警
        critical_alert = {
            'title': 'Database Down',
            'severity': 'critical',
            'component': 'database',
            'metric': 'connection_status',
            'value': 'down',
            'description': 'Primary database is unreachable'
        }

        critical_result = create_pagerduty_event(pagerduty_config, critical_alert)
        assert critical_result['success'], "Critical告警应该触发PagerDuty事件"
        assert 'event' in critical_result, "应该包含PagerDuty事件数据"

        event = critical_result['event']
        assert event['payload']['severity'] == 'critical', "事件严重程度应该是critical"
        assert event['payload']['summary'] == 'Database Down', "事件摘要应该正确"

        # 测试warning级别告警（不应该触发PagerDuty）
        warning_alert = {
            'title': 'High CPU Usage',
            'severity': 'warning',
            'component': 'application',
            'metric': 'cpu_percent',
            'value': 85.0,
            'description': 'CPU usage is high'
        }

        warning_result = create_pagerduty_event(pagerduty_config, warning_alert)
        assert not warning_result['success'], "Warning告警不应该触发PagerDuty事件"
        assert 'reason' in warning_result, "应该说明不触发的原因"


if __name__ == "__main__":
    pytest.main([__file__])
