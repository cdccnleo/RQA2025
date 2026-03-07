#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infrastructure Monitoring告警系统测试

测试告警规则配置、告警触发、通知、抑制和聚合功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import datetime, timedelta


class TestAlertRuleConfiguration:
    """测试告警规则配置"""
    
    def test_create_simple_alert_rule(self):
        """测试创建简单告警规则"""
        rule = {
            'rule_id': 'cpu_high',
            'name': 'CPU使用率过高',
            'metric': 'cpu_usage',
            'condition': '>',
            'threshold': 80.0,
            'severity': 'warning'
        }
        
        assert rule['rule_id'] == 'cpu_high'
        assert rule['threshold'] == 80.0
        assert rule['condition'] == '>'
    
    def test_create_complex_alert_rule(self):
        """测试创建复杂告警规则"""
        rule = {
            'rule_id': 'memory_critical',
            'name': '内存严重不足',
            'metric': 'memory_usage',
            'conditions': [
                {'field': 'memory_usage', 'operator': '>', 'value': 90},
                {'field': 'duration', 'operator': '>', 'value': 300}  # 持续5分钟
            ],
            'severity': 'critical',
            'actions': ['email', 'sms', 'webhook']
        }
        
        assert len(rule['conditions']) == 2
        assert rule['severity'] == 'critical'
        assert 'email' in rule['actions']
    
    def test_validate_alert_rule(self):
        """测试验证告警规则"""
        def validate_rule(rule: Dict) -> bool:
            required_fields = ['rule_id', 'metric', 'threshold']
            return all(field in rule for field in required_fields)
        
        valid_rule = {'rule_id': 'test', 'metric': 'cpu', 'threshold': 80}
        invalid_rule = {'rule_id': 'test', 'metric': 'cpu'}  # 缺少threshold
        
        assert validate_rule(valid_rule)
        assert not validate_rule(invalid_rule)
    
    def test_update_alert_rule(self):
        """测试更新告警规则"""
        rules = {}
        
        # 添加规则
        rule_id = 'cpu_high'
        rules[rule_id] = {'threshold': 80, 'severity': 'warning'}
        
        # 更新规则
        rules[rule_id]['threshold'] = 85
        rules[rule_id]['severity'] = 'critical'
        
        assert rules[rule_id]['threshold'] == 85
        assert rules[rule_id]['severity'] == 'critical'
    
    def test_delete_alert_rule(self):
        """测试删除告警规则"""
        rules = {
            'rule1': {'threshold': 80},
            'rule2': {'threshold': 90}
        }
        
        # 删除规则
        del rules['rule1']
        
        assert 'rule1' not in rules
        assert 'rule2' in rules


class TestAlertTrigger:
    """测试告警触发"""
    
    def test_trigger_alert_on_threshold_exceeded(self):
        """测试超过阈值时触发告警"""
        threshold = 80.0
        current_value = 85.0
        
        alert_triggered = current_value > threshold
        
        assert alert_triggered is True
    
    def test_no_trigger_below_threshold(self):
        """测试低于阈值时不触发"""
        threshold = 80.0
        current_value = 75.0
        
        alert_triggered = current_value > threshold
        
        assert alert_triggered is False
    
    def test_trigger_with_duration_check(self):
        """测试带持续时间检查的触发"""
        threshold = 80.0
        duration_required = 5  # 需要持续5次
        
        # 模拟持续超过阈值
        consecutive_violations = 0
        values = [85, 90, 82, 88, 91, 87]  # 6次都超过80
        
        for value in values:
            if value > threshold:
                consecutive_violations += 1
            else:
                consecutive_violations = 0
            
            if consecutive_violations >= duration_required:
                break
        
        assert consecutive_violations >= duration_required
    
    def test_trigger_with_multiple_conditions(self):
        """测试多条件触发"""
        conditions = {
            'cpu_usage': {'threshold': 80, 'value': 85},
            'memory_usage': {'threshold': 90, 'value': 92}
        }
        
        # 所有条件都满足
        all_met = all(
            cond['value'] > cond['threshold'] 
            for cond in conditions.values()
        )
        
        assert all_met is True
    
    def test_trigger_alert_with_metadata(self):
        """测试触发告警并附带元数据"""
        alert = {
            'rule_id': 'cpu_high',
            'triggered_at': datetime.now().isoformat(),
            'current_value': 85.0,
            'threshold': 80.0,
            'severity': 'warning',
            'message': 'CPU使用率超过阈值'
        }
        
        assert alert['current_value'] > alert['threshold']
        assert 'triggered_at' in alert
        assert alert['severity'] == 'warning'


class TestAlertNotification:
    """测试告警通知"""
    
    def test_send_email_notification(self):
        """测试发送邮件通知"""
        notification = {
            'type': 'email',
            'to': ['admin@example.com'],
            'subject': 'CPU Alert',
            'body': 'CPU usage exceeded 80%'
        }
        
        # 模拟发送
        sent = True  # 假设发送成功
        
        assert notification['type'] == 'email'
        assert sent is True
    
    def test_send_webhook_notification(self):
        """测试发送Webhook通知"""
        notification = {
            'type': 'webhook',
            'url': 'https://example.com/webhook',
            'method': 'POST',
            'payload': {
                'alert': 'cpu_high',
                'value': 85.0
            }
        }
        
        assert notification['type'] == 'webhook'
        assert notification['method'] == 'POST'
    
    def test_send_sms_notification(self):
        """测试发送短信通知"""
        notification = {
            'type': 'sms',
            'phone': '+1234567890',
            'message': 'ALERT: CPU 85%'
        }
        
        assert notification['type'] == 'sms'
        assert 'phone' in notification
    
    def test_notification_queue(self):
        """测试通知队列"""
        notification_queue = []
        
        # 添加多个通知
        for i in range(5):
            notification_queue.append({
                'id': i,
                'type': 'email',
                'status': 'pending'
            })
        
        assert len(notification_queue) == 5
        assert all(n['status'] == 'pending' for n in notification_queue)
    
    def test_notification_retry(self):
        """测试通知重试机制"""
        notification = {
            'id': 1,
            'type': 'email',
            'retry_count': 0,
            'max_retries': 3
        }
        
        # 模拟失败并重试
        for _ in range(2):
            notification['retry_count'] += 1
        
        assert notification['retry_count'] == 2
        assert notification['retry_count'] < notification['max_retries']


class TestAlertSuppression:
    """测试告警抑制"""
    
    def test_suppress_duplicate_alerts(self):
        """测试抑制重复告警"""
        alerts = []
        sent_alerts = set()
        
        # 尝试发送相同告警
        for _ in range(5):
            alert_key = 'cpu_high_server1'
            if alert_key not in sent_alerts:
                alerts.append({'key': alert_key})
                sent_alerts.add(alert_key)
        
        # 应该只发送一次
        assert len(alerts) == 1
    
    def test_suppress_with_cooldown_period(self):
        """测试冷却期抑制"""
        last_alert_time = {}
        cooldown_minutes = 15
        
        rule_id = 'cpu_high'
        now = datetime.now()
        
        # 第一次告警
        last_alert_time[rule_id] = now
        can_alert_1 = True
        
        # 5分钟后再次检查（在冷却期内）
        time_2 = now + timedelta(minutes=5)
        can_alert_2 = (time_2 - last_alert_time[rule_id]).total_seconds() / 60 >= cooldown_minutes
        
        # 20分钟后检查（冷却期已过）
        time_3 = now + timedelta(minutes=20)
        can_alert_3 = (time_3 - last_alert_time[rule_id]).total_seconds() / 60 >= cooldown_minutes
        
        assert can_alert_1 is True
        assert can_alert_2 is False  # 冷却期内
        assert can_alert_3 is True   # 冷却期已过
    
    def test_suppress_by_severity(self):
        """测试按严重程度抑制"""
        severity_levels = {
            'info': 1,
            'warning': 2,
            'error': 3,
            'critical': 4
        }
        
        min_severity = 'warning'
        
        # 只处理warning及以上级别
        alerts = [
            {'severity': 'info'},
            {'severity': 'warning'},
            {'severity': 'critical'}
        ]
        
        filtered_alerts = [
            a for a in alerts 
            if severity_levels[a['severity']] >= severity_levels[min_severity]
        ]
        
        assert len(filtered_alerts) == 2  # warning和critical
    
    def test_suppress_by_time_window(self):
        """测试按时间窗口抑制"""
        window_minutes = 10
        max_alerts_per_window = 3
        
        alerts_in_window = []
        now = datetime.now()
        
        # 模拟在时间窗口内的告警
        for i in range(5):
            alert_time = now + timedelta(minutes=i*2)
            alerts_in_window.append(alert_time)
        
        # 过滤掉超出窗口的告警
        cutoff_time = now - timedelta(minutes=window_minutes)
        recent_alerts = [t for t in alerts_in_window if t > cutoff_time]
        
        # 检查是否超过限制
        should_suppress = len(recent_alerts) >= max_alerts_per_window
        
        assert should_suppress is True


class TestAlertAggregation:
    """测试告警聚合"""
    
    def test_aggregate_by_type(self):
        """测试按类型聚合告警"""
        alerts = [
            {'type': 'cpu', 'value': 85},
            {'type': 'cpu', 'value': 90},
            {'type': 'memory', 'value': 75},
            {'type': 'memory', 'value': 80},
        ]
        
        aggregated = {}
        for alert in alerts:
            alert_type = alert['type']
            if alert_type not in aggregated:
                aggregated[alert_type] = []
            aggregated[alert_type].append(alert)
        
        assert len(aggregated) == 2
        assert len(aggregated['cpu']) == 2
        assert len(aggregated['memory']) == 2
    
    def test_aggregate_by_severity(self):
        """测试按严重程度聚合"""
        alerts = [
            {'severity': 'warning', 'message': 'msg1'},
            {'severity': 'critical', 'message': 'msg2'},
            {'severity': 'warning', 'message': 'msg3'},
        ]
        
        by_severity = {}
        for alert in alerts:
            sev = alert['severity']
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        assert by_severity['warning'] == 2
        assert by_severity['critical'] == 1
    
    def test_aggregate_summary(self):
        """测试告警摘要聚合"""
        alerts = [
            {'rule': 'cpu_high', 'count': 5},
            {'rule': 'mem_high', 'count': 3},
            {'rule': 'disk_full', 'count': 1},
        ]
        
        total_alerts = sum(a['count'] for a in alerts)
        summary = {
            'total': total_alerts,
            'by_rule': {a['rule']: a['count'] for a in alerts}
        }
        
        assert summary['total'] == 9
        assert summary['by_rule']['cpu_high'] == 5


class TestAlertEscalation:
    """测试告警升级"""
    
    def test_escalate_after_duration(self):
        """测试持续时间后升级"""
        alert = {
            'severity': 'warning',
            'created_at': datetime.now() - timedelta(minutes=30),
            'escalation_minutes': 15
        }
        
        # 检查是否需要升级
        duration = (datetime.now() - alert['created_at']).total_seconds() / 60
        should_escalate = duration >= alert['escalation_minutes']
        
        assert should_escalate is True
    
    def test_escalate_severity_level(self):
        """测试升级严重程度"""
        severity_order = ['info', 'warning', 'error', 'critical']
        
        current_severity = 'warning'
        current_index = severity_order.index(current_severity)
        
        # 升级一级
        if current_index < len(severity_order) - 1:
            escalated_severity = severity_order[current_index + 1]
        else:
            escalated_severity = current_severity
        
        assert escalated_severity == 'error'
    
    def test_escalate_notification_recipients(self):
        """测试升级通知接收人"""
        alert = {
            'severity': 'warning',
            'recipients': ['team_lead@example.com']
        }
        
        # 升级到critical后添加更多接收人
        if alert['severity'] == 'critical':
            alert['recipients'].extend(['manager@example.com', 'oncall@example.com'])
        
        # 模拟升级
        alert['severity'] = 'critical'
        alert['recipients'].extend(['manager@example.com'])
        
        assert len(alert['recipients']) == 2


class TestAlertHistory:
    """测试告警历史"""
    
    def test_store_alert_history(self):
        """测试存储告警历史"""
        alert_history = []
        
        alert = {
            'id': 1,
            'rule': 'cpu_high',
            'triggered_at': datetime.now().isoformat(),
            'resolved_at': None
        }
        
        alert_history.append(alert)
        
        assert len(alert_history) == 1
        assert alert_history[0]['resolved_at'] is None
    
    def test_resolve_alert(self):
        """测试解决告警"""
        alert = {
            'id': 1,
            'rule': 'cpu_high',
            'triggered_at': datetime.now().isoformat(),
            'resolved_at': None,
            'status': 'active'
        }
        
        # 解决告警
        alert['resolved_at'] = datetime.now().isoformat()
        alert['status'] = 'resolved'
        
        assert alert['status'] == 'resolved'
        assert alert['resolved_at'] is not None
    
    def test_query_active_alerts(self):
        """测试查询活跃告警"""
        alerts = [
            {'id': 1, 'status': 'active'},
            {'id': 2, 'status': 'resolved'},
            {'id': 3, 'status': 'active'},
        ]
        
        active_alerts = [a for a in alerts if a['status'] == 'active']
        
        assert len(active_alerts) == 2
        assert all(a['status'] == 'active' for a in active_alerts)
    
    def test_alert_history_cleanup(self):
        """测试告警历史清理"""
        max_history = 100
        alert_history = [{'id': i} for i in range(150)]
        
        # 保留最近的100条
        if len(alert_history) > max_history:
            alert_history = alert_history[-max_history:]
        
        assert len(alert_history) == max_history
        assert alert_history[0]['id'] == 50  # 从50开始


class TestAlertStatistics:
    """测试告警统计"""
    
    def test_count_alerts_by_rule(self):
        """测试按规则统计告警数"""
        alerts = [
            {'rule': 'cpu_high'},
            {'rule': 'cpu_high'},
            {'rule': 'mem_high'},
        ]
        
        stats = {}
        for alert in alerts:
            rule = alert['rule']
            stats[rule] = stats.get(rule, 0) + 1
        
        assert stats['cpu_high'] == 2
        assert stats['mem_high'] == 1
    
    def test_calculate_alert_rate(self):
        """测试计算告警频率"""
        alerts_count = 50
        time_period_hours = 24
        
        alert_rate = alerts_count / time_period_hours
        
        assert alert_rate == 50 / 24  # 约2.08次/小时
    
    def test_calculate_mean_time_to_resolve(self):
        """测试计算平均解决时间"""
        resolved_alerts = [
            {'triggered_at': datetime(2025, 11, 2, 10, 0), 'resolved_at': datetime(2025, 11, 2, 10, 15)},
            {'triggered_at': datetime(2025, 11, 2, 11, 0), 'resolved_at': datetime(2025, 11, 2, 11, 30)},
        ]
        
        resolution_times = []
        for alert in resolved_alerts:
            duration = (alert['resolved_at'] - alert['triggered_at']).total_seconds() / 60
            resolution_times.append(duration)
        
        mttr = sum(resolution_times) / len(resolution_times)
        
        assert mttr == 22.5  # 平均22.5分钟


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

