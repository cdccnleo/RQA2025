import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.monitoring.alert_manager import AlertManager, AlertRule, AlertThreshold

@pytest.fixture
def mock_config_manager():
    """创建mock的ConfigManager"""
    mock_cm = MagicMock()
    mock_cm.get_config.return_value = {
        'alert_rules': [
            {
                'name': 'cpu_high',
                'condition': 'cpu_usage > 80',
                'severity': 'warning',
                'notify_channels': ['email'],
                'cooldown': 300
            },
            {
                'name': 'memory_critical',
                'condition': 'memory_usage > 95',
                'severity': 'critical',
                'notify_channels': ['email', 'sms'],
                'cooldown': 60
            }
        ],
        'smtp': {
            'host': 'localhost',
            'port': 25,
            'user': 'test@example.com',
            'password': 'password',
            'from': 'alerts@rqa2025.com',
            'to': 'admin@rqa2025.com'
        },
        'sms': {
            'phone': '+1234567890'
        },
        'wechat': {
            'group': 'test_group'
        },
        'slack': {
            'channel': '#alerts'
        }
    }
    return mock_cm

@pytest.fixture
def alert_manager(mock_config_manager):
    """创建AlertManager实例"""
    config = {'test': 'config'}
    return AlertManager(config, config_manager=mock_config_manager)

class TestAlertManager:
    """AlertManager测试类"""

    def test_init_with_test_hook(self, mock_config_manager):
        """测试使用测试钩子初始化"""
        config = {'test': 'config'}
        manager = AlertManager(config, config_manager=mock_config_manager)
        
        assert manager.config_manager == mock_config_manager
        assert len(manager.alert_rules) == 2
        assert manager.alert_rules[0].name == 'cpu_high'
        assert manager.alert_rules[1].name == 'memory_critical'

    def test_init_without_test_hook(self):
        """测试不使用测试钩子初始化"""
        config = {'test': 'config'}
        with patch('src.infrastructure.monitoring.alert_manager.ConfigManager') as mock_cm_class:
            mock_cm_instance = MagicMock()
            mock_cm_class.return_value = mock_cm_instance
            mock_cm_instance.get_config.return_value = []
            
            manager = AlertManager(config)
            
            assert manager.config_manager == mock_cm_instance
            mock_cm_class.assert_called_once_with(config)

    def test_load_alert_rules(self, alert_manager):
        """测试加载告警规则"""
        assert len(alert_manager.alert_rules) == 2
        
        # 验证规则内容
        cpu_rule = alert_manager.alert_rules[0]
        assert cpu_rule.name == 'cpu_high'
        assert cpu_rule.condition == 'cpu_usage > 80'
        assert cpu_rule.severity == 'warning'
        assert cpu_rule.notify_channels == ['email']
        assert cpu_rule.cooldown == 300

    def test_check_metrics_no_alerts(self, alert_manager):
        """测试检查指标 - 无告警"""
        metrics = {'cpu_usage': 50, 'memory_usage': 60}
        alerts = alert_manager.check_metrics(metrics)
        
        assert len(alerts) == 0

    def test_check_metrics_with_alerts(self, alert_manager):
        """测试检查指标 - 有告警"""
        metrics = {'cpu_usage': 85, 'memory_usage': 97}
        
        with patch.object(alert_manager, 'notify') as mock_notify:
            alerts = alert_manager.check_metrics(metrics)
            
            assert len(alerts) == 2
            assert mock_notify.call_count == 2

    def test_evaluate_condition_simple(self, alert_manager):
        """测试条件评估 - 简单条件"""
        condition = 'cpu_usage > 80'
        metrics = {'cpu_usage': 85}
        
        result = alert_manager._evaluate_condition(condition, metrics)
        assert result is True

    def test_evaluate_condition_complex(self, alert_manager):
        """测试条件评估 - 复杂条件"""
        condition = 'cpu_usage > 80 and memory_usage > 90'
        metrics = {'cpu_usage': 85, 'memory_usage': 95}
        
        result = alert_manager._evaluate_condition(condition, metrics)
        assert result is True

    def test_evaluate_condition_false(self, alert_manager):
        """测试条件评估 - 条件不满足"""
        condition = 'cpu_usage > 80'
        metrics = {'cpu_usage': 70}
        
        result = alert_manager._evaluate_condition(condition, metrics)
        assert result is False

    def test_evaluate_condition_error(self, alert_manager):
        """测试条件评估 - 错误处理"""
        condition = 'invalid_condition'
        metrics = {'cpu_usage': 85}
        
        result = alert_manager._evaluate_condition(condition, metrics)
        assert result is False

    def test_notify_success(self, alert_manager):
        """测试通知发送 - 成功"""
        alert = {
            'rule': 'test_rule',
            'severity': 'warning',
            'metrics': {'cpu_usage': 85},
            'timestamp': time.time()
        }
        channels = ['email']
        
        with patch.object(alert_manager, '_send_email') as mock_send:
            result = alert_manager.notify(alert, channels)
            
            assert result is True
            mock_send.assert_called_once()

    def test_notify_failure(self, alert_manager):
        """测试通知发送 - 失败"""
        alert = {
            'rule': 'test_rule',
            'severity': 'warning',
            'metrics': {'cpu_usage': 85},
            'timestamp': time.time()
        }
        channels = ['unknown_channel']
        
        result = alert_manager.notify(alert, channels)
        assert result is False

    def test_format_alert_message(self, alert_manager):
        """测试告警消息格式化"""
        alert = {
            'rule': 'cpu_high',
            'severity': 'warning',
            'metrics': {'cpu_usage': 85, 'memory_usage': 70},
            'timestamp': time.time()
        }
        
        message = alert_manager._format_alert_message(alert)
        
        assert 'cpu_high' in message
        assert 'warning' in message
        assert 'cpu_usage: 85' in message
        assert 'memory_usage: 70' in message

    def test_format_metrics(self, alert_manager):
        """测试指标格式化"""
        metrics = {'cpu_usage': 85, 'memory_usage': 70}
        
        formatted = alert_manager._format_metrics(metrics)
        
        assert 'cpu_usage: 85' in formatted
        assert 'memory_usage: 70' in formatted

    def test_add_alert_rule(self, alert_manager):
        """测试添加告警规则"""
        new_rule = AlertRule(
            name='disk_full',
            condition='disk_usage > 90',
            severity='critical',
            notify_channels=['email', 'sms'],
            cooldown=120
        )
        
        initial_count = len(alert_manager.alert_rules)
        alert_manager.add_alert_rule(new_rule)
        
        assert len(alert_manager.alert_rules) == initial_count + 1
        assert alert_manager.alert_rules[-1].name == 'disk_full'

    def test_get_active_alerts(self, alert_manager):
        """测试获取活跃告警"""
        # 模拟一些告警历史
        alert_manager.alert_history['cpu_high'] = time.time() - 100  # 100秒前
        alert_manager.alert_history['memory_critical'] = time.time() - 30  # 30秒前
        
        active_alerts = alert_manager.get_active_alerts()
        
        # 应该返回最近触发的告警
        assert len(active_alerts) > 0

    @patch('smtplib.SMTP')
    def test_send_email(self, mock_smtp, alert_manager):
        """测试发送邮件"""
        message = "Test alert message"
        severity = "warning"
        
        alert_manager._send_email(message, severity)
        
        mock_smtp.assert_called_once()

    def test_send_sms(self, alert_manager):
        """测试发送短信"""
        message = "Test alert message"
        severity = "critical"
        
        with patch.object(alert_manager.config_manager, 'get_config') as mock_get:
            mock_get.return_value = {'phone': '+1234567890'}
            
            alert_manager._send_sms(message, severity)
            # 应该记录日志

    def test_send_wechat(self, alert_manager):
        """测试发送企业微信"""
        message = "Test alert message"
        severity = "warning"
        
        with patch.object(alert_manager.config_manager, 'get_config') as mock_get:
            mock_get.return_value = {'group': 'test_group'}
            
            alert_manager._send_wechat(message, severity)
            # 应该记录日志

    def test_send_slack(self, alert_manager):
        """测试发送Slack"""
        message = "Test alert message"
        severity = "critical"
        
        with patch.object(alert_manager.config_manager, 'get_config') as mock_get:
            mock_get.return_value = {'channel': '#alerts'}
            
            alert_manager._send_slack(message, severity)
            # 应该记录日志

    def test_cooldown_mechanism(self, alert_manager):
        """测试冷却机制"""
        metrics = {'cpu_usage': 85}
        
        # 第一次触发告警
        with patch.object(alert_manager, 'notify') as mock_notify:
            alerts1 = alert_manager.check_metrics(metrics)
            assert len(alerts1) > 0
            assert mock_notify.call_count > 0
        
        # 立即再次检查，应该因为冷却时间而不触发
        with patch.object(alert_manager, 'notify') as mock_notify:
            alerts2 = alert_manager.check_metrics(metrics)
            assert mock_notify.call_count == 0

    def test_load_config_from_manager(self, alert_manager, mock_config_manager):
        """测试从配置管理器加载配置"""
        # 清空现有规则
        alert_manager.alert_rules.clear()
        
        # 重新加载配置
        alert_manager.load_alert_rules()
        
        # 验证规则被加载
        assert len(alert_manager.alert_rules) == 2
        assert alert_manager.alert_rules[0].name == 'cpu_high'
        assert alert_manager.alert_rules[1].name == 'memory_critical' 