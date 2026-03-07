"""
资源管理配置类测试

测试覆盖:
- ResourceMonitorConfig: 资源监控配置
"""

import pytest

from src.infrastructure.resource.config.config_classes import ResourceMonitorConfig


class TestResourceMonitorConfig:
    """资源监控配置测试"""
    
    def test_create_with_defaults(self):
        """测试使用默认值创建配置"""
        config = ResourceMonitorConfig()
        
        assert config.monitor_interval == 60
        assert config.auto_scale is True
        assert config.max_resources == 100
        assert config.enable_cpu_monitoring is True
        assert config.enable_memory_monitoring is True
        assert config.enable_disk_monitoring is True
        assert config.history_size == 1000
        assert config.precision == 2
    
    def test_default_alert_thresholds(self):
        """测试默认告警阈值"""
        config = ResourceMonitorConfig()
        
        assert config.alert_threshold['cpu'] == 90.0
        assert config.alert_threshold['memory'] == 85.0
        assert config.alert_threshold['disk'] == 80.0
    
    def test_default_notification_channels(self):
        """测试默认通知渠道"""
        config = ResourceMonitorConfig()
        
        assert 'email' in config.notification_channels
        assert 'slack' in config.notification_channels
        assert len(config.notification_channels) == 2
    
    def test_default_thresholds(self):
        """测试默认阈值配置"""
        config = ResourceMonitorConfig()
        
        assert config.thresholds['cpu_warning'] == 80.0
        assert config.thresholds['memory_warning'] == 85.0
        assert config.thresholds['disk_warning'] == 90.0
    
    def test_backward_compatibility_attributes(self):
        """测试向后兼容属性"""
        config = ResourceMonitorConfig()
        
        assert config.cpu_threshold == 90.0
        assert config.memory_threshold == 85.0
        assert config.disk_threshold == 80.0
    
    def test_create_with_custom_values(self):
        """测试使用自定义值创建配置"""
        config = ResourceMonitorConfig(
            monitor_interval=30,
            auto_scale=False,
            max_resources=200,
            history_size=2000,
            precision=3
        )
        
        assert config.monitor_interval == 30
        assert config.auto_scale is False
        assert config.max_resources == 200
        assert config.history_size == 2000
        assert config.precision == 3
    
    def test_custom_alert_thresholds(self):
        """测试自定义告警阈值"""
        custom_thresholds = {
            'cpu': 80.0,
            'memory': 75.0,
            'disk': 70.0
        }
        
        config = ResourceMonitorConfig(
            alert_threshold=custom_thresholds
        )
        
        assert config.alert_threshold == custom_thresholds
        assert config.alert_threshold['cpu'] == 80.0
    
    def test_custom_notification_channels(self):
        """测试自定义通知渠道"""
        custom_channels = ['email', 'sms', 'webhook']
        
        config = ResourceMonitorConfig(
            notification_channels=custom_channels
        )
        
        assert config.notification_channels == custom_channels
        assert len(config.notification_channels) == 3
    
    def test_disable_monitoring(self):
        """测试禁用监控"""
        config = ResourceMonitorConfig(
            enable_cpu_monitoring=False,
            enable_memory_monitoring=False,
            enable_disk_monitoring=False
        )
        
        assert config.enable_cpu_monitoring is False
        assert config.enable_memory_monitoring is False
        assert config.enable_disk_monitoring is False
    
    def test_reasonable_monitor_interval(self):
        """测试监控间隔合理性"""
        config = ResourceMonitorConfig(monitor_interval=30)
        assert config.monitor_interval >= 1  # 至少1秒
        
        config2 = ResourceMonitorConfig(monitor_interval=300)
        assert config2.monitor_interval <= 600  # 不超过10分钟比较合理
    
    def test_threshold_consistency(self):
        """测试阈值一致性"""
        config = ResourceMonitorConfig()
        
        # alert_threshold和独立阈值属性应该保持一致性
        assert config.cpu_threshold == config.alert_threshold['cpu']
        assert config.memory_threshold == config.alert_threshold['memory']
        assert config.disk_threshold == config.alert_threshold['disk']
    
    def test_thresholds_in_valid_range(self):
        """测试阈值在有效范围内"""
        config = ResourceMonitorConfig()
        
        # 所有阈值应该在0-100之间（百分比）
        for key, value in config.alert_threshold.items():
            assert 0 <= value <= 100, f"{key} threshold {value} not in valid range [0, 100]"
        
        for key, value in config.thresholds.items():
            assert 0 <= value <= 100, f"{key} threshold {value} not in valid range [0, 100]"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

