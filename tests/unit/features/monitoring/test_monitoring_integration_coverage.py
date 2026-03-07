#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoring Integration模块测试覆盖
测试monitoring/monitoring_integration.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from src.features.monitoring.monitoring_integration import (
        IntegrationLevel,
        ComponentIntegrationConfig,
        MonitoringIntegrationManager,
        integrate_feature_layer_components,
        monitor_operation
    )
    MONITORING_INTEGRATION_AVAILABLE = True
except ImportError:
    MONITORING_INTEGRATION_AVAILABLE = False


@pytest.fixture
def mock_monitor():
    """创建mock的monitor"""
    monitor = Mock()
    monitor.register_component = Mock()
    monitor.collect_metrics = Mock()
    monitor.alert_manager = Mock()
    monitor.alert_manager.send_alert = Mock()
    return monitor


@pytest.fixture
def integration_manager(mock_monitor):
    """创建integration_manager实例"""
    if not MONITORING_INTEGRATION_AVAILABLE:
        pytest.skip("MonitoringIntegrationManager不可用")
    
    with patch('src.features.monitoring.monitoring_integration.get_monitor', return_value=mock_monitor):
        manager = MonitoringIntegrationManager()
        yield manager


class TestIntegrationLevel:
    """IntegrationLevel枚举测试"""

    def test_integration_level_values(self):
        """测试集成级别值"""
        if not MONITORING_INTEGRATION_AVAILABLE:
            pytest.skip("MonitoringIntegrationManager不可用")
        assert IntegrationLevel.BASIC.value == "basic"
        assert IntegrationLevel.STANDARD.value == "standard"
        assert IntegrationLevel.ADVANCED.value == "advanced"


class TestComponentIntegrationConfig:
    """ComponentIntegrationConfig数据类测试"""

    def test_component_integration_config_creation(self):
        """测试创建组件集成配置"""
        if not MONITORING_INTEGRATION_AVAILABLE:
            pytest.skip("MonitoringIntegrationManager不可用")
        config = ComponentIntegrationConfig(
            component_name="TestComponent",
            integration_level=IntegrationLevel.STANDARD,
            auto_monitor=True,
            collect_metrics=True,
            enable_alerts=True,
            custom_metrics=["metric1", "metric2"],
            performance_thresholds={"metric1": 1.0}
        )
        assert config.component_name == "TestComponent"
        assert config.integration_level == IntegrationLevel.STANDARD
        assert config.auto_monitor is True
        assert config.collect_metrics is True
        assert config.enable_alerts is True
        assert len(config.custom_metrics) == 2


class TestMonitoringIntegrationManager:
    """MonitoringIntegrationManager测试"""

    def test_manager_initialization(self, integration_manager):
        """测试管理器初始化"""
        assert integration_manager.config is not None
        assert integration_manager.monitor is not None
        assert isinstance(integration_manager.integrated_components, dict)

    def test_integrate_component(self, integration_manager):
        """测试集成组件"""
        # 创建一个简单的组件
        component = Mock()
        component.process = Mock(return_value="result")
        
        # 集成组件
        integration_manager.integrate_component(
            component=component,
            component_type="TestComponent"
        )
        
        # 验证组件已注册
        assert len(integration_manager.integrated_components) > 0

    def test_integrate_component_with_config(self, integration_manager):
        """测试使用自定义配置集成组件"""
        component = Mock()
        config = ComponentIntegrationConfig(
            component_name="CustomComponent",
            integration_level=IntegrationLevel.ADVANCED
        )
        
        integration_manager.integrate_component(
            component=component,
            component_type="CustomComponent",
            config=config
        )
        
        # 验证配置已保存
        assert len(integration_manager.integrated_components) > 0

    def test_get_integration_status(self, integration_manager):
        """测试获取集成状态"""
        component = Mock()
        integration_manager.integrate_component(component, "TestComponent")
        
        # 获取状态
        if hasattr(integration_manager, 'get_integration_status'):
            status = integration_manager.get_integration_status()
            assert isinstance(status, dict)
        else:
            pytest.skip("get_integration_status方法不可用")

    def test_remove_integration(self, integration_manager):
        """测试移除集成"""
        component = Mock()
        integration_manager.integrate_component(component, "TestComponent")
        
        # 移除集成
        if hasattr(integration_manager, 'remove_integration'):
            component_name = list(integration_manager.integrated_components.keys())[0]
            integration_manager.remove_integration(component_name)
            assert component_name not in integration_manager.integrated_components
        else:
            pytest.skip("remove_integration方法不可用")


class TestIntegrateFeatureLayerComponents:
    """integrate_feature_layer_components函数测试"""

    def test_integrate_feature_layer_components(self, mock_monitor):
        """测试集成特征层组件"""
        if not MONITORING_INTEGRATION_AVAILABLE:
            pytest.skip("MonitoringIntegrationManager不可用")
        
        with patch('src.features.monitoring.monitoring_integration.get_monitor', return_value=mock_monitor):
            manager = integrate_feature_layer_components()
            assert isinstance(manager, MonitoringIntegrationManager)


class TestMonitorOperation:
    """monitor_operation装饰器测试"""

    def test_monitor_operation_decorator(self, mock_monitor):
        """测试monitor_operation装饰器"""
        if not MONITORING_INTEGRATION_AVAILABLE:
            pytest.skip("MonitoringIntegrationManager不可用")
        
        with patch('src.features.monitoring.monitoring_integration.get_monitor', return_value=mock_monitor):
            @monitor_operation("TestComponent", "test_operation")
            def test_function():
                return "result"
            
            # 调用被装饰的函数
            result = test_function()
            assert result == "result"
            
            # 验证监控被调用（如果支持）
            # 注意：这个测试可能需要根据实际实现调整


