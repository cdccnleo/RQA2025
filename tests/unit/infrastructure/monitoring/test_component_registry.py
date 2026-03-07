#!/usr/bin/env python3
"""
RQA2025 基础设施层组件注册表单元测试
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import threading
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.infrastructure.monitoring.core.component_registry import (
    ComponentRegistry as InfrastructureComponentRegistry, ComponentMetadata, ComponentInstance,
    global_component_registry, register_component, get_component
)


class MockComponent:
    """模拟组件类"""

    def __init__(self, name: str = "test", config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.started = False
        self.stopped = False

    def start_monitoring(self):
        self.started = True

    def stop_monitoring(self):
        self.stopped = True

    def get_health_status(self):
        return {'status': 'healthy'}


class TestComponentMetadata:
    """测试组件元数据"""

    def test_initialization(self):
        """测试初始化"""
        metadata = ComponentMetadata(
            name="test_component",
            version="1.0.0",
            description="Test component",
            dependencies=["dep1", "dep2"],
            capabilities=["monitoring", "alerting"]
        )

        assert metadata.name == "test_component"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test component"
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.capabilities == ["monitoring", "alerting"]
        assert metadata.health_status == "unknown"
        assert metadata.last_health_check is None

    def test_to_dict(self):
        """测试转换为字典"""
        metadata = ComponentMetadata(
            name="test_component",
            version="1.0.0",
            description="Test component"
        )

        data = metadata.to_dict()
        assert data['name'] == "test_component"
        assert data['version'] == "1.0.0"
        assert data['description'] == "Test component"
        assert data['health_status'] == "unknown"

    def test_update_health_status(self):
        """测试更新健康状态"""
        metadata = ComponentMetadata("test", "1.0.0")
        metadata.update_health_status("healthy")

        assert metadata.health_status == "healthy"
        assert metadata.last_health_check is not None


class TestComponentInstance:
    """测试组件实例"""

    def test_initialization(self):
        """测试初始化"""
        metadata = ComponentMetadata("test", "1.0.0")
        config = {"param1": "value1"}
        instance = ComponentInstance(MockComponent, metadata, config)

        assert instance.component_class == MockComponent
        assert instance.metadata == metadata
        assert instance.config == config
        assert instance.instance is None
        assert not instance.is_running

    def test_create_instance(self):
        """测试创建实例"""
        metadata = ComponentMetadata("test", "1.0.0")
        config = {"name": "test_instance"}
        instance = ComponentInstance(MockComponent, metadata, config)

        component = instance.create_instance()
        assert isinstance(component, MockComponent)
        assert component.name == "test_instance"
        assert instance.instance == component

    def test_start_stop(self):
        """测试启动和停止"""
        metadata = ComponentMetadata("test", "1.0.0")
        instance = ComponentInstance(MockComponent, metadata)

        # 启动
        assert instance.start()
        assert instance.is_running
        assert instance.instance.started
        assert instance.start_count == 1

        # 停止
        assert instance.stop()
        assert not instance.is_running
        assert instance.instance.stopped

    def test_restart(self):
        """测试重启"""
        metadata = ComponentMetadata("test", "1.0.0")
        instance = ComponentInstance(MockComponent, metadata)

        # 初始启动
        instance.start()
        original_instance = instance.instance

        # 重启
        assert instance.restart()
        assert instance.is_running
        assert instance.start_count == 2

    def test_update_config(self):
        """测试更新配置"""
        metadata = ComponentMetadata("test", "1.0.0")
        config = {"param1": "value1"}
        instance = ComponentInstance(MockComponent, metadata, config)

        new_config = {"param2": "value2"}
        assert instance.update_config(new_config)

        assert instance.config["param1"] == "value1"
        assert instance.config["param2"] == "value2"

    def test_get_status(self):
        """测试获取状态"""
        metadata = ComponentMetadata("test", "1.0.0")
        config = {"param1": "value1"}
        instance = ComponentInstance(MockComponent, metadata, config)

        status = instance.get_status()
        assert status['name'] == "test"
        assert status['version'] == "1.0.0"
        assert not status['is_running']
        assert status['start_count'] == 0
        assert status['config_keys'] == ["param1"]


class TestInfrastructureComponentRegistry:
    """测试组件注册表"""

    def setup_method(self):
        """测试前准备"""
        self.registry = InfrastructureComponentRegistry()

    def teardown_method(self):
        """测试后清理"""
        # 清理所有组件
        for name in list(self.registry.components.keys()):
            self.registry.unregister_component(name)

    def test_register_component(self):
        """测试注册组件"""
        success = self.registry.register_component(
            name="test_component",
            component_class=MockComponent,
            version="1.0.0",
            description="Test component"
        )

        assert success
        assert "test_component" in self.registry.components
        assert "test_component" in self.registry.metadata

    def test_register_component_with_dependencies(self):
        """测试注册有依赖的组件"""
        # 先注册依赖组件
        self.registry.register_component("dep1", MockComponent)

        # 注册主组件
        success = self.registry.register_component(
            name="main_component",
            component_class=MockComponent,
            dependencies=["dep1"]
        )

        assert success

    def test_unregister_component(self):
        """测试注销组件"""
        self.registry.register_component("test", MockComponent)

        success = self.registry.unregister_component("test")
        assert success
        assert "test" not in self.registry.components
        assert "test" not in self.registry.metadata

    def test_get_component(self):
        """测试获取组件"""
        self.registry.register_component("test", MockComponent)

        component = self.registry.get_component("test")
        assert component is not None
        assert isinstance(component, ComponentInstance)

        # 获取不存在的组件
        component = self.registry.get_component("nonexistent")
        assert component is None

    def test_get_component_instance(self):
        """测试获取组件实例对象"""
        self.registry.register_component("test", MockComponent)

        instance = self.registry.get_component_instance("test")
        assert instance is None  # 还未创建实例

        # 启动组件后获取实例
        self.registry.start_component("test")
        instance = self.registry.get_component_instance("test")
        assert isinstance(instance, MockComponent)

    def test_list_components(self):
        """测试列出组件"""
        self.registry.register_component("comp1", MockComponent)
        self.registry.register_component("comp2", MockComponent)

        components = self.registry.list_components()
        assert len(components) == 2
        names = [c['name'] for c in components]
        assert "comp1" in names
        assert "comp2" in names

    def test_find_components_by_capability(self):
        """测试根据功能查找组件"""
        self.registry.register_component(
            "monitoring_comp",
            MockComponent,
            capabilities=["monitoring", "metrics"]
        )
        self.registry.register_component(
            "alerting_comp",
            MockComponent,
            capabilities=["alerting"]
        )

        monitoring_comps = self.registry.find_components_by_capability("monitoring")
        assert "monitoring_comp" in monitoring_comps
        assert "alerting_comp" not in monitoring_comps

    def test_start_stop_component(self):
        """测试启动和停止组件"""
        self.registry.register_component("test", MockComponent)

        # 启动
        assert self.registry.start_component("test")
        component = self.registry.get_component("test")
        assert component.is_running

        # 停止
        assert self.registry.stop_component("test")
        assert not component.is_running

    def test_restart_component(self):
        """测试重启组件"""
        self.registry.register_component("test", MockComponent)
        self.registry.start_component("test")

        original_start_count = self.registry.components["test"].start_count

        # 重启
        assert self.registry.restart_component("test")
        component = self.registry.get_component("test")
        assert component.is_running
        assert component.start_count == original_start_count + 1

    def test_update_component_config(self):
        """测试更新组件配置"""
        config = {"initial": "value"}
        self.registry.register_component("test", MockComponent, config=config)

        new_config = {"additional": "param"}
        assert self.registry.update_component_config("test", new_config)

        component = self.registry.get_component("test")
        assert component.config["initial"] == "value"
        assert component.config["additional"] == "param"

    def test_check_dependencies(self):
        """测试检查依赖关系"""
        # 注册依赖组件
        self.registry.register_component("dep1", MockComponent)

        # 注册主组件
        self.registry.register_component(
            "main",
            MockComponent,
            dependencies=["dep1", "missing_dep"]
        )

        deps_check = self.registry.check_dependencies("main")
        assert not deps_check['satisfied']
        assert "missing_dep" in deps_check['missing']
        assert deps_check['total_dependencies'] == 2

    def test_get_system_health(self):
        """测试获取系统健康状态"""
        # 注册一些组件
        self.registry.register_component("comp1", MockComponent)
        self.registry.register_component("comp2", MockComponent)

        health = self.registry.get_system_health()
        assert health['total_components'] == 2
        assert 'running_components' in health
        assert 'healthy_components' in health
        assert 'dependency_satisfaction' in health
        assert 'overall_health' in health

    @patch('builtins.open')
    @patch('json.dump')
    def test_save_registry_state(self, mock_json_dump, mock_open):
        """测试保存注册表状态"""
        self.registry.register_component("test", MockComponent)

        success = self.registry.save_registry_state("test.json")
        assert success
        mock_json_dump.assert_called_once()

    @patch('builtins.open')
    @patch('json.load')
    def test_load_registry_state(self, mock_json_load, mock_open):
        """测试加载注册表状态"""
        mock_json_load.return_value = {
            'timestamp': '2024-01-01T00:00:00',
            'components': {}
        }

        success = self.registry.load_registry_state("test.json")
        assert success
        mock_json_load.assert_called_once()


class TestGlobalFunctions:
    """测试全局函数"""

    def test_register_component_global(self):
        """测试全局注册函数"""
        success = register_component("global_test", MockComponent)
        assert success

        # 清理
        global_component_registry.unregister_component("global_test")

    def test_get_component_global(self):
        """测试全局获取函数"""
        register_component("global_test2", MockComponent)
        component = get_component("global_test2")
        assert component is None  # 未启动

        # 清理
        global_component_registry.unregister_component("global_test2")
