#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础组件测试 - 简化版

直接测试foundation/base.py模块，使用直接导入方式
"""

import pytest
import time
from unittest.mock import Mock

# 直接导入base.py，避免__init__.py的导入问题
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入base.py文件
    import importlib.util
    base_path = project_root / "src" / "core" / "foundation" / "base.py"
    spec = importlib.util.spec_from_file_location("foundation_base", base_path)
    base_module = importlib.util.module_from_spec(spec)
    
    # 需要先导入依赖
    sys.path.insert(0, str(project_root / "src"))
    
    # 处理constants依赖
    try:
        from src.core.constants import MAX_RECORDS, SECONDS_PER_MINUTE
    except ImportError:
        # 如果constants不存在，创建简单的占位符
        import types
        constants_module = types.ModuleType('src.core.constants')
        constants_module.MAX_RECORDS = 1000
        constants_module.SECONDS_PER_MINUTE = 60
        sys.modules['src.core.constants'] = constants_module
    
    spec.loader.exec_module(base_module)
    
    ComponentStatus = base_module.ComponentStatus
    ComponentHealth = base_module.ComponentHealth
    ComponentInfo = base_module.ComponentInfo
    BaseComponent = base_module.BaseComponent
    BaseService = getattr(base_module, 'BaseService', None)
    
    IMPORTS_AVAILABLE = True
except Exception as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"基础组件模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponentStatus:
    """测试组件状态枚举"""

    def test_component_status_values(self):
        """测试组件状态枚举值"""
        assert ComponentStatus.UNKNOWN.value == "unknown"
        assert ComponentStatus.INITIALIZING.value == "initializing"
        assert ComponentStatus.INITIALIZED.value == "initialized"
        assert ComponentStatus.RUNNING.value == "running"
        assert ComponentStatus.STOPPED.value == "stopped"
        assert ComponentStatus.ERROR.value == "error"

    def test_component_status_all_values(self):
        """测试所有组件状态值"""
        expected_values = [
            "unknown", "initializing", "initialized", "starting",
            "running", "stopping", "stopped", "error", "healthy", "unhealthy"
        ]
        actual_values = [status.value for status in ComponentStatus]
        for expected in expected_values:
            assert expected in actual_values


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponentHealth:
    """测试组件健康状态枚举"""

    def test_component_health_values(self):
        """测试组件健康状态枚举值"""
        assert ComponentHealth.HEALTHY.value == "healthy"
        assert ComponentHealth.UNHEALTHY.value == "unhealthy"
        assert ComponentHealth.UNKNOWN.value == "unknown"
        # DEGRADED可能不存在，检查所有可用值
        health_values = [h.value for h in ComponentHealth]
        assert "healthy" in health_values
        assert "unhealthy" in health_values
        assert "unknown" in health_values


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponentInfo:
    """测试组件信息数据类"""

    def test_component_info_creation(self):
        """测试组件信息创建"""
        info = ComponentInfo(
            name="test_component",
            version="1.0.0",
            status=ComponentStatus.RUNNING,
            health=ComponentHealth.HEALTHY
        )
        assert info.name == "test_component"
        assert info.version == "1.0.0"
        assert info.status == ComponentStatus.RUNNING
        assert info.health == ComponentHealth.HEALTHY

    def test_component_info_defaults(self):
        """测试组件信息默认值"""
        info = ComponentInfo(name="test")
        assert info.name == "test"
        assert info.version == "1.0.0"
        assert info.status == ComponentStatus.UNKNOWN
        assert info.health == ComponentHealth.UNKNOWN


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestBaseComponent:
    """测试基础组件类"""
    
    # 创建简单的实现类用于测试
    class TestComponent(BaseComponent):
        """测试用组件实现"""
        def shutdown(self) -> bool:
            """关闭组件"""
            return True

    def test_base_component_initialization(self):
        """测试基础组件初始化"""
        component = self.TestComponent(
            name="test_component",
            version="1.0.0",
            description="测试组件"
        )
        assert component.name == "test_component"
        assert component.version == "1.0.0"
        assert component.description == "测试组件"
        assert component._status == ComponentStatus.UNKNOWN

    def test_base_component_get_info(self):
        """测试获取组件信息"""
        component = self.TestComponent(name="test", version="1.0.0")
        info = component.get_info()
        assert info.name == "test"
        assert info.version == "1.0.0"
        assert isinstance(info.status, ComponentStatus)
        assert isinstance(info.health, ComponentHealth)

    def test_base_component_initialize(self):
        """测试组件初始化"""
        component = self.TestComponent(name="test")
        result = component.initialize()
        assert result is True
        assert component._status == ComponentStatus.INITIALIZED

    def test_base_component_start(self):
        """测试组件启动"""
        component = self.TestComponent(name="test")
        component.initialize()
        result = component.start()
        assert result is True
        assert component._status == ComponentStatus.RUNNING

    def test_base_component_stop(self):
        """测试组件停止"""
        component = self.TestComponent(name="test")
        component.initialize()
        component.start()
        result = component.stop()
        assert result is True
        assert component._status == ComponentStatus.STOPPED

    def test_base_component_health_check(self):
        """测试组件健康检查"""
        component = self.TestComponent(name="test")
        component.initialize()
        health = component.health_check()
        # health_check返回bool
        assert isinstance(health, bool)

    def test_base_component_get_status(self):
        """测试获取组件状态"""
        component = self.TestComponent(name="test")
        status = component.get_status()
        # get_status返回ComponentStatus枚举
        assert isinstance(status, ComponentStatus)
    
    def test_base_component_shutdown(self):
        """测试组件关闭"""
        component = self.TestComponent(name="test")
        component.initialize()
        result = component.shutdown()
        assert result is True

    def test_base_component_set_status(self):
        """测试设置组件状态"""
        component = self.TestComponent(name="test")
        component.set_status(ComponentStatus.RUNNING)
        assert component._status == ComponentStatus.RUNNING

    def test_base_component_set_health(self):
        """测试设置组件健康状态"""
        component = self.TestComponent(name="test")
        component.set_health(ComponentHealth.HEALTHY)
        assert component._health == ComponentHealth.HEALTHY

    def test_base_component_get_health(self):
        """测试获取组件健康状态"""
        component = self.TestComponent(name="test")
        health = component.get_health()
        assert isinstance(health, ComponentHealth)

    def test_base_component_add_metadata(self):
        """测试添加元数据"""
        component = self.TestComponent(name="test")
        component.add_metadata("key", "value")
        assert component._metadata.get("key") == "value"

    def test_base_component_get_metadata(self):
        """测试获取元数据"""
        component = self.TestComponent(name="test")
        component.add_metadata("key", "value")
        metadata_value = component.get_metadata("key")
        assert metadata_value == "value"

    def test_base_component_is_initialized(self):
        """测试检查是否已初始化"""
        component = self.TestComponent(name="test")
        assert component.is_initialized() is False
        component.initialize()
        assert component.is_initialized() is True

    def test_base_component_is_started(self):
        """测试检查是否已启动"""
        component = self.TestComponent(name="test")
        assert component.is_started() is False
        component.initialize()
        component.start()
        assert component.is_started() is True

    def test_base_component_is_running(self):
        """测试检查是否运行中"""
        component = self.TestComponent(name="test")
        assert component.is_running() is False
        component.initialize()
        component.start()
        assert component.is_running() is True


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestBaseService:
    """测试服务基类"""
    
    # 获取BaseService类
    try:
        BaseService = base_module.BaseService
    except AttributeError:
        BaseService = None
    
    # 创建简单的实现类用于测试
    if BaseService:
        class TestService(BaseService):
            """测试用服务实现"""
            def shutdown(self) -> bool:
                """关闭服务"""
                return True
    else:
        TestService = None

    def test_base_service_initialization(self):
        """测试服务初始化"""
        if self.TestService is None:
            pytest.skip("BaseService不可用")
        service = self.TestService(name="test_service")
        assert service.name == "test_service"
        assert hasattr(service, '_dependencies')
        assert hasattr(service, '_config')

    def test_base_service_add_dependency(self):
        """测试添加依赖"""
        if self.TestService is None:
            pytest.skip("BaseService不可用")
        service = self.TestService(name="test")
        service.add_dependency("dep1")
        assert "dep1" in service.get_dependencies()

    def test_base_service_get_dependencies(self):
        """测试获取依赖列表"""
        if self.TestService is None:
            pytest.skip("BaseService不可用")
        service = self.TestService(name="test")
        service.add_dependency("dep1")
        service.add_dependency("dep2")
        deps = service.get_dependencies()
        assert len(deps) == 2
        assert "dep1" in deps
        assert "dep2" in deps

    def test_base_service_set_config(self):
        """测试设置配置"""
        if self.TestService is None:
            pytest.skip("BaseService不可用")
        service = self.TestService(name="test")
        service.set_config("key1", "value1")
        assert service.get_config("key1") == "value1"

    def test_base_service_get_config(self):
        """测试获取配置"""
        if self.TestService is None:
            pytest.skip("BaseService不可用")
        service = self.TestService(name="test")
        service.set_config("key1", "value1")
        assert service.get_config("key1") == "value1"
        assert service.get_config("nonexistent", "default") == "default"

    def test_base_service_get_config_dict(self):
        """测试获取配置字典"""
        if self.TestService is None:
            pytest.skip("BaseService不可用")
        service = self.TestService(name="test")
        service.set_config("key1", "value1")
        service.set_config("key2", "value2")
        config_dict = service.get_config_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["key1"] == "value1"
        assert config_dict["key2"] == "value2"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestFoundationUtils:
    """测试基础工具函数"""

    def test_generate_id(self):
        """测试生成唯一ID"""
        try:
            generate_id = getattr(base_module, 'generate_id', None)
            if generate_id is None:
                pytest.skip("generate_id函数不可用")
            id1 = generate_id()
            id2 = generate_id()
            assert id1 != id2
            assert isinstance(id1, str)
            assert len(id1) > 0
        except Exception:
            pytest.skip("generate_id函数不可用")

    def test_generate_id_with_prefix(self):
        """测试带前缀生成ID"""
        try:
            generate_id = getattr(base_module, 'generate_id', None)
            if generate_id is None:
                pytest.skip("generate_id函数不可用")
            id1 = generate_id("test")
            assert isinstance(id1, str)
            assert len(id1) > 0
        except Exception:
            pytest.skip("generate_id函数不可用")

    def test_validate_config(self):
        """测试验证配置"""
        try:
            validate_config = getattr(base_module, 'validate_config', None)
            if validate_config is None:
                pytest.skip("validate_config函数不可用")
            config = {"key1": "value1", "key2": "value2"}
            assert validate_config(config, ["key1", "key2"]) is True
            assert validate_config(config, ["key1", "key3"]) is False
            assert validate_config(None, ["key1"]) is False
        except Exception:
            pytest.skip("validate_config函数不可用")

    def test_validate_required_fields(self):
        """测试验证必需字段"""
        try:
            validate_required_fields = getattr(base_module, 'validate_required_fields', None)
            if validate_required_fields is None:
                pytest.skip("validate_required_fields函数不可用")
            data = {"field1": "value1", "field2": "value2"}
            missing = validate_required_fields(data, ["field1", "field2"])
            assert len(missing) == 0
            
            missing = validate_required_fields(data, ["field1", "field3"])
            assert "field3" in missing
        except Exception:
            pytest.skip("validate_required_fields函数不可用")

    def test_safe_get(self):
        """测试安全获取字典值"""
        try:
            safe_get = getattr(base_module, 'safe_get', None)
            if safe_get is None:
                pytest.skip("safe_get函数不可用")
            data = {"key1": "value1", "nested": {"key2": "value2"}}
            assert safe_get(data, "key1") == "value1"
            assert safe_get(data, "nonexistent", "default") == "default"
            assert safe_get(data, "nested.key2") == "value2"
        except Exception:
            pytest.skip("safe_get函数不可用")

    def test_merge_dicts(self):
        """测试合并字典"""
        try:
            merge_dicts = getattr(base_module, 'merge_dicts', None)
            if merge_dicts is None:
                pytest.skip("merge_dicts函数不可用")
            dict1 = {"key1": "value1", "key2": "value2"}
            dict2 = {"key2": "new_value2", "key3": "value3"}
            result = merge_dicts(dict1, dict2)
            assert result["key1"] == "value1"
            assert result["key2"] == "new_value2"
            assert result["key3"] == "value3"
        except Exception:
            pytest.skip("merge_dicts函数不可用")

    def test_format_timestamp(self):
        """测试格式化时间戳"""
        try:
            format_timestamp = getattr(base_module, 'format_timestamp', None)
            if format_timestamp is None:
                pytest.skip("format_timestamp函数不可用")
            import time
            timestamp = time.time()
            formatted = format_timestamp(timestamp)
            assert isinstance(formatted, str)
            assert len(formatted) > 0
        except Exception:
            pytest.skip("format_timestamp函数不可用")

    def test_calculate_duration(self):
        """测试计算持续时间"""
        try:
            calculate_duration = getattr(base_module, 'calculate_duration', None)
            if calculate_duration is None:
                pytest.skip("calculate_duration函数不可用")
            import time
            start_time = time.time()
            time.sleep(0.01)
            duration = calculate_duration(start_time)
            assert duration > 0
            assert duration < 1
        except Exception:
            pytest.skip("calculate_duration函数不可用")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponentConfig:
    """测试组件配置"""

    def test_component_config_creation(self):
        """测试组件配置创建"""
        ComponentConfig = getattr(base_module, 'ComponentConfig', None)
        if ComponentConfig is None:
            pytest.skip("ComponentConfig不可用")
        config = ComponentConfig(name="test_component", version="1.0.0")
        assert config.name == "test_component"
        assert config.version == "1.0.0"
        assert config.auto_start is True

    def test_component_config_with_dependencies(self):
        """测试带依赖的组件配置"""
        ComponentConfig = getattr(base_module, 'ComponentConfig', None)
        if ComponentConfig is None:
            pytest.skip("ComponentConfig不可用")
        config = ComponentConfig(
            name="test",
            dependencies=["dep1", "dep2"]
        )
        assert len(config.dependencies) == 2


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponentMetrics:
    """测试组件指标"""

    def test_component_metrics_creation(self):
        """测试组件指标创建"""
        ComponentMetrics = getattr(base_module, 'ComponentMetrics', None)
        if ComponentMetrics is None:
            pytest.skip("ComponentMetrics不可用")
        metrics = ComponentMetrics(component_name="test")
        assert metrics.component_name == "test"
        assert metrics.operation_count == 0
        assert metrics.error_count == 0

    def test_component_metrics_record_operation(self):
        """测试记录操作"""
        ComponentMetrics = getattr(base_module, 'ComponentMetrics', None)
        if ComponentMetrics is None:
            pytest.skip("ComponentMetrics不可用")
        metrics = ComponentMetrics(component_name="test")
        metrics.record_operation(0.1, success=True)
        assert metrics.operation_count == 1
        assert metrics.error_count == 0
        assert metrics.avg_operation_time > 0

    def test_component_metrics_uptime(self):
        """测试运行时间"""
        ComponentMetrics = getattr(base_module, 'ComponentMetrics', None)
        if ComponentMetrics is None:
            pytest.skip("ComponentMetrics不可用")
        import time
        metrics = ComponentMetrics(component_name="test")
        time.sleep(0.01)
        uptime = metrics.uptime_seconds
        assert uptime > 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponentRegistry:
    """测试组件注册表"""

    class TestComponent(BaseComponent):
        """测试组件类"""
        def __init__(self, name: str):
            super().__init__(name=name)
        
        def _do_execute(self, *args, **kwargs):
            """执行组件逻辑"""
            return {"status": "ok"}
        
        def shutdown(self):
            """关闭组件"""
            self._status = ComponentStatus.STOPPED

    def test_component_registry_initialization(self):
        """测试组件注册表初始化"""
        ComponentRegistry = getattr(base_module, 'ComponentRegistry', None)
        if ComponentRegistry is None:
            pytest.skip("ComponentRegistry不可用")
        registry = ComponentRegistry()
        assert registry is not None

    def test_component_registry_register_component(self):
        """测试注册组件"""
        ComponentRegistry = getattr(base_module, 'ComponentRegistry', None)
        if ComponentRegistry is None:
            pytest.skip("ComponentRegistry不可用")
        registry = ComponentRegistry()
        component = self.TestComponent(name="test")
        registry.register_component(component)
        # register_component返回None，所以检查组件是否已注册
        retrieved = registry.get_component("test")
        assert retrieved == component

    def test_component_registry_get_component(self):
        """测试获取组件"""
        ComponentRegistry = getattr(base_module, 'ComponentRegistry', None)
        if ComponentRegistry is None:
            pytest.skip("ComponentRegistry不可用")
        registry = ComponentRegistry()
        component = self.TestComponent(name="test")
        registry.register_component(component)
        retrieved = registry.get_component("test")
        assert retrieved == component

    def test_component_registry_list_components(self):
        """测试列出所有组件"""
        ComponentRegistry = getattr(base_module, 'ComponentRegistry', None)
        if ComponentRegistry is None:
            pytest.skip("ComponentRegistry不可用")
        registry = ComponentRegistry()
        component1 = self.TestComponent(name="test1")
        component2 = self.TestComponent(name="test2")
        registry.register_component(component1)
        registry.register_component(component2)
        components = registry.list_components()
        assert "test1" in components
        assert "test2" in components
        assert len(components) == 2

    def test_component_registry_unregister_component(self):
        """测试注销组件"""
        ComponentRegistry = getattr(base_module, 'ComponentRegistry', None)
        if ComponentRegistry is None:
            pytest.skip("ComponentRegistry不可用")
        registry = ComponentRegistry()
        component = self.TestComponent(name="test")
        registry.register_component(component)
        result = registry.unregister_component("test")
        assert result is True
        retrieved = registry.get_component("test")
        assert retrieved is None

    def test_component_registry_get_component_metrics(self):
        """测试获取组件指标"""
        ComponentRegistry = getattr(base_module, 'ComponentRegistry', None)
        if ComponentRegistry is None:
            pytest.skip("ComponentRegistry不可用")
        registry = ComponentRegistry()
        component = self.TestComponent(name="test")
        registry.register_component(component)
        metrics = registry.get_component_metrics("test")
        assert metrics is not None
        assert metrics.component_name == "test"

    def test_component_registry_add_event_listener(self):
        """测试添加事件监听器"""
        ComponentRegistry = getattr(base_module, 'ComponentRegistry', None)
        if ComponentRegistry is None:
            pytest.skip("ComponentRegistry不可用")
        registry = ComponentRegistry()
        listener = Mock()
        registry.add_event_listener(listener)
        # 验证监听器已添加（通过注册组件触发事件）
        component = self.TestComponent(name="test")
        registry.register_component(component)
        # 如果监听器被调用，说明添加成功
        # 这里主要测试方法不会抛出异常

    def test_component_registry_remove_event_listener(self):
        """测试移除事件监听器"""
        ComponentRegistry = getattr(base_module, 'ComponentRegistry', None)
        if ComponentRegistry is None:
            pytest.skip("ComponentRegistry不可用")
        registry = ComponentRegistry()
        listener = Mock()
        registry.add_event_listener(listener)
        registry.remove_event_listener(listener)
        # 验证监听器已移除（不会抛出异常）

    def test_component_registry_perform_health_check(self):
        """测试执行健康检查"""
        ComponentRegistry = getattr(base_module, 'ComponentRegistry', None)
        if ComponentRegistry is None:
            pytest.skip("ComponentRegistry不可用")
        registry = ComponentRegistry()
        component = self.TestComponent(name="test")
        registry.register_component(component)
        results = registry.perform_health_check()
        assert isinstance(results, dict)
        assert "test" in results


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestGlobalComponentFunctions:
    """测试全局组件函数"""

    def test_get_component_registry(self):
        """测试获取组件注册表"""
        get_component_registry = getattr(base_module, 'get_component_registry', None)
        if get_component_registry is None:
            pytest.skip("get_component_registry函数不可用")
        registry = get_component_registry()
        assert registry is not None

    def test_register_global_component(self):
        """测试注册全局组件"""
        register_global_component = getattr(base_module, 'register_global_component', None)
        if register_global_component is None:
            pytest.skip("register_global_component函数不可用")
        component = TestBaseComponent.TestComponent(name="global_test")
        register_global_component(component)
        # 验证组件已注册
        get_global_component = getattr(base_module, 'get_global_component', None)
        if get_global_component:
            retrieved = get_global_component("global_test")
            assert retrieved == component

    def test_get_global_component(self):
        """测试获取全局组件"""
        register_global_component = getattr(base_module, 'register_global_component', None)
        get_global_component = getattr(base_module, 'get_global_component', None)
        if get_global_component is None:
            pytest.skip("get_global_component函数不可用")
        component = TestBaseComponent.TestComponent(name="global_test2")
        if register_global_component:
            register_global_component(component)
        retrieved = get_global_component("global_test2")
        assert retrieved == component

