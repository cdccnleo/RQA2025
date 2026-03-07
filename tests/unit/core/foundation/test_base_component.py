#!/usr/bin/env python3
"""
BaseComponent单元测试

测试组件基类的所有功能
"""

import pytest
from typing import Dict, Any

# 尝试导入所需模块
try:
    from src.core.foundation.base_component import BaseComponent
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestComponent(BaseComponent):
    """测试用组件类"""
    
    def __init__(self, name: str = "test", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.execute_count = 0
        self.init_called = False
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        self.init_called = True
        return config.get('init_success', True)
    
    def _do_execute(self, *args, **kwargs) -> Any:
        self.execute_count += 1
        return kwargs.get('return_value', 'executed')


class TestBaseComponent:
    """BaseComponent测试套件"""
    
    def test_component_creation(self):
        """测试组件创建"""
        component = TestComponent(name="test_component")
        
        assert component.name == "test_component"
        assert component._status == ComponentStatus.UNINITIALIZED
        assert component.config == {}
        assert component._error is None
    
    def test_component_initialization_success(self):
        """测试组件初始化成功"""
        component = TestComponent()
        config = {'init_success': True, 'key': 'value'}
        
        result = component.initialize(config)
        
        assert result is True
        assert component.init_called is True
        assert component._status == ComponentStatus.INITIALIZED
        assert component.config['key'] == 'value'
        assert component._initialized_at is not None
    
    def test_component_initialization_failure(self):
        """测试组件初始化失败"""
        component = TestComponent()
        config = {'init_success': False}
        
        result = component.initialize(config)
        
        assert result is False
        assert component._status == ComponentStatus.ERROR
    
    def test_component_execute_without_init(self):
        """测试未初始化就执行会抛出异常"""
        component = TestComponent()
        
        with pytest.raises(RuntimeError) as exc_info:
            component.execute()
        
        assert "未初始化" in str(exc_info.value)
    
    def test_component_execute_success(self):
        """测试组件执行成功"""
        component = TestComponent()
        component.initialize({})
        
        result = component.execute(return_value='test_result')
        
        assert result == 'test_result'
        assert component.execute_count == 1
        assert component._status == ComponentStatus.INITIALIZED
    
    def test_component_execute_error(self):
        """测试组件执行错误"""
        class ErrorComponent(BaseComponent):
            def _do_execute(self, *args, **kwargs):
                raise ValueError("Test error")
        
        component = ErrorComponent("error_comp")
        component.initialize({})
        
        with pytest.raises(ValueError):
            component.execute()
        
        assert component._status == ComponentStatus.ERROR
        assert component._error is not None
    
    def test_component_get_info(self):
        """测试获取组件信息"""
        component = TestComponent(name="info_test")
        component.initialize({'test_key': 'test_value'})
        
        info = component.get_info()
        
        assert info['name'] == 'info_test'
        assert info['type'] == 'TestComponent'
        assert info['status'] == ComponentStatus.INITIALIZED.value
        assert info['config']['test_key'] == 'test_value'
        assert info['created_at'] is not None
        assert info['initialized_at'] is not None
    
    def test_component_status_methods(self):
        """测试组件状态方法"""
        component = TestComponent()
        
        # 初始状态
        assert component.get_status() == ComponentStatus.UNINITIALIZED
        assert component.is_initialized() is False
        
        # 初始化后
        component.initialize({})
        assert component.get_status() == ComponentStatus.INITIALIZED
        assert component.is_initialized() is True
        
        # 重置后
        component.reset()
        assert component.get_status() == ComponentStatus.UNINITIALIZED
        assert component.is_initialized() is False
    
    def test_component_error_tracking(self):
        """测试错误跟踪"""
        class ErrorComponent(BaseComponent):
            def _do_execute(self, *args, **kwargs):
                raise ValueError("Test error")
        
        component = ErrorComponent("error_comp")
        component.initialize({})
        
        try:
            component.execute()
        except ValueError:
            pass
        
        error = component.get_error()
        assert error is not None
        assert isinstance(error, ValueError)
        assert "Test error" in str(error)
    
    def test_component_decorator(self):
        """测试组件装饰器"""
        @component("decorated_component")
        class DecoratedComponent(BaseComponent):
            def _do_execute(self, *args, **kwargs):
                return "decorated"
        
        assert hasattr(DecoratedComponent, '_component_name')
        assert DecoratedComponent._component_name == "decorated_component"


class TestComponentFactory:
    """ComponentFactory测试套件"""
    
    def test_factory_creation(self):
        """测试工厂创建"""
        factory = ComponentFactory()
        assert factory._components == {}
    
    def test_create_component_success(self):
        """测试创建组件成功"""
        factory = ComponentFactory()
        config = {'init_success': True}
        
        component = factory.create_component(
            "test_component",
            TestComponent,
            config
        )
        
        assert component is not None
        assert isinstance(component, TestComponent)
        assert component.name == "test_component"
        assert component.is_initialized()
    
    def test_create_component_failure(self):
        """测试创建组件失败"""
        factory = ComponentFactory()
        config = {'init_success': False}
        
        component = factory.create_component(
            "test_component",
            TestComponent,
            config
        )
        
        assert component is None
    
    def test_get_component(self):
        """测试获取组件"""
        factory = ComponentFactory()
        factory.create_component("test_comp", TestComponent, {})
        
        component = factory.get_component("test_comp")
        assert component is not None
        assert component.name == "test_comp"
        
        non_existent = factory.get_component("non_existent")
        assert non_existent is None
    
    def test_remove_component(self):
        """测试移除组件"""
        factory = ComponentFactory()
        factory.create_component("test_comp", TestComponent, {})
        
        result = factory.remove_component("test_comp")
        assert result is True
        assert factory.get_component("test_comp") is None
        
        result = factory.remove_component("non_existent")
        assert result is False
    
    def test_get_all_components(self):
        """测试获取所有组件"""
        factory = ComponentFactory()
        factory.create_component("comp1", TestComponent, {})
        factory.create_component("comp2", TestComponent, {})
        
        all_components = factory.get_all_components()
        assert len(all_components) == 2
        assert "comp1" in all_components
        assert "comp2" in all_components
    
    def test_clear_components(self):
        """测试清空所有组件"""
        factory = ComponentFactory()
        factory.create_component("comp1", TestComponent, {})
        factory.create_component("comp2", TestComponent, {})
        
        factory.clear()
        assert len(factory.get_all_components()) == 0


class TestComponentIntegration:
    """组件集成测试"""
    
    def test_component_lifecycle(self):
        """测试组件完整生命周期"""
        # 创建
        component = TestComponent("lifecycle_test")
        assert component._status == ComponentStatus.UNINITIALIZED
        
        # 初始化
        assert component.initialize({'data': 'test'})
        assert component._status == ComponentStatus.INITIALIZED
        
        # 执行
        result = component.execute(return_value='result1')
        assert result == 'result1'
        assert component.execute_count == 1
        
        # 再次执行
        result = component.execute(return_value='result2')
        assert result == 'result2'
        assert component.execute_count == 2
        
        # 重置
        component.reset()
        assert component._status == ComponentStatus.UNINITIALIZED
        assert component.execute_count == 2  # 计数不重置
        
        # 重新初始化
        assert component.initialize({})
        assert component._status == ComponentStatus.INITIALIZED
    
    def test_multiple_components_in_factory(self):
        """测试工厂中管理多个组件"""
        factory = ComponentFactory()
        
        # 创建多个不同配置的组件
        comp1 = factory.create_component("comp1", TestComponent, {'key1': 'value1'})
        comp2 = factory.create_component("comp2", TestComponent, {'key2': 'value2'})
        
        assert comp1 is not None
        assert comp2 is not None
        assert comp1.config['key1'] == 'value1'
        assert comp2.config['key2'] == 'value2'
        
        # 执行不同组件
        result1 = comp1.execute(return_value='r1')
        result2 = comp2.execute(return_value='r2')
        
        assert result1 == 'r1'
        assert result2 == 'r2'
        assert comp1.execute_count == 1
        assert comp2.execute_count == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

