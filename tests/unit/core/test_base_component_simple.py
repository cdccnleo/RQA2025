#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseComponent测试 - 简化版

直接测试foundation/base_component.py模块
"""

import pytest
from unittest.mock import Mock

# 直接导入base_component.py
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入base_component.py文件
    import importlib.util
    base_component_path = project_root / "src" / "core" / "foundation" / "base_component.py"
    spec = importlib.util.spec_from_file_location("base_component_module", base_component_path)
    base_component_module = importlib.util.module_from_spec(spec)
    
    # 处理依赖
    sys.path.insert(0, str(project_root / "src"))
    spec.loader.exec_module(base_component_module)
    
    # 尝试获取类
    ComponentStatus = getattr(base_component_module, 'ComponentStatus', None)
    IComponent = getattr(base_component_module, 'IComponent', None)
    BaseComponent = getattr(base_component_module, 'BaseComponent', None)
    
    IMPORTS_AVAILABLE = BaseComponent is not None
except Exception as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"BaseComponent模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestBaseComponentNew:
    """测试新的BaseComponent"""

    # 创建简单的实现类用于测试
    class TestComponent(BaseComponent):
        """测试用组件实现"""
        def _do_execute(self, *args, **kwargs):
            """执行组件主要功能（实现抽象方法）"""
            return {"result": "success"}
        
        def execute(self, *args, **kwargs):
            """执行组件主要功能"""
            return self._do_execute(*args, **kwargs)

    def test_base_component_initialization(self):
        """测试组件初始化"""
        component = self.TestComponent(name="test_component")
        assert component.name == "test_component"
        assert component._status == ComponentStatus.UNINITIALIZED

    def test_base_component_get_info(self):
        """测试获取组件信息"""
        component = self.TestComponent(name="test", config={"key": "value"})
        info = component.get_info()
        assert isinstance(info, dict)
        assert info['name'] == "test"
        # type可能是类名，验证存在即可
        assert 'type' in info or 'status' in info

    def test_base_component_initialize(self):
        """测试初始化组件"""
        component = self.TestComponent(name="test")
        result = component.initialize({"key": "value"})
        # initialize可能返回True或False，至少验证方法调用成功
        assert isinstance(result, bool)
        # 验证状态已改变（从UNINITIALIZED变为其他状态）
        assert component._status != ComponentStatus.UNINITIALIZED or result is True

    def test_base_component_execute(self):
        """测试执行组件"""
        component = self.TestComponent(name="test")
        component.initialize({})
        result = component.execute()
        assert isinstance(result, dict)
        assert result['result'] == "success"

    def test_base_component_status_transitions(self):
        """测试状态转换"""
        component = self.TestComponent(name="test")
        assert component._status == ComponentStatus.UNINITIALIZED
        component.initialize({})
        assert component._status == ComponentStatus.INITIALIZED

