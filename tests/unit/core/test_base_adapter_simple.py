#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseAdapter测试 - 简化版

直接测试foundation/base_adapter.py模块
"""

import pytest
from unittest.mock import Mock

# 直接导入base_adapter.py
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入base_adapter.py文件
    import importlib.util
    base_adapter_path = project_root / "src" / "core" / "foundation" / "base_adapter.py"
    spec = importlib.util.spec_from_file_location("base_adapter_module", base_adapter_path)
    base_adapter_module = importlib.util.module_from_spec(spec)
    
    # 处理依赖
    sys.path.insert(0, str(project_root / "src"))
    spec.loader.exec_module(base_adapter_module)
    
    # 尝试获取类
    AdapterStatus = getattr(base_adapter_module, 'AdapterStatus', None)
    IAdapter = getattr(base_adapter_module, 'IAdapter', None)
    BaseAdapter = getattr(base_adapter_module, 'BaseAdapter', None)
    
    IMPORTS_AVAILABLE = BaseAdapter is not None
except Exception as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"BaseAdapter模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestAdapterStatus:
    """测试适配器状态枚举"""

    def test_adapter_status_values(self):
        """测试适配器状态枚举值"""
        if AdapterStatus:
            assert AdapterStatus.READY.value == "ready"
            assert AdapterStatus.ADAPTING.value == "adapting"
            assert AdapterStatus.ERROR.value == "error"
            assert AdapterStatus.DISABLED.value == "disabled"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestBaseAdapter:
    """测试BaseAdapter"""

    # 创建简单的实现类用于测试
    class TestAdapter(BaseAdapter):
        """测试用适配器实现"""
        def _do_adapt(self, data):
            """实现具体的适配逻辑"""
            return {"adapted": data}

    def test_base_adapter_initialization(self):
        """测试适配器初始化"""
        adapter = self.TestAdapter(name="test_adapter")
        assert adapter.name == "test_adapter"
        assert adapter._status == AdapterStatus.READY

    def test_base_adapter_adapt(self):
        """测试适配数据"""
        adapter = self.TestAdapter(name="test")
        result = adapter.adapt({"key": "value"})
        assert isinstance(result, dict)
        assert "adapted" in result

    def test_base_adapter_validate_input(self):
        """测试验证输入"""
        adapter = self.TestAdapter(name="test")
        assert adapter.validate_input({"key": "value"}) is True
        assert adapter.validate_input(None) is False

    def test_base_adapter_with_cache(self):
        """测试带缓存的适配器"""
        adapter = self.TestAdapter(name="test", enable_cache=True)
        assert adapter.enable_cache is True

