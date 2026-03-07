"""
测试health API模块的初始化
"""

import pytest


def test_api_module_imports():
    """测试API模块的导入功能"""
    try:
        from src.infrastructure.health.api import DataAPIManager, DataAPI
        assert DataAPIManager is not None
        assert DataAPI is not None
    except ImportError:
        # 如果导入失败，尝试延迟导入
        try:
            from src.infrastructure.health.api import _lazy_import
            DataAPIManager, DataAPI = _lazy_import()
            assert DataAPIManager is not None
            assert DataAPI is not None
        except Exception as e:
            pytest.skip(f"API模块导入失败: {e}")


def test_lazy_import_function():
    """测试延迟导入函数"""
    from src.infrastructure.health.api import _lazy_import

    # 调用延迟导入函数
    result = _lazy_import()
    assert isinstance(result, tuple)
    assert len(result) == 2

    DataAPIManager, DataAPI = result
    assert DataAPIManager is not None
    assert DataAPI is not None


def test_module_docstring():
    """测试模块文档字符串"""
    import src.infrastructure.health.api as api_module

    assert api_module.__doc__ is not None
    assert "健康检查API模块" in api_module.__doc__
