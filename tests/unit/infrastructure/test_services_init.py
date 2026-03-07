#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试services_init模块

测试目标：提升services_init.py的覆盖率到100%
"""

import pytest

from src.infrastructure.services_init import __version__, __author__, __description__, __all__


class TestServicesInit:
    """测试services_init模块的元信息"""

    def test_version_info(self):
        """测试版本信息"""
        assert __version__ == "1.0.0"
        assert isinstance(__version__, str)

    def test_author_info(self):
        """测试作者信息"""
        assert __author__ == "RQA Team"
        assert isinstance(__author__, str)

    def test_description_info(self):
        """测试描述信息"""
        assert __description__ == "RQA服务层模块，提供完整的业务服务接口"
        assert isinstance(__description__, str)
        assert "服务层" in __description__
        assert "业务服务接口" in __description__

    def test_all_exports(self):
        """测试__all__导出列表"""
        assert isinstance(__all__, list)
        assert len(__all__) == 0  # 当前为空列表

    def test_module_attributes(self):
        """测试模块属性存在性"""
        # 验证所有预期的模块属性都存在
        import src.infrastructure.services_init as services_init

        assert hasattr(services_init, '__version__')
        assert hasattr(services_init, '__author__')
        assert hasattr(services_init, '__description__')
        assert hasattr(services_init, '__all__')

    def test_docstring_presence(self):
        """测试模块文档字符串"""
        import src.infrastructure.services_init as services_init

        assert services_init.__doc__ is not None
        assert "服务层统一导出模块" in services_init.__doc__
        assert "业务服务接口" in services_init.__doc__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
