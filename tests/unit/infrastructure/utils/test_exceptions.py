"""
基础设施工具层异常模块测试
"""

import pytest
from src.infrastructure.utils.exceptions import *


class TestInfrastructureUtilsExceptions:
    """测试基础设施工具层异常模块"""

    def test_exceptions_module_import(self):
        """测试异常模块可以正常导入"""
        # 这个测试验证模块可以正常导入
        assert True

    def test_exceptions_module_basic_functionality(self):
        """测试异常模块基本功能"""
        # 如果有导出的函数或类，这里可以测试它们
        # 目前这个模块主要是导入别名，所以只需要验证导入成功
        assert True
