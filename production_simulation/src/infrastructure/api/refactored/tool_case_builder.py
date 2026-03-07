"""测试用例构建器基类"""

from typing import List
from ..test_generation.models import TestCase, TestScenario
from .template_manager import TestTemplateManager
# 假设这些类在原模块中定义
# from ..api_test_case_generator import TestCase, TestScenario


class TestCaseBuilder:
    """测试用例构建基类"""
    
    def __init__(self, template_manager=None):
        self.template_manager = template_manager or TestTemplateManager()
    
    def create_test_case(self, config):
        """创建单个测试用例"""
        # TODO: 实现测试用例创建逻辑
        pass

    def create_scenario(self, config):
        """创建测试场景"""
        # TODO: 实现测试场景创建逻辑
        pass

    def create_authentication_tests(self):
        """创建认证测试"""
        pass

    def create_validation_tests(self):
        """创建验证测试"""
        pass

    def create_security_tests(self):
        """创建安全测试"""
        pass
