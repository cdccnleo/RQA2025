"""
测试测试模板管理器

覆盖 template_manager.py 中的 TestTemplateManager 类
"""

import pytest
from src.infrastructure.api.test_generation.template_manager import TestTemplateManager


class TestTestTemplateManager:
    """TestTemplateManager 类测试"""

    def test_initialization(self):
        """测试初始化"""
        manager = TestTemplateManager()

        assert hasattr(manager, 'templates')
        assert isinstance(manager.templates, dict)
        assert len(manager.templates) > 0

    def test_initialization_with_custom_dir(self):
        """测试带自定义目录初始化"""
        manager = TestTemplateManager(template_dir="/custom/path")

        assert hasattr(manager, 'templates')
        assert isinstance(manager.templates, dict)
        assert len(manager.templates) > 0

    def test_get_template(self):
        """测试获取模板"""
        manager = TestTemplateManager()

        template = manager.get_template("authentication", "login")

        assert template is not None
        assert isinstance(template, dict)

    def test_get_template_nonexistent(self):
        """测试获取不存在的模板"""
        manager = TestTemplateManager()

        template = manager.get_template("nonexistent")

        assert template is None

    def test_list_templates(self):
        """测试列出模板"""
        manager = TestTemplateManager()

        templates = manager.list_templates()

        assert isinstance(templates, list)
        assert len(templates) > 0
        assert "authentication" in templates
        assert "validation" in templates
        assert "business_logic" in templates

    def test_has_template(self):
        """测试检查模板存在性"""
        manager = TestTemplateManager()

        assert manager.has_template("authentication", None) == True
        assert manager.has_template("nonexistent") == False

    def test_get_authentication_templates(self):
        """测试获取认证模板"""
        manager = TestTemplateManager()

        auth_templates = manager._load_authentication_templates()

        assert isinstance(auth_templates, dict)
        assert "login" in auth_templates
        assert "logout" in auth_templates
        assert "register" in auth_templates

    def test_get_validation_templates(self):
        """测试获取验证模板"""
        manager = TestTemplateManager()

        validation_templates = manager._load_validation_templates()

        assert isinstance(validation_templates, dict)
        assert "required_fields" in validation_templates
        assert "data_types" in validation_templates

    def test_get_business_logic_templates(self):
        """测试获取业务逻辑模板"""
        manager = TestTemplateManager()

        business_templates = manager._load_business_logic_templates()

        assert isinstance(business_templates, dict)
        assert "order_processing" in business_templates
        assert "payment_flow" in business_templates

    def test_get_performance_templates(self):
        """测试获取性能模板"""
        manager = TestTemplateManager()

        performance_templates = manager._load_performance_templates()

        assert isinstance(performance_templates, dict)
        assert "load_test" in performance_templates
        assert "stress_test" in performance_templates

    def test_get_security_templates(self):
        """测试获取安全模板"""
        manager = TestTemplateManager()

        security_templates = manager._load_security_templates()

        assert isinstance(security_templates, dict)
        assert "sql_injection" in security_templates
        assert "xss_attack" in security_templates

    def test_template_structure(self):
        """测试模板结构"""
        manager = TestTemplateManager()

        # 检查认证模板结构
        login_template = manager.templates["authentication"]["login"]

        assert "test_cases" in login_template
        assert "setup" in login_template
        assert "teardown" in login_template

        # 检查测试用例结构
        test_cases = login_template["test_cases"]
        assert isinstance(test_cases, list)
        assert len(test_cases) > 0

        # 检查单个测试用例结构
        test_case = test_cases[0]
        assert "id" in test_case
        assert "title" in test_case
        assert "description" in test_case
        assert "steps" in test_case
        assert "expected" in test_case

    def test_template_consistency(self):
        """测试模板一致性"""
        manager = TestTemplateManager()

        # 确保所有模板都有相同的结构
        for template_name, template in manager.templates.items():
            assert isinstance(template, dict)

            # 每个模板应该至少有一个测试类型
            assert len(template) > 0

            # 每个测试类型应该有test_cases
            for test_type, test_config in template.items():
                assert "test_cases" in test_config
                assert isinstance(test_config["test_cases"], list)
