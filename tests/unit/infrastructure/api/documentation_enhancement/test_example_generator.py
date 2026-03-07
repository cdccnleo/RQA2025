"""
测试示例生成器

覆盖 example_generator.py 中的所有类和功能
"""

import pytest
from unittest.mock import Mock
from src.infrastructure.api.documentation_enhancement.example_generator import ExampleGenerator


class TestExampleGenerator:
    """ExampleGenerator 类测试"""

    def test_initialization(self):
        """测试初始化"""
        generator = ExampleGenerator()

        assert hasattr(generator, 'generate_request_example')
        assert hasattr(generator, 'generate_success_response_example')
        assert hasattr(generator, 'generate_error_response_example')

    def test_generate_request_example_empty_parameters(self):
        """测试生成空参数的请求示例"""
        generator = ExampleGenerator()

        # 创建一个mock endpoint对象
        endpoint = Mock()
        endpoint.parameters = []

        result = generator.generate_request_example(endpoint)

        assert isinstance(result, dict)
        assert result == {}

    def test_generate_request_example_with_parameters(self):
        """测试生成带参数的请求示例"""
        generator = ExampleGenerator()

        # 创建mock参数对象
        param1 = Mock()
        param1.name = "user_id"
        param1.example = "12345"
        param1.default = None

        param2 = Mock()
        param2.name = "limit"
        param2.example = None
        param2.default = 10

        param3 = Mock()
        param3.name = "offset"
        param3.example = None
        param3.default = None

        endpoint = Mock()
        endpoint.parameters = [param1, param2, param3]

        result = generator.generate_request_example(endpoint)

        assert result["user_id"] == "12345"
        assert result["limit"] == 10
        assert "offset" not in result

    def test_generate_success_response_example(self):
        """测试生成成功响应示例"""
        generator = ExampleGenerator()

        endpoint = Mock()
        endpoint.path = "/users"

        result = generator.generate_success_response_example(endpoint)

        assert isinstance(result, dict)
        assert result["success"] == True
        assert result["message"] == "操作成功"
        assert "data" in result
        assert result["timestamp"] == "2025-10-23T22:00:00Z"
        assert result["request_id"] == "req_abc123"

    def test_generate_error_response_example_400(self):
        """测试生成400错误响应示例"""
        generator = ExampleGenerator()

        result = generator.generate_error_response_example(400)

        assert isinstance(result, dict)
        assert result["success"] == False
        assert result["message"] == "请求参数错误"
        assert result["error"]["code"] == "E400"
        assert "details" in result["error"]

    def test_generate_error_response_example_404(self):
        """测试生成404错误响应示例"""
        generator = ExampleGenerator()

        result = generator.generate_error_response_example(404)

        assert isinstance(result, dict)
        assert result["success"] == False
        assert result["message"] == "资源不存在"
        assert result["error"]["code"] == "E404"

    def test_generate_error_response_example_500(self):
        """测试生成500错误响应示例"""
        generator = ExampleGenerator()

        result = generator.generate_error_response_example(500)

        assert isinstance(result, dict)
        assert result["success"] == False
        assert result["message"] == "服务器内部错误"
        assert result["error"]["code"] == "E500"

    def test_generate_error_response_example_unknown_status(self):
        """测试生成未知状态码的错误响应示例"""
        generator = ExampleGenerator()

        result = generator.generate_error_response_example(999)

        assert isinstance(result, dict)
        assert result["success"] == False
        assert result["message"] == "未知错误"
        assert result["error"]["code"] == "E999"


class TestExampleGeneratorIntegration:
    """ExampleGenerator 集成测试"""

    def test_complete_api_workflow(self):
        """测试完整的API工作流"""
        generator = ExampleGenerator()

        # 创建一个mock endpoint
        endpoint = Mock()
        endpoint.parameters = []
        endpoint.path = "/users"

        # 生成请求示例
        request_example = generator.generate_request_example(endpoint)

        # 生成成功响应示例
        success_response = generator.generate_success_response_example(endpoint)

        # 生成错误响应示例
        error_response = generator.generate_error_response_example(400)

        # 验证结果
        assert isinstance(request_example, dict)
        assert success_response["success"] == True
        assert error_response["success"] == False
        assert error_response["message"] == "请求参数错误"

    def test_parameter_priority_example_over_default(self):
        """测试参数优先级：example优先于default"""
        generator = ExampleGenerator()

        param = Mock()
        param.name = "test_param"
        param.example = "example_value"
        param.default = "default_value"

        endpoint = Mock()
        endpoint.parameters = [param]

        result = generator.generate_request_example(endpoint)

        # example应该优先于default
        assert result["test_param"] == "example_value"

    def test_parameter_fallback_to_default(self):
        """测试参数回退到default"""
        generator = ExampleGenerator()

        param = Mock()
        param.name = "test_param"
        param.example = None
        param.default = "default_value"

        endpoint = Mock()
        endpoint.parameters = [param]

        result = generator.generate_request_example(endpoint)

        assert result["test_param"] == "default_value"