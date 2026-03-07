"""
测试HTTP相关常量定义

覆盖 HTTPConstants 类的所有常量值
"""

import pytest
from src.infrastructure.constants.http_constants import HTTPConstants


class TestHTTPConstants:
    """HTTPConstants 单元测试"""

    def test_success_status_codes(self):
        """测试成功状态码"""
        assert HTTPConstants.OK == 200
        assert HTTPConstants.CREATED == 201
        assert HTTPConstants.ACCEPTED == 202
        assert HTTPConstants.NO_CONTENT == 204

    def test_redirection_status_codes(self):
        """测试重定向状态码"""
        assert HTTPConstants.MOVED_PERMANENTLY == 301
        assert HTTPConstants.FOUND == 302
        assert HTTPConstants.NOT_MODIFIED == 304

    def test_client_error_status_codes(self):
        """测试客户端错误状态码"""
        assert HTTPConstants.BAD_REQUEST == 400
        assert HTTPConstants.UNAUTHORIZED == 401
        assert HTTPConstants.FORBIDDEN == 403
        assert HTTPConstants.NOT_FOUND == 404
        assert HTTPConstants.METHOD_NOT_ALLOWED == 405
        assert HTTPConstants.CONFLICT == 409
        assert HTTPConstants.UNPROCESSABLE_ENTITY == 422
        assert HTTPConstants.TOO_MANY_REQUESTS == 429

    def test_server_error_status_codes(self):
        """测试服务器错误状态码"""
        assert HTTPConstants.INTERNAL_SERVER_ERROR == 500
        assert HTTPConstants.NOT_IMPLEMENTED == 501
        assert HTTPConstants.BAD_GATEWAY == 502
        assert HTTPConstants.SERVICE_UNAVAILABLE == 503
        assert HTTPConstants.GATEWAY_TIMEOUT == 504

    def test_http_methods(self):
        """测试HTTP方法"""
        assert HTTPConstants.METHOD_GET == 'GET'
        assert HTTPConstants.METHOD_POST == 'POST'
        assert HTTPConstants.METHOD_PUT == 'PUT'
        assert HTTPConstants.METHOD_DELETE == 'DELETE'
        assert HTTPConstants.METHOD_PATCH == 'PATCH'
        assert HTTPConstants.METHOD_HEAD == 'HEAD'
        assert HTTPConstants.METHOD_OPTIONS == 'OPTIONS'

    def test_content_types(self):
        """测试内容类型"""
        assert HTTPConstants.CONTENT_TYPE_JSON == 'application/json'
        assert HTTPConstants.CONTENT_TYPE_XML == 'application/xml'
        assert HTTPConstants.CONTENT_TYPE_FORM == 'application/x-www-form-urlencoded'
        assert HTTPConstants.CONTENT_TYPE_MULTIPART == 'multipart/form-data'
        assert HTTPConstants.CONTENT_TYPE_TEXT == 'text/plain'
        assert HTTPConstants.CONTENT_TYPE_HTML == 'text/html'

    def test_default_ports(self):
        """测试默认端口"""
        assert HTTPConstants.DEFAULT_HTTP_PORT == 80
        assert HTTPConstants.DEFAULT_HTTPS_PORT == 443
        assert HTTPConstants.DEFAULT_API_PORT == 5000
        assert HTTPConstants.DEFAULT_ADMIN_PORT == 8080

    def test_status_code_ranges(self):
        """测试状态码范围"""
        # 2xx 成功
        assert 200 <= HTTPConstants.OK <= 299
        assert 200 <= HTTPConstants.CREATED <= 299
        assert 200 <= HTTPConstants.ACCEPTED <= 299
        assert 200 <= HTTPConstants.NO_CONTENT <= 299

        # 3xx 重定向
        assert 300 <= HTTPConstants.MOVED_PERMANENTLY <= 399
        assert 300 <= HTTPConstants.FOUND <= 399
        assert 300 <= HTTPConstants.NOT_MODIFIED <= 399

        # 4xx 客户端错误
        assert 400 <= HTTPConstants.BAD_REQUEST <= 499
        assert 400 <= HTTPConstants.UNAUTHORIZED <= 499
        assert 400 <= HTTPConstants.FORBIDDEN <= 499
        assert 400 <= HTTPConstants.NOT_FOUND <= 499
        assert 400 <= HTTPConstants.METHOD_NOT_ALLOWED <= 499
        assert 400 <= HTTPConstants.CONFLICT <= 499
        assert 400 <= HTTPConstants.UNPROCESSABLE_ENTITY <= 499
        assert 400 <= HTTPConstants.TOO_MANY_REQUESTS <= 499

        # 5xx 服务器错误
        assert 500 <= HTTPConstants.INTERNAL_SERVER_ERROR <= 599
        assert 500 <= HTTPConstants.NOT_IMPLEMENTED <= 599
        assert 500 <= HTTPConstants.BAD_GATEWAY <= 599
        assert 500 <= HTTPConstants.SERVICE_UNAVAILABLE <= 599
        assert 500 <= HTTPConstants.GATEWAY_TIMEOUT <= 599

    def test_port_ranges(self):
        """测试端口范围"""
        # HTTP端口
        assert 1 <= HTTPConstants.DEFAULT_HTTP_PORT <= 65535
        assert 1 <= HTTPConstants.DEFAULT_HTTPS_PORT <= 65535

        # 应用端口
        assert 1 <= HTTPConstants.DEFAULT_API_PORT <= 65535
        assert 1 <= HTTPConstants.DEFAULT_ADMIN_PORT <= 65535

        # HTTPS端口应该大于HTTP端口
        assert HTTPConstants.DEFAULT_HTTPS_PORT > HTTPConstants.DEFAULT_HTTP_PORT

        # 应用端口应该大于标准端口
        assert HTTPConstants.DEFAULT_API_PORT > HTTPConstants.DEFAULT_HTTPS_PORT
        assert HTTPConstants.DEFAULT_ADMIN_PORT > HTTPConstants.DEFAULT_HTTPS_PORT

    def test_http_method_strings(self):
        """测试HTTP方法字符串"""
        methods = [
            HTTPConstants.METHOD_GET,
            HTTPConstants.METHOD_POST,
            HTTPConstants.METHOD_PUT,
            HTTPConstants.METHOD_DELETE,
            HTTPConstants.METHOD_PATCH,
            HTTPConstants.METHOD_HEAD,
            HTTPConstants.METHOD_OPTIONS
        ]

        for method in methods:
            assert isinstance(method, str)
            assert len(method) > 0
            assert method.isupper()  # HTTP方法通常是大写

    def test_content_type_strings(self):
        """测试内容类型字符串"""
        content_types = [
            HTTPConstants.CONTENT_TYPE_JSON,
            HTTPConstants.CONTENT_TYPE_XML,
            HTTPConstants.CONTENT_TYPE_FORM,
            HTTPConstants.CONTENT_TYPE_MULTIPART,
            HTTPConstants.CONTENT_TYPE_TEXT,
            HTTPConstants.CONTENT_TYPE_HTML
        ]

        for content_type in content_types:
            assert isinstance(content_type, str)
            assert len(content_type) > 0
            assert '/' in content_type  # MIME类型应该包含斜杠

    def test_common_status_codes(self):
        """测试常见状态码"""
        # 验证最常用的状态码
        assert HTTPConstants.OK == 200  # 最常用的成功状态码
        assert HTTPConstants.NOT_FOUND == 404  # 最常用的客户端错误
        assert HTTPConstants.INTERNAL_SERVER_ERROR == 500  # 最常用的服务器错误

    def test_standard_http_methods(self):
        """测试标准HTTP方法"""
        # 验证RESTful API常用的方法
        assert HTTPConstants.METHOD_GET == 'GET'
        assert HTTPConstants.METHOD_POST == 'POST'
        assert HTTPConstants.METHOD_PUT == 'PUT'
        assert HTTPConstants.METHOD_DELETE == 'DELETE'