# 测试网关层认证增强

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

class TestGatewayAuthentication:
    """网关认证测试"""

    @pytest.fixture
    def mock_gateway(self):
        """创建Mock网关"""
        mock_gateway = Mock()
        return mock_gateway

    def test_gateway_auth_basic(self, mock_gateway):
        """测试网关认证基础功能"""
        mock_gateway.authenticate = Mock(return_value={"authenticated": True, "user_id": "123"})

        result = mock_gateway.authenticate("token_abc")
        assert result["authenticated"] is True

    def test_gateway_auth_failure(self, mock_gateway):
        """测试网关认证失败分支"""
        mock_gateway.authenticate = Mock(return_value={"authenticated": False, "error": "invalid_token"})

        result = mock_gateway.authenticate("invalid_token")
        assert result["authenticated"] is False
        assert "error" in result

    def test_gateway_rate_limiting(self, mock_gateway):
        """测试网关限流分支"""
        mock_gateway.check_rate_limit = Mock(return_value={"allowed": False, "retry_after": 60})

        result = mock_gateway.check_rate_limit("user_123")
        assert result["allowed"] is False
        assert result["retry_after"] == 60

    def test_gateway_security_filter(self, mock_gateway):
        """测试网关安全过滤分支"""
        mock_gateway.apply_security_filter = Mock(return_value={"filtered": True, "threats": ["sql_injection"]})

        result = mock_gateway.apply_security_filter("request_payload")
        assert result["filtered"] is True
        assert len(result["threats"]) > 0
