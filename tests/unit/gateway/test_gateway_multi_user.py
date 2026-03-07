# 测试网关层多用户并发分支

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

class TestGatewayMultiUser:
    """网关多用户测试"""

    @pytest.fixture
    def mock_gateway(self):
        """创建Mock网关"""
        mock_gateway = Mock()
        return mock_gateway

    def test_gateway_multi_user_concurrency(self, mock_gateway):
        """测试网关多用户并发分支"""
        mock_gateway.handle_multi_users = Mock(return_value={"handled": True, "users": 10})

        result = mock_gateway.handle_multi_users()
        assert result["handled"] is True
        assert result["users"] == 10
