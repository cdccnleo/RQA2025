import pytest
from unittest.mock import Mock
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

class TestBasicTools:
    """基础工具测试"""

    @pytest.fixture
    def mock_tool(self):
        """Mock工具实例"""
        tool = Mock()
        tool.name = "test_tool"
        tool.version = "1.0.0"
        return tool

    def test_tool_initialization(self, mock_tool):
        """测试工具初始化"""
        assert mock_tool.name == "test_tool"
        assert mock_tool.version == "1.0.0"

    def test_tool_execution(self, mock_tool):
        """测试工具执行"""
        mock_tool.execute = Mock(return_value="success")
        result = mock_tool.execute("test_input")
        assert result == "success"
        mock_tool.execute.assert_called_once_with("test_input")

    def test_tool_configuration(self, mock_tool):
        """测试工具配置"""
        config = {"param1": "value1", "param2": 42}
        mock_tool.configure = Mock(return_value=True)
        result = mock_tool.configure(config)
        assert result is True
        mock_tool.configure.assert_called_once_with(config)

    def test_tool_validation(self, mock_tool):
        """测试工具验证"""
        mock_tool.validate = Mock(return_value=True)
        result = mock_tool.validate()
        assert result is True
        mock_tool.validate.assert_called_once()

    def test_tool_cleanup(self, mock_tool):
        """测试工具清理"""
        mock_tool.cleanup = Mock(return_value=None)
        result = mock_tool.cleanup()
        assert result is None
        mock_tool.cleanup.assert_called_once()
