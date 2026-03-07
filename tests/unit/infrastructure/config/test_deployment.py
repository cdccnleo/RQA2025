from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, mock_open
from src.infrastructure.config.tools.deployment import DeploymentManager
import yaml

class TestDeploymentManager:
    """测试部署管理器"""

    @pytest.fixture
    def mock_manager(self):
        """创建mock DeploymentManager实例"""
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data="data: yaml")) as mock_file, \
             patch('yaml.safe_load') as mock_yaml:
            manager = DeploymentManager(config={})
            return manager

    def test_load_environment_success(self, mock_manager):
        """测试成功加载环境 (覆盖行75-95)"""
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data="data: yaml")) as mock_file, \
             patch('yaml.safe_load') as mock_yaml:
            mock_exists.return_value = True
            mock_yaml.return_value = {'test': 'config'}
            result = mock_manager.load_environment('dev')
            assert result is True
            assert mock_manager.current_env == 'dev'

    def test_get_deployment_config(self, mock_manager):
        """测试获取部署配置 (覆盖行97-114)"""
        mock_manager.current_config = {'section': {'key': 'value'}}
        result = mock_manager.get_deployment_config('section.key')
        assert result == 'value'

    def test_validate_deployment(self, mock_manager):
        """测试验证部署 (覆盖行116-153)"""
        mock_manager.current_config = {
            'database': {'host': 'localhost', 'port': 5432},
            'trading': {'max_order_size': 100, 'default_slippage': 0.01},
            'monitoring': {},
            'security': {}
        }
        result = mock_manager.validate_deployment()
        assert result['valid'] is True

    def test_generate_deployment_script(self, mock_manager):
        """测试生成部署脚本 (覆盖行155-230)"""
        mock_manager.current_config = {
            'database': {'host': 'db_host', 'port': 5432, 'user': 'user', 'password': 'pass'},
            'trading': {'max_order_size': 10000, 'default_slippage': 0.001}
        }
        mock_manager.current_env = 'prod'
        result = mock_manager.generate_deployment_script('deploy.sh')
        assert result is True

    # 添加更多测试...
