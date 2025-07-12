import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.config.deployment_manager import DeploymentManager

@pytest.fixture
def mock_config_manager():
    """创建mock的ConfigManager"""
    mock_cm = MagicMock()
    mock_cm.get_config.return_value = {
        'deployment': {
            'environments': {
                'dev': 'config/deploy_dev.yaml',
                'test': 'config/deploy_test.yaml',
                'prod': 'config/deploy_prod.yaml'
            }
        }
    }
    return mock_cm

@pytest.fixture
def deployment_manager(mock_config_manager):
    """创建DeploymentManager实例"""
    config = {'test': 'config'}
    return DeploymentManager(config, config_manager=mock_config_manager)

@pytest.fixture
def sample_deploy_config():
    """示例部署配置"""
    return {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'user': 'postgres',
            'password': 'password'
        },
        'trading': {
            'max_order_size': 10000,
            'default_slippage': 0.001
        },
        'monitoring': {
            'enabled': True,
            'interval': 60
        },
        'security': {
            'level': 'high'
        }
    }

class TestDeploymentManager:
    """DeploymentManager测试类"""

    def test_init_with_test_hook(self, mock_config_manager):
        """测试使用测试钩子初始化"""
        config = {'test': 'config'}
        manager = DeploymentManager(config, config_manager=mock_config_manager)
        
        assert manager.config_manager == mock_config_manager
        assert manager.current_env is None
        assert manager.current_config is None

    def test_init_without_test_hook(self):
        """测试不使用测试钩子初始化"""
        config = {'test': 'config'}
        with patch('src.infrastructure.config.deployment_manager.ConfigManager') as mock_cm_class:
            mock_cm_instance = MagicMock()
            mock_cm_class.return_value = mock_cm_instance
            
            manager = DeploymentManager(config)
            
            assert manager.config_manager == mock_cm_instance
            mock_cm_class.assert_called_once_with(config)

    def test_load_environment_success(self, deployment_manager, sample_deploy_config):
        """测试加载环境配置 - 成功"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(sample_deploy_config, f)
            config_file = f.name
        
        try:
            # 临时修改环境配置路径
            deployment_manager.environments['test'] = config_file
            
            result = deployment_manager.load_environment('test')
            
            assert result is True
            assert deployment_manager.current_env == 'test'
            assert deployment_manager.current_config == sample_deploy_config
        finally:
            os.unlink(config_file)

    def test_load_environment_file_not_found(self, deployment_manager):
        """测试加载环境配置 - 文件不存在"""
        result = deployment_manager.load_environment('dev')
        
        assert result is False
        assert deployment_manager.current_env is None
        assert deployment_manager.current_config is None

    def test_load_environment_unknown_env(self, deployment_manager):
        """测试加载环境配置 - 未知环境"""
        result = deployment_manager.load_environment('unknown')
        
        assert result is False

    def test_get_deployment_config_success(self, deployment_manager, sample_deploy_config):
        """测试获取部署配置 - 成功"""
        deployment_manager.current_config = sample_deploy_config
        
        # 测试获取嵌套配置
        db_host = deployment_manager.get_deployment_config('database.host')
        assert db_host == 'localhost'
        
        # 测试获取交易配置
        max_order_size = deployment_manager.get_deployment_config('trading.max_order_size')
        assert max_order_size == 10000

    def test_get_deployment_config_no_config_loaded(self, deployment_manager):
        """测试获取部署配置 - 未加载配置"""
        value = deployment_manager.get_deployment_config('database.host', 'default')
        assert value == 'default'

    def test_get_deployment_config_key_not_found(self, deployment_manager, sample_deploy_config):
        """测试获取部署配置 - 键不存在"""
        deployment_manager.current_config = sample_deploy_config
        
        value = deployment_manager.get_deployment_config('unknown.key', 'default')
        assert value == 'default'

    def test_validate_deployment_success(self, deployment_manager, sample_deploy_config):
        """测试验证部署配置 - 成功"""
        deployment_manager.current_config = sample_deploy_config
        
        result = deployment_manager.validate_deployment()
        
        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_deployment_no_config(self, deployment_manager):
        """测试验证部署配置 - 无配置"""
        result = deployment_manager.validate_deployment()
        
        assert result['valid'] is False
        assert '未加载部署配置' in result['errors']

    def test_validate_deployment_missing_sections(self, deployment_manager):
        """测试验证部署配置 - 缺少配置段"""
        incomplete_config = {
            'database': {'host': 'localhost'},
            # 缺少 trading, monitoring, security
        }
        deployment_manager.current_config = incomplete_config
        
        result = deployment_manager.validate_deployment()
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('缺少必要配置段' in error for error in result['errors'])

    def test_validate_deployment_missing_db_params(self, deployment_manager):
        """测试验证部署配置 - 缺少数据库参数"""
        incomplete_config = {
            'database': {},  # 缺少 host 和 port
            'trading': {'max_order_size': 10000, 'default_slippage': 0.001},
            'monitoring': {'enabled': True},
            'security': {'level': 'high'}
        }
        deployment_manager.current_config = incomplete_config
        
        result = deployment_manager.validate_deployment()
        
        assert result['valid'] is False
        assert any('数据库配置缺少host参数' in error for error in result['errors'])
        assert any('数据库配置缺少port参数' in error for error in result['errors'])

    def test_validate_deployment_missing_trading_params(self, deployment_manager):
        """测试验证部署配置 - 缺少交易参数"""
        incomplete_config = {
            'database': {'host': 'localhost', 'port': 5432},
            'trading': {},  # 缺少 max_order_size 和 default_slippage
            'monitoring': {'enabled': True},
            'security': {'level': 'high'}
        }
        deployment_manager.current_config = incomplete_config
        
        result = deployment_manager.validate_deployment()
        
        assert result['valid'] is False
        assert any('交易配置缺少max_order_size参数' in error for error in result['errors'])
        assert any('交易配置缺少default_slippage参数' in error for error in result['errors'])

    def test_generate_deployment_script_success(self, deployment_manager, sample_deploy_config):
        """测试生成部署脚本 - 成功"""
        deployment_manager.current_config = sample_deploy_config
        deployment_manager.current_env = 'prod'
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            output_path = f.name
        
        try:
            result = deployment_manager.generate_deployment_script(output_path)
            
            assert result is True
            
            # 验证生成的脚本内容
            with open(output_path, 'r') as f:
                script_content = f.read()
                assert 'DB_HOST=localhost' in script_content
                assert 'DB_PORT=5432' in script_content
                assert 'MAX_ORDER_SIZE=10000' in script_content
        finally:
            os.unlink(output_path)

    def test_generate_deployment_script_no_config(self, deployment_manager):
        """测试生成部署脚本 - 无配置"""
        result = deployment_manager.generate_deployment_script('/tmp/test.sh')
        
        assert result is False

    def test_get_deployment_template_prod(self, deployment_manager):
        """测试获取部署模板 - 生产环境"""
        deployment_manager.current_env = 'prod'
        
        template = deployment_manager._get_deployment_template()
        
        assert '生产环境部署脚本' in template
        assert '--env=prod' in template

    def test_get_deployment_template_dev(self, deployment_manager):
        """测试获取部署模板 - 开发环境"""
        deployment_manager.current_env = 'dev'
        
        template = deployment_manager._get_deployment_template()
        
        assert '非生产环境部署脚本' in template
        assert '--env=dev' in template

    def test_fill_template_with_config(self, deployment_manager, sample_deploy_config):
        """测试填充模板 - 有配置"""
        deployment_manager.current_config = sample_deploy_config
        deployment_manager.current_env = 'test'
        
        template = "DB_HOST={db_host}, DB_PORT={db_port}, ENV={env}"
        filled = deployment_manager._fill_template(template)
        
        assert 'DB_HOST=localhost' in filled
        assert 'DB_PORT=5432' in filled
        assert 'ENV=test' in filled

    def test_fill_template_no_config(self, deployment_manager):
        """测试填充模板 - 无配置"""
        deployment_manager.current_config = None
        
        template = "DB_HOST={db_host}, DB_PORT={db_port}"
        filled = deployment_manager._fill_template(template)
        
        # 应该返回原始模板
        assert filled == template

    def test_get_environment_summary_with_config(self, deployment_manager, sample_deploy_config):
        """测试获取环境摘要 - 有配置"""
        deployment_manager.current_config = sample_deploy_config
        deployment_manager.current_env = 'prod'
        
        summary = deployment_manager.get_environment_summary()
        
        assert summary['environment'] == 'prod'
        assert summary['database']['host'] == 'localhost'
        assert summary['database']['port'] == 5432
        assert summary['trading']['max_order_size'] == 10000
        assert summary['security']['level'] == 'high'

    def test_get_environment_summary_no_config(self, deployment_manager):
        """测试获取环境摘要 - 无配置"""
        summary = deployment_manager.get_environment_summary()
        
        assert summary == {}

    def test_switch_environment_same_env(self, deployment_manager):
        """测试切换环境 - 相同环境"""
        deployment_manager.current_env = 'test'
        
        result = deployment_manager.switch_environment('test')
        
        assert result is True

    def test_switch_environment_success(self, deployment_manager, sample_deploy_config):
        """测试切换环境 - 成功"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(sample_deploy_config, f)
            config_file = f.name
        
        try:
            deployment_manager.environments['test'] = config_file
            
            result = deployment_manager.switch_environment('test')
            
            assert result is True
            assert deployment_manager.current_env == 'test'
        finally:
            os.unlink(config_file)

    def test_switch_environment_failure(self, deployment_manager):
        """测试切换环境 - 失败"""
        result = deployment_manager.switch_environment('unknown')
        
        assert result is False

    def test_load_config_from_manager(self, deployment_manager, mock_config_manager):
        """测试从配置管理器加载配置"""
        # 重新初始化配置管理器
        deployment_manager.config_manager = mock_config_manager
        
        # 验证配置管理器被正确设置
        assert deployment_manager.config_manager == mock_config_manager 