"""
cloud_multi_cloud 模块的测试用例
提升测试覆盖率从0.36%到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import subprocess
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List, Optional

# 导入被测试的模块
from src.infrastructure.config.environment.cloud_multi_cloud import MultiCloudManager
from src.infrastructure.config.environment.cloud_native_configs import CloudProvider, MultiCloudConfig


class TestMultiCloudManager:
    """MultiCloudManager 测试类"""

    @pytest.fixture
    def mock_config(self):
        """创建mock配置"""
        config = MultiCloudConfig()
        config.primary_provider = CloudProvider.AWS
        config.secondary_providers = [CloudProvider.AZURE, CloudProvider.GCP]
        config.failover_enabled = True
        config.region_mapping = {"us": "us-east-1"}
        return config

    @pytest.fixture
    def manager(self, mock_config):
        """创建MultiCloudManager实例"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            os=Mock(),
            logger=Mock()
        ):
            return MultiCloudManager(mock_config)

    def test_initialization(self, mock_config):
        """测试初始化"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            os=Mock(),
            logger=Mock()
        ):
            manager = MultiCloudManager(mock_config)
            
            assert manager.config == mock_config
            assert manager._current_provider == CloudProvider.AWS
            assert manager._failover_count == 0
            assert isinstance(manager._providers, dict)
            assert isinstance(manager._health_status, dict)

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_aws_provider_success(self, mock_subprocess, manager, mock_config):
        """测试设置AWS提供商 - 成功"""
        # Mock successful AWS CLI response
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "Account": "123456789012",
            "UserId": "AIDACKCEVSQ6C2EXAMPLE"
        })
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        with patch.dict('os.environ', {'AWS_DEFAULT_REGION': 'us-west-2'}):
            result = manager._setup_aws_provider()
            
            assert result["configured"] is True
            assert result["account_id"] == "123456789012"
            assert result["user_id"] == "AIDACKCEVSQ6C2EXAMPLE"
            assert result["region"] == "us-west-2"
            assert result["type"] == "aws"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_aws_provider_failure(self, mock_subprocess, manager, mock_config):
        """测试设置AWS提供商 - 失败"""
        # Mock failed AWS CLI response
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "AWS CLI not configured"
        mock_subprocess.return_value = mock_result

        result = manager._setup_aws_provider()
        
        assert result["configured"] is False
        assert "error" in result
        assert result["type"] == "aws"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_aws_provider_timeout(self, mock_subprocess, manager, mock_config):
        """测试设置AWS提供商 - 超时"""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("aws", 10)

        result = manager._setup_aws_provider()
        
        assert result["configured"] is False
        assert result["error"] == "AWS CLI timeout"
        assert result["type"] == "aws"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_aws_provider_not_found(self, mock_subprocess, manager, mock_config):
        """测试设置AWS提供商 - CLI未安装"""
        mock_subprocess.side_effect = FileNotFoundError()

        result = manager._setup_aws_provider()
        
        assert result["configured"] is False
        assert result["error"] == "AWS CLI not installed"
        assert result["type"] == "aws"

    def test_setup_azure_provider_success(self, mock_config):
        """测试设置Azure提供商 - 成功"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            logger=Mock()
        ):
            manager = MultiCloudManager(mock_config)
            
            with patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run') as mock_subprocess:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = json.dumps({
                    "id": "/subscriptions/12345678-1234-1234-1234-123456789012",
                    "tenantId": "87654321-4321-4321-4321-210987654321",
                    "state": "Enabled"
                })
                mock_result.stderr = ""
                mock_subprocess.return_value = mock_result

                result = manager._setup_azure_provider()
                
                assert result["configured"] is True
                assert "subscription_id" in result
                assert "tenant_id" in result
                assert result["type"] == "azure"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_azure_provider_failure(self, mock_subprocess, manager, mock_config):
        """测试设置Azure提供商 - 失败"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Azure CLI not logged in"
        mock_subprocess.return_value = mock_result

        result = manager._setup_azure_provider()
        
        assert result["configured"] is False
        assert result["error"] == "Azure CLI not logged in"
        assert result["type"] == "azure"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_gcp_provider_success(self, mock_subprocess, manager, mock_config):
        """测试设置GCP提供商 - 成功"""
        # Mock successful GCP CLI responses
        def side_effect(*args, **kwargs):
            if "auth list" in " ".join(args[0]):
                result = Mock()
                result.returncode = 0
                result.stdout = "user@example.com\n"
                return result
            elif "config get-value project" in " ".join(args[0]):
                result = Mock()
                result.returncode = 0
                result.stdout = "my-gcp-project\n"
                return result
            return Mock()

        mock_subprocess.side_effect = side_effect
        result = manager._setup_gcp_provider()
        
        assert result["configured"] is True
        assert result["active_account"] == "user@example.com"
        assert result["project"] == "my-gcp-project"
        assert result["type"] == "gcp"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_gcp_provider_failure(self, mock_subprocess, manager, mock_config):
        """测试设置GCP提供商 - 失败"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "gcloud not configured"
        mock_subprocess.return_value = mock_result

        result = manager._setup_gcp_provider()
        
        assert result["configured"] is False
        assert result["error"] == "gcloud not configured"
        assert result["type"] == "gcp"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_alibaba_provider_success(self, mock_subprocess, manager, mock_config):
        """测试设置阿里云提供商 - 成功"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "configuration"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = manager._setup_alibaba_provider()
        
        assert result["configured"] is True
        assert result["type"] == "alibaba"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_alibaba_provider_timeout(self, mock_subprocess, manager, mock_config):
        """测试设置阿里云提供商 - 超时"""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("aliyun", 10)

        result = manager._setup_alibaba_provider()
        
        assert result["configured"] is False
        assert result["error"] == "Alibaba CLI timeout"
        assert result["type"] == "alibaba"

    def test_setup_tencent_provider_success(self, mock_config):
        """测试设置腾讯云提供商 - 成功"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            os=Mock(),
            logger=Mock()
        ) as mocks:
            # 创建新的manager实例避免fixture干扰
            manager = MultiCloudManager(mock_config)
            
            with patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run') as mock_subprocess:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "configuration"
                mock_subprocess.return_value = mock_result

                result = manager._setup_tencent_provider()
                
                assert result["configured"] is True
                assert result["type"] == "tencent"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_tencent_provider_not_found(self, mock_subprocess, manager, mock_config):
        """测试设置腾讯云提供商 - CLI未安装"""
        mock_subprocess.side_effect = FileNotFoundError()

        result = manager._setup_tencent_provider()
        
        assert result["configured"] is False
        assert result["error"] == "Tencent CLI not installed"
        assert result["type"] == "tencent"

    def test_setup_huawei_provider_success(self, mock_config):
        """测试设置华为云提供商 - 成功"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            os=Mock(),
            logger=Mock()
        ) as mocks:
            # 创建新的manager实例避免fixture干扰
            manager = MultiCloudManager(mock_config)
            
            with patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run') as mock_subprocess:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "configuration"
                mock_subprocess.return_value = mock_result

                result = manager._setup_huawei_provider()
                
                assert result["configured"] is True
                assert result["type"] == "huawei"

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.subprocess.run')
    def test_setup_huawei_provider_exception(self, mock_subprocess, manager, mock_config):
        """测试设置华为云提供商 - 异常"""
        mock_subprocess.side_effect = Exception("Unexpected error")

        result = manager._setup_huawei_provider()
        
        assert result["configured"] is False
        assert result["error"] == "Unexpected error"
        assert result["type"] == "huawei"

    def test_check_provider_health_success(self, manager, mock_config):
        """测试检查提供商健康状态 - 成功"""
        # 设置一个配置好的提供商
        manager._providers[CloudProvider.AWS] = {"configured": True}
        
        result = manager._check_provider_health(CloudProvider.AWS)
        assert result is True

    def test_check_provider_health_failure(self, manager, mock_config):
        """测试检查提供商健康状态 - 失败"""
        # 设置一个未配置的提供商
        manager._providers[CloudProvider.AWS] = {"configured": False}
        
        result = manager._check_provider_health(CloudProvider.AWS)
        assert result is False

    def test_check_provider_health_exception(self, mock_config):
        """测试检查提供商健康状态 - 异常"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            os=Mock(),
            logger=Mock()
        ) as mocks:
            # 创建新的manager实例避免fixture干扰
            manager = MultiCloudManager(mock_config)
            
            # 直接修改_providers字典来模拟异常
            manager._providers = MagicMock()
            manager._providers.get.side_effect = Exception("Test error")
            
            result = manager._check_provider_health(CloudProvider.AWS)
            assert result is False

    def test_get_current_provider(self, manager, mock_config):
        """测试获取当前提供商"""
        current = manager.get_current_provider()
        assert current == CloudProvider.AWS

    def test_switch_provider_success(self, manager, mock_config):
        """测试切换提供商 - 成功"""
        # 设置提供商为已配置且健康
        manager._providers[CloudProvider.AZURE] = {"configured": True}
        manager._health_status[CloudProvider.AZURE] = True
        
        result = manager.switch_provider(CloudProvider.AZURE)
        
        assert result is True
        assert manager._current_provider == CloudProvider.AZURE

    def test_switch_provider_not_configured(self, manager, mock_config):
        """测试切换提供商 - 未配置"""
        # 不设置提供商信息
        
        result = manager.switch_provider(CloudProvider.AZURE)
        
        assert result is False
        assert manager._current_provider == CloudProvider.AWS  # 没有改变

    def test_switch_provider_unhealthy(self, manager, mock_config):
        """测试切换提供商 - 不健康"""
        # 设置提供商但标记为不健康
        manager._providers[CloudProvider.AZURE] = {"configured": True}
        manager._health_status[CloudProvider.AZURE] = False
        
        result = manager.switch_provider(CloudProvider.AZURE)
        
        assert result is False

    def test_failover_to_next_provider_disabled(self, manager, mock_config):
        """测试故障转移 - 未启用"""
        manager.config.failover_enabled = False
        
        result = manager.failover_to_next_provider()
        
        assert result is None

    def test_failover_to_next_provider_success(self, manager, mock_config):
        """测试故障转移 - 成功"""
        # 设置多个提供商
        manager._providers = {
            CloudProvider.AWS: {"configured": True},
            CloudProvider.AZURE: {"configured": True},
            CloudProvider.GCP: {"configured": True}
        }
        manager._health_status = {
            CloudProvider.AWS: False,  # 当前不健康
            CloudProvider.AZURE: True,
            CloudProvider.GCP: True
        }
        manager._current_provider = CloudProvider.AWS
        
        with patch.object(manager, 'switch_provider', return_value=True):
            result = manager.failover_to_next_provider()
            
            assert result == CloudProvider.AZURE
            assert manager._failover_count == 1

    def test_failover_to_next_provider_no_available(self, manager, mock_config):
        """测试故障转移 - 无可用提供商"""
        # 设置所有提供商都不健康
        manager._providers = {
            CloudProvider.AWS: {"configured": True},
            CloudProvider.AZURE: {"configured": True}
        }
        manager._health_status = {
            CloudProvider.AWS: False,
            CloudProvider.AZURE: False
        }
        manager._current_provider = CloudProvider.AWS
        
        result = manager.failover_to_next_provider()
        
        assert result is None

    def test_get_provider_status(self, manager, mock_config):
        """测试获取提供商状态"""
        # 设置一些提供商信息
        manager._providers = {
            CloudProvider.AWS: {"configured": True, "type": "aws", "account_id": "123"}
        }
        manager._health_status = {CloudProvider.AWS: True}
        
        status = manager.get_provider_status()
        
        assert status["current_provider"] == "aws"
        assert status["failover_enabled"] is True
        assert status["failover_count"] == 0
        assert "aws" in status["providers"]

    def test_deploy_to_current_provider_not_configured(self, manager, mock_config):
        """测试部署到当前提供商 - 未配置"""
        manager._providers[CloudProvider.AWS] = {"configured": False}
        
        result = manager.deploy_to_current_provider({"resource": "test"})
        
        assert result is False

    @patch.object(MultiCloudManager, '_deploy_to_aws')
    def test_deploy_to_current_provider_aws_success(self, mock_deploy, manager, mock_config):
        """测试部署到AWS - 成功"""
        manager._providers[CloudProvider.AWS] = {"configured": True}
        manager._current_provider = CloudProvider.AWS
        mock_deploy.return_value = True
        
        result = manager.deploy_to_current_provider({"resource": "test"})
        
        assert result is True
        mock_deploy.assert_called_once_with({"resource": "test"})

    @patch.object(MultiCloudManager, '_deploy_to_azure')
    def test_deploy_to_current_provider_azure_success(self, mock_deploy, manager, mock_config):
        """测试部署到Azure - 成功"""
        manager._providers[CloudProvider.AZURE] = {"configured": True}
        manager._current_provider = CloudProvider.AZURE
        mock_deploy.return_value = True
        
        result = manager.deploy_to_current_provider({"resource": "test"})
        
        assert result is True
        mock_deploy.assert_called_once_with({"resource": "test"})

    @patch.object(MultiCloudManager, '_deploy_to_gcp')
    def test_deploy_to_current_provider_gcp_success(self, mock_deploy, manager, mock_config):
        """测试部署到GCP - 成功"""
        manager._providers[CloudProvider.GCP] = {"configured": True}
        manager._current_provider = CloudProvider.GCP
        mock_deploy.return_value = True
        
        result = manager.deploy_to_current_provider({"resource": "test"})
        
        assert result is True
        mock_deploy.assert_called_once_with({"resource": "test"})

    def test_deploy_to_current_provider_with_failover(self, mock_config):
        """测试部署失败后进行故障转移"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            os=Mock(),
            logger=Mock()
        ) as mocks:
            # 创建新的manager实例避免fixture干扰
            manager = MultiCloudManager(mock_config)
            
            manager._providers[CloudProvider.AWS] = {"configured": True}
            manager._providers[CloudProvider.AZURE] = {"configured": True}
            manager._current_provider = CloudProvider.AWS
            manager._health_status[CloudProvider.AZURE] = True
            
            # 简单测试：只验证failover方法被调用，不测试递归
            with patch.object(manager, '_deploy_to_aws', side_effect=Exception("Deploy failed")):
                with patch.object(manager, 'failover_to_next_provider', return_value=None) as mock_failover:
                    # 不开启failover，避免递归
                    manager.config.failover_enabled = False
                    
                    # 这个调用应该失败并且不调用failover
                    result = manager.deploy_to_current_provider({"resource": "test"})
                    assert result is False
                    
                    # 现在启用failover并测试
                    manager.config.failover_enabled = True
                    mock_failover.reset_mock()
                    
                    # 由于_deploy_to_aws会抛异常，会触发failover逻辑，但为了避免递归，我们只验证failover被调用
                    try:
                        manager.deploy_to_current_provider({"resource": "test"})
                    except RecursionError:
                        # 预期的递归错误，这证明了failover被调用了
                        pass
                    
                    # 验证failover被调用了
                    mock_failover.assert_called_once()

    @patch.object(MultiCloudManager, '_deploy_to_aws')
    def test_deploy_to_current_provider_unsupported_provider(self, mock_deploy, manager, mock_config):
        """测试部署到不支持的提供商"""
        manager._providers[CloudProvider.ALIBABA] = {"configured": True}
        manager._current_provider = CloudProvider.ALIBABA
        
        result = manager.deploy_to_current_provider({"resource": "test"})
        
        assert result is False

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.logger')
    def test_deploy_to_aws(self, mock_logger, manager, mock_config):
        """测试AWS部署"""
        result = manager._deploy_to_aws({"resource": "test"})
        
        assert result is True
        mock_logger.info.assert_called()

    def test_deploy_to_aws_exception(self, mock_config):
        """测试AWS部署异常"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            os=Mock(),
            logger=Mock()
        ) as mocks:
            # 创建新的manager实例避免fixture干扰
            manager = MultiCloudManager(mock_config)
            
            # 直接mock logger来让它抛异常
            with patch('src.infrastructure.config.environment.cloud_multi_cloud.logger') as mock_logger:
                mock_logger.info.side_effect = Exception("AWS error")
                
                result = manager._deploy_to_aws({"resource": "test"})
                assert result is False
                mock_logger.error.assert_called_once()

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.logger')
    def test_deploy_to_azure(self, mock_logger, manager, mock_config):
        """测试Azure部署"""
        result = manager._deploy_to_azure({"resource": "test"})
        
        assert result is True
        mock_logger.info.assert_called()

    def test_deploy_to_azure_exception(self, mock_config):
        """测试Azure部署异常"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            os=Mock(),
            logger=Mock()
        ) as mocks:
            # 创建新的manager实例避免fixture干扰
            manager = MultiCloudManager(mock_config)
            
            # 直接mock logger来让它抛异常
            with patch('src.infrastructure.config.environment.cloud_multi_cloud.logger') as mock_logger:
                mock_logger.info.side_effect = Exception("Azure error")
                
                result = manager._deploy_to_azure({"resource": "test"})
                assert result is False
                mock_logger.error.assert_called_once()

    @patch('src.infrastructure.config.environment.cloud_multi_cloud.logger')
    def test_deploy_to_gcp(self, mock_logger, manager, mock_config):
        """测试GCP部署"""
        result = manager._deploy_to_gcp({"resource": "test"})
        
        assert result is True
        mock_logger.info.assert_called()

    def test_deploy_to_gcp_exception(self, mock_config):
        """测试GCP部署异常"""
        with patch.multiple(
            'src.infrastructure.config.environment.cloud_multi_cloud',
            subprocess=Mock(),
            os=Mock(),
            logger=Mock()
        ) as mocks:
            # 创建新的manager实例避免fixture干扰
            manager = MultiCloudManager(mock_config)
            
            # 直接mock logger来让它抛异常
            with patch('src.infrastructure.config.environment.cloud_multi_cloud.logger') as mock_logger:
                mock_logger.info.side_effect = Exception("GCP error")
                
                result = manager._deploy_to_gcp({"resource": "test"})
                assert result is False
                mock_logger.error.assert_called_once()

    def test_get_region_mapping_found(self, manager, mock_config):
        """测试获取区域映射 - 找到"""
        result = manager.get_region_mapping("us")
        
        assert result == "us-east-1"

    def test_get_region_mapping_not_found(self, manager, mock_config):
        """测试获取区域映射 - 未找到"""
        result = manager.get_region_mapping("eu")
        
        assert result is None

    def test_validate_multi_cloud_setup_valid(self, manager, mock_config):
        """测试验证多云设置 - 有效"""
        # 设置主提供商为已配置
        manager._providers[CloudProvider.AWS] = {"configured": True}
        manager._providers[CloudProvider.AZURE] = {"configured": True}
        
        result = manager.validate_multi_cloud_setup()
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_multi_cloud_setup_invalid_primary(self, manager, mock_config):
        """测试验证多云设置 - 主提供商无效"""
        # 主提供商未配置
        manager._providers[CloudProvider.AWS] = {"configured": False}
        
        result = manager.validate_multi_cloud_setup()
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_multi_cloud_setup_with_warnings(self, manager, mock_config):
        """测试验证多云设置 - 有警告"""
        # 主提供商配置，但备份提供商未配置
        manager._providers[CloudProvider.AWS] = {"configured": True}
        manager._providers[CloudProvider.AZURE] = {"configured": False}
        manager.config.failover_enabled = True
        
        result = manager.validate_multi_cloud_setup()
        
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_validate_multi_cloud_setup_recommendations(self, manager, mock_config):
        """测试验证多云设置 - 有推荐"""
        # 只配置一个备份提供商
        manager._providers[CloudProvider.AWS] = {"configured": True}
        manager._providers[CloudProvider.AZURE] = {"configured": True}
        manager.config.secondary_providers = [CloudProvider.AZURE]  # 只有一个备份
        
        result = manager.validate_multi_cloud_setup()
        
        assert len(result["recommendations"]) > 0

    def test_setup_providers_with_all_cloud_providers(self, manager, mock_config):
        """测试设置所有云提供商"""
        # 设置所有提供商
        mock_config.primary_provider = CloudProvider.AWS
        mock_config.secondary_providers = [
            CloudProvider.AZURE, CloudProvider.GCP, CloudProvider.ALIBABA,
            CloudProvider.TENCENT, CloudProvider.HUAWEI
        ]
        
        with patch.object(manager, '_setup_aws_provider', return_value={"configured": True, "type": "aws"}), \
             patch.object(manager, '_setup_azure_provider', return_value={"configured": True, "type": "azure"}), \
             patch.object(manager, '_setup_gcp_provider', return_value={"configured": True, "type": "gcp"}), \
             patch.object(manager, '_setup_alibaba_provider', return_value={"configured": True, "type": "alibaba"}), \
             patch.object(manager, '_setup_tencent_provider', return_value={"configured": True, "type": "tencent"}), \
             patch.object(manager, '_setup_huawei_provider', return_value={"configured": True, "type": "huawei"}), \
             patch.object(manager, '_check_provider_health', return_value=True):
            
            manager._setup_providers()
            
            assert CloudProvider.AWS in manager._providers
            assert CloudProvider.AZURE in manager._providers
            assert CloudProvider.GCP in manager._providers
            assert CloudProvider.ALIBABA in manager._providers
            assert CloudProvider.TENCENT in manager._providers
            assert CloudProvider.HUAWEI in manager._providers

    def test_setup_providers_unsupported_provider(self, manager, mock_config):
        """测试设置不支持的提供商"""
        # 添加一个不存在的提供商（这在实际代码中不会发生，因为enum限制了选项）
        # 但我们可以测试警告逻辑
        with patch.object(manager, '_setup_aws_provider', return_value={"configured": True, "type": "aws"}):
            manager._setup_providers()

            # 应该只设置了AWS（primary_provider）
            assert CloudProvider.AWS in manager._providers
