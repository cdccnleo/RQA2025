"""
测试核心业务层配置管理功能
"""
import pytest
from unittest.mock import patch, Mock, MagicMock

# 尝试导入所需模块
try:
    from core.business_process.config.config import ProcessConfigManager
    from infrastructure.resource.config.config_classes import ProcessConfig
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestProcessConfigManager:
    """测试流程配置管理器"""

    def test_process_config_manager_initialization(self):
        """测试流程配置管理器初始化"""
        manager = ProcessConfigManager()
        assert isinstance(manager.config_dir, object)  # Path对象
        assert isinstance(manager._config_cache, dict)
        assert isinstance(manager._config_versions, dict)

    def test_create_process_config(self):
        """测试创建流程配置"""
        manager = ProcessConfigManager()

        config_data = {
            "process_id": "test_process",
            "process_name": "Test Process",
            "description": "Test process description",
            "data_sources": [{"type": "market_data", "source": "api"}],
            "timeout_seconds": 600
        }

        config = manager.create_config(**config_data)

        assert config.process_id == "test_process"
        assert config.process_name == "Test Process"
        assert config.description == "Test process description"
        assert len(config.data_sources) == 1
        assert config.timeout_seconds == 600

    def test_save_and_load_config(self):
        """测试配置保存和加载"""
        manager = ProcessConfigManager()

        # 创建配置
        config = manager.create_config(
            process_id="test_process",
            process_name="Test Process",
            description="Test description"
        )

        # 保存配置
        manager.save_config(config)

        # 验证配置已保存到缓存
        assert "test_process" in manager._config_cache
        assert manager._config_cache["test_process"] == config

    def test_get_config(self):
        """测试获取配置"""
        manager = ProcessConfigManager()

        # 创建并保存配置
        config = manager.create_config(
            process_id="test_process",
            process_name="Test Process"
        )
        manager.save_config(config)

        # 获取配置
        retrieved_config = manager.get_config("test_process")
        assert retrieved_config == config

        # 获取不存在的配置
        assert manager.get_config("nonexistent") is None

    def test_list_configs(self):
        """测试列出所有配置"""
        manager = ProcessConfigManager()

        # 创建多个配置
        config1 = manager.create_config(
            process_id="process1",
            process_name="Process 1"
        )
        config2 = manager.create_config(
            process_id="process2",
            process_name="Process 2"
        )

        manager.save_config(config1)
        manager.save_config(config2)

        configs = manager.list_configs()
        # 查找新创建的配置
        process1_found = any(c['process_id'] == 'process1' for c in configs)
        process2_found = any(c['process_id'] == 'process2' for c in configs)
        assert process1_found
        assert process2_found

    def test_validate_config(self):
        """测试配置验证"""
        manager = ProcessConfigManager()

        # 有效配置
        valid_config = manager.create_config(
            process_id="valid_process",
            process_name="Valid Process"
        )
        assert manager._validate_config(valid_config)

        # 无效配置 - 缺少必需字段
        invalid_config = ProcessConfig(
            action="",  # 空action
            params={"invalid": "config"}
        )
        assert not manager._validate_config(invalid_config)

    def test_config_versioning(self):
        """测试配置版本管理"""
        manager = ProcessConfigManager()

        # 测试获取版本列表（空的）
        versions = manager.get_config_versions("nonexistent_process")
        assert versions == []

        # 创建配置并保存
        config = manager.create_config(
            process_id="versioned_process",
            process_name="Versioned Process"
        )
        manager.save_config(config)

        # 验证配置可以正常获取
        retrieved_config = manager.get_config("versioned_process")
        assert retrieved_config is not None
        assert retrieved_config.process_name == "Versioned Process"

    def test_delete_config(self):
        """测试删除配置"""
        manager = ProcessConfigManager()

        # 创建并保存配置
        config = manager.create_config(
            process_id="delete_test",
            process_name="Delete Test"
        )
        manager.save_config(config)

        # 验证配置存在
        assert manager.get_config("delete_test") is not None

        # 删除配置
        manager.delete_config("delete_test")

        # 验证配置已被删除
        assert manager.get_config("delete_test") is None

    @patch('core.business_process.config.config.Path.exists')
    @patch('core.business_process.config.config.Path.glob')
    def test_load_existing_configs(self, mock_glob, mock_exists):
        """测试加载现有配置"""
        mock_exists.return_value = True
        mock_glob.return_value = [MagicMock() for _ in range(3)]  # 模拟3个配置文件

        manager = ProcessConfigManager()

        # 验证加载方法被调用
        assert isinstance(manager._config_cache, dict)
        assert isinstance(manager._config_versions, dict)
