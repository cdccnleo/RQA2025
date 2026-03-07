"""
业务流程配置管理测试覆盖率补充

为config.py的未覆盖部分添加测试，提高覆盖率从18%到70%+
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

try:
    from src.core.business_process.config.config import ProcessConfigManager
    from src.core.business_process.models.models import ProcessConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="配置管理模块不可用")
class TestProcessConfigManagerCoverage:
    """测试ProcessConfigManager覆盖率补充"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ProcessConfigManager(self.temp_dir)

    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_config_success(self):
        """测试更新配置成功路径"""
        # 创建初始配置
        config = self.manager.create_config(
            process_id="test_process",
            process_name="Test Process",
            description="Test configuration"
        )
        assert config is not None

        # 更新配置
        updates = {
            "description": "Updated description",
            "retry_count": 5
        }
        updated_config = self.manager.update_config("test_process", updates)
        assert updated_config is not None
        assert updated_config.description == "Updated description"

    def test_update_config_not_found(self):
        """测试更新不存在的配置"""
        updates = {"description": "Updated"}
        result = self.manager.update_config("nonexistent", updates)
        assert result is None

    def test_update_config_validation_failure(self):
        """测试更新配置验证失败"""
        # 创建初始配置
        config = self.manager.create_config("test_process", "Test Process")
        assert config is not None

        # 尝试用无效数据更新 - 使用不存在的字段会抛出TypeError
        updates = {"nonexistent_field": "value"}
        with pytest.raises(TypeError):
            self.manager.update_config("test_process", updates)

    def test_delete_config_success(self):
        """测试删除配置成功"""
        # 创建配置
        config = self.manager.create_config("delete_test", "Delete Test")
        assert config is not None

        # 删除配置
        result = self.manager.delete_config("delete_test")
        assert result == True

        # 验证已删除
        retrieved = self.manager.get_config("delete_test")
        assert retrieved is None

    def test_delete_config_not_found(self):
        """测试删除不存在的配置"""
        result = self.manager.delete_config("nonexistent")
        assert result == True  # 删除不存在的配置被认为是成功的

    def test_list_configs_empty(self):
        """测试列出空配置列表"""
        configs = self.manager.list_configs()
        assert isinstance(configs, list)
        assert len(configs) == 0

    def test_list_configs_with_data(self):
        """测试列出配置列表"""
        # 创建多个配置
        config1 = self.manager.create_config("process1", "Process 1")
        config2 = self.manager.create_config("process2", "Process 2")

        assert config1 is not None
        assert config2 is not None

        configs = self.manager.list_configs()
        assert len(configs) >= 2

        # 验证配置信息
        config_ids = [c['process_id'] for c in configs]
        assert "process1" in config_ids
        assert "process2" in config_ids

    def test_get_config_versions_no_versions(self):
        """测试获取不存在的配置版本"""
        versions = self.manager.get_config_versions("nonexistent")
        assert versions == []

    def test_get_config_versions_with_versions(self):
        """测试获取配置版本"""
        # 创建配置
        config1 = self.manager.create_config("version_test", "Version Test")
        assert config1 is not None

        # 更新配置创建新版本
        self.manager.update_config("version_test", {"description": "Updated"})

        versions = self.manager.get_config_versions("version_test")
        assert isinstance(versions, list)
        assert len(versions) >= 1

    def test_save_config_success(self):
        """测试保存配置成功"""
        config = ProcessConfig(
            process_id="save_test",
            process_name="Save Test",
            description="Test for saving"
        )

        result = self.manager.save_config(config)
        assert result == True

        # 验证文件已创建
        config_file = Path(self.temp_dir) / "save_test.json"
        assert config_file.exists()

        # 验证可以重新加载
        loaded = self.manager.get_config("save_test")
        assert loaded is not None
        assert loaded.process_id == "save_test"

    def test_save_config_failure(self):
        """测试保存配置失败"""
        config = ProcessConfig(
            process_id="save_fail_test",
            process_name="Save Fail Test"
        )

        # Mock文件写入失败
        with patch('builtins.open', side_effect=IOError("Write failed")):
            result = self.manager.save_config(config)
            assert result == False

    def test_serialize_config(self):
        """测试序列化配置"""
        config = ProcessConfig(
            process_id="serialize_test",
            process_name="Serialize Test",
            description="Test serialization",
            data_sources=[{"type": "database", "connection": "test"}],
            feature_config={"enabled": True}
        )

        serialized = self.manager._serialize_config(config)
        assert isinstance(serialized, dict)
        assert serialized["process_id"] == "serialize_test"
        assert serialized["process_name"] == "Serialize Test"
        assert "data_sources" in serialized
        assert "feature_config" in serialized

    def test_deserialize_config(self):
        """测试反序列化配置"""
        data = {
            "process_id": "deserialize_test",
            "process_name": "Deserialize Test",
            "description": "Test deserialization",
            "version": "1.0.0",
            "data_sources": [{"type": "api", "endpoint": "test"}],
            "feature_config": {"enabled": False},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        config = self.manager._deserialize_config(data)
        assert isinstance(config, ProcessConfig)
        assert config.process_id == "deserialize_test"
        assert config.process_name == "Deserialize Test"
        assert config.data_sources == [{"type": "api", "endpoint": "test"}]

    def test_load_config_version_not_found(self):
        """测试加载不存在的配置版本"""
        result = self.manager._load_config_version("nonexistent", "1.0.0")
        assert result is None

    def test_load_config_version_success(self):
        """测试加载配置版本成功"""
        # 创建并保存配置
        config = self.manager.create_config("version_load_test", "Version Load Test")
        assert config is not None

        # 加载版本
        loaded = self.manager._load_config_version("version_load_test", config.version)
        assert loaded is not None
        assert loaded.process_id == "version_load_test"

    def test_validate_config_valid(self):
        """测试验证有效配置"""
        config = ProcessConfig(
            process_id="valid_test",
            process_name="Valid Test",
            description="Valid configuration"
        )

        result = self.manager._validate_config(config)
        assert result == True

    def test_validate_config_invalid(self):
        """测试验证无效配置"""
        # 创建无效配置（空的process_id）会抛出ValueError
        with pytest.raises(ValueError, match="process_id cannot be empty"):
            ProcessConfig(
                process_id="",  # 无效
                process_name="Invalid Test"
            )

    def test_validate_config_data_valid(self):
        """测试验证有效配置数据"""
        data = {
            "process_id": "data_valid_test",
            "process_name": "Data Valid Test",
            "version": "1.0.0"
        }

        result = self.manager._validate_config_data(data)
        assert result == True

    def test_validate_config_data_invalid(self):
        """测试验证无效配置数据"""
        data = {
            "process_name": "Missing ID Test"
            # 缺少process_id
        }

        result = self.manager._validate_config_data(data)
        assert result == False

    def test_increment_version(self):
        """测试版本递增"""
        # 测试正常版本
        new_version = self.manager._increment_version("1.0.0")
        assert new_version == "1.0.1"

        # 测试大版本
        new_version = self.manager._increment_version("2.5.9")
        assert new_version == "2.5.10"

    def test_get_config_hash(self):
        """测试获取配置哈希"""
        config = ProcessConfig(
            process_id="hash_test",
            process_name="Hash Test",
            description="Test for hashing"
        )

        hash_value = self.manager.get_config_hash(config)
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

        # 相同配置应该产生相同哈希
        config2 = ProcessConfig(
            process_id="hash_test",
            process_name="Hash Test",
            description="Test for hashing"
        )
        hash_value2 = self.manager.get_config_hash(config2)
        assert hash_value == hash_value2

    def test_compare_configs_identical(self):
        """测试比较相同配置"""
        config1 = ProcessConfig(
            process_id="compare_test",
            process_name="Compare Test",
            description="Same config"
        )
        config2 = ProcessConfig(
            process_id="compare_test",
            process_name="Compare Test",
            description="Same config"
        )

        result = self.manager.compare_configs(config1, config2)
        assert isinstance(result, dict)
        assert result.get("hash_changed") == False

    def test_compare_configs_different(self):
        """测试比较不同配置"""
        config1 = ProcessConfig(
            process_id="compare_test1",
            process_name="Compare Test 1",
            description="First config"
        )
        config2 = ProcessConfig(
            process_id="compare_test2",
            process_name="Compare Test 2",
            description="Second config"
        )

        result = self.manager.compare_configs(config1, config2)
        assert isinstance(result, dict)
        assert result.get("hash_changed") == True

    def test_load_existing_configs_with_invalid_json(self):
        """测试加载包含无效JSON的现有配置"""
        # 创建无效的JSON文件
        invalid_file = Path(self.temp_dir) / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")

        # 重新初始化管理器，应该处理无效JSON
        manager = ProcessConfigManager(self.temp_dir)
        # 不应该抛出异常，应该跳过无效文件

    def test_load_existing_configs_with_invalid_config(self):
        """测试加载包含无效配置的现有配置"""
        # 创建包含无效配置数据的JSON文件
        invalid_config_file = Path(self.temp_dir) / "invalid_config.json"
        with open(invalid_config_file, 'w') as f:
            json.dump({"invalid": "config"}, f)

        # 重新初始化管理器
        manager = ProcessConfigManager(self.temp_dir)
        # 应该跳过无效配置
