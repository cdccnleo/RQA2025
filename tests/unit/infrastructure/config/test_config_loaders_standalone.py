"""
配置加载器功能测试 (独立版本)
测试各种配置格式的加载功能，避免复杂导入
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """配置加载器基类"""

    def load(self, source: str) -> Dict[str, Any]:
        """加载配置"""
        raise NotImplementedError

    def save(self, config: Dict[str, Any], target: str) -> None:
        """保存配置"""
        raise NotImplementedError


class JSONConfigLoader(ConfigLoader):
    """JSON配置加载器"""

    def load(self, source: str) -> Dict[str, Any]:
        """从JSON文件加载配置"""
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {source}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def save(self, config: Dict[str, Any], target: str) -> None:
        """保存配置到JSON文件"""
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


class YAMLConfigLoader(ConfigLoader):
    """YAML配置加载器"""

    def load(self, source: str) -> Dict[str, Any]:
        """从YAML文件加载配置"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config loading")

        try:
            with open(source, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {source}")

    def save(self, config: Dict[str, Any], target: str) -> None:
        """保存配置到YAML文件"""
        try:
            import yaml
            with open(target, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            raise ImportError("PyYAML is required for YAML config saving")


class EnvironmentConfigLoader(ConfigLoader):
    """环境变量配置加载器"""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix.upper()

    def load(self, source: str) -> Dict[str, Any]:
        """从环境变量加载配置"""
        config = {}
        for key, value in os.environ.items():
            if not self.prefix or key.startswith(self.prefix):
                config_key = key[len(self.prefix):].lower() if self.prefix else key.lower()
                # Try to parse as JSON, fallback to string
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    config[config_key] = value
        return config

    def save(self, config: Dict[str, Any], target: str) -> None:
        """保存配置到环境变量 (实际上不保存，只是验证)"""
        # Environment variables are usually set externally
        pass


class ConfigLoaderFactory:
    """配置加载器工厂"""

    @staticmethod
    def create_loader(format_type: str, **kwargs) -> ConfigLoader:
        """创建配置加载器"""
        if format_type.lower() == 'json':
            return JSONConfigLoader()
        elif format_type.lower() == 'yaml':
            return YAMLConfigLoader()
        elif format_type.lower() == 'env':
            prefix = kwargs.get('prefix', '')
            return EnvironmentConfigLoader(prefix)
        else:
            raise ValueError(f"Unsupported config format: {format_type}")


class TestJSONConfigLoader:
    """JSON配置加载器测试"""

    def setup_method(self):
        """测试前准备"""
        self.loader = JSONConfigLoader()
        self.test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "admin",
                    "password": "secret"
                }
            },
            "logging": {
                "level": "INFO",
                "file": "/var/log/app.log"
            }
        }

    def test_load_valid_json(self):
        """测试加载有效的JSON配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f, indent=2)
            temp_path = f.name

        try:
            loaded_config = self.loader.load(temp_path)
            assert loaded_config == self.test_config
            assert loaded_config["database"]["host"] == "localhost"
            assert loaded_config["database"]["port"] == 5432
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json(self):
        """测试加载无效的JSON配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON format"):
                self.loader.load(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            self.loader.load("/nonexistent/file.json")

    def test_save_config(self):
        """测试保存配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            self.loader.save(self.test_config, temp_path)

            # Verify saved content
            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_content = json.load(f)

            assert saved_content == self.test_config
        finally:
            os.unlink(temp_path)


class TestYAMLConfigLoader:
    """YAML配置加载器测试"""

    def setup_method(self):
        """测试前准备"""
        self.loader = YAMLConfigLoader()
        self.test_config = {
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "ssl": True
            },
            "features": ["auth", "metrics", "health_check"],
            "version": "1.2.3"
        }

    def test_load_valid_yaml(self):
        """测试加载有效的YAML配置"""
        import yaml  # 直接导入，确保可用

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f, default_flow_style=False)
            temp_path = f.name

        try:
            loaded_config = self.loader.load(temp_path)
            assert loaded_config == self.test_config
            assert loaded_config["server"]["port"] == 8080
        finally:
            os.unlink(temp_path)

    def test_load_yaml_without_pyyaml(self):
        """测试在没有PyYAML的情况下加载YAML"""
        # Mock yaml import to raise ImportError
        import sys
        from unittest.mock import patch

        with patch.dict('sys.modules', {'yaml': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'yaml'")):
                # Create a dummy file so we don't get FileNotFoundError first
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    f.write("dummy: content")
                    temp_path = f.name

                try:
                    with pytest.raises(ImportError, match="PyYAML is required"):
                        self.loader.load(temp_path)
                finally:
                    os.unlink(temp_path)

    def test_load_nonexistent_yaml_file(self):
        """测试加载不存在的YAML文件"""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            self.loader.load("/nonexistent/file.yaml")


class TestEnvironmentConfigLoader:
    """环境变量配置加载器测试"""

    def setup_method(self):
        """测试前准备"""
        self.loader = EnvironmentConfigLoader("APP_")
        # Clean up any existing test environment variables
        test_keys = [k for k in os.environ.keys() if k.startswith("APP_")]
        for key in test_keys:
            del os.environ[key]

    def teardown_method(self):
        """测试后清理"""
        # Clean up test environment variables
        test_keys = [k for k in os.environ.keys() if k.startswith("APP_")]
        for key in test_keys:
            del os.environ[key]

    def test_load_env_variables_with_prefix(self):
        """测试加载带前缀的环境变量"""
        os.environ["APP_DATABASE_HOST"] = "prod-db.example.com"
        os.environ["APP_DATABASE_PORT"] = "5432"
        os.environ["APP_LOGGING_LEVEL"] = "DEBUG"
        os.environ["APP_FEATURES_ENABLED"] = '["auth", "metrics"]'

        config = self.loader.load("")

        assert config["database_host"] == "prod-db.example.com"
        assert config["database_port"] == 5432  # JSON parsing converts to int
        assert config["logging_level"] == "DEBUG"
        assert config["features_enabled"] == ["auth", "metrics"]

    def test_load_env_variables_without_prefix(self):
        """测试加载无前缀的环境变量"""
        loader = EnvironmentConfigLoader()
        os.environ["DATABASE_HOST"] = "localhost"
        os.environ["DATABASE_PORT"] = "3306"

        config = loader.load("")

        assert config["database_host"] == "localhost"
        assert config["database_port"] == 3306  # JSON parsing converts to int

        # Clean up
        del os.environ["DATABASE_HOST"]
        del os.environ["DATABASE_PORT"]

    def test_env_variable_json_parsing(self):
        """测试环境变量JSON解析"""
        os.environ["APP_CONFIG_JSON"] = '{"servers": ["server1", "server2"], "timeout": 30}'
        os.environ["APP_CONFIG_STRING"] = "simple_string"

        config = self.loader.load("")

        assert config["config_json"] == {"servers": ["server1", "server2"], "timeout": 30}
        assert config["config_string"] == "simple_string"


class TestConfigLoaderFactory:
    """配置加载器工厂测试"""

    def test_create_json_loader(self):
        """测试创建JSON加载器"""
        loader = ConfigLoaderFactory.create_loader("json")
        assert isinstance(loader, JSONConfigLoader)

    def test_create_yaml_loader(self):
        """测试创建YAML加载器"""
        loader = ConfigLoaderFactory.create_loader("YAML")
        assert isinstance(loader, YAMLConfigLoader)

    def test_create_env_loader(self):
        """测试创建环境变量加载器"""
        loader = ConfigLoaderFactory.create_loader("env", prefix="TEST_")
        assert isinstance(loader, EnvironmentConfigLoader)
        assert loader.prefix == "TEST_"

    def test_create_unsupported_format(self):
        """测试创建不支持的格式加载器"""
        with pytest.raises(ValueError, match="Unsupported config format"):
            ConfigLoaderFactory.create_loader("xml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
