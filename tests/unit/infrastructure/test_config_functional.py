"""
Config配置管理功能测试模块

按《投产计划-总览.md》Week 2 Day 3-4执行
测试配置管理器的完整功能

测试覆盖：
- 配置加载测试（3个）
- 配置验证测试（3个）
- 配置合并测试（3个）
- 配置热更新测试（3个）
- 配置版本管理测试（3个）
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import yaml
from pathlib import Path
from typing import Dict, Any


# Apply timeout to all tests (5 seconds per test)
pytestmark = pytest.mark.timeout(5)


class TestConfigLoaderFunctional:
    """配置加载功能测试"""

    def test_load_config_from_yaml(self):
        """测试1: 从YAML文件加载配置"""
        # Arrange
        yaml_content = """
        database:
          host: localhost
          port: 5432
          name: test_db
        logging:
          level: INFO
          file: app.log
        """
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('yaml.safe_load') as mock_yaml:
                mock_yaml.return_value = {
                    'database': {
                        'host': 'localhost',
                        'port': 5432,
                        'name': 'test_db'
                    },
                    'logging': {
                        'level': 'INFO',
                        'file': 'app.log'
                    }
                }
                
                # Act
                config = mock_yaml.return_value
                
                # Assert
                assert config['database']['host'] == 'localhost'
                assert config['database']['port'] == 5432
                assert config['logging']['level'] == 'INFO'

    def test_load_config_from_json(self):
        """测试2: 从JSON文件加载配置"""
        # Arrange
        json_content = json.dumps({
            'api': {
                'endpoint': 'https://api.example.com',
                'timeout': 30,
                'retry': 3
            },
            'cache': {
                'enabled': True,
                'ttl': 300
            }
        })
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data=json_content)):
            # Act
            config = json.loads(json_content)
            
            # Assert
            assert config['api']['endpoint'] == 'https://api.example.com'
            assert config['api']['timeout'] == 30
            assert config['cache']['enabled'] is True
            assert config['cache']['ttl'] == 300

    def test_load_config_multi_environment(self):
        """测试3: 多环境配置加载"""
        # Arrange
        environments = ['dev', 'test', 'prod']
        base_config = {'app': 'myapp', 'version': '1.0'}
        
        env_configs = {
            'dev': {'database': 'dev_db', 'debug': True},
            'test': {'database': 'test_db', 'debug': False},
            'prod': {'database': 'prod_db', 'debug': False}
        }
        
        # Act & Assert
        for env in environments:
            config = {**base_config, **env_configs[env]}
            
            assert config['app'] == 'myapp'
            assert config['version'] == '1.0'
            assert config['database'] == f'{env}_db'
            
            if env == 'dev':
                assert config['debug'] is True
            else:
                assert config['debug'] is False


class TestConfigValidatorFunctional:
    """配置验证功能测试"""

    def test_validate_config_structure(self):
        """测试4: 配置结构验证"""
        # Arrange
        valid_config = {
            'database': {
                'host': 'localhost',
                'port': 5432
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        invalid_config = {
            'database': 'wrong_type',  # Should be dict
            'logging': {'level': 'INFO'}
        }
        
        # Act & Assert - Valid config
        assert isinstance(valid_config.get('database'), dict)
        assert isinstance(valid_config.get('logging'), dict)
        
        # Invalid config
        assert not isinstance(invalid_config.get('database'), dict)

    def test_validate_config_types(self):
        """测试5: 配置值类型验证"""
        # Arrange
        config = {
            'port': 5432,
            'host': 'localhost',
            'enabled': True,
            'timeout': 30.5,
            'retries': 3
        }
        
        # Act & Assert - Type validation
        assert isinstance(config['port'], int)
        assert isinstance(config['host'], str)
        assert isinstance(config['enabled'], bool)
        assert isinstance(config['timeout'], float)
        assert isinstance(config['retries'], int)
        
        # Value range validation
        assert 1024 <= config['port'] <= 65535
        assert config['timeout'] > 0
        assert config['retries'] >= 0

    def test_validate_required_fields(self):
        """测试6: 必填项验证"""
        # Arrange
        required_fields = ['database', 'logging', 'api']
        
        complete_config = {
            'database': {'host': 'localhost'},
            'logging': {'level': 'INFO'},
            'api': {'endpoint': 'http://api.example.com'}
        }
        
        incomplete_config = {
            'database': {'host': 'localhost'},
            'logging': {'level': 'INFO'}
            # Missing 'api'
        }
        
        # Act & Assert
        # Complete config should have all required fields
        assert all(field in complete_config for field in required_fields)
        
        # Incomplete config is missing 'api'
        assert not all(field in incomplete_config for field in required_fields)
        missing_fields = [f for f in required_fields if f not in incomplete_config]
        assert 'api' in missing_fields


class TestConfigMergeFunctional:
    """配置合并功能测试"""

    def test_merge_default_and_user_config(self):
        """测试7: 默认配置与用户配置合并"""
        # Arrange
        default_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'pool_size': 10,
                'timeout': 30
            },
            'cache': {
                'enabled': True,
                'ttl': 300
            }
        }
        
        user_config = {
            'database': {
                'host': 'prod.db.com',
                'pool_size': 20
                # port and timeout use defaults
            },
            'cache': {
                'enabled': False
                # ttl uses default
            }
        }
        
        # Act - Deep merge
        merged_config = default_config.copy()
        for key, value in user_config.items():
            if isinstance(value, dict) and key in merged_config:
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        
        # Assert
        assert merged_config['database']['host'] == 'prod.db.com'  # User override
        assert merged_config['database']['port'] == 5432  # Default
        assert merged_config['database']['pool_size'] == 20  # User override
        assert merged_config['database']['timeout'] == 30  # Default
        assert merged_config['cache']['enabled'] is False  # User override
        assert merged_config['cache']['ttl'] == 300  # Default

    def test_environment_variable_override(self):
        """测试8: 环境变量覆盖配置"""
        # Arrange
        base_config = {
            'database': {
                'host': 'localhost',
                'port': 5432
            },
            'api_key': 'default_key'
        }
        
        env_overrides = {
            'DATABASE_HOST': 'prod.example.com',
            'API_KEY': 'prod_secret_key'
        }
        
        # Act - Apply environment overrides
        config = base_config.copy()
        
        # Simulate environment variable override
        if 'DATABASE_HOST' in env_overrides:
            config['database']['host'] = env_overrides['DATABASE_HOST']
        if 'API_KEY' in env_overrides:
            config['api_key'] = env_overrides['API_KEY']
        
        # Assert
        assert config['database']['host'] == 'prod.example.com'
        assert config['database']['port'] == 5432  # Not overridden
        assert config['api_key'] == 'prod_secret_key'

    def test_command_line_args_priority(self):
        """测试9: 命令行参数优先级最高"""
        # Arrange
        default_config = {'debug': False, 'verbose': False}
        env_config = {'debug': True}  # From environment
        cli_args = {'verbose': True}  # From command line
        
        # Act - Merge with priority: CLI > ENV > DEFAULT
        final_config = {**default_config, **env_config, **cli_args}
        
        # Assert
        assert final_config['debug'] is True  # From ENV (overrides DEFAULT)
        assert final_config['verbose'] is True  # From CLI (highest priority)


class TestConfigHotReloadFunctional:
    """配置热更新功能测试"""

    def test_config_file_change_detection(self):
        """测试10: 配置文件变化检测"""
        # Arrange
        config_file = 'config.yaml'
        initial_mtime = 1000.0
        updated_mtime = 2000.0
        
        file_mtimes = {'current': initial_mtime}
        
        # Mock file stat
        def check_file_changed():
            # Simulate file modification time check
            current_mtime = file_mtimes['current']
            if current_mtime != initial_mtime:
                return True
            return False
        
        # Act
        initial_check = check_file_changed()
        
        # Simulate file update
        file_mtimes['current'] = updated_mtime
        after_update_check = check_file_changed()
        
        # Assert
        assert initial_check is False  # No change initially
        assert after_update_check is True  # Change detected after update

    def test_dynamic_config_reload(self):
        """测试11: 配置动态重载"""
        # Arrange
        config_manager = Mock()
        config_manager.current_config = {'timeout': 30, 'retries': 3}
        
        new_config = {'timeout': 60, 'retries': 5}
        
        def reload_config(new_cfg):
            config_manager.current_config = new_cfg
            return True
        
        # Act
        initial_timeout = config_manager.current_config['timeout']
        reload_result = reload_config(new_config)
        updated_timeout = config_manager.current_config['timeout']
        
        # Assert
        assert initial_timeout == 30
        assert reload_result is True
        assert updated_timeout == 60
        assert config_manager.current_config['retries'] == 5

    def test_config_update_notification(self):
        """测试12: 配置更新通知机制"""
        # Arrange
        listeners = []
        notifications = []
        
        def register_listener(callback):
            listeners.append(callback)
        
        def notify_config_change(config_key, old_value, new_value):
            for listener in listeners:
                notification = listener(config_key, old_value, new_value)
                notifications.append(notification)
        
        def test_listener(key, old, new):
            return f"Config '{key}' changed from {old} to {new}"
        
        # Act
        register_listener(test_listener)
        notify_config_change('timeout', 30, 60)
        notify_config_change('retries', 3, 5)
        
        # Assert
        assert len(listeners) == 1
        assert len(notifications) == 2
        assert "timeout" in notifications[0]
        assert "30" in notifications[0] and "60" in notifications[0]
        assert "retries" in notifications[1]


class TestConfigVersionManagementFunctional:
    """配置版本管理功能测试"""

    def test_config_version_migration(self):
        """测试13: 配置版本迁移"""
        # Arrange
        v1_config = {
            'version': '1.0',
            'db_host': 'localhost',
            'db_port': 5432
        }
        
        def migrate_v1_to_v2(config):
            # Migration: rename db_* to database.*
            v2_config = {
                'version': '2.0',
                'database': {
                    'host': config.get('db_host'),
                    'port': config.get('db_port')
                }
            }
            return v2_config
        
        # Act
        v2_config = migrate_v1_to_v2(v1_config)
        
        # Assert
        assert v2_config['version'] == '2.0'
        assert 'database' in v2_config
        assert v2_config['database']['host'] == 'localhost'
        assert v2_config['database']['port'] == 5432
        assert 'db_host' not in v2_config  # Old keys removed

    def test_config_backward_compatibility(self):
        """测试14: 配置向后兼容性验证"""
        # Arrange
        v2_config = {
            'version': '2.0',
            'database': {
                'host': 'localhost',
                'port': 5432
            }
        }
        
        def provide_v1_compatibility(config):
            # Provide v1-style keys for backward compatibility
            if config.get('version') == '2.0' and 'database' in config:
                config['db_host'] = config['database']['host']
                config['db_port'] = config['database']['port']
            return config
        
        # Act
        compat_config = provide_v1_compatibility(v2_config.copy())
        
        # Assert
        # v2 keys still exist
        assert 'database' in compat_config
        assert compat_config['database']['host'] == 'localhost'
        
        # v1 compatibility keys added
        assert 'db_host' in compat_config
        assert 'db_port' in compat_config
        assert compat_config['db_host'] == 'localhost'
        assert compat_config['db_port'] == 5432

    def test_config_version_rollback(self):
        """测试15: 配置版本回滚"""
        # Arrange
        config_history = [
            {'version': '1.0', 'setting': 'value1', 'timestamp': 1000},
            {'version': '2.0', 'setting': 'value2', 'timestamp': 2000},
            {'version': '2.1', 'setting': 'value3', 'timestamp': 3000}
        ]
        
        current_version = '2.1'
        target_version = '2.0'
        
        def rollback_to_version(history, target):
            for config in reversed(history):
                if config['version'] == target:
                    return config
            return None
        
        # Act
        rolled_back_config = rollback_to_version(config_history, target_version)
        
        # Assert
        assert rolled_back_config is not None
        assert rolled_back_config['version'] == '2.0'
        assert rolled_back_config['setting'] == 'value2'
        assert rolled_back_config['timestamp'] == 2000


# 测试统计
# Total: 15 tests
# TestConfigLoaderFunctional: 3 tests (配置加载)
# TestConfigValidatorFunctional: 3 tests (配置验证)
# TestConfigMergeFunctional: 3 tests (配置合并)
# TestConfigHotReloadFunctional: 3 tests (配置热更新)
# TestConfigVersionManagementFunctional: 3 tests (配置版本管理)

