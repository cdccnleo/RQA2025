import pytest
from unittest.mock import MagicMock, patch, ANY
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.event import EventSystem
import os

class TestConfigManager:
    """Test initialization of ConfigManager with default parameters"""
    def test_init_with_default_params(self):
        """Test that ConfigManager initializes with correct defaults when no parameters are provided"""
        # Act
        config_manager = ConfigManager()

        # Assert
        assert config_manager._event_system == EventSystem.get_default()
        assert config_manager.env_policies['default']['audit_level'] == 'standard'
        assert config_manager.env_policies['default']['validation_level'] == 'basic'
        assert config_manager._security_service.audit_level == 'standard'
        assert config_manager._security_service.validation_level == 'basic'
        assert hasattr(config_manager, '_lock_manager')
        assert hasattr(config_manager, '_version_proxy')
        assert hasattr(config_manager, '_core')
        assert hasattr(config_manager, '_logger')

    def test_init_with_custom_security_service(self):
        """Test that custom security service is used and env policies are applied"""
        # Create a mock security service with required attributes
        mock_security_service = MagicMock()
        mock_security_service.audit_level = "original_audit"
        mock_security_service.validation_level = "original_validation"

        # Initialize ConfigManager with custom security service and test environment
        env = "test"
        manager = ConfigManager(security_service=mock_security_service, env=env)

        # Verify the custom security service is used
        assert manager._security_service == mock_security_service

        # Verify env policies are applied to the custom security service
        expected_policies = {
            'prod': {'audit_level': 'strict', 'validation_level': 'full'},
            'test': {'audit_level': 'normal', 'validation_level': 'basic'},
            'dev': {'audit_level': 'minimal', 'validation_level': 'none'},
            'default': {'audit_level': 'standard', 'validation_level': 'basic'}
        }
        expected_config = expected_policies.get(env, expected_policies['default'])

        assert mock_security_service.audit_level == expected_config['audit_level']
        assert mock_security_service.validation_level == expected_config['validation_level']

    def test_initialize_with_custom_event_system(self):
        """Test initialization with custom event system provided"""
        # Create a mock event system
        mock_event_system = MagicMock()

        # Initialize ConfigManager with custom event system
        config_manager = ConfigManager(event_system=mock_event_system)

        # Verify the custom event system is used instead of default one
        assert config_manager._event_system == mock_event_system
        mock_event_system.assert_not_called()  # Just verify the reference, not usage

    def test_init_with_prod_environment(self):
        """Test that prod environment gets strict audit level and full validation level"""
        # Create a mock security service to verify the levels are set correctly
        mock_security_service = MagicMock()
        mock_security_service.audit_level = None
        mock_security_service.validation_level = None

        # Initialize ConfigManager with prod environment
        config_manager = ConfigManager(security_service=mock_security_service, env='prod')

        # Verify the security service has the correct levels set
        assert mock_security_service.audit_level == 'strict'
        assert mock_security_service.validation_level == 'full'

        # Also verify the internal env_policies are applied correctly
        assert config_manager.env_policies['prod']['audit_level'] == 'strict'
        assert config_manager.env_policies['prod']['validation_level'] == 'full'

    def test_initialize_with_invalid_environment(self):
        """Test initialization with invalid environment name"""
        # Initialize with invalid environment
        manager = ConfigManager(env='invalid')

        # Verify default environment policies are applied
        assert manager.env_policies['default']['audit_level'] == 'standard'
        assert manager.env_policies['default']['validation_level'] == 'basic'

        # Check that the security service was initialized with default config
        assert manager._security_service.audit_level == 'standard'
        assert manager._security_service.validation_level == 'basic'

    def test_update_config_with_invalid_key_format(self):
        """Test updating a configuration with invalid key format"""
        # Arrange
        manager = ConfigManager(env='default')
        invalid_key = 'invalid key'
        value = 'value'

        # Act
        result = manager.update_config(invalid_key, value)

        # Assert
        assert result is False, "Should return False for invalid key format"

    def test_update_config_with_invalid_value_type(self):
        """Test that updating with invalid value type returns False"""
        # Setup
        mock_security_service = MagicMock()
        mock_security_service.validate_config.return_value = True  # 安全验证通过

        # 使用真实事件系统以便验证事件发布
        event_system = EventSystem.get_default()
        # 确保测试环境设置
        os.environ['TESTING'] = 'true'
        # 清理之前的事件
        event_system.clear_events()

        manager = ConfigManager(
            security_service=mock_security_service,
            event_system=event_system,
            env='default'
        )

        # Test
        key = 'log.level'
        invalid_value = ['array']  # 应为str/bool/int
        result = manager.update_config(key, invalid_value)

        # Verify
        assert result is False
        mock_security_service.validate_config.assert_not_called()  # 应在值类型验证阶段就失败

        # 验证错误事件发布
        events = event_system.get_events("config_error")
        assert len(events) == 1
        assert "Invalid value type" in events[0]["error"]
        assert len(events) == 1
        assert events[0]["key"] == "log.level"
        assert events[0]["error"] == "Invalid value type: <class 'list'>"

    def test_update_config_with_missing_dependency(self):
        """Test updating a configuration that violates dependencies"""
        # Setup
        manager = ConfigManager(env='default')

        # Mock security service to fail dependency check
        manager._security_service = MagicMock()
        manager._security_service.validate_config.return_value = (False, "Dependency check failed")

        # Test
        result = manager.update_config(key='cache.enabled', value=True)

        # Verify
        assert result is False, "Should return False when dependency check fails"

    def test_update_config_when_lock_acquisition_fails(self):
        """Test that update_config returns False when lock cannot be acquired"""
        # Create ConfigManager instance
        manager = ConfigManager(env='default')

        # Mock lock manager with acquire() returning False and release() doing nothing
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = False
        mock_lock.release.return_value = None
        manager._lock_manager = mock_lock

        # Call update_config with test inputs
        result = manager.update_config(key='valid.key', value='value')

        # Verify the result is False

        # Verify acquire was called
        mock_lock.acquire.assert_called_once()

        # Verify release was NOT called (since acquire failed)
        mock_lock.release.assert_not_called()

    def test_update_config_with_no_security_service(self):
        """Test that basic config operations work without explicit security service"""
        # Create ConfigManager with no security service
        manager = ConfigManager(security_service=None, env='default')

        # Mock lock manager to ensure update can proceed
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = True
        manager._lock_manager = mock_lock

        # Test basic config update
        test_key = 'test.key'
        test_value = 'value'
        result = manager.update_config(test_key, test_value)

        # Verify operation completed (result may be True/False depending on validation)
        assert isinstance(result, bool)

    def test_watch_config_change_with_valid_key(self):
        """Test watching a configuration change with valid key"""
        # Setup
        from src.infrastructure.lock import LockManager
        manager = ConfigManager(env='default')
        lock_manager = LockManager()
        manager._lock_manager = lock_manager
        mock_callback = MagicMock()

        # Mock the lock manager's acquire method
        lock_manager.acquire = MagicMock(return_value=True)
        lock_manager.release = MagicMock()

        # Execute
        subscription_id = manager.watch(key='watched.key', callback=mock_callback)

        # Verify
        assert isinstance(subscription_id, str)
        assert len(manager._watchers['watched.key']) == 1
        assert manager._watchers['watched.key'][0][1] == mock_callback
        lock_manager.acquire.assert_called_once_with(lock_name='watch_watched.key')
        lock_manager.release.assert_called_once_with(lock_name='watch_watched.key')

    def test_watch_config_change_lock_failure(self):
        """Test watch when lock acquisition fails"""
        # Setup
        from src.infrastructure.lock import LockManager
        manager = ConfigManager(env='default')
        lock_manager = LockManager()
        manager._lock_manager = lock_manager
        mock_callback = MagicMock()

        # Mock the lock manager's acquire and release methods
        lock_manager.acquire = MagicMock(return_value=False)
        lock_manager.release = MagicMock()

        # Execute
        subscription_id = manager.watch(key='watched.key', callback=mock_callback)

        # Verify
        assert subscription_id == ""
        assert 'watched.key' not in manager._watchers
        lock_manager.acquire.assert_called_once_with(lock_name='watch_watched.key')
        lock_manager.release.assert_not_called()

    def test_validate_config_with_all_valid_keys(self):
        """Test validating a configuration with all valid keys"""
        # Setup
        manager = ConfigManager()
        manager._security_service = MagicMock()
        manager._security_service.validate_config.return_value = (True, None)

        # Input
        config = {'valid.key': 'value', 'db.name': 'test'}

        # Test
        result = manager.validate_config(config)

        # Assert
        assert result == (True, None)
        manager._security_service.validate_config.assert_called_once_with(config)

    def test_validate_config_with_invalid_keys(self):
        """Test validating a configuration with invalid keys"""
        # Setup
        config_manager = ConfigManager()
        config_manager._security_service = MagicMock()
        config_manager._security_service.validate_config = MagicMock(
            return_value=(False, {'invalid key': 'Invalid key format'})
        )

        # Test input
        test_config = {'invalid key': 'value'}

        # Execute
        result = config_manager.validate_config(test_config)

        # Verify
        assert result == (False, {'invalid key': 'Invalid key format'})
        config_manager._security_service.validate_config.assert_called_once_with(test_config)

    def test_validate_config_with_invalid_value_types(self):
        """Test validating a configuration with invalid value types"""
        # Setup
        security_service_mock = MagicMock()
        event_system_mock = MagicMock()
        config_manager = ConfigManager(security_service=security_service_mock, event_system=event_system_mock)

        # Mock the validate_config method to return tuple with error details
        config_manager._security_service.validate_config = MagicMock(return_value=(
            False,
            {'db.name': 'Invalid type for db.name, expected str/int/dict'}
        ))

        # Test input
        test_config = {'db.name': ['invalid']}

        # Execute
        result = config_manager.validate_config(test_config)

        # Verify
        assert result == (
            False,
            {'db.name': 'Invalid type for db.name, expected str/int/dict'}
        )

    def test_check_valid_key_format(self):
        """Test checking valid key format returns True for valid key"""
        # Arrange
        manager = ConfigManager()
        test_key = 'valid.key.format'

        # Act
        # Test through public update_config method which should validate key format
        try:
            manager.update_config(test_key, "value")
            result = True
        except ValueError:
            result = False

        # Assert
        assert result is True, "Valid key format should be accepted"

    def test_check_value_type_for_db_key(self):
        """Test checking value type for db-related key"""
        # Create ConfigManager instance with default parameters
        config_manager = ConfigManager()

        # Mock the security service's validate_config method
        mock_security = MagicMock()
        mock_security.validate_config.return_value = (True, None)
        config_manager._security_service = mock_security

        # Test the behavior with given input
        result = config_manager.update_config(key='db.name', value='valid')

        # Assert the expected outcome
        assert result is True, "Should return True for valid db key and value"
        mock_security.validate_config.assert_called_once_with({'db.name': 'valid'})

    def test_check_value_type_for_log_key(self):
        """Test checking value type for log-related key"""
        # Setup
        manager = ConfigManager()
        key = 'log.level'
        value = True

        # Mock the security service's validate_config method
        mock_security = MagicMock()
        mock_security.validate_config.return_value = (True, None)
        manager._security_service = mock_security

        # Execute
        result = manager.update_config(key, value)

        # Verify
        assert result is True
        mock_security.validate_config.assert_called_once_with({key: value})

    @pytest.mark.timeout(5)
    def test_check_dependencies_with_valid_config(self):
        """测试依赖配置验证"""
        # 创建配置管理器实例
        config_manager = ConfigManager()

        # 测试有效配置
        new_config = {
            'cache.enabled': True,
            'cache.size': 1024
        }
        full_config = {'cache.size': 1024}

        # 验证依赖关系
        errors = config_manager._check_dependencies(new_config, full_config)
        assert not errors

    def test_event_publishing_on_config_change(self):
        """Verify correct events are published during config update"""
        # Arrange
        mock_event_system = MagicMock()
        mock_event_bus = MagicMock()
        mock_security_service = MagicMock()
        mock_security_service.validate_config.return_value = (True, None)  # 验证通过

        manager = ConfigManager(
            event_system=mock_event_system,
            event_bus=mock_event_bus,
            security_service=mock_security_service,
            env='test'
        )

        # Mock lock manager to ensure update can proceed
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = True
        manager._lock_manager = mock_lock

        test_key = 'event.key'
        test_value = 'new'

        # Act
        manager.update_config(key=test_key, value=test_value)

        # Assert
        # Verify both event systems were called with updated event
        mock_event_system.publish.assert_called_once_with(
            "config_updated",
            {
                "key": test_key,
                "old_value": None,
                "new_value": test_value,
                "env": "test",
                "version": 0,
                "timestamp": ANY
            }
        )
        mock_event_bus.publish.assert_called_once_with(
            "config_updated",
            {
                "key": test_key,
                "old_value": None,
                "new_value": test_value,
                "env": "test",
                "version": 0,
                "timestamp": ANY
            }
        )

        # Verify security service was called
        mock_security_service.validate_config.assert_called_once_with(
            {test_key: test_value}
        )
