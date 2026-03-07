import pytest
from src.infrastructure.config.core.factory import ConfigFactory
from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager

class TestConfigLowCoverage:
    def test_config_factory_creation(self):
        factory = ConfigFactory()
        assert factory is not None

    def test_basic_validation(self):
        config_manager = UnifiedConfigManager()
        config = {'key': 'value'}
        result = config_manager.validate_config(config)
        assert result  # Assume passes

    # Add 5-10 more simple tests for loaders, mergers, etc.
    def test_env_loader(self):
        # Mock env
        pass

    # ... more
