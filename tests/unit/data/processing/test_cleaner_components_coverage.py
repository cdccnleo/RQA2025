import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from unittest.mock import Mock, patch
import logging

from src.data.processing.cleaner_components import (
    ComponentFactory,
    CleanerComponent,
    CleanerComponentFactory
)


def test_cleaner_component_factory_create_component_exception(monkeypatch):
    """测试ComponentFactory.create_component的异常处理（24-26行）"""
    factory = ComponentFactory()
    
    # Mock _create_component_instance to raise exception
    def failing_create(component_type, config):
        raise Exception("Cannot create component")
    
    monkeypatch.setattr(factory, '_create_component_instance', failing_create)
    
    # Create component - should handle exception
    result = factory.create_component('test_type', {})
    assert result is None


def test_cleaner_component_factory_create_component_initialize_false(monkeypatch):
    """测试ComponentFactory.create_component中initialize返回False（22行）"""
    factory = ComponentFactory()
    
    # Mock _create_component_instance to return component with initialize=False
    mock_component = Mock()
    mock_component.initialize.return_value = False
    
    def mock_create(component_type, config):
        return mock_component
    
    monkeypatch.setattr(factory, '_create_component_instance', mock_create)
    
    # Create component - should return None when initialize returns False
    result = factory.create_component('test_type', {})
    assert result is None


def test_cleaner_component_factory_create_component_success(monkeypatch):
    """测试ComponentFactory.create_component成功返回component（22行）"""
    factory = ComponentFactory()
    
    # Mock _create_component_instance to return component with initialize=True
    mock_component = Mock()
    mock_component.initialize.return_value = True
    
    def mock_create(component_type, config):
        return mock_component
    
    monkeypatch.setattr(factory, '_create_component_instance', mock_create)
    
    # Create component - should return component when initialize returns True
    result = factory.create_component('test_type', {})
    assert result is not None
    assert result == mock_component


def test_cleaner_component_factory_create_component_component_none(monkeypatch):
    """测试ComponentFactory.create_component中component为None（22行）"""
    factory = ComponentFactory()
    
    # Mock _create_component_instance to return None
    def mock_create(component_type, config):
        return None
    
    monkeypatch.setattr(factory, '_create_component_instance', mock_create)
    
    # Create component - should return None when component is None
    result = factory.create_component('test_type', {})
    assert result is None


def test_cleaner_component_process_exception(monkeypatch):
    """测试CleanerComponent.process的异常处理（104-105行）"""
    component = CleanerComponent(cleaner_id=1)
    
    # Mock component_name to raise exception when accessed
    original_name = component.component_name
    
    # Create a property that raises exception
    class FailingName:
        def __str__(self):
            raise Exception("Name access failed")
    
    failing_name = FailingName()
    
    # Use setattr to replace the attribute
    setattr(component, 'component_name', failing_name)
    
    try:
        # Process should handle exception
        result = component.process({'test': 'data'})
        assert result is not None
        assert result['status'] == 'error'
        assert 'error' in result
    finally:
        # Restore original name
        setattr(component, 'component_name', original_name)

