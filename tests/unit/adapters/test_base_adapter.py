# -*- coding: utf-8 -*-
"""
基础适配器单元测试
测试 SecureConfigManager, BaseAdapter, DataAdapter 等核心类
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.adapters.base.base_adapter import (
    SecureConfigManager,
    BaseAdapter,
    DataAdapter,
    MockAdapter
)


class TestSecureConfigManager:
    """安全配置管理器测试"""

    def test_secure_config_manager_initialization(self):
        """测试安全配置管理器初始化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_file = os.path.join(temp_dir, 'test_key')

            manager = SecureConfigManager(key_file)
            assert manager.key_file == key_file
            assert hasattr(manager, 'cipher')
            assert hasattr(manager, 'encryption_key')

    def test_encrypt_decrypt_sensitive_data(self):
        """测试敏感数据加密解密"""
        manager = SecureConfigManager()

        test_data = "test_sensitive_data"
        encrypted = manager.encrypt_sensitive_data(test_data)
        decrypted = manager.decrypt_sensitive_data(encrypted)

        assert decrypted == test_data
        assert encrypted != test_data  # 确保数据被加密

    def test_secure_config_operations(self):
        """测试安全配置操作"""
        manager = SecureConfigManager()

        # 测试配置安全处理
        test_config = {"api_key": "test_key", "secret": "test_secret"}
        secured_config = manager.secure_config(test_config)

        assert isinstance(secured_config, dict)
        assert "api_key" in secured_config

    def test_secure_config_with_invalid_data(self):
        """测试无效数据的安全配置"""
        manager = SecureConfigManager()

        # 测试解密无效数据
        with pytest.raises(Exception):
            manager.decrypt_data("invalid_encrypted_data")


class TestBaseAdapter:
    """基础适配器测试"""

    def test_base_adapter_initialization(self):
        """测试基础适配器初始化"""
        config = {"test_param": "test_value"}
        adapter = BaseAdapter(config)

        assert adapter.config == config
        assert hasattr(adapter, 'logger')

    def test_base_adapter_health_check(self):
        """测试基础适配器健康检查"""
        adapter = BaseAdapter({})

        # 基础实现应该返回健康状态
        health_status = adapter.health_check()
        assert isinstance(health_status, dict)
        assert "status" in health_status

    def test_base_adapter_config_handling(self):
        """测试配置处理"""
        config = {"param1": "value1", "param2": "value2"}
        adapter = BaseAdapter(config)

        assert adapter.config == config


class TestDataAdapter:
    """数据适配器测试"""

    def test_data_adapter_initialization(self):
        """测试数据适配器初始化"""
        config = {"data_source": "test_db"}
        adapter = DataAdapter(config)

        assert adapter.config == config
        assert isinstance(adapter, BaseAdapter)

    def test_data_adapter_connect_disconnect(self):
        """测试数据适配器连接断开"""
        adapter = DataAdapter({})

        # 抽象方法应该抛出NotImplementedError
        with pytest.raises(NotImplementedError):
            adapter.connect()

        with pytest.raises(NotImplementedError):
            adapter.disconnect()

    def test_data_adapter_data_operations(self):
        """测试数据操作"""
        adapter = DataAdapter({})

        # 测试数据获取（DataAdapter是抽象类）
        with pytest.raises(NotImplementedError):
            adapter.get_data(query="test_query")


class TestMockAdapter:
    """模拟适配器测试"""

    def test_mock_adapter_initialization(self):
        """测试模拟适配器初始化"""
        config = {"mock_data": "test"}
        adapter = MockAdapter(config)

        assert adapter.config == config
        assert isinstance(adapter, BaseAdapter)

    def test_mock_adapter_data_operations(self):
        """测试模拟适配器数据操作"""
        adapter = MockAdapter({})

        # 模拟数据操作应该成功
        data = adapter.get_data(query="test_query")
        assert data is not None
        assert isinstance(data, dict)

        # MockAdapter没有save_data方法，所以不测试

    def test_mock_adapter_health_check(self):
        """测试模拟适配器健康检查"""
        adapter = MockAdapter({})

        # 先连接
        adapter.connect()

        health = adapter.health_check()
        assert health["status"] == "healthy"
        assert health["adapter_type"] == "MockAdapter"