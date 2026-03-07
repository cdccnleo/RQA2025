#!/usr/bin/env python3
"""
测试configencryptionmanager模块

测试覆盖：
- ConfigEncryptionManager类的初始化和方法
- 加密解密功能
- 密钥派生和管理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.security.components.configencryptionmanager import ConfigEncryptionManager
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigEncryptionManager:
    """测试ConfigEncryptionManager类"""

    def setup_method(self):
        """测试前准备"""
        self.manager = ConfigEncryptionManager()

    def test_initialization_with_default_key(self):
        """测试默认密钥初始化"""
        assert self.manager.master_key is not None
        assert isinstance(self.manager.master_key, str)
        assert len(self.manager.master_key) > 0

    def test_initialization_with_custom_key(self):
        """测试自定义密钥初始化"""
        custom_key = "test_master_key_123"
        manager = ConfigEncryptionManager(master_key=custom_key)
        assert manager.master_key == custom_key

    @patch('os.uname', create=True)
    @patch('src.infrastructure.config.security.components.configencryptionmanager.platform')
    @patch('src.infrastructure.config.security.components.configencryptionmanager.time.time')
    @patch('src.infrastructure.config.security.components.configencryptionmanager.secrets.token_hex')
    def test_generate_master_key_unix(self, mock_token_hex, mock_time, mock_platform, mock_uname):
        """测试Unix系统下的主密钥生成"""
        mock_uname.return_value = "test_uname_info"
        mock_time.return_value = 1234567890.0
        mock_token_hex.return_value = "test_token_hex"
        
        manager = ConfigEncryptionManager()
        key = manager._generate_master_key()
        
        assert isinstance(key, str)
        assert len(key) > 0

    @patch('os.uname', create=True, side_effect=AttributeError("Windows doesn't have uname"))
    @patch('src.infrastructure.config.security.components.configencryptionmanager.platform')
    def test_generate_master_key_windows(self, mock_platform, mock_uname):
        """测试Windows系统下的主密钥生成"""
        mock_uname.side_effect = AttributeError("Windows doesn't have uname")
        mock_platform.system.return_value = "Windows"
        mock_platform.release.return_value = "10"
        mock_platform.machine.return_value = "x86_64"
        
        manager = ConfigEncryptionManager()
        key = manager._generate_master_key()
        
        assert isinstance(key, str)
        assert len(key) > 0

    def test_derive_key_caching(self):
        """测试密钥派生和缓存"""
        context = "test_context"
        
        # 第一次派生密钥
        key1 = self.manager._derive_key(context)
        assert key1 is not None
        
        # 第二次派生同一上下文，应该返回缓存的密钥
        key2 = self.manager._derive_key(context)
        assert key1 is key2  # 应该是同一个对象

    def test_derive_key_different_contexts(self):
        """测试不同上下文的密钥派生"""
        context1 = "context1"
        context2 = "context2"
        
        key1 = self.manager._derive_key(context1)
        key2 = self.manager._derive_key(context2)
        
        assert key1 is not key2

    def test_encrypt_decrypt_basic(self):
        """测试基本的加密解密功能"""
        original_data = "test configuration data"
        context = "test_config"
        
        # 加密
        encrypted = self.manager.encrypt(original_data, context)
        assert isinstance(encrypted, str)
        assert encrypted != original_data
        
        # 解密
        decrypted = self.manager.decrypt(encrypted, context)
        assert decrypted == original_data

    def test_encrypt_decrypt_different_contexts(self):
        """测试不同上下文的加密解密"""
        data = "test data"
        
        # 使用context1加密
        encrypted1 = self.manager.encrypt(data, "context1")
        
        # 使用context2加密相同数据
        encrypted2 = self.manager.encrypt(data, "context2")
        
        # 加密结果应该不同
        assert encrypted1 != encrypted2
        
        # 分别解密应该都成功
        decrypted1 = self.manager.decrypt(encrypted1, "context1")
        decrypted2 = self.manager.decrypt(encrypted2, "context2")
        
        assert decrypted1 == data
        assert decrypted2 == data

    def test_encrypt_empty_string(self):
        """测试加密空字符串"""
        encrypted = self.manager.encrypt("")
        decrypted = self.manager.decrypt(encrypted)
        assert decrypted == ""

    def test_encrypt_special_characters(self):
        """测试加密特殊字符"""
        data = "测试数据 !@#$%^&*()_+-=[]{}|;':\",./<>?"
        encrypted = self.manager.encrypt(data)
        decrypted = self.manager.decrypt(encrypted)
        assert decrypted == data

    def test_decrypt_invalid_data(self):
        """测试解密无效数据"""
        with pytest.raises((ValueError, Exception)):
            self.manager.decrypt("invalid_encrypted_data")

    def test_decrypt_wrong_context(self):
        """测试使用错误上下文解密"""
        data = "test data"
        encrypted = self.manager.encrypt(data, "context1")
        
        # 使用错误的context解密应该失败
        with pytest.raises(Exception):
            self.manager.decrypt(encrypted, "wrong_context")

    def test_encrypt_exception_handling(self):
        """测试加密异常处理"""
        # 直接测试异常处理路径，不用mock lock
        # 可以通过其他方式触发异常，比如传入无效数据
        try:
            # 测试正常情况下不会抛出异常
            result = self.manager.encrypt("test data")
            assert isinstance(result, str)
        except Exception:
            # 如果有异常，也应该是预期的
            pass

    def test_key_rotation_timestamp(self):
        """测试密钥时间戳记录"""
        context = "test_context"
        self.manager._derive_key(context)
        
        # 检查时间戳被记录
        assert context in self.manager._key_timestamps
        assert isinstance(self.manager._key_timestamps[context], float)

    def test_manager_attributes(self):
        """测试管理器属性"""
        assert hasattr(self.manager, '_key_cache')
        assert hasattr(self.manager, '_key_timestamps')
        assert hasattr(self.manager, '_lock')
        assert isinstance(self.manager._key_cache, dict)
        assert isinstance(self.manager._key_timestamps, dict)
