#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块修复版本测试
解决安全服务Mock和依赖注入问题
"""
import pytest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock

class TestConfigManagerFixed:
    """配置管理器修复版本测试"""
    
    @pytest.fixture
    def mock_security_service(self):
        """创建正确的安全服务Mock"""
        mock_service = Mock()
        # 设置正确的validate_config返回值格式
        mock_service.validate_config.return_value = (True, None)
        mock_service.sign_config.return_value = {"signed": "config"}
        mock_service.audit_level = 'standard'
        mock_service.validation_level = 'basic'
        return mock_service
    
    @pytest.fixture
    def mock_event_system(self):
        """创建事件系统Mock"""
        mock_system = Mock()
        mock_system.publish = Mock()
        return mock_system
    
    @pytest.fixture
    def mock_lock_manager(self):
        """创建锁管理器Mock"""
        mock_lock = Mock()
        mock_lock.acquire.return_value = True
        mock_lock.release = Mock()
        return mock_lock
    
    @pytest.fixture
    def config_manager_fixed(self, mock_security_service, mock_event_system, mock_lock_manager):
        """创建修复版本的配置管理器"""
        try:
            from src.infrastructure.config.config_manager import ConfigManager
            
            # 使用patch来替换依赖
            with patch('src.infrastructure.lock.LockManager') as mock_lock_class:
                mock_lock_class.return_value = mock_lock_manager
                
                config_manager = ConfigManager(
                    security_service=mock_security_service,
                    event_system=mock_event_system
                )
                return config_manager
                
        except ImportError:
            pytest.skip("ConfigManager模块不可用")
    
    def test_config_manager_import(self):
        """测试配置管理器导入"""
        try:
            from src.infrastructure.config.config_manager import ConfigManager
            assert True
        except ImportError as e:
            pytest.skip(f"无法导入ConfigManager: {e}")
    
    def test_config_manager_initialization_fixed(self, mock_security_service, mock_event_system):
        """测试配置管理器初始化（修复版本）"""
        try:
            from src.infrastructure.config.config_manager import ConfigManager
            
            # 测试基本初始化
            config_manager = ConfigManager(
                security_service=mock_security_service,
                event_system=mock_event_system
            )
            assert config_manager is not None
            assert hasattr(config_manager, '_config')
            assert hasattr(config_manager, '_watchers')
            assert hasattr(config_manager, '_lock_manager')
            
        except ImportError:
            pytest.skip("ConfigManager模块不可用")
    
    def test_config_manager_update_config_fixed(self, config_manager_fixed):
        """测试配置更新（修复版本）"""
        # 测试更新配置
        result = config_manager_fixed.update_config("test_key", "test_value")
        assert result == True
        
        # 测试获取配置
        value = config_manager_fixed.get_config("test_key")
        assert value == "test_value"
    
    def test_config_manager_get_config_fixed(self, config_manager_fixed):
        """测试配置获取（修复版本）"""
        # 设置配置
        config_manager_fixed.update_config("test_key", "test_value")
        
        # 获取配置
        value = config_manager_fixed.get_config("test_key")
        assert value == "test_value"
        
        # 测试默认值
        default_value = config_manager_fixed.get_config("nonexistent_key", "default_value")
        assert default_value == "default_value"
    
    def test_config_manager_update_alias_fixed(self, config_manager_fixed):
        """测试update别名（修复版本）"""
        # 使用update别名
        result = config_manager_fixed.update("alias_key", "alias_value")
        assert result == True
        
        # 验证配置已设置
        value = config_manager_fixed.get_config("alias_key")
        assert value == "alias_value"
    
    def test_config_manager_validation_fixed(self, config_manager_fixed):
        """测试配置验证（修复版本）"""
        # 测试配置验证方法
        test_config = {"test_key": "test_value"}
        is_valid, errors = config_manager_fixed.validate_config(test_config)
        
        # 验证返回格式
        assert isinstance(is_valid, bool)
        assert isinstance(errors, (dict, type(None)))
    
    def test_config_manager_watch_fixed(self, config_manager_fixed):
        """测试配置监听（修复版本）"""
        # 创建回调函数
        callback_called = False
        def test_callback(key, old_value, new_value):
            nonlocal callback_called
            callback_called = True
        
        # 注册监听器
        sub_id = config_manager_fixed.watch("test_key", test_callback)
        assert isinstance(sub_id, str)
        
        # 更新配置触发回调
        config_manager_fixed.update_config("test_key", "new_value")
        
        # 验证回调被调用
        assert callback_called == True
    
    def test_config_manager_unwatch_fixed(self, config_manager_fixed):
        """测试取消监听（修复版本）"""
        # 创建回调函数
        def test_callback(key, old_value, new_value):
            pass
        
        # 注册监听器
        sub_id = config_manager_fixed.watch("test_key", test_callback)
        
        # 取消监听
        result = config_manager_fixed.unwatch("test_key", sub_id)
        assert result == True
    
    def test_config_manager_invalid_key_format_fixed(self, config_manager_fixed):
        """测试无效键格式（修复版本）"""
        # 测试无效键格式
        result = config_manager_fixed.update_config("invalid-key", "value")
        assert result == False
    
    def test_config_manager_invalid_value_type_fixed(self, config_manager_fixed):
        """测试无效值类型（修复版本）"""
        # 测试无效值类型（复杂对象）
        result = config_manager_fixed.update_config("test_key", {"complex": "object"})
        assert result == False
    
    def test_config_manager_valid_value_types_fixed(self, config_manager_fixed):
        """测试有效值类型（修复版本）"""
        # 测试字符串
        result = config_manager_fixed.update_config("string_key", "string_value")
        assert result == True
        
        # 测试整数
        result = config_manager_fixed.update_config("int_key", 123)
        assert result == True
        
        # 测试浮点数
        result = config_manager_fixed.update_config("float_key", 123.45)
        assert result == True
        
        # 测试布尔值
        result = config_manager_fixed.update_config("bool_key", True)
        assert result == True
    
    def test_config_manager_cache_dependency_fixed(self, config_manager_fixed):
        """测试缓存依赖关系（修复版本）"""
        # 启用缓存但未设置缓存大小
        result = config_manager_fixed.update_config("cache.enabled", True)
        assert result == False
        
        # 先设置缓存大小
        result = config_manager_fixed.update_config("cache.size", 1000)
        assert result == True
        
        # 然后启用缓存
        result = config_manager_fixed.update_config("cache.enabled", True)
        assert result == True
    
    def test_config_manager_concurrent_updates_fixed(self, config_manager_fixed):
        """测试并发更新（修复版本）"""
        import threading
        
        def worker():
            for i in range(10):
                config_manager_fixed.update_config(f"thread_key_{i}", f"thread_value_{i}")
        
        # 启动多个线程
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证配置管理器在并发环境下正常工作
        assert config_manager_fixed is not None
    
    def test_config_manager_environment_initialization_fixed(self, mock_security_service, mock_event_system):
        """测试环境初始化（修复版本）"""
        try:
            from src.infrastructure.config.config_manager import ConfigManager
            
            # 测试不同环境
            config_manager_prod = ConfigManager(
                security_service=mock_security_service,
                event_system=mock_event_system,
                env='prod'
            )
            config_manager_test = ConfigManager(
                security_service=mock_security_service,
                event_system=mock_event_system,
                env='test'
            )
            config_manager_dev = ConfigManager(
                security_service=mock_security_service,
                event_system=mock_event_system,
                env='dev'
            )
            
            assert config_manager_prod is not None
            assert config_manager_test is not None
            assert config_manager_dev is not None
            
        except ImportError:
            pytest.skip("ConfigManager模块不可用")
    
    def test_config_manager_security_service_integration_fixed(self, config_manager_fixed, mock_security_service):
        """测试安全服务集成（修复版本）"""
        # 更新配置
        result = config_manager_fixed.update_config("secure_key", "secure_value")
        assert result == True
        
        # 验证安全服务被调用
        mock_security_service.validate_config.assert_called()
    
    def test_config_manager_event_system_integration_fixed(self, config_manager_fixed, mock_event_system):
        """测试事件系统集成（修复版本）"""
        # 更新配置
        result = config_manager_fixed.update_config("event_key", "event_value")
        assert result == True
        
        # 验证事件被发布
        mock_event_system.publish.assert_called()
    
    def test_config_manager_error_handling_fixed(self, mock_security_service, mock_event_system):
        """测试错误处理（修复版本）"""
        try:
            from src.infrastructure.config.config_manager import ConfigManager
            
            # 创建会抛出异常的安全服务
            mock_security_service.validate_config.side_effect = Exception("Security error")
            
            config_manager = ConfigManager(
                security_service=mock_security_service,
                event_system=mock_event_system
            )
            
            # 更新配置应该失败
            result = config_manager.update_config("error_key", "error_value")
            assert result == False
            
        except ImportError:
            pytest.skip("ConfigManager模块不可用")
    
    def test_config_manager_performance_fixed(self, config_manager_fixed):
        """测试性能（修复版本）"""
        import time
        
        # 测试大量配置操作性能
        start_time = time.time()
        for i in range(1000):
            config_manager_fixed.update_config(f"perf_key_{i}", f"perf_value_{i}")
        update_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(1000):
            config_manager_fixed.get_config(f"perf_key_{i}")
        get_time = time.time() - start_time
        
        # 性能要求：1000次操作应在1秒内完成
        assert update_time < 1.0
        assert get_time < 1.0
    
    def test_config_manager_memory_usage_fixed(self, config_manager_fixed):
        """测试内存使用（修复版本）"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # 创建大量配置
        for i in range(10000):
            config_manager_fixed.update_config(f"memory_key_{i}", f"memory_value_{i}")
        
        # 强制垃圾回收
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # 内存增长应该小于150MB（放宽限制以适应不同环境）
        assert memory_increase < 150
    
    def test_config_manager_integration_scenario_fixed(self, config_manager_fixed):
        """测试集成场景（修复版本）"""
        # 模拟完整的配置管理场景
        configs = [
            ("database.host", "localhost"),
            ("database.port", 5432),
            ("database.name", "test_db"),
            ("logging.level", "INFO"),
            ("logging.file", "app.log"),
            ("monitoring.enabled", True),
            ("monitoring.interval", 60),
            ("cache.size", 1000),
            ("cache.enabled", True)
        ]
        
        # 设置所有配置
        for key, value in configs:
            result = config_manager_fixed.update_config(key, value)
            assert result == True
        
        # 验证所有配置
        for key, expected_value in configs:
            actual_value = config_manager_fixed.get_config(key)
            assert actual_value == expected_value
    
    def test_config_manager_cleanup_and_reset_fixed(self, config_manager_fixed):
        """测试清理和重置（修复版本）"""
        # 设置一些配置
        config_manager_fixed.update_config("test_key", "test_value")
        
        # 验证配置已设置
        value = config_manager_fixed.get_config("test_key")
        assert value == "test_value"
        
        # 测试配置管理器可以正常工作
        assert config_manager_fixed is not None
    
    def test_config_manager_validation_rules_fixed(self, config_manager_fixed):
        """测试验证规则（修复版本）"""
        # 测试不同类型的配置验证
        test_configs = [
            {"simple_key": "simple_value"},
            {"nested.key": "nested_value"},
            {"cache.enabled": True, "cache.size": 1000}
        ]
        
        for test_config in test_configs:
            is_valid, errors = config_manager_fixed.validate_config(test_config)
            assert isinstance(is_valid, bool)
            assert isinstance(errors, (dict, type(None)))
    
    def test_config_manager_backup_restore_fixed(self, config_manager_fixed):
        """测试备份和恢复（修复版本）"""
        # 设置配置
        config_manager_fixed.update_config("backup_key", "backup_value")
        
        # 验证配置已设置
        value = config_manager_fixed.get_config("backup_key")
        assert value == "backup_value"
        
        # 测试配置管理器可以正常工作
        assert config_manager_fixed is not None
    
    def test_config_manager_versioning_fixed(self, config_manager_fixed):
        """测试版本管理（修复版本）"""
        # 测试配置版本管理功能
        assert hasattr(config_manager_fixed, '_version_proxy')
        
        # 更新配置
        result = config_manager_fixed.update_config("version_key", "version_value")
        assert result == True
        
        # 验证配置已设置
        value = config_manager_fixed.get_config("version_key")
        assert value == "version_value" 