#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施核心模块单元测试
专注于现有可用的基础设施模块
"""
import pytest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 导入现有的基础设施模块
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.config.factory import ConfigFactory
from src.infrastructure.config.exceptions import ConfigError
from src.infrastructure.config.simple_cache import SimpleCache
from src.infrastructure.config.security_service import SecurityService
from src.infrastructure.config.paths import ConfigPaths
from src.infrastructure.config.schema import ConfigSchema

class TestInfrastructureCore:
    """基础设施核心测试"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """创建临时配置目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config_manager(self):
        """创建配置管理器实例"""
        return ConfigManager()
    
    def test_config_manager_basic(self, config_manager):
        """测试配置管理器基础功能"""
        # 测试配置设置和获取
        config_manager.set("test.key", "test_value")
        assert config_manager.get("test.key") == "test_value"
        
        # 测试默认值
        assert config_manager.get("nonexistent.key", "default") == "default"
        
        # 测试嵌套配置
        config_manager.set("nested.config.value", 123)
        assert config_manager.get("nested.config.value") == 123
    
    def test_config_manager_from_dict(self, config_manager):
        """测试从字典加载配置"""
        config_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db"
            },
            "redis": {
                "host": "localhost",
                "port": 6379
            }
        }
        
        config_manager.load_from_dict(config_data)
        
        assert config_manager.get("database.host") == "localhost"
        assert config_manager.get("database.port") == 5432
        assert config_manager.get("redis.host") == "localhost"
        assert config_manager.get("redis.port") == 6379
    
    def test_config_manager_from_file(self, config_manager, temp_config_dir):
        """测试从文件加载配置"""
        config_file = os.path.join(temp_config_dir, "test_config.json")
        config_data = {
            "app": {"name": "RQA2025", "version": "1.0.0"},
            "features": {"enabled": True}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config_manager.load_from_file(config_file)
        
        assert config_manager.get("app.name") == "RQA2025"
        assert config_manager.get("app.version") == "1.0.0"
        assert config_manager.get("features.enabled") is True
    
    def test_config_factory(self):
        """测试配置工厂"""
        factory = ConfigFactory()
        
        # 测试创建配置管理器
        config_manager = factory.create_config_manager()
        assert isinstance(config_manager, ConfigManager)
        
        # 测试创建缓存服务
        cache_service = factory.create_cache_service()
        assert cache_service is not None
    
    def test_simple_cache(self):
        """测试简单缓存"""
        cache = SimpleCache()
        
        # 测试缓存设置和获取
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # 测试缓存过期
        cache.set("expire_key", "expire_value", ttl=1)
        time.sleep(2)
        assert cache.get("expire_key") is None
        
        # 测试缓存删除
        cache.set("delete_key", "delete_value")
        cache.delete("delete_key")
        assert cache.get("delete_key") is None
    
    def test_security_service(self):
        """测试安全服务"""
        security_service = SecurityService()
        
        # 测试数据加密
        sensitive_data = "敏感信息：密码123456"
        encrypted_data = security_service.encrypt_config(sensitive_data)
        assert encrypted_data != sensitive_data
        
        # 测试数据解密
        decrypted_data = security_service.decrypt_config(encrypted_data)
        assert decrypted_data == sensitive_data
    
    def test_config_paths(self):
        """测试配置路径"""
        paths = ConfigPaths()
        
        # 测试获取配置目录
        config_dir = paths.get_config_dir()
        assert os.path.exists(config_dir)
        
        # 测试获取日志目录
        log_dir = paths.get_log_dir()
        assert os.path.exists(log_dir)
        
        # 测试获取数据目录
        data_dir = paths.get_data_dir()
        assert os.path.exists(data_dir)
    
    def test_config_schema(self):
        """测试配置模式"""
        schema = ConfigSchema()
        
        # 测试模式验证
        valid_config = {
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        assert schema.validate(valid_config) is True
        
        # 测试无效配置
        invalid_config = {
            "database": {
                "port": "invalid_port"  # 应该是整数
            }
        }
        
        # 应该抛出验证错误
        with pytest.raises(Exception):
            schema.validate(invalid_config)
    
    def test_config_error_handling(self, config_manager):
        """测试配置错误处理"""
        # 测试文件不存在错误
        with pytest.raises(ConfigError):
            config_manager.load_from_file("nonexistent_file.json")
        
        # 测试无效JSON错误
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()
            with pytest.raises(ConfigError):
                config_manager.load_from_file(f.name)
            os.unlink(f.name)
    
    def test_config_performance(self, config_manager):
        """测试配置性能"""
        import time
        
        # 测试大量配置设置性能
        start_time = time.time()
        for i in range(1000):
            config_manager.set(f"key_{i}", f"value_{i}")
        set_time = time.time() - start_time
        
        # 性能要求：设置1000个配置项应在0.1秒内完成
        assert set_time < 0.1
        
        # 测试大量配置获取性能
        start_time = time.time()
        for i in range(1000):
            config_manager.get(f"key_{i}")
        get_time = time.time() - start_time
        
        # 性能要求：获取1000个配置项应在0.1秒内完成
        assert get_time < 0.1
    
    def test_config_concurrency(self, config_manager):
        """测试配置并发访问"""
        import threading
        
        # 并发设置配置
        def set_config():
            for i in range(100):
                config_manager.set(f"concurrent_key_{i}", f"value_{i}")
        
        # 启动多个线程
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=set_config)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有配置都被设置
        for i in range(100):
            assert config_manager.get(f"concurrent_key_{i}") == f"value_{i}"
    
    def test_config_validation(self, config_manager):
        """测试配置验证"""
        # 测试有效配置
        valid_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db"
            }
        }
        
        config_manager.load_from_dict(valid_config)
        assert config_manager.validate() is True
        
        # 测试配置更新
        config_manager.set("database.port", 5433)
        assert config_manager.get("database.port") == 5433
    
    def test_config_export_import(self, config_manager):
        """测试配置导出和导入"""
        # 设置一些配置
        config_data = {
            "export": {"data": "test_value"},
            "nested": {"level1": {"level2": "nested_value"}}
        }
        config_manager.load_from_dict(config_data)
        
        # 导出配置
        exported_data = config_manager.export_config()
        assert "export" in exported_data
        assert "nested" in exported_data
        
        # 创建新的配置管理器并导入
        new_config_manager = ConfigManager()
        new_config_manager.import_config(exported_data)
        
        assert new_config_manager.get("export.data") == "test_value"
        assert new_config_manager.get("nested.level1.level2") == "nested_value"
    
    def test_config_backup_restore(self, config_manager):
        """测试配置备份和恢复"""
        # 设置原始配置
        original_config = {"backup": {"test": True}}
        config_manager.load_from_dict(original_config)
        
        # 创建备份
        backup_data = config_manager.create_backup()
        assert "backup" in backup_data
        
        # 修改配置
        config_manager.set("backup.test", False)
        assert config_manager.get("backup.test") is False
        
        # 恢复备份
        config_manager.restore_from_backup(backup_data)
        assert config_manager.get("backup.test") is True
    
    def test_config_hot_reload(self, config_manager, temp_config_dir):
        """测试配置热重载"""
        config_file = os.path.join(temp_config_dir, "hot_reload.json")
        
        # 创建初始配置
        initial_config = {"hot_reload": {"enabled": True}}
        with open(config_file, 'w') as f:
            json.dump(initial_config, f)
        
        config_manager.load_from_file(config_file)
        assert config_manager.get("hot_reload.enabled") is True
        
        # 修改配置文件
        updated_config = {"hot_reload": {"enabled": False}}
        with open(config_file, 'w') as f:
            json.dump(updated_config, f)
        
        # 重新加载配置
        config_manager.reload()
        assert config_manager.get("hot_reload.enabled") is False
    
    def test_config_encryption(self, config_manager):
        """测试配置加密"""
        security_service = SecurityService()
        
        # 加密敏感配置
        sensitive_config = {
            "database": {
                "password": "secret123",
                "api_key": "key456"
            }
        }
        
        encrypted_config = security_service.encrypt_config(sensitive_config)
        assert encrypted_config != sensitive_config
        
        # 解密配置
        decrypted_config = security_service.decrypt_config(encrypted_config)
        assert decrypted_config["database"]["password"] == "secret123"
        assert decrypted_config["database"]["api_key"] == "key456"
    
    def test_config_memory_management(self, config_manager):
        """测试配置内存管理"""
        # 设置大量配置
        for i in range(10000):
            config_manager.set(f"memory_test_key_{i}", f"memory_test_value_{i}")
        
        # 验证内存使用在合理范围内
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # 内存使用应该小于100MB
        assert memory_usage < 100
        
        # 清理配置
        for i in range(10000):
            config_manager.delete(f"memory_test_key_{i}")
    
    def test_config_error_recovery(self, config_manager):
        """测试配置错误恢复"""
        # 模拟配置系统错误
        with patch.object(config_manager, '_set_config', side_effect=Exception("设置错误")):
            # 应该能够优雅地处理错误
            config_manager.set("error_test", "value")
        
        # 系统应该继续正常工作
        config_manager.set("recovery_test", "value")
        assert config_manager.get("recovery_test") == "value" 