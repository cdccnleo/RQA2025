#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚焦测试推进脚本
针对基础设施层核心模块进行测试覆盖提升，重点关注生产就绪状态
"""

from pathlib import Path


class FocusedTestAdvancement:
    """聚焦测试推进器"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"

        # 核心模块优先级（按生产就绪重要性排序）
        self.critical_modules = {
            "config_manager": {
                "file": "src/infrastructure/config/config_manager.py",
                "current_coverage": 14.32,
                "target_coverage": 85,
                "priority": "critical",
                "description": "配置管理核心模块"
            },
            "logger": {
                "file": "src/infrastructure/utils/logger.py",
                "current_coverage": 60.81,
                "target_coverage": 90,
                "priority": "critical",
                "description": "日志管理核心模块"
            },
            "error_handler": {
                "file": "src/infrastructure/error/error_handler.py",
                "current_coverage": 32.08,
                "target_coverage": 85,
                "priority": "critical",
                "description": "错误处理核心模块"
            },
            "circuit_breaker": {
                "file": "src/infrastructure/error/circuit_breaker.py",
                "current_coverage": 34.33,
                "target_coverage": 80,
                "priority": "high",
                "description": "断路器模式模块"
            },
            "thread_safe_cache": {
                "file": "src/infrastructure/cache/thread_safe_cache.py",
                "current_coverage": 25.66,
                "target_coverage": 85,
                "priority": "high",
                "description": "线程安全缓存模块"
            },
            "database_manager": {
                "file": "src/infrastructure/database/database_manager.py",
                "current_coverage": 42.86,
                "target_coverage": 75,
                "priority": "high",
                "description": "数据库管理模块"
            }
        }

    def create_focused_tests(self, module_name: str) -> None:
        """为指定模块创建聚焦测试"""
        print(f"📝 为 {module_name} 模块创建聚焦测试...")

        module_info = self.critical_modules[module_name]

        if module_name == "config_manager":
            self._create_config_manager_focused_tests()
        elif module_name == "logger":
            self._create_logger_focused_tests()
        elif module_name == "error_handler":
            self._create_error_handler_focused_tests()
        elif module_name == "circuit_breaker":
            self._create_circuit_breaker_focused_tests()
        elif module_name == "thread_safe_cache":
            self._create_cache_focused_tests()
        elif module_name == "database_manager":
            self._create_database_focused_tests()

    def _create_config_manager_focused_tests(self):
        """创建配置管理器聚焦测试"""
        test_content = '''"""
配置管理器聚焦测试
专注于配置加载、验证、热重载等核心功能
"""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.infrastructure.config.unified_manager import UnifiedConfigManager as ConfigManager

class TestConfigManagerFocused:
    """配置管理器聚焦测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.json"
        
        # 创建基础测试配置
        self.test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db"
            },
            "logging": {
                "level": "INFO",
                "file": "test.log"
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        config_manager = ConfigManager()
        assert config_manager is not None
        assert hasattr(config_manager, 'config')
    
    def test_config_loading_from_file(self):
        """测试从文件加载配置"""
        config_manager = ConfigManager()
        
        # 测试加载配置
        config = config_manager.load_config(str(self.config_file))
        assert config is not None
        
        # 验证配置内容
        assert config.get("database.host") == "localhost"
        assert config.get("database.port") == 5432
        assert config.get("logging.level") == "INFO"
    
    def test_config_get_method(self):
        """测试配置获取方法"""
        config_manager = ConfigManager()
        config_manager.load_config(str(self.config_file))
        
        # 测试获取配置值
        host = config_manager.get("database.host")
        port = config_manager.get("database.port")
        level = config_manager.get("logging.level")
        
        assert host == "localhost"
        assert port == 5432
        assert level == "INFO"
    
    def test_config_set_method(self):
        """测试配置设置方法"""
        config_manager = ConfigManager()
        config_manager.load_config(str(self.config_file))
        
        # 设置新配置
        config_manager.set("new.key", "new_value")
        config_manager.set("database.host", "new_host")
        
        # 验证设置结果
        assert config_manager.get("new.key") == "new_value"
        assert config_manager.get("database.host") == "new_host"
    
    def test_config_validation(self):
        """测试配置验证"""
        config_manager = ConfigManager()
        
        # 测试有效配置
        valid_config = {"database": {"host": "localhost"}}
        assert config_manager.validate_config(valid_config) is True
        
        # 测试无效配置
        invalid_config = {"database": {"host": None}}
        with pytest.raises(Exception):
            config_manager.validate_config(invalid_config)
    
    def test_config_error_handling(self):
        """测试配置错误处理"""
        config_manager = ConfigManager()
        
        # 测试文件不存在
        with pytest.raises(Exception):
            config_manager.load_config("nonexistent.json")
        
        # 测试无效JSON
        invalid_file = Path(self.temp_dir) / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(Exception):
            config_manager.load_config(str(invalid_file))
    
    def test_config_performance(self):
        """测试配置性能"""
        config_manager = ConfigManager()
        
        import time
        start_time = time.time()
        
        # 多次加载配置
        for _ in range(100):
            config = config_manager.load_config(str(self.config_file))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能要求：100次加载应在1秒内完成
        assert duration < 1.0
    
    def test_config_concurrency(self):
        """测试配置并发访问"""
        import threading
        import time
        
        config_manager = ConfigManager()
        config_manager.load_config(str(self.config_file))
        
        results = []
        errors = []
        
        def reader_thread():
            """读取线程"""
            try:
                for _ in range(50):
                    value = config_manager.get("database.host")
                    results.append(value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def writer_thread():
            """写入线程"""
            try:
                for i in range(10):
                    config_manager.set(f"test.key_{i}", f"value_{i}")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        # 启动多个线程
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader_thread))
        threads.append(threading.Thread(target=writer_thread))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 验证没有错误
        assert len(errors) == 0
        assert len(results) == 250  # 5个线程 * 50次读取
'''

        # 写入测试文件
        test_dir = self.tests_path / "unit" / "infrastructure" / "config"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_file = test_dir / "test_config_manager_focused.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"✅ 已创建配置管理器聚焦测试: {test_file}")

    def _create_logger_focused_tests(self):
        """创建日志管理器聚焦测试"""
        test_content = '''"""
日志管理器聚焦测试
专注于日志记录、格式化、级别控制等核心功能
"""

import pytest
import tempfile
import logging
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.infrastructure.utils.logger import get_logger, LoggerFactory, configure_logging

class TestLoggerFocused:
    """日志管理器聚焦测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logger_creation(self):
        """测试日志记录器创建"""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"
        assert logger.level == logging.INFO
    
    def test_logger_factory(self):
        """测试日志工厂"""
        factory = LoggerFactory()
        logger = factory.create_logger("factory_test")
        assert logger is not None
        assert logger.name == "factory_test"
    
    def test_logging_configuration(self):
        """测试日志配置"""
        configure_logging(
            level="DEBUG",
            log_file=str(self.log_file)
        )
        
        logger = get_logger("config_test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # 验证日志文件存在
        assert self.log_file.exists()
        
        # 验证日志内容
        with open(self.log_file, 'r') as f:
            content = f.read()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content
    
    def test_log_level_control(self):
        """测试日志级别控制"""
        # 设置INFO级别
        configure_logging(level="INFO")
        logger = get_logger("level_test")
        
        # DEBUG消息不应该被记录
        logger.debug("Debug message")
        
        with open(self.log_file, 'r') as f:
            content = f.read()
            assert "Debug message" not in content
        
        # INFO消息应该被记录
        logger.info("Info message")
        
        with open(self.log_file, 'r') as f:
            content = f.read()
            assert "Info message" in content
    
    def test_log_formatting(self):
        """测试日志格式化"""
        configure_logging(
            log_file=str(self.log_file),
            level="INFO"
        )
        
        logger = get_logger("format_test")
        logger.info("Test message")
        
        # 验证日志格式
        with open(self.log_file, 'r') as f:
            content = f.read()
            # 验证包含时间戳
            assert re.search(r'\d{4}-\d{2}-\d{2}', content)
            # 验证包含日志级别
            assert "INFO" in content
            # 验证包含模块名
            assert "format_test" in content
    
    def test_log_performance(self):
        """测试日志性能"""
        configure_logging(
            log_file=str(self.log_file),
            level="INFO"
        )
        
        logger = get_logger("performance_test")
        
        import time
        start_time = time.time()
        
        # 写入大量日志
        for i in range(1000):
            logger.info(f"Performance test message {i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能要求：1000条日志应在2秒内完成
        assert duration < 2.0
    
    def test_log_error_handling(self):
        """测试日志错误处理"""
        # 测试无效日志文件路径
        with pytest.raises(Exception):
            configure_logging(log_file="/invalid/path/test.log")
        
        # 测试无效日志级别
        with pytest.raises(ValueError):
            configure_logging(level="INVALID_LEVEL")
'''

        # 写入测试文件
        test_dir = self.tests_path / "unit" / "infrastructure" / "utils"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_file = test_dir / "test_logger_focused.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"✅ 已创建日志管理器聚焦测试: {test_file}")

    def _create_error_handler_focused_tests(self):
        """创建错误处理器聚焦测试"""
        test_content = '''"""
错误处理器聚焦测试
专注于异常捕获、处理、重试等核心功能
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.error.error_handler import ErrorHandler

class TestErrorHandlerFocused:
    """错误处理器聚焦测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.error_handler = ErrorHandler()
    
    def test_error_handler_basic(self):
        """测试错误处理器基础功能"""
        # 测试正常函数
        def normal_function():
            return "success"
        
        result = self.error_handler.handle(normal_function)
        assert result == "success"
        
        # 测试异常函数
        def error_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            self.error_handler.handle(error_function)
    
    def test_error_handler_with_fallback(self):
        """测试带回退的错误处理"""
        def error_function():
            raise ValueError("Test error")
        
        def fallback_function():
            return "fallback"
        
        result = self.error_handler.handle_with_fallback(
            error_function, 
            fallback_function
        )
        assert result == "fallback"
    
    def test_error_handler_logging(self):
        """测试错误处理器日志记录"""
        def error_function():
            raise ValueError("Logged error")
        
        with patch.object(self.error_handler, 'logger') as mock_logger:
            with pytest.raises(ValueError):
                self.error_handler.handle(error_function)
            
            # 验证错误被记录
            mock_logger.error.assert_called()
    
    def test_error_handler_performance(self):
        """测试错误处理性能"""
        def fast_function():
            return "fast"
        
        import time
        start_time = time.time()
        
        # 执行大量错误处理操作
        for _ in range(1000):
            self.error_handler.handle(fast_function)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能要求：1000次操作应在1秒内完成
        assert duration < 1.0
    
    def test_error_handler_concurrency(self):
        """测试错误处理并发"""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker_function(worker_id):
            """工作函数"""
            try:
                if worker_id % 2 == 0:
                    # 偶数线程成功
                    return f"success_{worker_id}"
                else:
                    # 奇数线程失败
                    raise ValueError(f"error_{worker_id}")
            except Exception as e:
                errors.append(e)
                raise
        
        def worker_thread(worker_id):
            """工作线程"""
            try:
                result = self.error_handler.handle(
                    lambda: worker_function(worker_id)
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 启动多个线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(results) == 5  # 5个成功线程
        assert len(errors) == 5   # 5个失败线程
'''

        # 写入测试文件
        test_dir = self.tests_path / "unit" / "infrastructure" / "error"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_file = test_dir / "test_error_handler_focused.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"✅ 已创建错误处理器聚焦测试: {test_file}")

    def _create_circuit_breaker_focused_tests(self):
        """创建断路器聚焦测试"""
        test_content = '''"""
断路器聚焦测试
专注于断路器状态转换、熔断逻辑等核心功能
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.error.circuit_breaker import CircuitBreaker

class TestCircuitBreakerFocused:
    """断路器聚焦测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5
        )
    
    def test_circuit_breaker_initial_state(self):
        """测试断路器初始状态"""
        assert self.circuit_breaker.state == "CLOSED"
        assert self.circuit_breaker.failure_count == 0
    
    def test_circuit_breaker_successful_calls(self):
        """测试断路器成功调用"""
        def success_function():
            return "success"
        
        # 多次成功调用
        for _ in range(5):
            result = self.circuit_breaker.call(success_function)
            assert result == "success"
        
        # 状态应该保持关闭
        assert self.circuit_breaker.state == "CLOSED"
        assert self.circuit_breaker.failure_count == 0
    
    def test_circuit_breaker_failure_threshold(self):
        """测试断路器失败阈值"""
        def failing_function():
            raise ValueError("Service unavailable")
        
        # 前几次调用应该失败
        for _ in range(3):
            with pytest.raises(ValueError):
                self.circuit_breaker.call(failing_function)
        
        # 断路器应该打开
        assert self.circuit_breaker.state == "OPEN"
        assert self.circuit_breaker.failure_count >= 3
    
    def test_circuit_breaker_open_state(self):
        """测试断路器打开状态"""
        def failing_function():
            raise ValueError("Service unavailable")
        
        # 触发断路器打开
        for _ in range(3):
            with pytest.raises(ValueError):
                self.circuit_breaker.call(failing_function)
        
        # 后续调用应该被拒绝
        with pytest.raises(Exception):
            self.circuit_breaker.call(failing_function)
    
    def test_circuit_breaker_recovery(self):
        """测试断路器恢复"""
        def failing_function():
            raise ValueError("Service unavailable")
        
        def success_function():
            return "success"
        
        # 触发断路器打开
        for _ in range(3):
            with pytest.raises(ValueError):
                self.circuit_breaker.call(failing_function)
        
        # 等待恢复超时
        time.sleep(6)
        
        # 断路器应该进入半开状态
        assert self.circuit_breaker.state == "HALF_OPEN"
        
        # 成功调用应该关闭断路器
        result = self.circuit_breaker.call(success_function)
        assert result == "success"
        assert self.circuit_breaker.state == "CLOSED"
    
    def test_circuit_breaker_half_open_failure(self):
        """测试断路器半开状态下的失败"""
        def failing_function():
            raise ValueError("Service unavailable")
        
        # 触发断路器打开
        for _ in range(3):
            with pytest.raises(ValueError):
                self.circuit_breaker.call(failing_function)
        
        # 等待恢复超时
        time.sleep(6)
        
        # 半开状态下再次失败
        with pytest.raises(ValueError):
            self.circuit_breaker.call(failing_function)
        
        # 断路器应该重新打开
        assert self.circuit_breaker.state == "OPEN"
    
    def test_circuit_breaker_performance(self):
        """测试断路器性能"""
        def fast_function():
            return "fast"
        
        import time
        start_time = time.time()
        
        # 执行大量调用
        for _ in range(1000):
            result = self.circuit_breaker.call(fast_function)
            assert result == "fast"
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能要求：1000次调用应在1秒内完成
        assert duration < 1.0
'''

        # 写入测试文件
        test_dir = self.tests_path / "unit" / "infrastructure" / "error"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_file = test_dir / "test_circuit_breaker_focused.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"✅ 已创建断路器聚焦测试: {test_file}")

    def _create_cache_focused_tests(self):
        """创建缓存聚焦测试"""
        test_content = '''"""
线程安全缓存聚焦测试
专注于缓存操作、并发安全、性能等核心功能
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.cache.thread_safe_cache import ThreadSafeCache

class TestThreadSafeCacheFocused:
    """线程安全缓存聚焦测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.cache = ThreadSafeCache(max_size=100)
    
    def test_cache_basic_operations(self):
        """测试缓存基本操作"""
        # 测试设置和获取
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        assert self.cache.get("key1") == "value1"
        assert self.cache.get("key2") == "value2"
        
        # 测试更新
        self.cache.set("key1", "updated_value")
        assert self.cache.get("key1") == "updated_value"
    
    def test_cache_missing_key(self):
        """测试缓存缺失键"""
        # 获取不存在的键
        result = self.cache.get("nonexistent")
        assert result is None
        
        # 测试默认值
        result = self.cache.get("nonexistent", default="default_value")
        assert result == "default_value"
    
    def test_cache_size_limit(self):
        """测试缓存大小限制"""
        # 填充缓存到限制
        for i in range(100):
            self.cache.set(f"key{i}", f"value{i}")
        
        # 验证缓存大小
        assert len(self.cache) == 100
        
        # 添加新键应该触发淘汰
        self.cache.set("new_key", "new_value")
        
        # 验证旧键被淘汰
        assert self.cache.get("key0") is None
        assert self.cache.get("new_key") == "new_value"
    
    def test_cache_concurrency(self):
        """测试缓存并发安全"""
        results = []
        errors = []
        
        def writer_thread(thread_id):
            """写入线程"""
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    self.cache.set(key, value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def reader_thread(thread_id):
            """读取线程"""
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    value = self.cache.get(key)
                    results.append(value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # 启动多个线程
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer_thread, args=(i,)))
            threads.append(threading.Thread(target=reader_thread, args=(i,)))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 验证没有错误
        assert len(errors) == 0
        assert len(results) == 500  # 5个读取线程 * 100次读取
    
    def test_cache_performance(self):
        """测试缓存性能"""
        import time
        
        # 预热缓存
        for i in range(100):
            self.cache.set(f"key{i}", f"value{i}")
        
        start_time = time.time()
        
        # 执行大量操作
        for i in range(1000):
            self.cache.set(f"perf_key{i}", f"perf_value{i}")
            self.cache.get(f"perf_key{i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能要求：1000次操作应在1秒内完成
        assert duration < 1.0
    
    def test_cache_clear(self):
        """测试缓存清空"""
        # 添加一些数据
        for i in range(10):
            self.cache.set(f"key{i}", f"value{i}")
        
        assert len(self.cache) == 10
        
        # 清空缓存
        self.cache.clear()
        
        assert len(self.cache) == 0
        
        # 验证数据被清空
        for i in range(10):
            assert self.cache.get(f"key{i}") is None
    
    def test_cache_delete(self):
        """测试缓存删除"""
        # 添加数据
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        assert self.cache.get("key1") == "value1"
        assert self.cache.get("key2") == "value2"
        
        # 删除一个键
        self.cache.delete("key1")
        
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") == "value2"
'''

        # 写入测试文件
        test_dir = self.tests_path / "unit" / "infrastructure" / "cache"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_file = test_dir / "test_thread_safe_cache_focused.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"✅ 已创建缓存聚焦测试: {test_file}")

    def _create_database_focused_tests(self):
        """创建数据库管理器聚焦测试"""
        test_content = '''"""
数据库管理器聚焦测试
专注于数据库连接、查询、事务等核心功能
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.infrastructure.database.database_manager import DatabaseManager

class TestDatabaseManagerFocused:
    """数据库管理器聚焦测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_file = Path(self.temp_dir) / "test.db"
        
        self.db_manager = DatabaseManager()
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_manager_initialization(self):
        """测试数据库管理器初始化"""
        assert self.db_manager is not None
        assert hasattr(self.db_manager, 'connection')
    
    def test_database_connection(self):
        """测试数据库连接"""
        # 测试连接创建
        connection = self.db_manager.create_connection(str(self.db_file))
        assert connection is not None
        
        # 测试连接关闭
        self.db_manager.close_connection(connection)
    
    def test_database_query_execution(self):
        """测试数据库查询执行"""
        connection = self.db_manager.create_connection(str(self.db_file))
        
        # 创建测试表
        self.db_manager.execute_query(connection, """
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        """)
        
        # 插入数据
        self.db_manager.execute_query(connection, """
            INSERT INTO test_table (id, name, value) VALUES (1, 'test', 123.45)
        """)
        
        # 查询数据
        result = self.db_manager.execute_query(connection, """
            SELECT * FROM test_table WHERE id = 1
        """)
        
        assert len(result) == 1
        assert result[0][1] == 'test'
        assert result[0][2] == 123.45
        
        self.db_manager.close_connection(connection)
    
    def test_database_transaction(self):
        """测试数据库事务"""
        connection = self.db_manager.create_connection(str(self.db_file))
        
        # 创建测试表
        self.db_manager.execute_query(connection, """
            CREATE TABLE transaction_test (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        
        # 开始事务
        self.db_manager.begin_transaction(connection)
        
        # 执行事务操作
        self.db_manager.execute_query(connection, """
            INSERT INTO transaction_test (id, name) VALUES (1, 'transaction1')
        """)
        
        self.db_manager.execute_query(connection, """
            INSERT INTO transaction_test (id, name) VALUES (2, 'transaction2')
        """)
        
        # 提交事务
        self.db_manager.commit_transaction(connection)
        
        # 验证数据
        result = self.db_manager.execute_query(connection, """
            SELECT * FROM transaction_test
        """)
        
        assert len(result) == 2
        
        self.db_manager.close_connection(connection)
    
    def test_database_error_handling(self):
        """测试数据库错误处理"""
        connection = self.db_manager.create_connection(str(self.db_file))
        
        # 测试无效查询
        with pytest.raises(Exception):
            self.db_manager.execute_query(connection, "INVALID SQL QUERY")
        
        # 测试无效事务
        with pytest.raises(Exception):
            self.db_manager.begin_transaction(connection)
            self.db_manager.execute_query(connection, "INVALID SQL")
            self.db_manager.commit_transaction(connection)
        
        self.db_manager.close_connection(connection)
    
    def test_database_performance(self):
        """测试数据库性能"""
        connection = self.db_manager.create_connection(str(self.db_file))
        
        # 创建测试表
        self.db_manager.execute_query(connection, """
            CREATE TABLE performance_test (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)
        
        import time
        start_time = time.time()
        
        # 执行大量插入
        for i in range(1000):
            self.db_manager.execute_query(connection, f"""
                INSERT INTO performance_test (id, data) VALUES ({i}, 'data_{i}')
            """)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能要求：1000次插入应在5秒内完成
        assert duration < 5.0
        
        self.db_manager.close_connection(connection)
    
    def test_database_concurrency(self):
        """测试数据库并发"""
        import threading
        import time
        
        def worker_thread(thread_id):
            """工作线程"""
            connection = self.db_manager.create_connection(str(self.db_file))
            
            try:
                for i in range(10):
                    self.db_manager.execute_query(connection, f"""
                        INSERT INTO concurrency_test (id, thread_id, data) 
                        VALUES ({thread_id * 100 + i}, {thread_id}, 'data_{i}')
                    """)
                    time.sleep(0.001)
            finally:
                self.db_manager.close_connection(connection)
        
        # 创建测试表
        connection = self.db_manager.create_connection(str(self.db_file))
        self.db_manager.execute_query(connection, """
            CREATE TABLE concurrency_test (
                id INTEGER PRIMARY KEY,
                thread_id INTEGER,
                data TEXT
            )
        """)
        self.db_manager.close_connection(connection)
        
        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证数据
        connection = self.db_manager.create_connection(str(self.db_file))
        result = self.db_manager.execute_query(connection, """
            SELECT COUNT(*) FROM concurrency_test
        """)
        
        assert result[0][0] == 50  # 5个线程 * 10次插入
        
        self.db_manager.close_connection(connection)
'''

        # 写入测试文件
        test_dir = self.tests_path / "unit" / "infrastructure" / "database"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_file = test_dir / "test_database_manager_focused.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"✅ 已创建数据库管理器聚焦测试: {test_file}")

    def run_focused_advancement(self):
        """运行聚焦测试推进"""
        print("🚀 开始聚焦测试推进")
        print("=" * 60)

        # 为每个核心模块创建聚焦测试
        for module_name in self.critical_modules.keys():
            print(f"📝 处理模块: {module_name}")
            self.create_focused_tests(module_name)

        print("=" * 60)
        print("🎉 聚焦测试推进完成!")
        print("")
        print("📋 下一步行动:")
        print("1. 运行新创建的聚焦测试")
        print("2. 分析测试结果和覆盖率")
        print("3. 根据结果调整测试用例")
        print("4. 持续监控和维护测试质量")


def main():
    """主函数"""
    advancement = FocusedTestAdvancement()
    advancement.run_focused_advancement()


if __name__ == "__main__":
    main()
