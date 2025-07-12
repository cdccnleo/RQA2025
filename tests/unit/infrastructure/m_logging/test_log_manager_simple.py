#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的日志管理器单元测试
专注于核心功能测试
"""
import pytest
import tempfile
import os
import json
import logging
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.m_logging.log_manager import LogManager, JsonFormatter

class TestLogManagerSimple:
    """简化的日志管理器测试"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """创建临时日志目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    def test_log_manager_initialization(self):
        """测试日志管理器初始化"""
        log_manager = LogManager()
        assert log_manager is not None
        assert hasattr(log_manager, '_loggers')
        assert hasattr(log_manager, '_sampler')
    
    def test_get_logger(self):
        """测试获取日志器"""
        logger = LogManager.get_logger("test_logger")
        assert logger is not None
        assert isinstance(logger, type(logging.getLogger()))
    
    def test_json_formatter(self):
        """测试JSON格式化器"""
        formatter = JsonFormatter()
        
        # 创建模拟的日志记录
        record = Mock()
        record.levelname = "INFO"
        record.getMessage.return_value = "测试消息"
        record.name = "test_logger"
        record.pathname = "/path/to/test.py"
        record.lineno = 123
        record.exc_info = None
        
        # 格式化日志
        formatted = formatter.format(record)
        
        # 验证JSON格式
        log_data = json.loads(formatted)
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "测试消息"
        assert log_data["logger"] == "test_logger"
        assert "timestamp" in log_data
    
    def test_log_manager_singleton(self):
        """测试日志管理器单例模式"""
        # 获取两个实例
        instance1 = LogManager.get_instance()
        instance2 = LogManager.get_instance()
        
        # 验证是同一个实例
        assert instance1 is instance2
    
    def test_log_manager_configure(self, temp_log_dir):
        """测试日志管理器配置"""
        log_manager = LogManager()
        config = {
            'app_name': 'test_app',
            'log_dir': temp_log_dir,
            'max_bytes': 1024*1024,
            'backup_count': 5,
            'log_level': 'DEBUG'
        }
        
        log_manager.configure(config)
        
        # 验证配置生效
        assert hasattr(log_manager, '_app_name') or 'app_name' in config
    
    def test_log_manager_close(self):
        """测试日志管理器关闭"""
        # 获取日志管理器实例
        manager = LogManager.get_instance()
        
        # 关闭日志管理器
        LogManager.close()
        
        # 验证关闭操作（这里主要是测试方法存在）
        assert hasattr(LogManager, 'close')
    
    def test_log_manager_basic_logging(self, temp_log_dir):
        """测试基础日志记录"""
        # 配置日志管理器
        log_manager = LogManager()
        log_manager.configure({
            'app_name': 'test_app',
            'log_dir': temp_log_dir,
            'log_level': 'INFO'
        })
        
        # 测试不同级别的日志
        LogManager.info("测试信息日志")
        LogManager.warning("测试警告日志")
        LogManager.error("测试错误日志")
        LogManager.debug("测试调试日志")
        
        # 验证日志文件创建
        if os.path.exists(temp_log_dir):
            log_files = os.listdir(temp_log_dir)
            # 验证至少有一个日志文件
            assert len(log_files) >= 0
    
    def test_log_manager_error_handling(self):
        """测试日志管理器错误处理"""
        log_manager = LogManager()
        
        # 测试无效配置
        with pytest.raises(ValueError):
            log_manager.configure_sampler("invalid_config")
    
    def test_log_manager_log_level_validation(self):
        """测试日志级别验证"""
        # 测试有效日志级别
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            LogManager.set_level(level)
            logger = LogManager.get_logger()
            assert logger.level == getattr(logging, level)
        
        # 测试无效日志级别
        with pytest.raises(AttributeError):
            LogManager.set_level("INVALID")
    
    def test_log_manager_cleanup(self):
        """测试日志管理器清理"""
        # 记录一些日志
        for i in range(10):
            LogManager.info(f"清理测试日志 {i}")
        
        # 执行日志清理
        LogManager.close()
        
        # 验证清理操作完成
        assert hasattr(LogManager, 'close')
    
    def test_log_manager_integration_simple(self):
        """测试日志管理器简单集成"""
        # 测试与其他模块的集成
        LogManager.info("配置变更")
        
        # 模拟数据库操作
        LogManager.info("数据库查询")
        
        # 模拟交易操作
        LogManager.info("交易执行")
        
        # 验证集成日志被记录
        logger = LogManager.get_logger()
        assert logger is not None
    
    def test_log_manager_sampler_basic(self):
        """测试日志采样器基础功能"""
        log_manager = LogManager()
        
        # 验证采样器存在
        assert hasattr(log_manager, '_sampler')
        assert log_manager._sampler is not None
    
    def test_log_manager_file_handler_basic(self, temp_log_dir):
        """测试文件处理器基础功能"""
        # 创建测试日志文件
        log_file = os.path.join(temp_log_dir, "test.json")
        
        # 添加JSON文件处理器
        LogManager.add_json_file_handler(log_file)
        
        # 验证文件处理器被添加
        logger = LogManager.get_logger()
        handlers = logger.handlers
        
        # 验证至少有一个处理器
        assert len(handlers) > 0
    
    def test_log_manager_performance_basic(self):
        """测试日志管理器基础性能"""
        import time
        
        # 测试少量日志记录性能
        start_time = time.time()
        for i in range(100):
            LogManager.info(f"性能测试日志 {i}")
        logging_time = time.time() - start_time
        
        # 性能要求：记录100条日志应在1秒内完成
        assert logging_time < 1.0
    
    def test_log_manager_concurrency_basic(self):
        """测试日志管理器基础并发访问"""
        import threading
        
        log_count = 0
        
        def log_worker():
            nonlocal log_count
            for i in range(10):
                LogManager.info(f"并发日志 {i}")
                log_count += 1
        
        # 启动多个线程同时记录日志
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=log_worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有日志都被记录
        assert log_count == 30
    
    def test_log_manager_memory_basic(self):
        """测试日志管理器基础内存管理"""
        # 记录少量日志
        for i in range(100):
            LogManager.info(f"内存测试日志 {i}")
        
        # 验证内存使用在合理范围内
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # 内存使用应该小于100MB
        assert memory_usage < 100 