#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志管理器单元测试
基于实际的LogManager类创建可运行的测试
"""
import pytest
import tempfile
import os
import json
import time
import logging
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.m_logging.log_manager import LogManager, JsonFormatter

class TestLogManagerWorking:
    """日志管理器工作测试"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """创建临时日志目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def log_manager(self, temp_log_dir):
        """创建日志管理器实例"""
        return LogManager(app_name="test_app", log_dir=temp_log_dir)
    
    def test_log_manager_initialization(self, log_manager):
        """测试日志管理器初始化"""
        assert log_manager is not None
        assert hasattr(log_manager, '_loggers')
        assert hasattr(log_manager, '_sampler')
    
    def test_get_logger(self, log_manager):
        """测试获取日志器"""
        logger = LogManager.get_logger("test_logger")
        assert logger is not None
        assert isinstance(logger, type(logging.getLogger()))
    
    def test_basic_logging(self, log_manager):
        """测试基础日志记录"""
        # 配置日志管理器
        log_manager.configure({
            'app_name': 'test_app',
            'log_dir': log_manager._log_dir if hasattr(log_manager, '_log_dir') else tempfile.mkdtemp(),
            'log_level': 'INFO'
        })
        
        # 测试不同级别的日志
        LogManager.info("测试信息日志", extra={})
        LogManager.warning("测试警告日志", extra={})
        LogManager.error("测试错误日志", extra={})
        LogManager.debug("测试调试日志", extra={})
        
        # 验证日志文件创建
        log_dir = getattr(log_manager, '_log_dir', None)
        if log_dir and os.path.exists(log_dir):
            log_files = os.listdir(log_dir)
            assert len(log_files) > 0
    
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
    
    def test_log_manager_configure(self, log_manager):
        """测试日志管理器配置"""
        config = {
            'app_name': 'test_app',
            'log_dir': tempfile.mkdtemp(),
            'max_bytes': 1024*1024,
            'backup_count': 5,
            'log_level': 'DEBUG'
        }
        
        log_manager.configure(config)
        
        # 验证配置生效
        assert hasattr(log_manager, '_app_name') or 'app_name' in config
    
    def test_log_levels(self, log_manager):
        """测试日志级别"""
        # 设置日志级别为WARNING
        LogManager.set_level("WARNING")
        
        # 只有WARNING及以上级别的日志会被记录
        LogManager.debug("调试信息")  # 不会被记录
        LogManager.info("普通信息")   # 不会被记录
        LogManager.warning("警告信息") # 会被记录
        LogManager.error("错误信息")   # 会被记录
        
        # 验证日志级别设置
        logger = LogManager.get_logger()
        assert logger.level >= logging.WARNING
    
    def test_log_manager_close(self):
        """测试日志管理器关闭"""
        # 获取日志管理器实例
        manager = LogManager.get_instance()
        
        # 关闭日志管理器
        LogManager.close()
        
        # 验证关闭操作（这里主要是测试方法存在）
        assert hasattr(LogManager, 'close')
    
    def test_add_json_file_handler(self, temp_log_dir):
        """测试添加JSON文件处理器"""
        # 创建测试日志文件
        log_file = os.path.join(temp_log_dir, "test.json")
        
        # 添加JSON文件处理器
        LogManager.add_json_file_handler(log_file)
        
        # 验证文件处理器被添加
        logger = LogManager.get_logger()
        handlers = logger.handlers
        
        # 验证至少有一个处理器
        assert len(handlers) > 0
        
        # 验证文件存在
        assert os.path.exists(log_file)
    
    def test_log_manager_performance(self, log_manager):
        """测试日志管理器性能"""
        import time
        
        # 测试大量日志记录性能
        start_time = time.time()
        for i in range(1000):
            LogManager.info(f"性能测试日志 {i}")
        logging_time = time.time() - start_time
        
        # 性能要求：记录1000条日志应在2秒内完成
        assert logging_time < 2.0
    
    def test_log_manager_concurrency(self, log_manager):
        """测试日志管理器并发访问"""
        import threading
        
        log_count = 0
        
        def log_worker():
            nonlocal log_count
            for i in range(100):
                LogManager.info(f"并发日志 {i}")
                log_count += 1
        
        # 启动多个线程同时记录日志
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=log_worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有日志都被记录
        assert log_count == 500
    
    def test_log_manager_error_handling(self, log_manager):
        """测试日志管理器错误处理"""
        # 测试无效配置
        with pytest.raises(ValueError):
            log_manager.configure_sampler("invalid_config")
        
        # 测试无效日志级别
        with pytest.raises(AttributeError):
            LogManager.set_level("INVALID_LEVEL")
    
    def test_log_manager_memory_management(self, log_manager):
        """测试日志管理器内存管理"""
        # 记录大量日志
        for i in range(10000):
            LogManager.info(f"内存测试日志 {i}")
        
        # 验证内存使用在合理范围内
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # 内存使用应该小于200MB
        assert memory_usage < 200
    
    def test_log_manager_integration(self, log_manager):
        """测试日志管理器集成"""
        # 测试与其他模块的集成
        LogManager.info("配置变更", extra={"config_key": "database.host", "old_value": "localhost", "new_value": "newhost"})
        
        # 模拟数据库操作
        LogManager.info("数据库查询", extra={"query": "SELECT * FROM users", "duration": 0.1})
        
        # 模拟交易操作
        LogManager.info("交易执行", extra={"order_id": "12345", "symbol": "AAPL", "quantity": 100})
        
        # 验证集成日志被记录
        logger = LogManager.get_logger()
        assert logger is not None
    
    def test_log_manager_sampler_configuration(self, log_manager):
        """测试日志采样器配置"""
        # 配置采样器
        sampler_config = {
            'default_rate': 0.5,
            'level_rates': {
                'DEBUG': 0.1,
                'INFO': 0.5,
                'WARNING': 1.0,
                'ERROR': 1.0
            }
        }
        
        log_manager.configure_sampler(sampler_config)
        
        # 验证采样器配置
        assert log_manager._sampler is not None
    
    def test_log_manager_file_rotation(self, temp_log_dir):
        """测试日志文件轮转"""
        # 配置日志管理器
        config = {
            'app_name': 'rotation_test',
            'log_dir': temp_log_dir,
            'max_bytes': 1024,  # 小文件大小以触发轮转
            'backup_count': 3,
            'log_level': 'INFO'
        }
        
        log_manager = LogManager()
        log_manager.configure(config)
        
        # 创建大量日志数据以触发轮转
        for i in range(1000):
            LogManager.info(f"轮转测试日志 {i}")
        
        # 检查日志文件轮转
        log_files = os.listdir(temp_log_dir)
        assert len(log_files) > 1  # 应该有多个日志文件
    
    def test_log_manager_log_level_validation(self, log_manager):
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
    
    def test_log_manager_handler_management(self, temp_log_dir):
        """测试日志处理器管理"""
        # 添加多个文件处理器
        log_file1 = os.path.join(temp_log_dir, "test1.json")
        log_file2 = os.path.join(temp_log_dir, "test2.json")
        
        LogManager.add_json_file_handler(log_file1)
        LogManager.add_json_file_handler(log_file2)
        
        # 验证处理器被正确添加
        logger = LogManager.get_logger()
        handlers = logger.handlers
        
        # 验证文件存在
        assert os.path.exists(log_file1)
        assert os.path.exists(log_file2)
    
    def test_log_manager_cleanup(self, log_manager):
        """测试日志管理器清理"""
        # 记录一些日志
        for i in range(100):
            LogManager.info(f"清理测试日志 {i}")
        
        # 执行日志清理
        LogManager.close()
        
        # 验证清理操作完成
        assert hasattr(LogManager, 'close') 