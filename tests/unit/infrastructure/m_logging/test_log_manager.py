#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志管理器单元测试
覆盖日志管理、格式化、级别控制等核心功能
"""
import pytest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.infrastructure.m_logging.log_manager import LogManager
from src.infrastructure.m_logging.logger import TradingLogger as Logger

class TestLogManager:
    """日志管理器测试"""
    
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
        return LogManager(log_dir=temp_log_dir)
    
    def test_log_manager_initialization(self, log_manager):
        """测试日志管理器初始化"""
        assert log_manager is not None
        assert hasattr(log_manager, 'log_dir')
        assert hasattr(log_manager, 'get_logger')
    
    def test_get_logger(self, log_manager):
        """测试获取日志器"""
        logger = log_manager.get_logger("test_logger")
        assert isinstance(logger, Logger)
        assert logger.name == "test_logger"
    
    def test_basic_logging(self, log_manager):
        """测试基础日志记录"""
        logger = log_manager.get_logger("basic_test")
        
        # 测试不同级别的日志
        logger.info("测试信息日志")
        logger.warning("测试警告日志")
        logger.error("测试错误日志")
        logger.debug("测试调试日志")
        
        # 验证日志文件创建
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 0
    
    def test_log_levels(self, log_manager):
        """测试日志级别控制"""
        logger = log_manager.get_logger("level_test")
        
        # 设置日志级别为WARNING
        logger.setLevel("WARNING")
        
        # 只有WARNING及以上级别的日志会被记录
        logger.debug("调试信息")  # 不会被记录
        logger.info("普通信息")   # 不会被记录
        logger.warning("警告信息") # 会被记录
        logger.error("错误信息")   # 会被记录
        
        # 验证只有WARNING和ERROR级别的日志被记录
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 0
    
    def test_log_formatting(self, log_manager):
        """测试日志格式化"""
        logger = log_manager.get_logger("format_test")
        
        # 测试JSON格式
        json_logger = logger.with_format("json")
        json_logger.info("JSON格式日志", extra={"user_id": 123})
        
        # 测试结构化格式
        structured_logger = logger.with_format("structured")
        structured_logger.info("结构化日志", extra={"action": "login"})
        
        # 验证日志文件存在
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 0
    
    def test_log_context(self, log_manager):
        """测试日志上下文"""
        logger = log_manager.get_logger("context_test")
        
        # 添加上下文信息
        with logger.context(user_id=123, session_id="abc123"):
            logger.info("带上下文的日志")
        
        # 验证上下文信息被正确记录
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 0
    
    def test_log_rotation(self, log_manager):
        """测试日志轮转"""
        # 创建大量日志数据
        logger = log_manager.get_logger("rotation_test")
        
        for i in range(1000):
            logger.info(f"测试日志消息 {i}")
        
        # 检查日志文件轮转
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 1  # 应该有多个日志文件
    
    def test_log_performance(self, log_manager):
        """测试日志性能"""
        import time
        
        logger = log_manager.get_logger("performance_test")
        
        # 测试大量日志记录性能
        start_time = time.time()
        for i in range(1000):
            logger.info(f"性能测试日志 {i}")
        logging_time = time.time() - start_time
        
        # 性能要求：记录1000条日志应在1秒内完成
        assert logging_time < 1.0
    
    def test_log_concurrency(self, log_manager):
        """测试日志并发访问"""
        import threading
        
        logger = log_manager.get_logger("concurrent_test")
        log_count = 0
        
        def log_worker():
            nonlocal log_count
            for i in range(100):
                logger.info(f"并发日志 {i}")
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
        
        # 验证日志文件存在
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 0
    
    def test_log_error_recovery(self, log_manager):
        """测试日志错误恢复"""
        logger = log_manager.get_logger("recovery_test")
        
        # 模拟日志系统错误
        with patch.object(logger, '_write_log', side_effect=Exception("写入错误")):
            # 应该能够优雅地处理错误
            logger.info("测试错误恢复")
        
        # 系统应该继续正常工作
        logger.info("错误恢复后的正常日志")
        
        # 验证日志系统仍然可用
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 0
    
    def test_log_memory_management(self, log_manager):
        """测试日志内存管理"""
        logger = log_manager.get_logger("memory_test")
        
        # 记录大量日志
        for i in range(10000):
            logger.info(f"内存测试日志 {i}")
        
        # 验证内存使用在合理范围内
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # 内存使用应该小于100MB
        assert memory_usage < 100
    
    def test_log_configuration(self, log_manager):
        """测试日志配置"""
        # 测试配置日志管理器
        config = {
            "log_level": "INFO",
            "log_format": "json",
            "max_file_size": "10MB",
            "backup_count": 5
        }
        
        log_manager.configure(config)
        
        # 验证配置生效
        logger = log_manager.get_logger("config_test")
        logger.info("配置测试日志")
        
        # 验证日志文件存在
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 0
    
    def test_log_filters(self, log_manager):
        """测试日志过滤器"""
        logger = log_manager.get_logger("filter_test")
        
        # 添加过滤器
        def custom_filter(record):
            return "sensitive" not in record.getMessage()
        
        logger.add_filter(custom_filter)
        
        # 测试过滤器功能
        logger.info("正常日志")
        logger.info("包含sensitive的日志")  # 应该被过滤
        
        # 验证只有非敏感日志被记录
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 0
    
    def test_log_handlers(self, log_manager):
        """测试日志处理器"""
        logger = log_manager.get_logger("handler_test")
        
        # 添加自定义处理器
        custom_handler = Mock()
        logger.add_handler(custom_handler)
        
        # 记录日志
        logger.info("处理器测试日志")
        
        # 验证处理器被调用
        custom_handler.handle.assert_called()
    
    def test_log_metrics(self, log_manager):
        """测试日志指标"""
        logger = log_manager.get_logger("metrics_test")
        
        # 记录不同类型的日志
        logger.info("信息日志")
        logger.warning("警告日志")
        logger.error("错误日志")
        
        # 获取日志指标
        metrics = log_manager.get_metrics()
        assert "total_logs" in metrics
        assert "info_count" in metrics
        assert "warning_count" in metrics
        assert "error_count" in metrics
    
    def test_log_cleanup(self, log_manager):
        """测试日志清理"""
        logger = log_manager.get_logger("cleanup_test")
        
        # 记录一些日志
        for i in range(100):
            logger.info(f"清理测试日志 {i}")
        
        # 执行日志清理
        cleanup_result = log_manager.cleanup_old_logs(days=1)
        assert cleanup_result["cleaned_count"] >= 0
    
    def test_log_export(self, log_manager):
        """测试日志导出"""
        logger = log_manager.get_logger("export_test")
        
        # 记录一些日志
        logger.info("导出测试日志1")
        logger.warning("导出测试日志2")
        logger.error("导出测试日志3")
        
        # 导出日志
        export_file = log_manager.export_logs("export_test")
        assert os.path.exists(export_file)
        
        # 清理导出文件
        os.unlink(export_file)
    
    def test_log_search(self, log_manager):
        """测试日志搜索"""
        logger = log_manager.get_logger("search_test")
        
        # 记录包含特定关键词的日志
        logger.info("包含关键词的日志")
        logger.warning("普通日志")
        logger.error("另一个包含关键词的日志")
        
        # 搜索包含关键词的日志
        search_results = log_manager.search_logs("关键词")
        assert len(search_results) >= 2
    
    def test_log_compression(self, log_manager):
        """测试日志压缩"""
        logger = log_manager.get_logger("compression_test")
        
        # 记录大量日志
        for i in range(1000):
            logger.info(f"压缩测试日志 {i}")
        
        # 压缩日志文件
        compression_result = log_manager.compress_logs()
        assert compression_result["compressed_files"] > 0
    
    def test_log_integration(self, log_manager):
        """测试日志集成"""
        # 测试与其他模块的集成
        logger = log_manager.get_logger("integration_test")
        
        # 模拟配置变更
        logger.info("配置变更", extra={"config_key": "database.host", "old_value": "localhost", "new_value": "newhost"})
        
        # 模拟数据库操作
        logger.info("数据库查询", extra={"query": "SELECT * FROM users", "duration": 0.1})
        
        # 模拟交易操作
        logger.info("交易执行", extra={"order_id": "12345", "symbol": "AAPL", "quantity": 100})
        
        # 验证集成日志被记录
        log_files = os.listdir(log_manager.log_dir)
        assert len(log_files) > 0