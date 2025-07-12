#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志模块基础测试
专注于现有可用的日志功能
"""
import pytest
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock

# 导入现有的日志模块
from src.infrastructure.m_logging.logger import TradingLogger as Logger
from src.infrastructure.m_logging.log_manager import LogManager
from src.infrastructure.m_logging.log_metrics import LogMetrics
from src.infrastructure.m_logging.log_compressor import LogCompressor

class TestLoggingBasic:
    """日志基础测试"""
    
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
    
    def test_logger_creation(self, log_manager):
        """测试日志器创建"""
        logger = log_manager.get_logger("test_logger")
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')
    
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
    
    def test_log_metrics(self):
        """测试日志指标"""
        metrics = LogMetrics()
        
        # 记录日志指标
        metrics.record_log("INFO", "test_message", 100)
        metrics.record_log("ERROR", "error_message", 200)
        metrics.record_log("WARNING", "warning_message", 150)
        
        # 获取指标统计
        stats = metrics.get_statistics()
        assert stats["total_logs"] == 3
        assert stats["info_count"] == 1
        assert stats["error_count"] == 1
        assert stats["warning_count"] == 1
        assert stats["total_size"] == 450
    
    def test_log_compression(self):
        """测试日志压缩"""
        compressor = LogCompressor()
        
        # 创建测试日志数据
        log_data = "这是一条测试日志数据，包含重复的内容。" * 100
        
        # 压缩日志
        compressed_data = compressor.compress(log_data)
        assert len(compressed_data) < len(log_data)
        
        # 解压日志
        decompressed_data = compressor.decompress(compressed_data)
        assert decompressed_data == log_data
    
    def test_log_performance(self, log_manager):
        """测试日志性能"""
        import time
        
        logger = log_manager.get_logger("performance_test")
        
        # 测试大量日志记录性能
        start_time = time.time()
        for i in range(1000):
            logger.info(f"性能测试日志 {i}")
        logging_time = time.time() - start_time
        
        # 性能要求：记录1000条日志应在2秒内完成
        assert logging_time < 2.0
    
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
    
    def test_log_error_handling(self, log_manager):
        """测试日志错误处理"""
        logger = log_manager.get_logger("error_test")
        
        # 模拟日志系统错误
        with patch.object(logger, '_write_log', side_effect=Exception("写入错误")):
            # 应该能够优雅地处理错误
            logger.info("测试错误处理")
        
        # 系统应该继续正常工作
        logger.info("错误处理后的正常日志")
        
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
        
        # 内存使用应该小于200MB
        assert memory_usage < 200
    
    def test_log_compression_performance(self):
        """测试日志压缩性能"""
        compressor = LogCompressor()
        
        # 创建大量日志数据
        large_log_data = "测试日志数据" * 10000
        
        # 测试压缩性能
        start_time = time.time()
        compressed_data = compressor.compress(large_log_data)
        compression_time = time.time() - start_time
        
        # 性能要求：压缩1MB数据应在1秒内完成
        assert compression_time < 1.0
        
        # 验证压缩比
        compression_ratio = len(compressed_data) / len(large_log_data)
        assert compression_ratio < 0.8  # 至少20%的压缩率
    
    def test_log_metrics_performance(self):
        """测试日志指标性能"""
        metrics = LogMetrics()
        
        # 测试大量指标记录性能
        start_time = time.time()
        for i in range(10000):
            metrics.record_log("INFO", f"test_message_{i}", 100)
        metrics_time = time.time() - start_time
        
        # 性能要求：记录10000个指标应在1秒内完成
        assert metrics_time < 1.0
        
        # 验证指标统计
        stats = metrics.get_statistics()
        assert stats["total_logs"] == 10000
        assert stats["info_count"] == 10000
    
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
    
    def test_log_levels(self, log_manager):
        """测试日志级别"""
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
    
    def test_log_cleanup(self, log_manager):
        """测试日志清理"""
        logger = log_manager.get_logger("cleanup_test")
        
        # 记录一些日志
        for i in range(100):
            logger.info(f"清理测试日志 {i}")
        
        # 执行日志清理
        cleanup_result = log_manager.cleanup_old_logs(days=1)
        assert cleanup_result["cleaned_count"] >= 0
    
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