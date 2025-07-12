import os
import re
import sys
import tempfile
import shutil
import pytest
import logging
import multiprocessing
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.infrastructure.m_logging.log_manager import LogManager, ConcurrentRotatingFileHandler

class TestLogManagerBoundaryConditions:
    """测试LogManager的边界条件"""

    @pytest.fixture
    def temp_log_dir(self):
        """创建临时日志目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # 手动清理临时目录
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_rotation_at_exact_size(self, temp_log_dir):
        """测试日志文件达到精确大小时轮转"""
        # 设置很小的轮转大小(1KB)
        manager = LogManager(
            app_name="test_rotation",
            log_dir=temp_log_dir,
            max_bytes=1024,
            backup_count=3
        )

        # 写入刚好达到轮转大小的日志
        logger = logging.getLogger("test_rotation")
        message = "x" * 512  # 每条消息512字节
        for _ in range(2):  # 2*512=1024字节
            logger.info(message)

        # 验证是否创建了轮转文件
        log_file = os.path.join(temp_log_dir, "test_rotation.log")
        assert os.path.exists(log_file)
        assert os.path.exists(log_file + ".1")
        assert os.path.getsize(log_file) <= 1024

        manager.close()

    def test_extreme_log_level_filtering(self, temp_log_dir):
        """测试极端日志级别过滤"""
        manager = LogManager(
            app_name="test_levels",
            log_dir=temp_log_dir,
            log_level=logging.CRITICAL  # 只记录CRITICAL级别
        )

        logger = logging.getLogger("test_levels")
        logger.debug("Debug message - should not appear")
        logger.info("Info message - should not appear")
        logger.warning("Warning message - should not appear")
        logger.error("Error message - should not appear")
        logger.critical("Critical message - should appear")

        # 验证日志文件内容
        log_file = os.path.join(temp_log_dir, "test_levels.log")
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Critical message" in content
            assert "Debug message" not in content

        manager.close()

    def test_exception_handler_boundary(self, temp_log_dir):
        """测试异常处理器的边界情况"""
        manager = LogManager(
            app_name="test_exceptions",
            log_dir=temp_log_dir
        )

        logger = logging.getLogger("test_exceptions")

        # 测试各种异常类型
        try:
            1/0
        except Exception:
            logger.exception("Division by zero")

        try:
            raise ValueError("Test value error")
        except Exception:
            logger.exception("Value error")

        # 验证异常是否被记录
        log_file = os.path.join(temp_log_dir, "test_exceptions.log")
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Division by zero" in content
            assert "Value error" in content

        manager.close()

    def test_log_sampling_boundary(self, temp_log_dir):
        """测试日志采样边界条件"""
        manager = LogManager(
            app_name="test_sampling",
            log_dir=temp_log_dir,
            log_level="INFO"
        )
        
        # 获取日志记录器
        logger = manager.get_logger()
        
        # 写入多条日志
        for i in range(100):
            logger.info(f"Sample test message {i}")
        
        # 验证日志文件存在
        log_file = os.path.join(temp_log_dir, "test_sampling.log")
        assert os.path.exists(log_file)
        
        # 验证有日志内容
        with open(log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
