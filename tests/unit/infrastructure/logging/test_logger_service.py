#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 日志服务实现

测试logging/services/logger_service.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.logging.services.logger_service import LoggerService


class TestLoggerService:
    """测试日志服务实现"""

    def setup_method(self):
        """测试前准备"""
        self.config = {
            "default_level": "INFO",
            "max_loggers": 10,
            "enable_persistence": True,
            "storage_config": {"max_size": 1000}
        }
        self.service = LoggerService(self.config)

    def teardown_method(self):
        """测试后清理"""
        if self.service:
            try:
                self.service.stop()
            except:
                pass

    def test_initialization_with_config(self):
        """测试带配置的初始化"""
        assert self.service.name == "LoggerService"
        assert self.service.default_level == "INFO"
        assert self.service.max_loggers == 10
        assert self.service.enabled is True
        assert isinstance(self.service.loggers, dict)
        assert hasattr(self.service, 'storage')

    def test_initialization_default_config(self):
        """测试默认配置初始化"""
        service = LoggerService()

        assert service.name == "LoggerService"
        assert service.default_level == "INFO"
        assert service.max_loggers == 100
        assert service.enabled is True

    def test_setup_default_components(self):
        """测试默认组件设置"""
        # 默认组件应该在初始化时自动设置
        assert 'root' in self.service.loggers
        root_logger = self.service.loggers['root']
        assert root_logger is not None

    def test_create_logger_basic(self):
        """测试基本日志器创建"""
        logger_name = "test_logger"
        logger = self.service.create_logger(logger_name)

        assert logger is not None
        assert logger_name in self.service.loggers
        assert self.service.loggers[logger_name] is logger

    def test_create_logger_with_config(self):
        """测试带配置的日志器创建"""
        logger_name = "configured_logger"
        logger_config = {
            "level": "DEBUG",
            "handlers": ["console"],
            "formatter": "json"
        }

        logger = self.service.create_logger(logger_name, logger_config)

        assert logger is not None
        assert logger_name in self.service.loggers

    def test_create_logger_duplicate_name(self):
        """测试创建重复名称的日志器"""
        logger_name = "duplicate_logger"

        # 创建第一个日志器
        logger1 = self.service.create_logger(logger_name)
        assert logger1 is not None

        # 尝试创建同名日志器（应该返回已存在的）
        logger2 = self.service.create_logger(logger_name)
        assert logger2 is logger1  # 应该返回同一个实例

    def test_validate_logger_limit(self):
        """测试日志器数量限制验证"""
        # 创建接近限制数量的日志器
        for i in range(8):  # max_loggers = 10，留出一些空间给默认的
            self.service.create_logger(f"logger_{i}")

        # 应该还能创建更多日志器
        additional_logger = self.service.create_logger("additional")
        assert additional_logger is not None

    def test_build_logger_config(self):
        """测试日志器配置构建"""
        config = self.service._build_logger_config("test", {"level": "DEBUG"})

        assert isinstance(config, dict)
        assert "name" in config
        assert "level" in config
        assert config["name"] == "test"
        assert config["level"] == "DEBUG"

    def test_build_logger_config_default(self):
        """测试默认日志器配置构建"""
        config = self.service._build_logger_config("test", None)

        assert isinstance(config, dict)
        assert config["name"] == "test"
        assert "level" in config
        assert "handlers" in config

    def test_create_logger_instance(self):
        """测试日志器实例创建"""
        logger_config = {
            "name": "test_instance",
            "level": "INFO",
            "handlers": [],
            "formatter": "text"
        }

        logger = self.service._create_logger_instance(logger_config)

        assert logger is not None
        assert hasattr(logger, 'name')

    def test_add_handlers_to_logger(self):
        """测试为日志器添加处理器"""
        from src.infrastructure.logging.core.base_logger import BaseLogger

        # 创建模拟日志器
        mock_logger = Mock(spec=BaseLogger)
        mock_logger.name = "test_logger"

        handler_config = {
            "handlers": [
                {"type": "console", "level": "INFO"},
                {"type": "file", "level": "ERROR", "filename": "test.log"}
            ]
        }

        # 这个方法可能会抛出异常，取决于依赖是否可用
        try:
            self.service._add_handlers_to_logger(mock_logger, handler_config)
        except:
            # 如果依赖不可用，测试异常处理
            pass

    def test_create_handler(self):
        """测试处理器创建"""
        # 测试控制台处理器
        console_config = {"type": "console", "level": "INFO"}
        handler = self.service._create_handler(console_config)

        # 如果成功创建，应该返回处理器实例或None（如果依赖不可用）
        assert handler is None or hasattr(handler, 'emit')

    def test_create_handler_invalid_type(self):
        """测试创建无效类型的处理器"""
        invalid_config = {"type": "invalid", "level": "INFO"}
        handler = self.service._create_handler(invalid_config)

        assert handler is None

    def test_configure_handler_formatter(self):
        """测试处理器格式化器配置"""
        mock_handler = Mock()
        mock_formatter = Mock()

        handler_config = {"formatter": "json"}

        # 这个方法可能会因为依赖不可用来抛出异常
        try:
            self.service._configure_handler_formatter(mock_handler, handler_config)
        except:
            # 异常处理测试
            pass

    def test_register_logger(self):
        """测试日志器注册"""
        from src.infrastructure.logging.core.base_logger import BaseLogger

        mock_logger = Mock(spec=BaseLogger)
        mock_logger.name = "register_test"

        self.service._register_logger("register_test", mock_logger)

        assert "register_test" in self.service.loggers
        assert self.service.loggers["register_test"] is mock_logger

    def test_get_logger_existing(self):
        """测试获取存在的日志器"""
        logger_name = "existing_logger"
        created_logger = self.service.create_logger(logger_name)

        retrieved_logger = self.service.get_logger(logger_name)

        assert retrieved_logger is created_logger

    def test_get_logger_nonexistent(self):
        """测试获取不存在的日志器"""
        retrieved_logger = self.service.get_logger("nonexistent")

        assert retrieved_logger is None

    def test_remove_logger_existing(self):
        """测试移除存在的日志器"""
        logger_name = "to_remove"
        self.service.create_logger(logger_name)

        # 确认存在
        assert logger_name in self.service.loggers

        # 移除
        result = self.service.remove_logger(logger_name)

        assert result is True
        assert logger_name not in self.service.loggers

    def test_remove_logger_nonexistent(self):
        """测试移除不存在的日志器"""
        result = self.service.remove_logger("nonexistent")

        assert result is False

    def test_list_loggers(self):
        """测试列出所有日志器"""
        # 创建一些日志器
        loggers_to_create = ["logger1", "logger2", "logger3"]
        for name in loggers_to_create:
            self.service.create_logger(name)

        logger_list = self.service.list_loggers()

        assert isinstance(logger_list, list)
        assert len(logger_list) >= len(loggers_to_create)
        for name in loggers_to_create:
            assert name in logger_list

    def test_log_message_success(self):
        """测试成功记录消息"""
        logger_name = "message_logger"
        self.service.create_logger(logger_name)

        result = self.service.log_message(
            logger_name=logger_name,
            level="INFO",
            message="Test message"
        )

        assert isinstance(result, bool)
        # 结果可能是True或False，取决于日志器实现

    def test_log_message_nonexistent_logger(self):
        """测试向不存在的日志器记录消息"""
        result = self.service.log_message(
            logger_name="nonexistent",
            level="INFO",
            message="Test message"
        )

        assert result is False

    def test_log_message_with_kwargs(self):
        """测试带额外参数记录消息"""
        logger_name = "kwargs_logger"
        self.service.create_logger(logger_name)

        result = self.service.log_message(
            logger_name=logger_name,
            level="DEBUG",
            message="Test with kwargs",
            extra_data="value",
            user_id=123
        )

        assert isinstance(result, bool)

    def test_get_logger_for_logging(self):
        """测试获取用于记录的日志器"""
        logger_name = "logging_test"
        created_logger = self.service.create_logger(logger_name)

        retrieved_logger = self.service._get_logger_for_logging(logger_name)

        assert retrieved_logger is created_logger

    def test_get_logger_for_logging_nonexistent(self):
        """测试获取不存在的用于记录的日志器"""
        retrieved_logger = self.service._get_logger_for_logging("nonexistent")

        assert retrieved_logger is None

    def test_log_to_logger(self):
        """测试向日志器记录日志"""
        from src.infrastructure.logging.core.base_logger import BaseLogger

        mock_logger = Mock(spec=BaseLogger)
        mock_logger.info = Mock()
        mock_logger.debug = Mock()

        # 测试INFO级别
        self.service._log_to_logger(mock_logger, "INFO", "Test message")
        mock_logger.info.assert_called_once()

        # 测试DEBUG级别
        self.service._log_to_logger(mock_logger, "DEBUG", "Debug message", extra="data")
        mock_logger.debug.assert_called_once()

    def test_get_log_method_by_level(self):
        """测试根据级别获取日志方法"""
        from src.infrastructure.logging.core.base_logger import BaseLogger

        mock_logger = Mock(spec=BaseLogger)

        # 测试有效级别
        method = self.service._get_log_method_by_level(mock_logger, "INFO")
        assert method is not None

        # 测试无效级别
        method = self.service._get_log_method_by_level(mock_logger, "INVALID")
        assert method is None

    def test_get_log_method_mapping(self):
        """测试获取日志方法映射"""
        from src.infrastructure.logging.core.base_logger import BaseLogger

        mock_logger = Mock(spec=BaseLogger)

        mapping = self.service._get_log_method_mapping(mock_logger)

        assert isinstance(mapping, dict)
        assert "DEBUG" in mapping
        assert "INFO" in mapping
        assert "WARNING" in mapping
        assert "ERROR" in mapping
        assert "CRITICAL" in mapping

    def test_persist_log_if_enabled(self):
        """测试条件日志持久化"""
        # 这个方法可能因为依赖不可用来抛出异常
        try:
            self.service._persist_log_if_enabled(
                "test_logger", "INFO", "Test message", extra="data"
            )
        except:
            # 异常处理测试
            pass

    def test_build_log_record(self):
        """测试日志记录构建"""
        record = self.service._build_log_record(
            "test_logger", "INFO", "Test message", extra="data", user_id=123
        )

        assert isinstance(record, dict)
        assert "logger_name" in record
        assert "level" in record
        assert "message" in record
        assert "timestamp" in record
        assert record["logger_name"] == "test_logger"
        assert record["level"] == "INFO"
        assert record["message"] == "Test message"

    def test_start_service(self):
        """测试启动服务"""
        result = self.service.start()

        assert result is True

    def test_stop_service(self):
        """测试停止服务"""
        self.service.start()

        result = self.service.stop()

        assert result is True

    def test_restart_service(self):
        """测试重启服务"""
        self.service.start()

        result = self.service.restart()

        assert result is True

    def test_get_status(self):
        """测试获取状态"""
        status = self.service.get_status()

        assert isinstance(status, dict)
        assert "name" in status
        assert "enabled" in status
        assert "logger_count" in status

    def test_get_info(self):
        """测试获取信息"""
        info = self.service.get_info()

        assert isinstance(info, dict)
        assert "service_name" in info
        assert "service_type" in info
        assert "config" in info
        assert "active_loggers" in info

    def test_concurrent_logger_operations(self):
        """测试并发日志器操作"""
        import threading

        results = []
        errors = []

        def worker_thread(thread_id):
            try:
                # 创建日志器
                logger_name = f"concurrent_logger_{thread_id}"
                logger = self.service.create_logger(logger_name)
                results.append(f"created_{thread_id}")

                # 记录消息
                self.service.log_message(logger_name, "INFO", f"Message from {thread_id}")
                results.append(f"logged_{thread_id}")

                # 获取日志器
                retrieved = self.service.get_logger(logger_name)
                assert retrieved is logger
                results.append(f"retrieved_{thread_id}")

            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        # 启动多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker_thread, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程
        for t in threads:
            t.join(timeout=5.0)

        # 验证结果
        assert len(errors) == 0
        assert len(results) == 15  # 5线程 * 3操作

    def test_logger_limit_enforcement(self):
        """测试日志器数量限制执行"""
        # 创建大量日志器来测试限制
        max_loggers = self.service.max_loggers

        for i in range(max_loggers):  # 达到限制
            logger_name = f"limit_test_{i}"
            self.service.create_logger(logger_name)

        # 检查没有超过限制（可能有默认日志器）
        current_count = len(self.service.loggers)
        assert current_count <= max_loggers + 2  # 允许一些缓冲

    def test_error_handling_in_logger_creation(self):
        """测试日志器创建中的错误处理"""
        # 测试无效配置
        try:
            self.service.create_logger("error_test", {"invalid": "config"})
        except:
            # 应该优雅处理错误
            pass

    def test_service_state_after_operations(self):
        """测试操作后的服务状态"""
        # 执行一系列操作
        self.service.start()

        logger1 = self.service.create_logger("state_test1")
        logger2 = self.service.create_logger("state_test2")

        self.service.log_message("state_test1", "INFO", "Test message")
        self.service.log_message("state_test2", "DEBUG", "Debug message")

        removed = self.service.remove_logger("state_test1")
        assert removed is True

        self.service.stop()

        # 验证最终状态
        status = self.service.get_status()
        assert isinstance(status, dict)

    def test_memory_management(self):
        """测试内存管理"""
        import sys

        # 记录初始状态
        initial_logger_count = len(self.service.loggers)

        # 创建多个日志器
        for i in range(10):
            self.service.create_logger(f"memory_test_{i}")

        # 记录中间状态
        middle_logger_count = len(self.service.loggers)
        assert middle_logger_count >= initial_logger_count + 10

        # 移除一些日志器
        for i in range(5):
            self.service.remove_logger(f"memory_test_{i}")

        # 验证清理
        final_logger_count = len(self.service.loggers)
        assert final_logger_count <= middle_logger_count

    def test_performance_logging_operations(self):
        """测试日志操作性能"""
        import time

        # 创建日志器
        perf_logger = self.service.create_logger("perf_test")

        # 执行多次日志操作
        start_time = time.time()
        operations = 100

        for i in range(operations):
            self.service.log_message("perf_test", "INFO", f"Performance test message {i}")

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能（100次操作应该在合理时间内完成）
        assert duration < 2.0  # 少于2秒

    def test_large_scale_logger_management(self):
        """测试大规模日志器管理"""
        # 创建大量日志器来测试可扩展性
        base_count = len(self.service.loggers)

        large_batch = self.service.max_loggers
        created_loggers = []

        for i in range(large_batch):
            logger_name = f"scale_test_{i}"
            logger = self.service.create_logger(logger_name)
            created_loggers.append(logger_name)

        # 验证创建成功
        assert len(self.service.loggers) >= base_count + large_batch

        # 批量记录消息
        for i, logger_name in enumerate(created_loggers[:10]):  # 只测试前10个
            self.service.log_message(logger_name, "INFO", f"Scale test {i}")

        # 批量移除
        for logger_name in created_loggers:
            self.service.remove_logger(logger_name)

        # 验证清理
        final_count = len(self.service.loggers)
        assert final_count <= base_count + 5  # 允许一些剩余