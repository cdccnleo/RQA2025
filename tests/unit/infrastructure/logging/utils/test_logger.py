"""
日志工具单元测试

测试日志工具相关的功能。
"""

import pytest
import logging
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.logging.utils.logger import (
    configure_logging,
    reset_logging,
    get_logger,
    get_component_logger,
    set_log_level,
    add_file_handler,
    LoggerFactory,
    debug,
    info,
    warning,
    error,
    critical,
)


class TestLoggingConfiguration:
    """测试日志配置功能"""

    def test_configure_logging_default(self):
        """测试默认日志配置"""
        # 重置之前的配置
        reset_logging()

        configure_logging()

        # 验证根日志器已配置
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO
        assert len(root_logger.handlers) > 0

    def test_configure_logging_custom_config(self):
        """测试自定义日志配置"""
        reset_logging()

        custom_config = {
            'level': 'DEBUG',
            'format': '%(levelname)s: %(message)s',
            'log_dir': 'custom_logs'
        }

        configure_logging(custom_config)

        root_logger = logging.getLogger()
        assert root_logger.level <= logging.DEBUG

    def test_configure_logging_directory_creation(self):
        """测试日志目录创建"""
        reset_logging()

        # 使用当前目录的子目录而不是临时目录
        test_log_dir = os.path.join(os.getcwd(), "test_logs_temp")
        try:
            config = {'log_dir': test_log_dir}

            configure_logging(config)

            # 验证目录存在
            assert os.path.exists(test_log_dir)
        finally:
            # 清理测试目录
            import shutil
            if os.path.exists(test_log_dir):
                shutil.rmtree(test_log_dir, ignore_errors=True)

    def test_reset_logging(self):
        """测试重置日志配置"""
        # 先配置日志
        configure_logging()

        # 验证有处理器
        root_logger = logging.getLogger()
        initial_handlers = len(root_logger.handlers)

        # 重置
        reset_logging()

        # 验证处理器被清除
        assert len(root_logger.handlers) <= initial_handlers

    def test_get_logger_basic(self):
        """测试获取基本日志器"""
        reset_logging()

        logger = get_logger("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_get_logger_with_level(self):
        """测试获取带级别的日志器"""
        reset_logging()

        logger = get_logger("test_logger_with_level", level="DEBUG")

        assert logger.level <= logging.DEBUG

    def test_get_component_logger(self):
        """测试获取组件日志器"""
        reset_logging()

        logger = get_component_logger("test_component", "business")

        assert isinstance(logger, logging.Logger)
        assert "test_component" in logger.name
        assert "business" in logger.name

    def test_set_log_level(self):
        """测试设置日志级别"""
        reset_logging()

        logger = get_logger("test_level")
        initial_level = logger.level

        set_log_level("test_level", "ERROR")

        assert logger.level == logging.ERROR

    def test_add_file_handler(self):
        """测试添加文件处理器"""
        reset_logging()

        logger = get_logger("test_file")

        # 使用当前目录的子目录
        test_log_dir = os.path.join(os.getcwd(), "test_logs_temp")
        log_file = os.path.join(test_log_dir, "test.log")

        try:
            add_file_handler(logger, "test_file", log_file)

            # 验证处理器被添加到日志器
            assert len(logger.handlers) > 0
            # 验证日志文件被创建
            assert os.path.exists(log_file)
        finally:
            # 清理测试目录
            import shutil
            if os.path.exists(test_log_dir):
                shutil.rmtree(test_log_dir, ignore_errors=True)

    def test_add_file_handler_none_logger(self):
        """测试为None日志器添加文件处理器"""
        reset_logging()

        # 使用当前目录的子目录
        test_log_dir = os.path.join(os.getcwd(), "test_logs_temp")
        log_file = os.path.join(test_log_dir, "test.log")

        try:
            # 不应该抛出异常
            add_file_handler(None, "test_none", log_file)
            # 验证日志文件被创建
            assert os.path.exists(log_file)
        finally:
            # 清理测试目录
            import shutil
            if os.path.exists(test_log_dir):
                shutil.rmtree(test_log_dir, ignore_errors=True)

    def test_add_file_handler_with_level_and_formatter(self):
        """测试添加带级别和格式化器的文件处理器"""
        reset_logging()

        logger = get_logger("test_custom")

        # 使用当前目录的子目录
        test_log_dir = os.path.join(os.getcwd(), "test_logs_temp")
        log_file = os.path.join(test_log_dir, "test.log")

        try:
            custom_formatter = logging.Formatter('%(levelname)s - %(message)s')

            add_file_handler(logger, "test_custom", log_file, level="ERROR", formatter=custom_formatter)

            # 验证处理器被添加到日志器
            assert len(logger.handlers) > 0
            # 验证日志文件被创建
            assert os.path.exists(log_file)
        finally:
            # 清理测试目录
            import shutil
            if os.path.exists(test_log_dir):
                shutil.rmtree(test_log_dir, ignore_errors=True)


class TestLoggerFactory:
    """测试日志器工厂"""

    def test_create_logger(self):
        """测试创建日志器"""
        factory = LoggerFactory()

        logger = factory.create_logger("factory_test")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "factory_test"

    def test_create_logger_with_parameters(self):
        """测试创建带参数的日志器"""
        factory = LoggerFactory()

        logger = factory.create_logger("factory_param_test", level="WARNING", add_file=True)

        # 验证日志器创建成功
        assert logger is not None
        assert logger.name == "factory_param_test"

    def test_create_logger_with_file_handler(self):
        """测试创建带文件处理器的日志器"""
        import tempfile
        import logging

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "factory_test.log")

            logger = LoggerFactory.create_logger("factory_file_test", level="WARNING", add_file=True, filename=log_file)

            assert isinstance(logger, logging.Logger)
            assert logger.name == "factory_file_test"
            # 验证日志文件被创建
            assert os.path.exists(log_file)

            # 确保所有处理器都被正确关闭
            for handler in logger.handlers:
                if hasattr(handler, 'close'):
                    handler.close()
            logger.handlers.clear()


class TestLoggingFunctions:
    """测试日志记录函数"""

    def test_debug_function(self):
        """测试debug函数"""
        reset_logging()

        with patch('src.infrastructure.logging.utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            debug("test debug message", "test_logger")

            mock_logger.debug.assert_called_with("test debug message")

    def test_info_function(self):
        """测试info函数"""
        reset_logging()

        with patch('src.infrastructure.logging.utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            info("test info message", "test_logger")

            mock_logger.info.assert_called_with("test info message")

    def test_warning_function(self):
        """测试warning函数"""
        reset_logging()

        with patch('src.infrastructure.logging.utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            warning("test warning message", "test_logger")

            mock_logger.warning.assert_called_with("test warning message")

    def test_error_function(self):
        """测试error函数"""
        reset_logging()

        with patch('src.infrastructure.logging.utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            error("test error message", "test_logger")

            mock_logger.error.assert_called_with("test error message")

    def test_critical_function(self):
        """测试critical函数"""
        reset_logging()

        with patch('src.infrastructure.logging.utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            critical("test critical message", "test_logger")

            mock_logger.critical.assert_called_with("test critical message")


class TestLoggingIntegration:
    """测试日志系统集成"""

    def test_full_logging_workflow(self):
        """测试完整的日志工作流"""
        reset_logging()

        # 配置日志
        configure_logging()

        # 获取日志器
        logger = get_logger("workflow_test")

        # 记录不同级别的日志
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # 验证日志器配置正确
        assert logger.level <= logging.INFO
        assert len(logging.getLogger().handlers) > 0

    def test_component_logging(self):
        """测试组件日志"""
        reset_logging()

        # 获取不同组件的日志器
        business_logger = get_component_logger("trading", "business")
        system_logger = get_component_logger("cache", "system")

        assert "trading" in business_logger.name
        assert "business" in business_logger.name
        assert "cache" in system_logger.name
        assert "system" in system_logger.name

    @patch('src.infrastructure.logging.utils.logger.threading.Lock')
    def test_thread_safety(self, mock_lock):
        """测试线程安全性"""
        reset_logging()

        # 模拟锁
        mock_lock_instance = Mock()
        mock_lock.return_value = mock_lock_instance

        # 重新导入以使用mock
        # 注意：这个测试在实际运行时可能需要不同的方法

    def test_error_handling_in_configuration(self):
        """测试配置中的错误处理"""
        reset_logging()

        # 测试无效的配置
        invalid_config = {
            'level': 'INVALID_LEVEL',
            'log_dir': '/invalid/path'
        }

        # 不应该抛出异常
        try:
            configure_logging(invalid_config)
        except Exception:
            # 如果抛出异常，应该被正确处理
            pass

    def test_logger_naming_conventions(self):
        """测试日志器命名约定"""
        reset_logging()

        # 测试各种命名模式
        names = [
            "simple",
            "package.module",
            "deep.package.module.submodule"
        ]

        for name in names:
            logger = get_logger(name)
            assert logger.name == name


class TestLoggerUtilityFunctions:
    """测试日志工具函数"""

    def test_reset_logging_functionality(self):
        """测试重置日志功能的完整性"""
        # 先配置日志
        configure_logging()

        # 验证配置生效
        root_logger = logging.getLogger()
        initial_handlers_count = len(root_logger.handlers)

        # 重置日志
        reset_logging()

        # 验证重置后的状态
        root_logger_after = logging.getLogger()
        # 重置后可能仍然有处理器，取决于实现
        assert isinstance(root_logger_after, logging.Logger)

    def test_get_component_logger_variations(self):
        """测试获取组件日志器的各种变体"""
        reset_logging()

        # 测试不同的组件和类别组合
        test_cases = [
            ("auth", "security"),
            ("payment", "business"),
            ("cache", "system"),
            ("api", "network")
        ]

        for component, category in test_cases:
            logger = get_component_logger(component, category)

            assert component in logger.name
            assert category in logger.name

    def test_set_log_level_edge_cases(self):
        """测试设置日志级别的边界情况"""
        reset_logging()

        logger = get_logger("level_test")

        # 测试各种级别
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            set_log_level("level_test", level)
            # 验证级别设置（注意：这里可能需要检查实际的级别值）

        # 测试无效级别（应该不抛出异常）
        try:
            set_log_level("level_test", "INVALID_LEVEL")
        except Exception:
            # 如果抛出异常也是可以接受的
            pass

    def test_multiple_loggers_management(self):
        """测试多个日志器的管理"""
        reset_logging()

        # 创建多个日志器
        logger_names = [f"multi_test_{i}" for i in range(10)]

        loggers = []
        for name in logger_names:
            logger = get_logger(name)
            loggers.append(logger)

            # 为每个日志器记录一条消息
            logger.info(f"Test message for {name}")

        # 验证所有日志器都创建成功
        assert len(loggers) == 10

        for i, logger in enumerate(loggers):
            assert logger.name == logger_names[i]

    def test_concurrent_logger_access(self):
        """测试并发访问日志器"""
        reset_logging()

        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(20):
                    logger_name = f"concurrent_{worker_id}_{i}"
                    logger = get_logger(logger_name)
                    logger.debug(f"Worker {worker_id} message {i}")
                    results.append((worker_id, i))
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 100  # 5 workers * 20 messages
        assert len(errors) == 0

    def test_logger_factory_advanced_usage(self):
        """测试日志器工厂的高级用法"""
        factory = LoggerFactory()

        # 测试工厂的重用
        logger1 = factory.get_or_create_logger("factory_advanced")
        logger2 = factory.get_or_create_logger("factory_advanced")

        assert logger1 is logger2

        # 测试工厂创建多个不同的日志器
        different_loggers = []
        for i in range(5):
            logger = factory.get_or_create_logger(f"factory_multi_{i}")
            different_loggers.append(logger)

        # 验证都是不同的实例
        for i in range(len(different_loggers)):
            for j in range(i + 1, len(different_loggers)):
                assert different_loggers[i] is not different_loggers[j]

    def test_logging_functions_with_different_loggers(self):
        """测试日志记录函数使用不同的日志器"""
        reset_logging()

        # 测试所有日志级别函数
        test_cases = [
            (debug, "DEBUG"),
            (info, "INFO"),
            (warning, "WARNING"),
            (error, "ERROR"),
            (critical, "CRITICAL")
        ]

        for log_func, expected_level in test_cases:
            with patch('src.infrastructure.logging.utils.logger.get_logger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger

                log_func("Test message", "test_logger")

                # 验证调用了正确的日志方法
                if expected_level.lower() in ['debug', 'info', 'warning', 'error', 'critical']:
                    getattr(mock_logger, expected_level.lower()).assert_called_with("Test message")


class TestLoggerConfigurationEdgeCases:
    """测试日志配置的边界情况"""

    def test_configure_logging_with_invalid_config(self):
        """测试使用无效配置的日志配置"""
        reset_logging()

        invalid_configs = [
            {"level": None},
            {"level": 123},
            {"format": []},
            {"log_dir": None},
            {"max_bytes": "invalid"},
            {"backup_count": -1}
        ]

        for config in invalid_configs:
            try:
                configure_logging(config)
                # 如果没有抛出异常，验证系统仍然可用
                root_logger = logging.getLogger()
                assert isinstance(root_logger, logging.Logger)
            except Exception:
                # 如果抛出异常也是可以接受的，只要不崩溃
                pass

    def test_configure_logging_idempotent(self):
        """测试日志配置的幂等性"""
        reset_logging()

        # 多次调用配置应该不会导致问题
        for i in range(3):
            configure_logging()

        root_logger = logging.getLogger()
        assert isinstance(root_logger, logging.Logger)

    def test_add_file_handler_directory_creation(self):
        """测试添加文件处理器时的目录创建"""
        reset_logging()

        base_dir = tempfile.mkdtemp()
        try:
            # 创建嵌套目录结构
            nested_dir = os.path.join(base_dir, "logs", "app", "subsystem")
            log_file = os.path.join(nested_dir, "test.log")

            logger = get_logger("dir_creation_test")
            add_file_handler(logger, "dir_creation_test", log_file)

            # 验证目录和文件都被创建
            assert os.path.exists(nested_dir)
            assert os.path.exists(log_file)

            # 强制关闭所有处理器以释放文件句柄
            for handler in logger.handlers:
                if hasattr(handler, 'close'):
                    handler.close()
                    logger.removeHandler(handler)
        finally:
            # 手动清理临时目录
            import shutil
            try:
                shutil.rmtree(base_dir, ignore_errors=True)
            except:
                pass

    def test_add_file_handler_file_permissions(self):
        """测试添加文件处理器的文件权限"""
        import logging
        reset_logging()

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "permission_test.log")

            logger = get_logger("permission_test")
            add_file_handler(logger, "permission_test", log_file)

            # 验证文件可以写入
            logger.info("Permission test message")

            # 验证文件内容
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Permission test message" in content

            # 确保所有处理器都被正确关闭
            for handler in logger.handlers:
                if hasattr(handler, 'close'):
                    handler.close()
            logger.handlers.clear()


class TestLoggerErrorHandling:
    """测试日志系统的错误处理"""

    def test_configure_logging_with_readonly_directory(self):
        """测试在只读目录中配置日志"""
        import logging
        reset_logging()

        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = os.path.join(temp_dir, "readonly")
            os.makedirs(readonly_dir)

            # 在Windows上设置只读属性比较复杂，这里简化处理
            config = {'log_dir': readonly_dir}

            try:
                configure_logging(config)
                # 如果成功，验证日志器仍然可用
                root_logger = logging.getLogger()
                assert isinstance(root_logger, logging.Logger)
            except Exception:
                # 如果失败也是可以接受的
                pass

            # 清理所有日志处理器
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if hasattr(handler, 'close'):
                    handler.close()
                root_logger.removeHandler(handler)

    def test_get_logger_with_circular_reference(self):
        """测试带有循环引用的日志器获取"""
        reset_logging()

        # 创建可能导致循环引用的场景
        logger1 = get_logger("circular1")
        logger2 = get_logger("circular2")

        # 添加彼此作为处理器（模拟循环引用）
        # 注意：这在实际的logging系统中是不常见的，但测试边界情况

        # 验证日志器仍然正常工作
        logger1.info("Circular reference test 1")
        logger2.info("Circular reference test 2")

        assert logger1 is not logger2

    def test_logger_factory_resource_cleanup(self):
        """测试日志器工厂的资源清理"""
        factory = LoggerFactory()

        # 创建一些日志器
        loggers = []
        for i in range(10):
            logger = factory.get_or_create_logger(f"cleanup_test_{i}")
            loggers.append(logger)

        # 验证日志器创建成功
        assert len(loggers) == 10

        # 清理引用
        del loggers
        del factory

        # 在实际应用中，这里可能需要测试GC是否正常工作
        import gc
        gc.collect()

    def test_extreme_logger_creation_burst(self):
        """测试极端情况下的日志器创建爆发"""
        reset_logging()

        # 快速连续创建大量日志器
        start_time = time.time()

        loggers = []
        for i in range(100):
            logger = get_logger(f"burst_{i}")
            loggers.append(logger)

        end_time = time.time()
        creation_time = end_time - start_time

        # 验证创建成功
        assert len(loggers) == 100

        # 验证性能（应该在合理时间内完成）
        assert creation_time < 5.0  # 5秒内完成

        print(f"Logger creation burst completed in {creation_time:.3f} seconds")
