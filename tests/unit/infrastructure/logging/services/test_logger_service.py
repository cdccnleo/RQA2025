"""
Logger Service 单元测试

测试日志服务的核心功能。
"""

import pytest
import logging
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

from src.infrastructure.logging.services.logger_service import LoggerService, LoggerWrapper
from src.infrastructure.logging.core.interfaces import LogLevel
from src.infrastructure.logging.core.base_logger import BaseLogger


class TestLoggerWrapper:
    """测试LoggerWrapper类"""

    def test_init(self):
        """测试LoggerWrapper初始化"""
        mock_logger = Mock()
        mock_logger.name = "test_logger"
        wrapper = LoggerWrapper(mock_logger)

        assert wrapper.logger == mock_logger
        assert wrapper.name == "test_logger"
        assert wrapper._handlers == []

    def test_log_methods(self):
        """测试各种日志级别方法"""
        mock_logger = Mock()
        wrapper = LoggerWrapper(mock_logger)

        # 测试debug
        wrapper.debug("test debug")
        mock_logger.debug.assert_called_once_with("test debug")

        # 测试info
        wrapper.info("test info")
        mock_logger.info.assert_called_once_with("test info")

        # 测试warning
        wrapper.warning("test warning")
        mock_logger.warning.assert_called_once_with("test warning")

        # 测试error
        wrapper.error("test error")
        mock_logger.error.assert_called_once_with("test error")

        # 测试critical
        wrapper.critical("test critical")
        mock_logger.critical.assert_called_once_with("test critical")

    def test_log_method(self):
        """测试通用log方法"""
        mock_logger = Mock()
        wrapper = LoggerWrapper(mock_logger)

        # 测试正常级别
        wrapper.log("info", "test message")
        mock_logger.info.assert_called_once_with("test message")

        # 测试不存在的级别（应该使用info）
        wrapper.log("unknown", "test message")
        mock_logger.info.assert_called_with("test message")

    def test_add_handler(self):
        """测试添加处理器"""
        mock_logger = Mock()
        wrapper = LoggerWrapper(mock_logger)

        mock_handler = Mock()
        wrapper.add_handler(mock_handler)

        assert mock_handler in wrapper._handlers

        # 再次添加同一个处理器，不应该重复
        wrapper.add_handler(mock_handler)
        assert wrapper._handlers.count(mock_handler) == 1

    def test_remove_handler(self):
        """测试移除处理器"""
        mock_logger = Mock()
        wrapper = LoggerWrapper(mock_logger)

        mock_handler = Mock()
        wrapper.add_handler(mock_handler)
        assert mock_handler in wrapper._handlers

        wrapper.remove_handler(mock_handler)
        assert mock_handler not in wrapper._handlers

    def test_shutdown(self):
        """测试关闭"""
        mock_logger = Mock()
        wrapper = LoggerWrapper(mock_logger)

        mock_handler1 = Mock()
        mock_handler2 = Mock()
        wrapper.add_handler(mock_handler1)
        wrapper.add_handler(mock_handler2)

        # 验证添加成功
        assert len(wrapper._handlers) == 2

        wrapper.shutdown()

        # 验证shutdown后handlers被清空
        assert len(wrapper._handlers) == 0
        # 验证logger的removeHandler被调用
        mock_logger.removeHandler.assert_any_call(mock_handler1)
        mock_logger.removeHandler.assert_any_call(mock_handler2)


class TestLoggerService:
    """测试LoggerService类"""

    def test_init_default(self):
        """测试默认初始化"""
        service = LoggerService()

        assert service.default_level == "INFO"
        assert service.max_loggers == 100
        assert service.enable_persistence is True
        assert service.auto_create_missing is True
        assert isinstance(service.loggers, dict)
        assert "root" in service.loggers

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'default_level': 'DEBUG',
            'max_loggers': 50,
            'enable_persistence': False,
            'auto_create_missing': False,
            'log_file': 'custom.log'
        }
        service = LoggerService(config)

        assert service.default_level == "DEBUG"
        assert service.max_loggers == 50
        assert service.enable_persistence is False
        assert service.auto_create_missing is False

    def test_create_logger_basic(self):
        """测试创建基本logger"""
        service = LoggerService()
        logger = service.create_logger("test_logger")

        assert isinstance(logger, LoggerWrapper)
        assert logger.logger.name == "test_logger"

    def test_create_logger_with_config(self):
        """测试创建带配置的logger"""
        service = LoggerService()
        config = {
            'level': 'DEBUG',
            'handlers': ['console']
        }

        logger = service.create_logger("test_logger", config)
        assert logger is not None
        assert "test_logger" in service.loggers

    def test_get_logger_existing(self):
        """测试获取已存在的logger"""
        service = LoggerService()
        # 先创建
        logger1 = service.create_logger("test_logger")

        # 再获取
        logger2 = service.get_logger("test_logger")

        assert logger1 is logger2

    def test_get_logger_nonexistent_auto_create(self):
        """测试获取不存在的logger（自动创建）"""
        service = LoggerService()
        logger = service.get_logger("new_logger")

        assert logger is not None
        assert "new_logger" in service.loggers

    def test_get_logger_nonexistent_no_auto_create(self):
        """测试获取不存在的logger（不自动创建）"""
        config = {'auto_create_missing': False}
        service = LoggerService(config)

        logger = service.get_logger("nonexistent")
        assert logger is None

    def test_remove_logger(self):
        """测试移除logger"""
        service = LoggerService()
        service.create_logger("test_logger")

        assert "test_logger" in service.loggers

        result = service.remove_logger("test_logger")
        assert result is True
        assert "test_logger" not in service.loggers

    def test_remove_nonexistent_logger(self):
        """测试移除不存在的logger"""
        service = LoggerService()
        result = service.remove_logger("nonexistent")

        assert result is False

    def test_list_loggers(self):
        """测试列出所有loggers"""
        service = LoggerService()
        service.create_logger("logger1")
        service.create_logger("logger2")

        loggers = service.list_loggers()

        assert "logger1" in loggers
        assert "logger2" in loggers
        assert "root" in loggers

    def test_log_message_success(self):
        """测试成功记录日志消息"""
        service = LoggerService()
        service.create_logger("test_logger")

        result = service.log_message("test_logger", "info", "test message")

        assert result is True

    def test_log_message_nonexistent_logger(self):
        """测试向不存在的logger记录消息（自动创建）"""
        service = LoggerService()

        result = service.log_message("nonexistent", "info", "test message")

        # 默认情况下会自动创建logger，所以应该成功
        assert result is True
        assert "nonexistent" in service.loggers

    def test_log_message_invalid_level(self):
        """测试无效日志级别"""
        service = LoggerService()
        service.create_logger("test_logger")

        result = service.log_message("test_logger", "invalid_level", "test message")

        assert result is False

    @patch('src.infrastructure.logging.services.logger_service.MemoryStorage')
    def test_persist_log(self, mock_storage_class):
        """测试日志持久化"""
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        service = LoggerService({'enable_persistence': True})
        service.create_logger("test_logger")

        service.log_message("test_logger", "info", "test message")

        # 验证存储被调用
        mock_storage.store.assert_called()

    @patch('src.infrastructure.logging.services.logger_service.MemoryStorage')
    def test_no_persist_when_disabled(self, mock_storage_class):
        """测试禁用持久化时不存储"""
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        service = LoggerService({'enable_persistence': False})
        service.create_logger("test_logger")

        service.log_message("test_logger", "info", "test message")

        # 验证存储没有被调用
        mock_storage.store.assert_not_called()

    def test_start_service(self):
        """测试启动服务"""
        service = LoggerService()
        result = service.start()

        assert result is True
        assert service.is_running

    def test_stop_service(self):
        """测试停止服务"""
        service = LoggerService()
        service.start()

        result = service.stop()

        assert result is True
        assert not service.is_running

    def test_get_status(self):
        """测试获取服务状态"""
        service = LoggerService()
        service.create_logger("test_logger")

        status = service.get_status()

        # 检查状态包含必要字段（来自BaseService）
        assert 'enabled' in status
        assert 'default_level' in status

    def test_get_info(self):
        """测试获取服务信息"""
        service = LoggerService()

        info = service.get_info()

        # 检查LoggerService特有的字段
        assert 'active_loggers' in info
        assert 'capabilities' in info

    def test_logger_limit_validation(self):
        """测试logger数量限制"""
        config = {'max_loggers': 3}  # 包含root logger
        service = LoggerService(config)

        # 创建允许数量的logger
        service.create_logger("logger1")
        service.create_logger("logger2")

        # 应该有3个logger（包含root）
        assert len(service.loggers) == 3

    def test_default_handlers_setup(self):
        """测试默认处理器设置"""
        service = LoggerService()

        # 验证root logger存在并有处理器
        assert "root" in service.loggers
        root_logger = service.loggers["root"]
        # 验证LoggerWrapper有handlers
        assert hasattr(root_logger, '_handlers')

    def test_build_log_record(self):
        """测试构建日志记录"""
        service = LoggerService()

        record = service._build_log_record("test_logger", "info", "test message", extra="data")

        assert record['logger_name'] == "test_logger"
        assert record['level'] == "info"
        assert record['message'] == "test message"
        assert record['extra'] == {"extra": "data"}
        assert 'timestamp' in record

    def test_log_method_mapping(self):
        """测试日志方法映射"""
        service = LoggerService()
        mock_logger = Mock()

        mapping = service._get_log_method_mapping(mock_logger)

        # 检查大写的日志级别
        assert 'DEBUG' in mapping
        assert 'INFO' in mapping
        assert 'WARNING' in mapping
        assert 'ERROR' in mapping
        assert 'CRITICAL' in mapping

        # 验证映射的方法是可调用的
        assert callable(mapping['INFO'])
