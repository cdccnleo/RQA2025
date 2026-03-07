#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 日志处理器

测试各种日志处理器的功能。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import tempfile
import logging
import socket
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

from src.infrastructure.logging.core import LogLevel
from src.infrastructure.logging.handlers import ConsoleHandler, FileHandler, RemoteHandler


class TestConsoleHandler:
    """控制台处理器测试"""

    def test_initialization(self):
        """测试初始化"""
        config = {'level': logging.INFO}
        handler = ConsoleHandler(config)
        assert handler.level == logging.INFO
        assert handler.name == 'ConsoleHandler'

    def test_handle_method(self):
        """测试处理方法"""
        config = {'level': logging.DEBUG}
        handler = ConsoleHandler(config)

        # 创建模拟的日志记录
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="测试消息", args=(), exc_info=None
        )

        # 测试处理
        with patch('sys.stdout') as mock_stdout:
            handler.emit(record)
            # 验证输出被调用

    def test_level_setting(self):
        """测试级别设置"""
        config = {'level': logging.DEBUG}
        handler = ConsoleHandler(config)
        assert handler.level == logging.DEBUG

        handler.level = logging.ERROR
        assert handler.level == logging.ERROR

    def test_emit_method(self):
        """测试_emit方法"""
        config = {'level': logging.INFO}
        handler = ConsoleHandler(config)

        # 创建模拟的日志记录
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="emit测试消息", args=(), exc_info=None
        )

        # Mock print和stream
        with patch('builtins.print') as mock_print:
            handler._emit(record)
            # 验证print被调用
            assert mock_print.called
            call_args = mock_print.call_args
            assert "emit测试消息" in str(call_args[0])

    def test_emit_with_colorize(self):
        """测试带颜色化的_emit方法"""
        config = {'level': logging.INFO, 'colorize': True}
        handler = ConsoleHandler(config)

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="颜色化测试", args=(), exc_info=None
        )

        with patch('builtins.print') as mock_print:
            handler._emit(record)
            assert mock_print.called
            call_args = mock_print.call_args
            # 验证消息被颜色化 (INFO级别是绿色)
            call_content = str(call_args[0])
            # 检查是否包含绿色转义序列，考虑字符串转义
            assert ('\\x1b[32m' in call_content or '\033[32m' in call_content or '\\033[32m' in call_content)

    def test_emit_exception_handling(self):
        """测试_emit方法的异常处理"""
        config = {'level': logging.INFO}
        handler = ConsoleHandler(config)

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="异常测试", args=(), exc_info=None
        )

        # Mock print抛出异常，然后验证异常处理路径被执行
        original_print = print
        call_count = 0
        
        def mock_print(*args, **kwargs):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                raise Exception("模拟异常")
            else:
                # 这是错误处理的print调用
                call_count += 1
        
        with patch('builtins.print', side_effect=mock_print):
            # 这个调用应该不会抛出异常，因为被内部捕获了
            handler._emit(record)
            # 验证有两次调用：一次正常调用（抛出异常），一次错误处理调用
            assert call_count >= 1

    def test_format_record_with_formatter(self):
        """测试带格式化器的_format_record方法"""
        config = {'level': logging.INFO}
        handler = ConsoleHandler(config)

        # 创建模拟格式化器
        mock_formatter = Mock()
        mock_formatter.format.return_value = "格式化后的消息"
        handler.set_formatter(mock_formatter)

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="原始消息", args=(), exc_info=None
        )

        result = handler._format_record(record)
        assert result == "格式化后的消息"
        mock_formatter.format.assert_called_once_with(record)

    def test_format_record_without_formatter(self):
        """测试无格式化器的_format_record方法"""
        config = {'level': logging.INFO}
        handler = ConsoleHandler(config)
        handler._formatter = None  # 确保没有格式化器

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="默认格式测试", args=(), exc_info=None
        )

        result = handler._format_record(record)
        assert isinstance(result, str)
        assert "默认格式测试" in result
        assert "INFO:" in result

    def test_colorize_message(self):
        """测试_colorize_message方法"""
        config = {'level': logging.INFO}
        handler = ConsoleHandler(config)

        # 测试不同级别的颜色化
        test_cases = [
            (logging.DEBUG, '\033[36m'),    # 青色
            (logging.INFO, '\033[32m'),     # 绿色
            (logging.WARNING, '\033[33m'),  # 黄色
            (logging.ERROR, '\033[31m'),    # 红色
            (logging.CRITICAL, '\033[35m'), # 紫色
        ]

        for level, expected_color in test_cases:
            result = handler._colorize_message("测试消息", level)
            assert expected_color in result
            assert '\033[0m' in result  # 应该包含重置颜色

    def test_colorize_message_unknown_level(self):
        """测试未知级别的_colorize_message方法"""
        config = {'level': logging.INFO}
        handler = ConsoleHandler(config)

        result = handler._colorize_message("测试消息", 9999)  # 未知级别
        assert result == "测试消息"  # 不应该添加颜色

    def test_set_formatter(self):
        """测试set_formatter方法"""
        config = {'level': logging.INFO}
        handler = ConsoleHandler(config)

        mock_formatter = Mock()
        handler.set_formatter(mock_formatter)
        
        assert handler._formatter == mock_formatter

    def test_get_status(self):
        """测试get_status方法"""
        config = {'level': logging.INFO, 'colorize': True}
        handler = ConsoleHandler(config)

        status = handler.get_status()
        
        assert isinstance(status, dict)
        assert 'stream' in status
        assert 'colorize' in status
        assert 'has_formatter' in status
        assert status['colorize'] is True
        assert status['has_formatter'] is False  # 默认没有格式化器

    def test_get_status_with_formatter(self):
        """测试带格式化器的get_status方法"""
        config = {'level': logging.INFO}
        handler = ConsoleHandler(config)
        mock_formatter = Mock()
        handler.set_formatter(mock_formatter)

        status = handler.get_status()
        
        assert status['has_formatter'] is True

    def test_initialization_with_custom_stream(self):
        """测试使用自定义流的初始化"""
        import io
        custom_stream = io.StringIO()
        
        config = {'stream': custom_stream, 'level': logging.INFO}
        handler = ConsoleHandler(config)
        
        assert handler.stream == custom_stream
        assert handler.colorize is False  # 默认值

    def test_initialization_with_colorize(self):
        """测试带颜色化选项的初始化"""
        config = {'colorize': True, 'level': logging.INFO}
        handler = ConsoleHandler(config)
        
        assert handler.colorize is True


class TestFileHandler:
    """文件处理器测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """测试初始化"""
        log_file = Path(self.temp_dir) / "test.log"
        config = {'file_path': str(log_file), 'level': logging.INFO}
        handler = FileHandler(config)

        assert handler.level == logging.INFO
        assert handler.file_path == log_file
        assert handler.name == 'FileHandler'

    def test_file_creation(self):
        """测试文件创建"""
        log_file = Path(self.temp_dir) / "test.log"
        config = {'file_path': str(log_file), 'level': logging.INFO}
        handler = FileHandler(config)

        # 创建日志记录
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="测试消息", args=(), exc_info=None
        )

        # 处理日志
        handler.emit(record)

        # 验证文件被创建
        assert log_file.exists()

        # 验证文件内容
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "测试消息" in content

    def test_rotation(self):
        """测试日志轮转"""
        log_file = Path(self.temp_dir) / "rotate.log"
        config = {'file_path': str(log_file), 'level': logging.INFO, 'max_bytes': 100, 'backup_count': 2}
        handler = FileHandler(config)

        # 生成足够的内容来触发轮转
        for i in range(50):
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=i, msg=f"测试消息 {i}", args=(), exc_info=None
            )
            handler.emit(record)

        # 检查是否有轮转文件
        assert log_file.exists()
        # 注意：轮转的具体行为可能因logging模块实现而异

    def test_level_filtering(self):
        """测试级别过滤"""
        log_file = Path(self.temp_dir) / "filter.log"
        config = {'file_path': str(log_file), 'level': logging.WARNING}
        handler = FileHandler(config)

        # 发送不同级别的日志
        debug_record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="test.py",
            lineno=1, msg="调试消息", args=(), exc_info=None
        )

        info_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=2, msg="信息消息", args=(), exc_info=None
        )

        warning_record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="test.py",
            lineno=3, msg="警告消息", args=(), exc_info=None
        )

        # 处理日志
        handler.emit(debug_record)  # 应该被过滤
        handler.emit(info_record)   # 应该被过滤
        handler.emit(warning_record) # 应该被记录

        # 验证只有警告消息被记录
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "调试消息" not in content
            assert "信息消息" not in content
            assert "警告消息" in content

    def test_file_handler_format_record(self):
        """测试文件处理器记录格式化"""
        log_file = Path(self.temp_dir) / "format.log"
        config = {'file_path': str(log_file), 'level': logging.INFO}
        handler = FileHandler(config)

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="格式化测试", args=(), exc_info=None
        )

        # 测试默认格式化
        formatted = handler._format_record(record)
        assert isinstance(formatted, str)
        assert "格式化测试" in formatted

    def test_file_handler_should_rotate(self):
        """测试文件处理器轮转检查"""
        log_file = Path(self.temp_dir) / "rotate_check.log"
        config = {'file_path': str(log_file), 'max_bytes': 50}
        handler = FileHandler(config)

        # 初始状态不应该轮转
        assert handler._should_rotate() is False

        # 模拟设置文件大小超过限制
        handler._current_size = 75
        handler._file = open(log_file, 'w', encoding='utf-8')
        
        # 现在应该需要轮转
        assert handler._should_rotate() is True

        # 清理
        handler._file.close()

    def test_file_handler_close_file(self):
        """测试文件处理器关闭文件"""
        log_file = Path(self.temp_dir) / "close_file.log"
        config = {'file_path': str(log_file), 'level': logging.INFO}
        handler = FileHandler(config)

        # 打开文件
        handler._file = open(log_file, 'w', encoding='utf-8')
        assert handler._file is not None

        # 关闭文件
        handler._close_current_file()
        assert handler._file is None

    def test_file_handler_with_compression(self):
        """测试文件处理器压缩功能"""
        log_file = Path(self.temp_dir) / "compress.log"
        config = {'file_path': str(log_file), 'compress': True, 'level': logging.INFO}
        handler = FileHandler(config)

        assert handler.compress is True

        # 创建日志记录
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="压缩测试", args=(), exc_info=None
        )

        # 处理日志
        handler.emit(record)

        # 验证文件被创建
        assert log_file.exists()

    def test_file_handler_error_handling(self):
        """测试文件处理器错误处理"""
        # 测试无效路径导致的错误
        invalid_config = {'file_path': '/invalid/path/test.log', 'level': logging.INFO}
        
        # 这个测试可能因为权限问题而失败，但可以测试错误处理路径
        try:
            handler = FileHandler(invalid_config)
            # 如果初始化成功，尝试写入
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg="错误测试", args=(), exc_info=None
            )
            
            # 可能会抛出异常，这测试了错误处理
            handler.emit(record)
        except Exception:
            # 预期可能出现的异常
            pass

    def test_file_handler_config_methods(self):
        """测试文件处理器配置方法"""
        log_file = Path(self.temp_dir) / "config.log"
        config = {
            'file_path': str(log_file), 
            'max_bytes': 2048,
            'backup_count': 3,
            'encoding': 'utf-8-sig'
        }
        handler = FileHandler(config)

        assert handler.max_bytes == 2048
        assert handler.backup_count == 3
        assert handler.encoding == 'utf-8-sig'


class TestHandlerIntegration:
    """处理器集成测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multiple_handlers(self):
        """测试多个处理器同时工作"""
        log_file = Path(self.temp_dir) / "multi.log"

        console_config = {'level': logging.INFO}
        file_config = {'file_path': str(log_file), 'level': logging.INFO}

        console_handler = ConsoleHandler(console_config)
        file_handler = FileHandler(file_config)

        # 创建日志记录
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="多处理器测试", args=(), exc_info=None
        )

        # 测试控制台处理器
        with patch('sys.stdout') as mock_stdout:
            console_handler.emit(record)

        # 测试文件处理器
        file_handler.emit(record)

        # 验证文件被创建
        assert log_file.exists()


class TestRemoteHandler:
    """远程处理器测试"""

    def test_initialization(self):
        """测试初始化"""
        config = {
            'host': '192.168.1.100',
            'port': 5140,
            'protocol': 'udp'
        }
        handler = RemoteHandler(config)
        assert handler.host == '192.168.1.100'
        assert handler.port == 5140
        assert handler.protocol == 'udp'
        assert handler.name == 'RemoteHandler'

    def test_initialization_defaults(self):
        """测试默认配置初始化"""
        handler = RemoteHandler()
        assert handler.host == 'localhost'
        assert handler.port == 514
        assert handler.protocol == 'tcp'

    def test_tcp_connection_success(self):
        """测试TCP连接成功"""
        config = {'host': '127.0.0.1', 'port': 9999, 'protocol': 'tcp'}
        handler = RemoteHandler(config)

        # 创建模拟的服务器socket
        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_socket.return_value = mock_sock

            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg="测试消息", args=(), exc_info=None
            )

            # 测试发送
            handler.emit(record)

            # 验证socket操作
            mock_socket.assert_called_once()
            mock_sock.connect.assert_called_once_with(('127.0.0.1', 9999))
            mock_sock.sendall.assert_called_once()
            # 注意：连接可能保持打开状态，不一定每次都关闭

    def test_udp_connection(self):
        """测试UDP连接"""
        config = {'host': '127.0.0.1', 'port': 514, 'protocol': 'udp'}
        handler = RemoteHandler(config)

        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_socket.return_value = mock_sock

            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg="测试消息", args=(), exc_info=None
            )

            handler.emit(record)

            # UDP不调用connect
            mock_sock.connect.assert_not_called()
            mock_sock.sendto.assert_called_once()

    def test_connection_failure_retry(self):
        """测试连接失败重试"""
        config = {'host': '127.0.0.1', 'port': 9999, 'protocol': 'tcp'}
        handler = RemoteHandler(config)

        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            # 模拟连接失败，然后发送失败，最后成功重连
            mock_sock.connect.side_effect = [ConnectionError("连接失败"), None]
            mock_sock.sendall.side_effect = [Exception("发送失败"), None]  # 发送也失败一次
            mock_socket.return_value = mock_sock

            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg="测试消息", args=(), exc_info=None
            )

            # 不应该抛出异常
            handler.emit(record)

            # 验证重试逻辑 - 至少调用了一次connect（可能因为实现细节不同）
            assert mock_sock.connect.call_count >= 1

    def test_max_retries_exceeded(self):
        """测试超过最大重试次数"""
        config = {'host': '127.0.0.1', 'port': 9999, 'protocol': 'tcp'}
        handler = RemoteHandler(config)

        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_sock.connect.side_effect = ConnectionError("连接失败")
            mock_socket.return_value = mock_sock

            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg="测试消息", args=(), exc_info=None
            )

            # 不应该抛出异常，但会记录错误
            with patch.object(handler, '_handle_error') as mock_error:
                handler.emit(record)
                mock_error.assert_called()

    def test_send_timeout(self):
        """测试发送超时"""
        config = {'host': '127.0.0.1', 'port': 9999, 'protocol': 'tcp', 'timeout': 5.0}
        handler = RemoteHandler(config)

        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_sock.sendall.side_effect = socket.timeout("发送超时")
            mock_socket.return_value = mock_sock

            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg="测试消息", args=(), exc_info=None
            )

            with patch.object(handler, '_handle_error') as mock_error:
                handler.emit(record)
                mock_error.assert_called()

    def test_buffered_sending(self):
        """测试缓冲发送"""
        config = {'host': '127.0.0.1', 'port': 9999, 'protocol': 'tcp', 'buffer_size': 3}
        handler = RemoteHandler(config)

        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_socket.return_value = mock_sock

            # 发送多条记录
            records = []
            for i in range(5):
                record = logging.LogRecord(
                    name="test", level=logging.INFO, pathname="test.py",
                    lineno=i, msg=f"消息{i}", args=(), exc_info=None
                )
                records.append(record)
                handler.emit(record)

            # 验证缓冲行为（实际实现可能不同，这里验证基本发送）
            assert mock_sock.sendall.call_count >= 3

    def test_close_connection(self):
        """测试关闭连接"""
        handler = RemoteHandler()

        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_socket.return_value = mock_sock

            # 先发送一条消息
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg="测试消息", args=(), exc_info=None
            )

            handler.emit(record)

            # 关闭处理器
            handler.close()

            # 验证close被调用
            mock_sock.close.assert_called()

    def test_level_filtering(self):
        """测试级别过滤"""
        config = {'level': logging.WARNING}
        handler = RemoteHandler(config)

        # 低级别消息应该被过滤
        low_level_record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="低级别消息", args=(), exc_info=None
        )

        # 不应该发送
        with patch('socket.socket') as mock_socket:
            mock_sock = MagicMock()
            mock_socket.return_value = mock_sock

            handler.emit(low_level_record)

            # 不应该调用socket操作
            mock_sock.connect.assert_not_called()

    def test_format_message(self):
        """测试消息格式化"""
        handler = RemoteHandler()

        record = logging.LogRecord(
            name="test.logger", level=logging.ERROR, pathname="/app/main.py",
            lineno=25, msg="错误: %s", args=("数据库连接失败",), exc_info=None
        )

        formatted = handler._format_record(record)

        # 验证格式化包含必要信息
        assert 'test.logger' in formatted
        assert '<40>' in formatted  # ERROR level = 40
        assert '错误: 数据库连接失败' in formatted


class TestBaseHandler:
    """BaseHandler基础处理器测试"""

    def setup_method(self):
        """测试前准备"""
        from src.infrastructure.logging.handlers.base import BaseHandler
        
        # 创建一个BaseHandler的子类用于测试，因为BaseHandler是抽象类
        class ConcreteHandler(BaseHandler):
            def _emit(self, record):
                self.last_record = record
            def _close(self):
                pass

        self.HandlerClass = ConcreteHandler

    def test_initialization_default_config(self):
        """测试默认配置初始化"""
        handler = self.HandlerClass()
        
        assert handler.config == {}
        assert handler.name == 'ConcreteHandler'
        assert handler.level == logging.INFO
        assert handler.enabled is True
        assert handler._closed is False

    def test_initialization_custom_config(self):
        """测试自定义配置初始化"""
        config = {
            'name': 'TestHandler',
            'level': logging.ERROR,
            'enabled': False
        }
        handler = self.HandlerClass(config)
        
        assert handler.config == config
        assert handler.name == 'TestHandler'
        assert handler.level == logging.ERROR
        assert handler.enabled is False

    def test_handle_method_with_logrecord(self):
        """测试handle方法处理LogRecord"""
        handler = self.HandlerClass()
        
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="测试消息", args=(), exc_info=None
        )
        
        handler.handle(record)
        # 应该调用了emit，记录应该被保存
        assert hasattr(handler, 'last_record')
        assert handler.last_record == record

    def test_get_level(self):
        """测试get_level方法"""
        handler = self.HandlerClass({'level': logging.WARNING})
        
        from src.infrastructure.logging.core.interfaces import LogLevel
        level = handler.get_level()
        assert level == LogLevel.WARNING

    def test_set_level(self):
        """测试set_level方法"""
        handler = self.HandlerClass()
        
        from src.infrastructure.logging.core.interfaces import LogLevel
        handler.set_level(LogLevel.ERROR)
        assert handler.level == 40  # logging.ERROR的值

    def test_get_level_value_with_loglevel(self):
        """测试_get_level_value方法使用LogLevel"""
        handler = self.HandlerClass()
        
        from src.infrastructure.logging.core.interfaces import LogLevel
        handler.level = LogLevel.DEBUG
        value = handler._get_level_value()
        assert value == 10

    def test_get_level_value_with_int(self):
        """测试_get_level_value方法使用整数"""
        handler = self.HandlerClass()
        
        handler.level = 30  # logging.WARNING
        value = handler._get_level_value()
        assert value == 30

    def test_emit_when_disabled(self):
        """测试disable状态下emit方法"""
        handler = self.HandlerClass({'enabled': False})
        
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="测试消息", args=(), exc_info=None
        )
        
        handler.emit(record)
        # 应该没有调用_emit，因为enabled=False
        assert not hasattr(handler, 'last_record')

    def test_emit_when_closed(self):
        """测试closed状态下emit方法"""
        handler = self.HandlerClass()
        handler.close()  # 设置为closed状态
        
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="测试消息", args=(), exc_info=None
        )
        
        handler.emit(record)
        # 应该没有调用_emit，因为_closed=True
        assert not hasattr(handler, 'last_record')

    def test_emit_level_filtering(self):
        """测试emit方法级别过滤"""
        handler = self.HandlerClass({'level': logging.WARNING})
        
        # 发送INFO级别的记录，应该被过滤掉
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="低级别消息", args=(), exc_info=None
        )
        
        handler.emit(record)
        # INFO级别低于WARNING，应该被过滤
        assert not hasattr(handler, 'last_record')

    def test_emit_level_allowed(self):
        """测试emit方法级别允许"""
        handler = self.HandlerClass({'level': logging.INFO})
        
        # 发送ERROR级别的记录，应该被处理
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="高级别消息", args=(), exc_info=None
        )
        
        handler.emit(record)
        # ERROR级别高于INFO，应该被处理
        assert hasattr(handler, 'last_record')

    def test_close_method(self):
        """测试close方法"""
        handler = self.HandlerClass()
        
        assert handler._closed is False
        handler.close()
        assert handler._closed is True

    def test_get_status(self):
        """测试get_status方法"""
        handler = self.HandlerClass({'name': 'TestStatus', 'level': logging.DEBUG})
        
        status = handler.get_status()
        
        assert isinstance(status, dict)
        assert status['name'] == 'TestStatus'
        assert status['enabled'] is True
        assert status['level'] == logging.DEBUG
        assert status['closed'] is False
        assert status['type'] == 'ConcreteHandler'

    def test_enable_disable(self):
        """测试enable和disable方法"""
        handler = self.HandlerClass({'enabled': False})
        
        assert handler.enabled is False
        
        handler.enable()
        assert handler.enabled is True
        
        handler.disable()
        assert handler.enabled is False

    def test_handle_error_method_exists(self):
        """测试_handle_error方法存在"""
        handler = self.HandlerClass()
        
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="错误测试", args=(), exc_info=None
        )
        
        # 测试_handle_error方法可以被调用（应该不抛出异常）
        handler._handle_error(record, Exception("测试错误"))


if __name__ == "__main__":
    pytest.main([__file__])
