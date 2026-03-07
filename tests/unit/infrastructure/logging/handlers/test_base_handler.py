"""
测试基础日志处理器

覆盖 base.py 中的 BaseHandler 类
"""

import logging
from unittest.mock import Mock, patch
from src.infrastructure.logging.handlers.base import BaseHandler
from src.infrastructure.logging.core.interfaces import LogLevel


class TestBaseHandler:
    """BaseHandler 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        handler = BaseHandler()

        assert handler.config == {}
        assert handler.name == "BaseHandler"
        assert handler.level == logging.INFO
        assert handler.enabled == True
        assert handler._closed == False

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'name': 'CustomHandler',
            'level': logging.DEBUG,
            'enabled': False
        }
        handler = BaseHandler(config)

        assert handler.config == config
        assert handler.name == "CustomHandler"
        assert handler.level == logging.DEBUG
        assert handler.enabled == False

    def test_handle_with_log_record(self):
        """测试处理LogRecord"""
        handler = BaseHandler()

        with patch.object(handler, 'emit') as mock_emit:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Test message",
                args=(),
                exc_info=None
            )

            handler.handle(record)
            mock_emit.assert_called_once_with(record)

    def test_handle_with_other_record(self):
        """测试处理其他格式记录"""
        handler = BaseHandler()

        with patch.object(handler, 'emit') as mock_emit:
            record = "some other record format"

            handler.handle(record)
            mock_emit.assert_called_once_with(record)

    def test_get_level(self):
        """测试获取级别"""
        handler = BaseHandler()
        handler.level = 20  # INFO level

        assert handler.get_level() == LogLevel.INFO

        handler.level = 40  # ERROR level
        assert handler.get_level() == LogLevel.ERROR

        handler.level = 99  # Unknown level
        assert handler.get_level() == LogLevel.INFO  # Default

    def test_set_level(self):
        """测试设置级别"""
        handler = BaseHandler()

        handler.set_level(LogLevel.DEBUG)
        assert handler.level == 10

        handler.set_level(LogLevel.WARNING)
        assert handler.level == 30

        handler.set_level(LogLevel.CRITICAL)
        assert handler.level == 50

    def test_get_level_value_with_enum(self):
        """测试获取级别值（枚举）"""
        handler = BaseHandler()
        handler.level = LogLevel.DEBUG

        assert handler._get_level_value() == 10

        handler.level = LogLevel.ERROR
        assert handler._get_level_value() == 40

    def test_get_level_value_with_int(self):
        """测试获取级别值（整数）"""
        handler = BaseHandler()
        handler.level = 25

        assert handler._get_level_value() == 25

    def test_emit_disabled_handler(self):
        """测试禁用处理器不发出日志"""
        handler = BaseHandler()
        handler.enabled = False

        with patch.object(handler, '_emit') as mock_emit:
            record = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)
            handler.emit(record)
            mock_emit.assert_not_called()

    def test_emit_closed_handler(self):
        """测试关闭处理器不发出日志"""
        handler = BaseHandler()
        handler._closed = True

        with patch.object(handler, '_emit') as mock_emit:
            record = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)
            handler.emit(record)
            mock_emit.assert_not_called()

    def test_emit_level_filtering(self):
        """测试级别过滤"""
        handler = BaseHandler()
        handler.level = logging.WARNING  # 30

        with patch.object(handler, '_emit') as mock_emit:
            # 低于当前级别的日志不应该发出
            debug_record = logging.LogRecord("test", logging.DEBUG, "", 0, "", (), None)  # 10
            info_record = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)    # 20

            handler.emit(debug_record)
            handler.emit(info_record)

            mock_emit.assert_not_called()

            # 高于等于当前级别的日志应该发出
            warning_record = logging.LogRecord("test", logging.WARNING, "", 0, "", (), None)  # 30
            error_record = logging.LogRecord("test", logging.ERROR, "", 0, "", (), None)      # 40

            handler.emit(warning_record)
            handler.emit(error_record)

            assert mock_emit.call_count == 2

    def test_emit_error_handling(self):
        """测试发出错误处理"""
        handler = BaseHandler()

        record = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)

        with patch.object(handler, '_emit', side_effect=Exception("Emit failed")) as mock_emit:
            with patch.object(handler, '_handle_error') as mock_handle_error:
                handler.emit(record)

                mock_emit.assert_called_once_with(record)
                mock_handle_error.assert_called_once_with(record, mock_emit.side_effect)

    def test_emit_method_exists(self):
        """测试_emit方法存在"""
        handler = BaseHandler()

        # BaseHandler提供了_emit的默认实现（空实现）
        record = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)
        # 不应该抛出NotImplementedError
        try:
            handler._emit(record)
            assert True  # 期望的行为 - 方法存在且可以调用
        except NotImplementedError:
            assert False, "BaseHandler should implement _emit method"

    def test_close(self):
        """测试关闭处理器"""
        handler = BaseHandler()

        with patch.object(handler, '_close') as mock_close:
            handler.close()

            assert handler._closed == True
            mock_close.assert_called_once()

    def test_get_status(self):
        """测试获取状态"""
        config = {'name': 'TestHandler', 'level': logging.DEBUG}
        handler = BaseHandler(config)

        status = handler.get_status()

        assert status['name'] == 'TestHandler'
        assert status['enabled'] == True
        assert status['level'] == logging.DEBUG
        assert status['closed'] == False
        assert status['type'] == 'BaseHandler'

    def test_enable_disable(self):
        """测试启用和禁用"""
        handler = BaseHandler()

        handler.disable()
        assert handler.enabled == False

        handler.enable()
        assert handler.enabled == True

    def test_handle_error_default(self):
        """测试默认错误处理"""
        handler = BaseHandler()
        record = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)
        error = Exception("Test error")

        # 默认实现应该是静默的，不会抛出异常
        handler._handle_error(record, error)
