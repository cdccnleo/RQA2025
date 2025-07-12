import pytest
import asyncio
from unittest.mock import patch, MagicMock, mock_open
import logging
import os
import tempfile
from src.infrastructure.m_logging.logger import TradingLogger, SecureLogger, AsyncLogWriter, AsyncTimedRotatingFileHandler

@pytest.fixture
def logger_config():
    return {
        'base_level': logging.INFO,
        'log_dir': tempfile.gettempdir(),
        'rotation': 'midnight',
        'backup_count': 2,
        'max_workers': 2,
        'security': {'secret_key': 'testkey'}
    }

@pytest.mark.asyncio
async def test_trading_logger_singleton_and_log(logger_config):
    # 清理单例
    TradingLogger._instance = None
    # Patch AsyncTimedRotatingFileHandler，避免真实文件IO和异步任务
    with patch('src.infrastructure.m_logging.logger.AsyncTimedRotatingFileHandler') as mock_handler:
        mock_handler.return_value = MagicMock()
        logger = await TradingLogger.initialize(logger_config)
        assert TradingLogger._instance is logger
        # Patch logger.logger.info
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log('INFO', 'test message', user='alice')
            mock_info.assert_called()
        await logger.shutdown()
        # 清理handlers，防止资源残留
        logger.logger.handlers.clear()
        TradingLogger._instance = None

@pytest.mark.asyncio
async def test_trading_logger_signature(logger_config):
    TradingLogger._instance = None
    logger = await TradingLogger.initialize(logger_config)
    # 验证签名
    message = 'secure log'
    log_entry = {'timestamp': '2024-01-01T00:00:00', 'message': message}
    signature = logger.secure_logger.sign_log(log_entry)
    assert isinstance(signature, str)
    assert len(signature) == 64  # sha256 hex长度
    await logger.shutdown()
    TradingLogger._instance = None

@pytest.mark.asyncio
async def test_async_log_writer_batch(monkeypatch):
    writer = AsyncLogWriter(max_workers=1)
    # Patch _write_batch为同步Mock
    called = {}
    def fake_write_batch(batch):
        called['count'] = len(batch)
    monkeypatch.setattr(writer, '_write_batch', fake_write_batch)
    await writer.start()
    # 填充队列
    for i in range(5):
        await writer.queue.put({'filepath': os.path.join(tempfile.gettempdir(), 'dummy.log'), 'message': f'msg{i}'})
    await asyncio.sleep(0.2)
    await writer.stop()
    assert called['count'] == 5

@pytest.mark.asyncio
async def test_async_timed_rotating_file_handler_emit(monkeypatch):
    # Patch emit父类为Mock
    with patch('src.infrastructure.m_logging.logger.TimedRotatingFileHandler.emit') as mock_emit:
        handler = AsyncTimedRotatingFileHandler(filename=os.path.join(tempfile.gettempdir(), 'dummy.log'), when='midnight', backupCount=1)
        record = logging.LogRecord('trading', logging.INFO, __file__, 1, 'msg', None, None)
        handler.emit(record)
        await asyncio.sleep(0.1)
        # 关闭handler
        await handler.close()
        assert mock_emit.called

@pytest.mark.asyncio
async def test_trading_logger_async_shutdown(logger_config):
    TradingLogger._instance = None
    logger = await TradingLogger.initialize(logger_config)
    await logger.shutdown()
    TradingLogger._instance = None 