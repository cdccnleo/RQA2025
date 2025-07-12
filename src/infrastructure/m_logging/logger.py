import datetime
import logging
import os
import asyncio
import hmac
import hashlib
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor

class AsyncLogWriter:
    """异步日志写入器"""
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.queue = asyncio.Queue()
        self._running = False

    async def start(self):
        """启动异步写入任务"""
        self._running = True
        asyncio.create_task(self._process_queue())

    async def stop(self):
        """停止写入器并等待队列清空"""
        self._running = False
        await self.queue.join()

    async def _process_queue(self):
        """处理日志队列"""
        while self._running or not self.queue.empty():
            batch = []
            try:
                # 批量获取日志条目
                while len(batch) < 100:  # 每批最多100条
                    log_entry = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=0.1
                    )
                    batch.append(log_entry)
            except asyncio.TimeoutError:
                pass

            if batch:
                # 在线程池中执行同步IO
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._write_batch,
                    batch
                )
                for _ in batch:
                    self.queue.task_done()

    def _write_batch(self, batch: List[dict]):
        """批量写入日志"""
        with open(batch[0]['filepath'], 'a') as f:
            for entry in batch:
                f.write(entry['message'] + '\n')

class SecureLogger:
    """安全日志增强组件"""
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()

    def sign_log(self, log_entry: dict) -> str:
        """生成日志签名"""
        msg = f"{log_entry['timestamp']}{log_entry['message']}".encode()
        return hmac.new(
            key=self.secret_key,
            msg=msg,
            digestmod=hashlib.sha256
        ).hexdigest()

class TradingLogger:
    """增强版交易日志器"""
    _instance = None

    def __init__(self, config: Dict):
        if TradingLogger._instance is not None:
            raise Exception("Logger is a singleton!")

        self.config = config
        self.async_writer = AsyncLogWriter(
            max_workers=config.get('max_workers', 4)
        )
        self.secure_logger = SecureLogger(
            config['security']['secret_key']
        )
        self._setup_loggers()

        TradingLogger._instance = self

    def _setup_loggers(self):
        """初始化日志处理器"""
        # 核心日志配置
        self.logger = logging.getLogger('trading')
        self.logger.setLevel(self.config['base_level'])

        # 异步文件处理器
        file_handler = AsyncTimedRotatingFileHandler(
            filename=os.path.join(
                self.config['log_dir'],
                'trading.log'
            ),
            when=self.config['rotation'],
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        ))
        self.logger.addHandler(file_handler)

    @classmethod
    async def initialize(cls, config: Dict):
        """异步初始化"""
        logger = cls(config)
        await logger.async_writer.start()
        return logger

    async def shutdown(self):
        """安全关闭"""
        await self.async_writer.stop()

    def log(self, level: str, message: str, **kwargs):
        """记录带签名的日志"""
        log_entry = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'signature': self.secure_logger.sign_log({
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'message': message
            })
        }
        log_entry.update(kwargs)

        getattr(self.logger, level.lower())(message, extra=log_entry)

class AsyncTimedRotatingFileHandler(TimedRotatingFileHandler):
    """异步时间滚动文件处理器"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_queue = asyncio.Queue()
        self._writer_task = asyncio.create_task(self._write_loop())

    async def _write_loop(self):
        """异步写入循环"""
        while True:
            record = await self._log_queue.get()
            super().emit(record)
            self._log_queue.task_done()

    def emit(self, record):
        """异步提交日志记录"""
        self._log_queue.put_nowait(record)

    async def close(self):
        """安全关闭处理器"""
        await self._log_queue.join()
        self._writer_task.cancel()
        super().close()
