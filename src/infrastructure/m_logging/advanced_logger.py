import asyncio
import datetime
import random
import time
from typing import Dict
from .log_compressor import TradingHoursAwareCompressor
from .backpressure import AdaptiveBackpressure, TradingSampler, BackpressureError
from .performance_monitor import LoggingMetrics

class EnhancedTradingLogger:
    """增强版交易日志器（集成压缩/背压/采样）"""

    def __init__(self, config: Dict):
        """
        Args:
            config: {
                'compression': {...},
                'backpressure': {...},
                'sampling': {...},
                'security': {...}
            }
        """
        # 核心组件初始化
        self.compressor = TradingHoursAwareCompressor(config['compression'])
        self.backpressure = AdaptiveBackpressure(config['backpressure'])
        self.sampler = TradingSampler(config['sampling'])
        self.metrics = LoggingMetrics()

        # 运行时状态
        self._queue = asyncio.Queue(maxsize=config['backpressure'].get('max_queue', 10000))
        self._is_running = False

    async def start(self):
        """启动日志处理循环"""
        self._is_running = True
        asyncio.create_task(self._process_loop())

    async def stop(self):
        """安全停止"""
        self._is_running = False
        await self._queue.join()

    async def log(self, level: str, message: str, **kwargs):
        """记录增强日志"""
        # 1. 采样决策
        if not self._should_sample():
            return

        # 2. 背压保护
        try:
            await self.backpressure.protect()
        except BackpressureError:
            self.metrics.record_drop()
            return

        # 3. 异步处理
        log_entry = self._build_entry(level, message, kwargs)
        await self._queue.put(log_entry)

    async def _process_loop(self):
        """日志处理主循环"""
        while self._is_running or not self._queue.empty():
            entry = await self._queue.get()

            try:
                # 压缩处理
                if self.compressor.should_compress():
                    entry = self._compress_entry(entry)

                # 写入存储
                await self._write_entry(entry)

            except Exception as e:
                self.metrics.record_error(type(e).__name__)
            finally:
                self._queue.task_done()

    def _should_sample(self) -> bool:
        """决定是否采样当前日志"""
        rate = self.sampler.get_sample_rate()
        sampled = random.random() < rate
        self.metrics.record_sample(rate, sampled)
        return sampled

    def _build_entry(self, level: str, message: str, meta: Dict) -> Dict:
        """构建日志条目"""
        return {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'meta': meta,
            'signature': self._sign_entry(message, meta)
        }

    def _compress_entry(self, entry: Dict) -> Dict:
        """压缩日志条目"""
        original_size = len(entry['message'])
        compressed = self.compressor.compress(entry['message'].encode())
        entry['message'] = compressed
        entry['compression'] = {
            'algorithm': 'zstd',
            'ratio': original_size / len(compressed)
        }
        self.metrics.record_compression(original_size, len(compressed))
        return entry

    def _sign_entry(self, message: str, meta: Dict) -> str:
        """生成日志签名"""
        # 简单实现
        return f"signature_{hash(message)}"

    async def _write_entry(self, entry: Dict):
        """写入日志存储"""
        # 实现略...
        pass

class EmergencySwitch:
    """应急开关（熔断机制）"""

    def __init__(self):
        self._is_triggered = False

    def trigger(self):
        """触发熔断"""
        self._is_triggered = True

    def reset(self):
        """重置"""
        self._is_triggered = False

    def check(self) -> bool:
        """检查状态"""
        return not self._is_triggered

class AdvancedLogger:
    """高级日志记录器"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = EnhancedTradingLogger({
            'compression': {},
            'backpressure': {},
            'sampling': {},
            'security': {}
        })
    
    def log_structured(self, event_type: str, data: Dict):
        """记录结构化日志"""
        message = f"{event_type}: {data}"
        asyncio.create_task(self.logger.log("INFO", message))
    
    def performance_log(self, operation: str):
        """性能日志上下文管理器"""
        return PerformanceLogContext(self, operation)
    
    def log_exception(self, error_type: str, exception: Exception):
        """记录异常日志"""
        message = f"{error_type}: {str(exception)}"
        asyncio.create_task(self.logger.log("ERROR", message))

class PerformanceLogContext:
    """性能日志上下文管理器"""
    
    def __init__(self, logger: AdvancedLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            message = f"Performance: {self.operation} took {duration:.3f}s"
            asyncio.create_task(self.logger.logger.log("INFO", message))
