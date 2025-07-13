import zstandard as zstd
import threading
from datetime import datetime
from typing import Optional, Dict
import psutil

class LogCompressor:
    """符合《证券期货业数据规范》的日志压缩模块"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 压缩配置 {
                'algorithm': 'zstd',
                'level': 3,
                'chunk_size': 1048576,
                'thread_safe': True
            }
        """
        self.config = config
        self.compressor = zstd.ZstdCompressor(
            level=config.get('level', 3),
            threads=config.get('threads', 2)
        )
        self.chunk_size = config.get('chunk_size', 1024*1024)
        self.lock = threading.Lock() if config.get('thread_safe', True) else None
        self.strategy = config.get('strategy', 'default')

    def compress(self, data: bytes) -> bytes:
        """压缩日志数据"""
        if self.lock:
            with self.lock:
                return self.compressor.compress(data)
        return self.compressor.compress(data)

    def stream_compress(self, input_file: str, output_file: str):
        """流式压缩日志文件"""
        cctx = zstd.ZstdCompressor()
        with open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                cctx.copy_stream(f_in, f_out)

    def should_compress(self) -> bool:
        """判断是否应该压缩"""
        now = datetime.now().time()
        # 交易时段不压缩，其他时段压缩
        trading_start = datetime.strptime('09:30', '%H:%M').time()
        trading_end = datetime.strptime('15:00', '%H:%M').time()
        
        if trading_start <= now <= trading_end:
            return False  # 交易时段不压缩
        return True  # 其他时段压缩

    def auto_select_strategy(self):
        """根据系统负载自动选择策略"""
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > 70:
            self.strategy = 'light'
        else:
            self.strategy = 'aggressive'

class TradingHoursAwareCompressor(LogCompressor):
    """交易时段感知的智能压缩器"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.trading_hours = config.get('trading_hours', {
            'morning': ('09:30', '11:30'),
            'afternoon': ('13:00', '15:00'),
            'night': ('21:00', '02:30'),
            'close': ('15:00', '21:00')
        })

    def should_compress(self) -> bool:
        """根据交易时段决定压缩策略"""
        now = datetime.now().time()
        if self.trading_hours['night'][0] <= now < self.trading_hours['morning'][0]:
            return True  # 夜盘时段高压缩
        elif self.trading_hours['morning'][0] <= now < self.trading_hours['close'][0]:
            return False  # 交易时段不压缩
        return True  # 其他时段压缩

class CompressionManager:
    """压缩策略管理器"""

    def __init__(self):
        self.compressors = {}
        self.current_strategy = None

    def register_strategy(self, name: str, compressor: LogCompressor):
        """注册压缩策略"""
        self.compressors[name] = compressor

    def auto_select_strategy(self):
        """根据系统负载自动选择策略"""
        # 根据CPU和内存使用率动态选择
        if get_cpu_usage() > 70:
            self.current_strategy = self.compressors.get('light')
        else:
            self.current_strategy = self.compressors.get('aggressive')

def get_cpu_usage() -> float:
    """获取CPU使用率"""
    return psutil.cpu_percent()
