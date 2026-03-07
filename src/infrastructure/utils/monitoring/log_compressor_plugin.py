
import psutil
import threading
import zstandard as zstd

from datetime import datetime
from typing import Dict, Optional, Any
"""
基础设施层 - 日志系统组件

log_compressor_plugin 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""


class LogCompressorPlugin:
    """符合《证券期货业数据规范》的日志压缩模块"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 压缩配置 {}
                'algorithm': 'zstd',
                'level': 3,
                'chunk_size': 1048576,
                'thread_safe': True

        """
        self.config = config
        self.compressor = zstd.ZstdCompressor()
        level = (config.get("level", 3),)
        threads = config.get("threads", 2)

        self.chunk_size = config.get("chunk_size", 1024 * 1024)
        self.lock = threading.Lock() if config.get("thread_safe", True) else None
        self.strategy = config.get("strategy", "default")
        self.current_strategy = None  # 添加current_strategy属性

    def compress(self, data: bytes) -> bytes:
        """压缩日志数据"""
        if self.lock:
            with self.lock:
                return self.compressor.compress(data)
        return self.compressor.compress(data)

    def stream_compress(self, input_file: str, output_file: str):
        """流式压缩日志文件"""
        try:
            cctx = zstd.ZstdCompressor()
            with open(input_file, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    cctx.copy_stream(f_in, f_out)
        except FileNotFoundError as e:
            print(f"输入文件不存在: {input_file}")
            raise
        except PermissionError as e:
            print(f"文件权限错误: {e}")
            raise
        except Exception as e:
            print(f"流式压缩失败: {e}")
            raise

    def should_compress(self) -> bool:
        """判断是否应该压缩"""
        now = datetime.now().time()
        # 交易时段不压缩，其他时段压缩
        trading_start = datetime.strptime("09:30", "%H:%M").time()
        trading_end = datetime.strptime("15:00", "%H:%M").time()

        # 修复边界逻辑：在交易时段内（包括边界）不压缩
        if trading_start <= now <= trading_end:
            return False  # 交易时段不压缩
        return True  # 其他时段压缩

    def auto_select_strategy(self):
        """根据系统负载自动选择策略"""
        try:
            cpu_usage = psutil.cpu_percent()
            if cpu_usage > 70:
                self.current_strategy = "light"  # 修复：设置current_strategy而不是strategy
                return "light"
            else:
                self.current_strategy = "aggressive"  # 修复：设置current_strategy而不是strategy
                return "aggressive"
        except Exception as e:
            print(f"获取CPU使用率失败: {e}，使用默认策略")
            self.current_strategy = "default"
            return "default"
    
    def decompress(self, data: bytes) -> bytes:
        """解压缩数据"""
        dctx = zstd.ZstdDecompressor()
        if self.lock:
            with self.lock:
                return dctx.decompress(data)
        return dctx.decompress(data)
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        return {
            'algorithm': self.config.get('algorithm', 'zstd') if self.config else 'zstd',
            'level': self.config.get('level', 3) if self.config else 3,
            'chunk_size': self.chunk_size,
            'thread_safe': self.lock is not None,
            'strategy': self.strategy,
            'total_compressed_bytes': 0,
            'total_decompressed_bytes': 0,
            'compression_ratio': 0.0,
            'current_strategy': self.current_strategy or self.strategy
        }
    
    def get_supported_algorithms(self) -> list:
        """获取支持的压缩算法"""
        return ['zstd', 'gzip', 'bz2', 'lzma']
    
    def is_compression_effective(self, original_size: int, compressed_size: int) -> bool:
        """判断压缩是否有效"""
        if original_size == 0:
            return False
        compression_ratio = compressed_size / original_size
        return compression_ratio < 0.9  # 压缩率小于90%才认为有效
    
    def update_strategy(self, new_strategy: str) -> None:
        """更新压缩策略"""
        self.strategy = new_strategy
        self.current_strategy = new_strategy
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        required_keys = ['algorithm', 'level', 'chunk_size']
        return all(key in config for key in required_keys)


class TradingHoursAwareCompressor(LogCompressorPlugin):
    """交易时段感知的智能压缩器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)
        self.trading_hours = config.get(
            "trading_hours",
            {
                "morning": ("09:30", "11:30"),
                "afternoon": ("13:00", "15:00"),
                "night": ("21:00", "02:30"),
                "close": ("15:00", "21:00"),
            },
        )

    def should_compress(self) -> bool:
        """根据交易时段决定压缩策略"""
        now = datetime.now().time()
        # 将字符串时间转换为datetime.time对象进行比较
        night_start = datetime.strptime(self.trading_hours["night"][0], "%H:%M").time()
        morning_start = datetime.strptime(self.trading_hours["morning"][0], "%H:%M").time()
        close_start = datetime.strptime(self.trading_hours["close"][0], "%H:%M").time()

        if night_start <= now < morning_start:
            return True  # 夜盘时段高压缩
        elif morning_start <= now < close_start:
            return False  # 交易时段不压缩
        return True  # 其他时段压缩


class CompressionManager:
    """压缩策略管理器"""

    def __init__(self):

        self.compressors = {}
        self.current_strategy = None

    def register_strategy(self, name: str, compressor: LogCompressorPlugin):
        """注册压缩策略"""
        self.compressors[name] = compressor

    def auto_select_strategy(self):
        """根据系统负载自动选择策略"""
        # 根据CPU和内存使用率动态选择
        if psutil.cpu_percent(interval=1) > 70:
            self.current_strategy = self.compressors.get("light")
        else:
            self.current_strategy = self.compressors.get("aggressive")

    def get_cpu_usage() -> float:
        """获取CPU使用率"""
        return psutil.cpu_percent()
