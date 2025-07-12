import time
from typing import Dict, Optional
from pydantic import BaseModel, validator, confloat, conint

from src.infrastructure.m_logging.advanced_logger import EnhancedTradingLogger


class CompressionConfig(BaseModel):
    """压缩配置验证模型"""
    algorithm: str = "zstd"
    level: conint(ge=1, le=22) = 3
    chunk_size: conint(ge=1024, le=10485760) = 1048576
    trading_hours: Dict[str, bool]

    @validator('algorithm')
    def check_algorithm(cls, v):
        if v not in ["zstd", "lz4", "gzip"]:
            raise ValueError("只支持 zstd/lz4/gzip 压缩算法")
        return v

class BackpressureConfig(BaseModel):
    """背压配置验证模型"""
    initial_rate: conint(ge=100, le=100000) = 1000
    max_rate: conint(ge=1000, le=500000) = 10000
    window_size: conint(ge=10, le=600) = 60
    backoff_factor: confloat(ge=0.1, le=0.9) = 0.5
    max_queue: conint(ge=1000, le=1000000) = 10000

    @validator('max_rate')
    def check_max_rate(cls, v, values):
        if 'initial_rate' in values and v < values['initial_rate']:
            raise ValueError("最大速率不能小于初始速率")
        return v

class SamplingConfig(BaseModel):
    """采样配置验证模型"""
    base_sample_rate: confloat(ge=0.01, le=1.0) = 1.0
    trading_hours: Dict[str, confloat(ge=0.01, le=1.0)]

    @validator('trading_hours')
    def check_hours(cls, v):
        required = ["night", "morning", "afternoon"]
        if not all(k in v for k in required):
            raise ValueError(f"必须包含所有交易时段配置: {required}")
        return v

class LoggingConfigValidator:
    """配置验证器"""

    @classmethod
    def validate_full_config(cls, config: Dict) -> Dict:
        """验证完整日志配置"""
        return {
            'compression': CompressionConfig(**config['compression']).dict(),
            'backpressure': BackpressureConfig(**config['backpressure']).dict(),
            'sampling': SamplingConfig(**config['sampling']).dict(),
            'security': config['security']  # 安全配置单独验证
        }

class LoadTestRunner:
    """压力测试运行器"""

    def __init__(self, config: Dict):
        self.config = config
        self.results = {}

    async def run_throughput_test(self, duration: int = 60):
        """吞吐量测试"""
        logger = EnhancedTradingLogger(self.config)
        await logger.start()

        start_time = time.monotonic()
        count = 0

        while time.monotonic() - start_time < duration:
            await logger.log("INFO", f"压力测试消息 {count}")
            count += 1

        await logger.stop()
        self.results['throughput'] = count / duration

    async def run_backpressure_test(self, burst_size: int = 100000):
        """背压测试"""
        # 实现略...
        pass

class CircuitBreakerConfig(BaseModel):
    """熔断器配置"""
    cpu_threshold: confloat(ge=0.5, le=0.95) = 0.8
    mem_threshold: confloat(ge=0.5, le=0.95) = 0.8
    recovery_time: conint(ge=10, le=3600) = 300
    enable_auto_reset: bool = True
