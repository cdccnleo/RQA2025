import asyncio
import datetime
import time
from typing import Optional, Dict
from prometheus_client import Gauge, Counter

class TokenBucket:
    """令牌桶流量控制器"""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: 令牌生成速率(个/秒)
            capacity: 桶容量
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_time = time.monotonic()

        # 监控指标
        self.metrics = {
            'tokens': Gauge('log_bucket_tokens', '当前令牌数'),
            'drops': Counter('log_backpressure_drops', '丢弃日志数')
        }

    async def consume(self, tokens: int = 1) -> bool:
        """消费令牌，返回是否允许通过"""
        now = time.monotonic()
        elapsed = now - self._last_time
        self._last_time = now

        # 添加新令牌
        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self._rate
        )
        self.metrics['tokens'].set(self._tokens)

        # 检查令牌是否足够
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True

        self.metrics['drops'].inc()
        return False

class AdaptiveBackpressure:
    """自适应背压控制器"""

    def __init__(self, config: Dict):
        """
        Args:
            config: {
                'initial_rate': 1000,  # 默认1000
                'max_rate': 10000,     # 默认10000
                'window_size': 60,     # 默认60
                'backoff_factor': 0.5  # 默认0.5
            }
        """
        # 设置默认值
        initial_rate = config.get('initial_rate', 1000)
        max_rate = config.get('max_rate', 10000)
        window_size = config.get('window_size', 60)
        backoff_factor = config.get('backoff_factor', 0.5)
        
        self.bucket = TokenBucket(
            rate=initial_rate,
            capacity=initial_rate * 2
        )
        self.max_rate = config['max_rate']
        self.window_size = config['window_size']
        self.backoff_factor = config['backoff_factor']
        self._rates = []

    async def adjust_rate(self, current_load: float):
        """根据系统负载动态调整速率"""
        self._rates.append(current_load)
        if len(self._rates) > self.window_size:
            self._rates.pop(0)

        avg_load = sum(self._rates) / len(self._rates)
        new_rate = min(
            self.max_rate,
            self.bucket._rate * (1 + (0.8 - avg_load))
        )
        self.bucket._rate = max(100, new_rate)  # 保持最小100/s

    async def protect(self):
        """保护系统免受过载"""
        if not await self.bucket.consume():
            # 触发降级策略
            await self._apply_backoff()
            raise BackpressureError("系统过载，拒绝日志写入")

    async def _apply_backoff(self):
        """指数退避"""
        self.bucket._rate *= self.backoff_factor
        await asyncio.sleep(1)

class BackpressureError(Exception):
    """背压异常"""
    pass

class TradingSampler:
    """交易时段动态采样器"""

    TRADING_SCHEDULE = {
        'night': ((21, 0), (2, 30)),   # 夜盘时段
        'morning': ((9, 30), (11, 30)), # 早盘
        'afternoon': ((13, 0), (15, 0)) # 午盘
    }

    def __init__(self, config: Dict):
        """
        Args:
            config: {
                'base_sample_rate': 1.0,  # 默认1.0(全采样)
                'trading_hours': {  # 可选，默认为空
                    'night': 0.2,
                    'morning': 0.8,
                    'afternoon': 0.6
                }
            }
        """
        self.base_rate = config.get('base_sample_rate', 1.0)  # 默认全采样
        self.rates = config.get('trading_hours', {})  # 默认为空字典

    def current_period(self) -> str:
        """获取当前交易时段"""
        now = datetime.now()
        current_time = now.hour + now.minute/60

        for period, ((sh, sm), (eh, em)) in self.TRADING_SCHEDULE.items():
            start = sh + sm/60
            end = eh + em/60
            if start <= current_time < end:
                return period
        return 'off'  # 非交易时段

    def get_sample_rate(self) -> float:
        """获取当前采样率"""
        period = self.current_period()
        return self.base_rate * self.rates.get(period, 0.3)

class BackpressureHandler:
    """背压处理器"""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.backpressure = AdaptiveBackpressure({
            'initial_rate': 1000,
            'max_rate': 10000,
            'window_size': 60,
            'backoff_factor': 0.5
        })
    
    def handle_log(self, message: str) -> bool:
        """处理日志消息，返回是否成功"""
        try:
            if self.queue.qsize() < self.max_queue_size:
                asyncio.create_task(self.queue.put(message))
                return True
            else:
                return False
        except Exception:
            return False
    
    async def process_logs(self):
        """处理队列中的日志"""
        while True:
            try:
                message = await self.queue.get()
                await self.backpressure.protect()
                # 处理日志消息
                await self._process_message(message)
            except BackpressureError:
                # 系统过载，丢弃消息
                pass
            except Exception as e:
                # 记录错误
                print(f"Error processing log: {e}")
    
    async def _process_message(self, message: str):
        """处理单个日志消息"""
        # 实际处理逻辑
        pass
