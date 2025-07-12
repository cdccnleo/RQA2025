import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
import asyncio
import os
from pathlib import Path

class MarketDataDeduplicator:
    """优化的行情数据去重处理器"""

    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: 去重时间窗口(秒)
        """
        self.window_size = window_size
        self.last_hashes: Dict[str, tuple] = {}  # symbol: (hash, timestamp)

    def _generate_hash(self, tick_data: Dict) -> str:
        """生成行情数据指纹"""
        key_fields = {
            'symbol': tick_data['symbol'],
            'price': tick_data['price'],
            'volume': tick_data['volume'],
            'bid': tick_data.get('bid', []),
            'ask': tick_data.get('ask', [])
        }
        return hashlib.sha256(str(key_fields).encode()).hexdigest()

    def is_duplicate(self, tick_data: Dict) -> bool:
        """判断是否重复数据"""
        symbol = tick_data['symbol']
        current_hash = self._generate_hash(tick_data)
        now = datetime.now()

        if symbol in self.last_hashes:
            last_hash, timestamp = self.last_hashes[symbol]
            if current_hash == last_hash and (now - timestamp).seconds < self.window_size:
                return True

        self.last_hashes[symbol] = (current_hash, now)
        return False

class TradingHoursAwareCircuitBreaker:
    """交易时段感知的智能熔断器"""

    def __init__(self, schedule: Dict):
        """
        Args:
            schedule: 交易时段配置 {
                'morning': {'start': '09:30', 'end': '11:30', 'threshold': 0.8},
                'afternoon': {'start': '13:00', 'end': '15:00', 'threshold': 0.7}
            }
        """
        self.schedule = self._parse_schedule(schedule)
        self.default_threshold = 0.9
        self._load_threshold = 0.0

    def _parse_schedule(self, config: Dict) -> Dict:
        """解析交易时段配置"""
        parsed = {}
        for period, params in config.items():
            start = datetime.strptime(params['start'], '%H:%M').time()
            end = datetime.strptime(params['end'], '%H:%M').time()
            parsed[period] = {
                'start': start,
                'end': end,
                'threshold': params['threshold']
            }
        return parsed

    def update_load(self, current_load: float):
        """更新系统负载指标"""
        self._load_threshold = current_load

    def should_trigger(self) -> bool:
        """判断是否触发熔断"""
        current_threshold = self._get_current_threshold()
        return self._load_threshold >= current_threshold

    def _get_current_threshold(self) -> float:
        """获取当前时段熔断阈值"""
        now = datetime.now().time()
        for period in self.schedule.values():
            if period['start'] <= now < period['end']:
                return period['threshold']
        return self.default_threshold

class LogShardManager:
    """日志分片存储管理器"""

    def __init__(self, base_path: str, sharding_rules: Dict):
        """
        Args:
            base_path: 基础存储路径
            sharding_rules: 分片规则 {
                'by_symbol': True,
                'by_date': {'format': '%Y%m%d', 'interval': 'day'}
            }
        """
        self.base_path = Path(base_path)
        self.sharding_rules = sharding_rules
        self.active_shards: Set[Path] = set()

    def get_shard_path(self, log_entry: Dict) -> Path:
        """获取分片存储路径"""
        path = self.base_path

        # 按交易品种分片
        if self.sharding_rules.get('by_symbol') and 'symbol' in log_entry:
            path = path / log_entry['symbol']

        # 按日期分片
        if 'by_date' in self.sharding_rules:
            fmt = self.sharding_rules['by_date']['format']
            path = path / datetime.now().strftime(fmt)

        self.active_shards.add(path)
        return path

    async def rotate_shards(self):
        """分片轮转操作"""
        for shard in self.active_shards:
            if self._should_rotate(shard):
                await self._compress_shard(shard)

    def _should_rotate(self, shard: Path) -> bool:
        """判断是否需要轮转"""
        if not shard.exists():
            return False

        # 按大小轮转 (100MB)
        return os.path.getsize(shard) > 100 * 1024 * 1024

    async def _compress_shard(self, shard: Path):
        """压缩分片文件"""
        # 实现略...
        pass

class OptimizedLogger:
    """集成优化组件后的日志处理器"""

    def __init__(self, config: Dict):
        self.deduplicator = MarketDataDeduplicator(
            config.get('deduplication_window', 3))
        self.circuit_breaker = TradingHoursAwareCircuitBreaker(
            config['trading_hours'])
        self.shard_manager = LogShardManager(
            config['log_storage']['path'],
            config['log_storage']['sharding'])

    async def log_market_data(self, tick_data: Dict):
        """记录行情数据（带去重检查）"""
        if self.deduplicator.is_duplicate(tick_data):
            return

        if self.circuit_breaker.should_trigger():
            await self._handle_backpressure()
            return

        shard_path = self.shard_manager.get_shard_path(tick_data)
        await self._write_to_shard(shard_path, tick_data)

    async def _handle_backpressure(self):
        """处理背压情况"""
        # 实现略...
        pass

    async def _write_to_shard(self, path: Path, data: Dict):
        """写入分片文件"""
        # 实现略...
        pass
