import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional

class MarketDataDeduplicator:
    """行情数据去重处理器"""

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: 去重时间窗口(秒)
        """
        self.window_size = window_size
        self.last_hashes = {}

    def _generate_hash(self, tick_data: Dict) -> str:
        """生成行情数据指纹"""
        key_fields = {
            'symbol': tick_data['symbol'],
            'price': tick_data['price'],
            'volume': tick_data['volume']
        }
        return hashlib.md5(str(key_fields).encode()).hexdigest()

    def is_duplicate(self, tick_data: Dict) -> bool:
        """判断是否重复数据"""
        symbol = tick_data['symbol']
        current_hash = self._generate_hash(tick_data)

        # 检查时间窗口内的哈希记录
        now = datetime.now()
        if symbol in self.last_hashes:
            last_hash, timestamp = self.last_hashes[symbol]
            if current_hash == last_hash and now - timestamp < timedelta(seconds=self.window_size):
                return True

        # 更新最新记录
        self.last_hashes[symbol] = (current_hash, now)
        return False

class TradingHoursAwareCircuitBreaker:
    """交易时段感知熔断器"""

    def __init__(self, schedule: Dict):
        """
        Args:
            schedule: 交易时段配置 {
                'morning': {'start': '09:30', 'end': '11:30', 'threshold': 0.8},
                'afternoon': {'start': '13:00', 'end': '15:00', 'threshold': 0.7}
            }
        """
        self.schedule = schedule
        self.current_threshold = 0.9  # 默认阈值

    def _get_current_period(self) -> Optional[str]:
        """获取当前交易时段"""
        now = datetime.now().time()
        for period, config in self.schedule.items():
            start = datetime.strptime(config['start'], '%H:%M').time()
            end = datetime.strptime(config['end'], '%H:%M').time()
            if start <= now < end:
                return period
        return None

    def should_trigger(self, current_load: float) -> bool:
        """判断是否触发熔断"""
        period = self._get_current_period()
        if period:
            self.current_threshold = self.schedule[period]['threshold']
        return current_load >= self.current_threshold

class LogShardingManager:
    """日志分片存储管理器"""

    def __init__(self, shard_rules: Dict):
        """
        Args:
            shard_rules: 分片规则 {
                'by_symbol': ['symbol'],  # 按品种分片
                'by_date': {'format': '%Y%m%d', 'interval': 'day'}  # 按时间分片
            }
        """
        self.shard_rules = shard_rules

    def get_shard_path(self, log_data: Dict) -> str:
        """获取分片存储路径"""
        path_parts = []

        # 按品种分片
        if 'by_symbol' in self.shard_rules:
            for field in self.shard_rules['by_symbol']:
                if field in log_data:
                    path_parts.append(str(log_data[field]))

        # 按时间分片
        if 'by_date' in self.shard_rules:
            fmt = self.shard_rules['by_date']['format']
            path_parts.append(datetime.now().strftime(fmt))

        return '/'.join(path_parts)

    def auto_merge_shards(self, older_than: int = 7):
        """自动合并旧分片"""
        # 实现略...
        pass
