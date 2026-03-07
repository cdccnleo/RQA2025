"""
optimized_components 模块

提供 optimized_components 相关功能和接口。
"""

import logging
import os

# 导入统一的ComponentFactory基类
import hashlib

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.components.core.base_components import ComponentFactory
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
"""
基础设施层 - Optimized组件统一实现

使用统一的ComponentFactory基类，提供Optimized组件的工厂模式实现。
"""

# ComponentFactory, IComponentFactory 已通过其他方式获取

logger = logging.getLogger(__name__)


class IOptimizedComponent(ABC):
    """优化组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def get_component_id(self) -> int:
        """获取组件ID"""


class OptimizedComponent(IOptimizedComponent):
    """统一优化组件实现"""

    def __init__(self, component_id: int, component_type: str = "Optimized"):
        """初始化组件"""
        self.component_id = component_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{component_id}"
        self.creation_time = datetime.now()

    def get_component_id(self) -> int:
        """获取组件ID"""
        return self.component_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一优化组件实现",
            "version": "2.0.0",
            "type": "unified_optimized_component",
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "component_id": self.component_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_optimized_processing",
            }

            return result
        except Exception as e:
            return {
                "component_id": self.component_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good",
        }


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
            "symbol": tick_data["symbol"],
            "price": tick_data["price"],
            "volume": tick_data["volume"],
            "bid": tick_data.get("bid", []),
            "ask": tick_data.get("ask", []),
        }

        return hashlib.sha256(str(key_fields).encode()).hexdigest()

    def is_duplicate(self, tick_data: Dict) -> bool:
        """判断是否重复数据"""
        symbol = tick_data["symbol"]
        current_hash = self._generate_hash(tick_data)
        now = datetime.now()

        last_hash, timestamp = self.last_hashes.get(symbol, (None, None))

        if (
            last_hash is not None
            and timestamp is not None
            and current_hash == last_hash
            and (now - timestamp).total_seconds() < self.window_size
        ):
            return True

        self.last_hashes[symbol] = (current_hash, now)
        return False


class TradingHoursAwareCircuitBreaker:
    """交易时段感知的智能熔断器"""

    def __init__(self, schedule: Dict):
        """
        Args:
            schedule: 交易时段配置 {}
                'morning': {'start': '09:30', 'end': '11:30', 'threshold': 0.8},
                'afternoon': {'start': '13:00', 'end': '15:00', 'threshold': 0.7}

        """
        self.schedule = self._parse_schedule(schedule)
        self.default_threshold = 0.9
        self._load_threshold = 0.0

    def _parse_schedule(self, config: Dict) -> Dict:
        """解析交易时段配置"""
        parsed = {}
        for period, params in config.items():
            start = datetime.strptime(params["start"], "%H:%M").time()
            end = datetime.strptime(params["end"], "%H:%M").time()
            parsed[period] = {
                "start": start,
                "end": end,
                "threshold": params["threshold"],
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
            if period["start"] <= now < period["end"]:
                return period["threshold"]
        return self.default_threshold


class LogShardManager:
    """日志分片存储管理器"""

    def __init__(self, base_path: str, sharding_rules: Dict):
        """
        Args:
            base_path: 基础存储路径
            sharding_rules: 分片规则 {}
                'by_symbol': True,
                'by_date': {'format': '%Y % m % d', 'interval': 'day'}

        """
        self.base_path = Path(base_path)
        self.sharding_rules = sharding_rules
        self.active_shards: Set[Path] = set()

    def get_shard_path(self, log_entry: Dict) -> Path:
        """获取分片存储路径"""
        path = self.base_path

        # 按交易品种分片
        if self.sharding_rules.get("by_symbol") and "symbol" in log_entry:
            path = path / log_entry["symbol"]

        # 按日期分片
        if "by_date" in self.sharding_rules:
            fmt = self.sharding_rules["by_date"]["format"]
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


class OptimizedLogger:
    """集成优化组件后的日志处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.deduplicator = MarketDataDeduplicator(config.get("deduplication_window", 3))
        self.circuit_breaker = TradingHoursAwareCircuitBreaker(config["trading_hours"])
        self.shard_manager = LogShardManager(
            config["log_storage"]["path"], config["log_storage"]["sharding"])

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

    async def _write_to_shard(self, path: Path, data: Dict):
        """写入分片文件"""
        # 实现略...


class OptimizedComponentFactory(ComponentFactory):
    """优化组件工厂"""

    # 支持的组件ID列表 (基于文件中的类数量)
    def __init__(self):
        super().__init__()
        # 注册组件工厂函数

    SUPPORTED_COMPONENT_IDS = [1, 2, 3, 4, 5]  # 5个主要组件类

    @staticmethod
    def create_component(component_id: int) -> OptimizedComponent:
        """创建指定ID的组件"""
        if component_id not in OptimizedComponentFactory.SUPPORTED_COMPONENT_IDS:
            raise ValueError(
                f"不支持的组件ID: {component_id}。支持的ID: {OptimizedComponentFactory.SUPPORTED_COMPONENT_IDS}"
            )

        return OptimizedComponent(component_id, "Optimized")

    @staticmethod
    def get_available_components() -> List[int]:
        """获取所有可用的组件ID"""
        return sorted(list(OptimizedComponentFactory.SUPPORTED_COMPONENT_IDS))

    @staticmethod
    def create_all_components() -> Dict[int, OptimizedComponent]:
        """创建所有可用组件"""
        return {
            component_id: OptimizedComponent(component_id, "Optimized")
            for component_id in OptimizedComponentFactory.SUPPORTED_COMPONENT_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "OptimizedComponentFactory",
            "version": "2.0.0",
            "total_components": len(OptimizedComponentFactory.SUPPORTED_COMPONENT_IDS),
            "supported_ids": sorted(list(OptimizedComponentFactory.SUPPORTED_COMPONENT_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "基础设施工具组件工厂，整合优化组件",
            "optimization_type": "infrastructure_utils",
            "original_classes": 5,
        }

# 向后兼容：创建旧的组件实例函数


def create_optimized_component_1():

    return OptimizedComponentFactory.create_component(1)


def create_optimized_component_2():

    return OptimizedComponentFactory.create_component(2)


def create_optimized_component_3():

    return OptimizedComponentFactory.create_component(3)


def create_optimized_component_4():

    return OptimizedComponentFactory.create_component(4)


def create_optimized_component_5():

    return OptimizedComponentFactory.create_component(5)


__all__ = [
    "IOptimizedComponent",
    "OptimizedComponent",
    "OptimizedComponentFactory",
    "create_optimized_component_1",
    "create_optimized_component_2",
    "create_optimized_component_3",
    "create_optimized_component_4",
    "create_optimized_component_5",
]
