"""
多市场数据同步模块 - 生产环境实现
"""
# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
import pytz
import numpy as np

from src.infrastructure.logging import get_infrastructure_logger

logger = get_infrastructure_logger('multi_market_sync')


class DataType(Enum):

    """数据类型"""
    TICK = "tick"
    OHLC = "ohlc"
    business = "business"
    QUOTE = "quote"
    sequence = "orderbook"


class SyncType(Enum):

    """同步类型"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    HISTORICAL = "historical"


class SyncStatus(Enum):

    """同步状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MarketData:

    """市场数据"""
    market_id: str
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    timezone: str
    currency: str
    data_type: DataType
    source: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['data_type'] = self.data_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class MarketConfig:

    """市场配置"""
    market_id: str
    market_name: str
    timezone: str
    base_currency: str
    data_sources: List[str]
    sync_frequency: int  # 秒
    priority: int  # 1 - 10
    # 可选字段：交易时段（用于初始化内置市场时传入）
    trading_hours: Optional[Dict[str, List[str]]] = None
    # 可选字段：业务分组
    business: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class SyncTask:

    """同步任务"""
    task_id: str
    market_id: str
    sync_type: SyncType
    status: SyncStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    records_synced: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['sync_type'] = self.sync_type.value
        data['status'] = self.status.value
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


class GlobalMarketDataManager:

    """全球市场数据管理器"""

    def __init__(self):

        self.market_data: Dict[str, List[MarketData]] = {}
        self.market_configs: Dict[str, MarketConfig] = {}
        self.data_registry: Dict[str, Dict[str, Any]] = {}
        logger.info("全球市场数据管理器初始化完成")

    def register_market(self, market_config: MarketConfig) -> bool:
        """注册市场"""
        self.market_configs[market_config.market_id] = market_config
        self.market_data[market_config.market_id] = []
        self.data_registry[market_config.market_id] = {
            'last_sync': None,
            'data_count': 0,
            'last_update': None
        }
        logger.info(f"注册市场: {market_config.market_name}")
        return True

    def add_market_data(self, market_id: str, data: MarketData) -> bool:
        """添加市场数据"""
        if market_id not in self.market_data:
            logger.warning(f"市场未注册: {market_id}")
            return False

        self.market_data[market_id].append(data)
        self.data_registry[market_id]['data_count'] += 1
        self.data_registry[market_id]['last_update'] = datetime.now()

        # 限制数据量，避免内存溢出
        if len(self.market_data[market_id]) > 10000:
            self.market_data[market_id] = self.market_data[market_id][-5000:]

        return True

    def get_market_data(self, market_id: str,


                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        data_type: Optional[DataType] = None) -> List[MarketData]:
        """获取市场数据"""
        if market_id not in self.market_data:
            return []

        data = self.market_data[market_id]

        # 时间过滤
        if start_time:
            data = [d for d in data if d.timestamp >= start_time]
        if end_time:
            data = [d for d in data if d.timestamp <= end_time]

        # 类型过滤
        if data_type:
            data = [d for d in data if d.data_type == data_type]

        return data

    def get_market_statistics(self, market_id: str) -> Dict[str, Any]:
        """获取市场统计信息"""
        if market_id not in self.market_data:
            return {}

        data = self.market_data[market_id]
        if not data:
            return {'market_id': market_id, 'data_count': 0}

        prices = [d.price for d in data]
        volumes = [d.volume for d in data]

        return {
            'market_id': market_id,
            'data_count': len(data),
            'price_stats': {
                'min': min(prices),
                'max': max(prices),
                'avg': sum(prices) / len(prices),
                'std': np.std(prices)
            },
            'volume_stats': {
                'total': sum(volumes),
                'avg': sum(volumes) / len(volumes)
            },
            'last_update': self.data_registry[market_id]['last_update'].isoformat() if self.data_registry[market_id]['last_update'] else None
        }


class CrossTimezoneSynchronizer:

    """跨时区同步器"""

    def __init__(self):

        self.timezone_mappings: Dict[str, str] = {}
        self.sync_schedules: Dict[str, Dict[str, Any]] = {}
        logger.info("跨时区同步器初始化完成")

    def set_timezone_mapping(self, market_id: str, timezone: str) -> bool:
        """设置时区映射"""
        self.timezone_mappings[market_id] = timezone
        logger.info(f"设置时区映射: {market_id} -> {timezone}")
        return True

    def convert_timezone(self, timestamp: datetime, from_timezone: str,


                         to_timezone: str) -> datetime:
        """转换时区"""
        from_tz = pytz.timezone(from_timezone)
        to_tz = pytz.timezone(to_timezone)

        # 确保时间戳有时区信息
        if timestamp.tzinfo is None:
            timestamp = from_tz.localize(timestamp)

        return timestamp.astimezone(to_tz)

    def schedule_sync(self, market_id: str, target_timezone: str,


                      sync_frequency: int) -> str:
        """安排同步"""
        schedule_id = str(uuid.uuid4())
        self.sync_schedules[schedule_id] = {
            'market_id': market_id,
            'target_timezone': target_timezone,
            'sync_frequency': sync_frequency,
            'last_sync': None,
            'next_sync': datetime.now() + timedelta(seconds=sync_frequency)
        }
        logger.info(f"安排同步: {market_id} -> {target_timezone}")
        return schedule_id

    def get_sync_schedules(self) -> List[Dict[str, Any]]:
        """获取同步计划"""
        return [
            {
                'schedule_id': schedule_id,
                'market_id': schedule['market_id'],
                'target_timezone': schedule['target_timezone'],
                'sync_frequency': schedule['sync_frequency'],
                'last_sync': schedule['last_sync'].isoformat() if schedule['last_sync'] else None,
                'next_sync': schedule['next_sync'].isoformat() if schedule['next_sync'] else None
            }
            for schedule_id, schedule in self.sync_schedules.items()
        ]


class MultiCurrencyProcessor:

    """多货币处理器"""

    def __init__(self):

        self.exchange_rates: Dict[str, Dict[str, float]] = {}
        self.rate_history: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("多货币处理器初始化完成")

    def set_exchange_rate(self, from_currency: str, to_currency: str,


                          rate: float, timestamp: datetime) -> bool:
        """设置汇率"""
        if from_currency not in self.exchange_rates:
            self.exchange_rates[from_currency] = {}

        self.exchange_rates[from_currency][to_currency] = rate

        # 记录汇率历史
        if from_currency not in self.rate_history:
            self.rate_history[from_currency] = []

        self.rate_history[from_currency].append({
            'to_currency': to_currency,
            'rate': rate,
            'timestamp': timestamp.isoformat()
        })

        logger.info(f"设置汇率: {from_currency} -> {to_currency} = {rate}")
        return True

    def convert_currency(self, amount: float, from_currency: str,


                         to_currency: str) -> Optional[float]:
        """转换货币"""
        if from_currency == to_currency:
            return amount

        if from_currency not in self.exchange_rates:
            logger.warning(f"汇率不存在: {from_currency}")
            return None

        if to_currency not in self.exchange_rates[from_currency]:
            logger.warning(f"汇率不存在: {from_currency} -> {to_currency}")
            return None

        rate = self.exchange_rates[from_currency][to_currency]
        return amount * rate

    def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """获取汇率"""
        if from_currency == to_currency:
            return 1.0

        if from_currency not in self.exchange_rates:
            return None

        return self.exchange_rates[from_currency].get(to_currency)

    def get_rate_history(self, from_currency: str, to_currency: str,


                         days: int = 30) -> List[Dict[str, Any]]:
        """获取汇率历史"""
        if from_currency not in self.rate_history:
            return []

        cutoff_date = datetime.now() - timedelta(days=days)
        history = [
            record for record in self.rate_history[from_currency]
            if record['to_currency'] == to_currency
            and datetime.fromisoformat(record['timestamp']) >= cutoff_date
        ]

        return history


class MultiMarketSyncManager:

    """多市场同步管理器"""

    def __init__(self):

        self.global_manager = GlobalMarketDataManager()
        self.timezone_synchronizer = CrossTimezoneSynchronizer()
        self.currency_processor = MultiCurrencyProcessor()
        self.sync_tasks: Dict[str, SyncTask] = {}
        self.sync_metrics: Dict[str, Dict[str, Any]] = {}
        logger.info("多市场同步管理器初始化完成")

    def initialize_markets(self) -> Dict[str, Any]:
        """初始化市场"""
        # 注册主要市场
        markets = [
            MarketConfig(
                market_id="SHANGHAI",
                market_name="上海证券交易所",
                timezone="Asia / Shanghai",
                base_currency="CNY",
                trading_hours={
                    "monday": ["09:30", "15:00"],
                    "tuesday": ["09:30", "15:00"],
                    "wednesday": ["09:30", "15:00"],
                    "thursday": ["09:30", "15:00"],
                    "friday": ["09:30", "15:00"]
                },
                data_sources=["wind", "tushare"],
                sync_frequency=60,
                priority=10
            ),
            MarketConfig(
                market_id="SHENZHEN",
                market_name="深圳证券交易所",
                timezone="Asia / Shanghai",
                base_currency="CNY",
                trading_hours={
                    "monday": ["09:30", "15:00"],
                    "tuesday": ["09:30", "15:00"],
                    "wednesday": ["09:30", "15:00"],
                    "thursday": ["09:30", "15:00"],
                    "friday": ["09:30", "15:00"]
                },
                data_sources=["wind", "tushare"],
                sync_frequency=60,
                priority=10
            ),
            MarketConfig(
                market_id="NYSE",
                market_name="纽约证券交易所",
                timezone="America / New_York",
                base_currency="USD",
                trading_hours={
                    "monday": ["09:30", "16:00"],
                    "tuesday": ["09:30", "16:00"],
                    "wednesday": ["09:30", "16:00"],
                    "thursday": ["09:30", "16:00"],
                    "friday": ["09:30", "16:00"]
                },
                data_sources=["yahoo", "alpha_vantage"],
                sync_frequency=300,
                priority=8
            )
        ]

        for market in markets:
            self.global_manager.register_market(market)
            self.timezone_synchronizer.set_timezone_mapping(
                market.market_id, market.timezone
            )

        # 设置基础汇率
        self.currency_processor.set_exchange_rate("CNY", "USD", 0.14, datetime.now())
        self.currency_processor.set_exchange_rate("USD", "CNY", 7.1, datetime.now())

        return {
            'markets_registered': len(markets),
            'timezone_mappings': len(self.timezone_synchronizer.timezone_mappings),
            'exchange_rates_set': len(self.currency_processor.exchange_rates)
        }

    def start_sync_task(self, market_id: str, sync_type: SyncType) -> str:
        """启动同步任务"""
        task_id = str(uuid.uuid4())
        task = SyncTask(
            task_id=task_id,
            market_id=market_id,
            sync_type=sync_type,
            status=SyncStatus.RUNNING,
            start_time=datetime.now()
        )

        self.sync_tasks[task_id] = task
        logger.info(f"启动同步任务: {market_id} - {sync_type.value}")
        return task_id

    def complete_sync_task(self, task_id: str, records_synced: int,


                           error_count: int = 0) -> bool:
        """完成同步任务"""
        if task_id not in self.sync_tasks:
            return False

        task = self.sync_tasks[task_id]
        task.status = SyncStatus.COMPLETED if error_count == 0 else SyncStatus.FAILED
        task.end_time = datetime.now()
        task.records_synced = records_synced
        task.error_count = error_count

        # 更新同步指标
        self.sync_metrics[task_id] = {
            'market_id': task.market_id,
            'sync_type': task.sync_type.value,
            'duration': (task.end_time - task.start_time).total_seconds(),
            'records_synced': records_synced,
            'error_count': error_count,
            'success_rate': (records_synced - error_count) / records_synced if records_synced > 0 else 0
        }

        logger.info(f"完成同步任务: {task_id}")
        return True

    def get_sync_report(self) -> Dict[str, Any]:
        """获取同步报告"""
        active_tasks = [task for task in self.sync_tasks.values()
                        if task.status == SyncStatus.RUNNING]
        completed_tasks = [task for task in self.sync_tasks.values()
                           if task.status == SyncStatus.COMPLETED]
        failed_tasks = [task for task in self.sync_tasks.values()
                        if task.status == SyncStatus.FAILED]

        total_records = sum(task.records_synced for task in completed_tasks)
        total_errors = sum(task.error_count for task in completed_tasks + failed_tasks)

        return {
            'report_date': datetime.now().isoformat(),
            'active_tasks_count': len(active_tasks),
            'completed_tasks_count': len(completed_tasks),
            'failed_tasks_count': len(failed_tasks),
            'total_records_synced': total_records,
            'total_errors': total_errors,
            'overall_success_rate': (total_records - total_errors) / total_records if total_records > 0 else 0,
            'market_statistics': {
                market_id: self.global_manager.get_market_statistics(market_id)
                for market_id in self.global_manager.market_data.keys()
            }
        }
