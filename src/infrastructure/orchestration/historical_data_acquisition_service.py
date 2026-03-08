#!/usr/bin/env python3
"""
历史数据采集服务
支持策略回测所需的10年历史数据采集
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import json

from src.infrastructure.orchestration.data_quality_manager import DataQualityResult
from src.core.persistence.timescale_storage import TimescaleStorage
from src.core.cache.redis_cache import RedisCache
from src.infrastructure.integration.akshare_service import get_akshare_service

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """数据源类型"""
    AKSHARE = "akshare"
    YAHOO = "yahoo"
    TUSHARE = "tushare"
    LOCAL_BACKUP = "local_backup"
    WIND = "wind"
    EASTMONEY = "eastmoney"


class DataQualityLevel(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"  # 完整度>95%, 准确性>99%
    GOOD = "good"          # 完整度>90%, 准确性>95%
    ACCEPTABLE = "acceptable"  # 完整度>80%, 准确性>90%
    POOR = "poor"          # 完整度<80% 或 准确性<90%
    INVALID = "invalid"    # 数据无效或严重缺失


@dataclass
class HistoricalDataConfig:
    """历史数据采集配置"""
    data_source: DataSourceType
    symbol: str
    start_date: datetime
    end_date: datetime
    data_type: str = "stock"  # stock, index, fund, fundamental, etc.
    priority: int = 1  # 1-5, 1最高
    max_retry_count: int = 3
    timeout_seconds: int = 30
    quality_threshold: float = 0.85  # 质量阈值
    fallback_sources: List[DataSourceType] = field(default_factory=list)


@dataclass
class DataSourceResult:
    """数据源采集结果"""
    source: DataSourceType
    symbol: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    quality_level: DataQualityLevel = DataQualityLevel.INVALID
    collection_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoricalDataBatch:
    """历史数据批次"""
    batch_id: str
    year: int
    symbol: str
    config: HistoricalDataConfig
    results: List[DataSourceResult] = field(default_factory=list)
    best_result: Optional[DataSourceResult] = None
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class DataSourceAdapter(ABC):
    """数据源适配器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def collect_historical_data(self, config: HistoricalDataConfig) -> DataSourceResult:
        """采集历史数据"""
        pass

    @abstractmethod
    def get_supported_data_types(self) -> List[str]:
        """获取支持的数据类型"""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> Set[str]:
        """获取支持的标的列表"""
        pass

    def validate_config(self, config: HistoricalDataConfig) -> bool:
        """验证配置"""
        return True


class AKShareAdapter(DataSourceAdapter):
    """AKShare数据源适配器"""

    def get_supported_data_types(self) -> List[str]:
        return ["stock", "index", "fund", "bond", "futures"]

    def get_supported_symbols(self) -> Set[str]:
        # 这里应该从AKShare获取实际支持的标的列表
        # 暂时返回示例数据
        return {"000001.SZ", "600036.SH", "000858.SZ"}

    async def collect_historical_data(self, config: HistoricalDataConfig) -> DataSourceResult:
        """从AKShare采集历史数据"""
        start_time = datetime.now()

        try:
            # 使用无缝切换机制采集真实数据
            self.logger.info(f"从AKShare采集 {config.symbol} 数据: {config.start_date} - {config.end_date}")

            # 调用真实数据采集逻辑
            real_data = await self._collect_real_data_with_fallback(config)

            collection_time = (datetime.now() - start_time).total_seconds()

            # 评估数据质量
            quality_score, quality_level = self._assess_data_quality(real_data, config)

            result = DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol=config.symbol,
                data=real_data,
                quality_score=quality_score,
                quality_level=quality_level,
                collection_time=collection_time,
                metadata={
                    "data_points": len(real_data),
                    "date_range": f"{config.start_date.date()} - {config.end_date.date()}",
                    "source_version": "akshare_1.0"
                }
            )

            self.logger.info(f"AKShare采集完成: {config.symbol}, {len(real_data)}条数据, 质量: {quality_level.value}")

            return result

        except Exception as e:
            collection_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"AKShare采集失败 {config.symbol}: {e}")

            return DataSourceResult(
                source=DataSourceType.AKSHARE,
                symbol=config.symbol,
                quality_score=0.0,
                quality_level=DataQualityLevel.INVALID,
                collection_time=collection_time,
                error_message=str(e)
            )

    def _generate_mock_data(self, config: HistoricalDataConfig) -> List[Dict[str, Any]]:
        """生成模拟数据用于测试"""
        data = []
        current_date = config.start_date

        while current_date <= config.end_date:
            if current_date.weekday() < 5:  # 周一到周五
                data.append({
                    "symbol": config.symbol,
                    "date": current_date.date(),
                    "open": 100.0 + (current_date - config.start_date).days * 0.1,
                    "high": 105.0 + (current_date - config.start_date).days * 0.1,
                    "low": 95.0 + (current_date - config.start_date).days * 0.1,
                    "close": 102.0 + (current_date - config.start_date).days * 0.1,
                    "volume": 1000000 + (current_date - config.start_date).days * 1000,
                    "amount": 102000000.0 + (current_date - config.start_date).days * 10000,
                    "source": "akshare",
                    "timestamp": current_date
                })
            current_date += timedelta(days=1)

        return data

    async def _collect_real_data_with_fallback(self, config: HistoricalDataConfig) -> List[Dict[str, Any]]:
        """使用新的AKShare服务采集真实数据"""
        # 格式化日期
        start_date_str = config.start_date.strftime('%Y%m%d')
        end_date_str = config.end_date.strftime('%Y%m%d')
        
        try:
            # 使用统一的AKShare服务
            akshare_service = get_akshare_service()
            
            # 调用统一的get_stock_data方法
            df = await akshare_service.get_stock_data(
                symbol=config.symbol,
                start_date=start_date_str,
                end_date=end_date_str,
                adjust="qfq"
            )
            
            if df is not None and not df.empty:
                # 转换为标准格式
                real_data = akshare_service.convert_to_standard_format(df)
                
                # 添加标的代码
                for record in real_data:
                    record['symbol'] = config.symbol
                
                return real_data
            else:
                self.logger.warning("⚠️ AKShare服务返回空数据")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ AKShare服务调用失败: {e}")
            return []

    def _assess_data_quality(self, data: List[Dict[str, Any]], config: HistoricalDataConfig) -> Tuple[float, DataQualityLevel]:
        """评估数据质量"""
        if not data:
            return 0.0, DataQualityLevel.INVALID

        total_days = (config.end_date - config.start_date).days + 1
        trading_days = len([d for d in data if d.get('volume', 0) > 0])

        completeness = trading_days / max(total_days * 0.7, 1)  # 假设70%为交易日

        # 检查数据完整性
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        field_completeness = sum(
            1 for item in data
            if all(item.get(field) is not None for field in required_fields)
        ) / len(data) if data else 0

        # 检查数据合理性（价格不能为负，成交量不能为负等）
        valid_records = sum(
            1 for item in data
            if (item.get('open', 0) > 0 and item.get('close', 0) > 0 and
                item.get('volume', 0) >= 0 and item.get('amount', 0) >= 0 and
                item.get('high', 0) >= item.get('low', 0))
        ) / len(data) if data else 0

        quality_score = (completeness * 0.4 + field_completeness * 0.3 + valid_records * 0.3)

        if quality_score >= 0.95:
            level = DataQualityLevel.EXCELLENT
        elif quality_score >= 0.90:
            level = DataQualityLevel.GOOD
        elif quality_score >= 0.80:
            level = DataQualityLevel.ACCEPTABLE
        elif quality_score >= 0.50:
            level = DataQualityLevel.POOR
        else:
            level = DataQualityLevel.INVALID

        return quality_score, level


class YahooAdapter(DataSourceAdapter):
    """Yahoo Finance数据源适配器"""

    def get_supported_data_types(self) -> List[str]:
        return ["stock", "index", "etf"]

    def get_supported_symbols(self) -> Set[str]:
        # 主要支持美股和部分国际指数
        return {"AAPL", "GOOGL", "MSFT", "^GSPC", "^IXIC"}

    async def collect_historical_data(self, config: HistoricalDataConfig) -> DataSourceResult:
        """从Yahoo Finance采集历史数据"""
        start_time = datetime.now()

        try:
            self.logger.info(f"从Yahoo采集 {config.symbol} 数据: {config.start_date} - {config.end_date}")

            # 这里应该调用yfinance库
            # 示例实现
            mock_data = self._generate_mock_data(config)

            collection_time = (datetime.now() - start_time).total_seconds()
            quality_score, quality_level = self._assess_data_quality(mock_data, config)

            return DataSourceResult(
                source=DataSourceType.YAHOO,
                symbol=config.symbol,
                data=mock_data,
                quality_score=quality_score,
                quality_level=quality_level,
                collection_time=collection_time,
                metadata={
                    "data_points": len(mock_data),
                    "date_range": f"{config.start_date.date()} - {config.end_date.date()}",
                    "adjusted_close": True
                }
            )

        except Exception as e:
            collection_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Yahoo采集失败 {config.symbol}: {e}")

            return DataSourceResult(
                source=DataSourceType.YAHOO,
                symbol=config.symbol,
                quality_score=0.0,
                quality_level=DataQualityLevel.INVALID,
                collection_time=collection_time,
                error_message=str(e)
            )

    def _generate_mock_data(self, config: HistoricalDataConfig) -> List[Dict[str, Any]]:
        """生成模拟数据"""
        data = []
        current_date = config.start_date

        while current_date <= config.end_date:
            if current_date.weekday() < 5:
                data.append({
                    "symbol": config.symbol,
                    "date": current_date.date(),
                    "open": 150.0 + (current_date - config.start_date).days * 0.05,
                    "high": 155.0 + (current_date - config.start_date).days * 0.05,
                    "low": 145.0 + (current_date - config.start_date).days * 0.05,
                    "close": 152.0 + (current_date - config.start_date).days * 0.05,
                    "adj_close": 152.0 + (current_date - config.start_date).days * 0.05,
                    "volume": 50000000 + (current_date - config.start_date).days * 50000,
                    "source": "yahoo",
                    "timestamp": current_date
                })
            current_date += timedelta(days=1)

        return data

    def _assess_data_quality(self, data: List[Dict[str, Any]], config: HistoricalDataConfig) -> Tuple[float, DataQualityLevel]:
        """评估数据质量"""
        if not data:
            return 0.0, DataQualityLevel.INVALID

        # Yahoo通常数据质量较高
        completeness = len(data) / max((config.end_date - config.start_date).days * 0.7, 1)

        # 检查必需字段
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        field_completeness = sum(
            1 for item in data
            if all(item.get(field) is not None for field in required_fields)
        ) / len(data)

        quality_score = (completeness * 0.5 + field_completeness * 0.5)

        if quality_score >= 0.95:
            level = DataQualityLevel.EXCELLENT
        elif quality_score >= 0.90:
            level = DataQualityLevel.GOOD
        elif quality_score >= 0.80:
            level = DataQualityLevel.ACCEPTABLE
        elif quality_score >= 0.60:
            level = DataQualityLevel.POOR
        else:
            level = DataQualityLevel.INVALID

        return quality_score, level


class LocalBackupAdapter(DataSourceAdapter):
    """本地备份数据源适配器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backup_dir = Path(config.get('backup_dir', './data/backup'))

    def get_supported_data_types(self) -> List[str]:
        return ["stock", "index", "fund", "bond", "futures"]

    def get_supported_symbols(self) -> Set[str]:
        """从备份目录扫描支持的标的"""
        if not self.backup_dir.exists():
            return set()

        symbols = set()
        for file_path in self.backup_dir.glob("*.csv"):
            # 从文件名提取标的代码
            symbol = file_path.stem.split('_')[0]
            symbols.add(symbol)

        return symbols

    async def collect_historical_data(self, config: HistoricalDataConfig) -> DataSourceResult:
        """从本地备份采集历史数据"""
        start_time = datetime.now()

        try:
            self.logger.info(f"从本地备份采集 {config.symbol} 数据: {config.start_date} - {config.end_date}")

            # 从备份文件加载数据
            data = self._load_from_backup(config)

            collection_time = (datetime.now() - start_time).total_seconds()
            quality_score, quality_level = self._assess_data_quality(data, config)

            return DataSourceResult(
                source=DataSourceType.LOCAL_BACKUP,
                symbol=config.symbol,
                data=data,
                quality_score=quality_score,
                quality_level=quality_level,
                collection_time=collection_time,
                metadata={
                    "data_points": len(data),
                    "backup_file": f"{config.symbol}_{config.start_date.year}.csv",
                    "cache_hit": True
                }
            )

        except Exception as e:
            collection_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"本地备份采集失败 {config.symbol}: {e}")

            return DataSourceResult(
                source=DataSourceType.LOCAL_BACKUP,
                symbol=config.symbol,
                quality_score=0.0,
                quality_level=DataQualityLevel.INVALID,
                collection_time=collection_time,
                error_message=str(e)
            )

    def _load_from_backup(self, config: HistoricalDataConfig) -> List[Dict[str, Any]]:
        """从备份文件加载数据"""
        data = []
        backup_file = self.backup_dir / f"{config.symbol}_{config.start_date.year}.csv"

        if not backup_file.exists():
            return data

        try:
            import csv
            with open(backup_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 解析日期
                    row_date = datetime.strptime(row['date'], '%Y-%m-%d').date()

                    # 检查日期范围
                    if config.start_date.date() <= row_date <= config.end_date.date():
                        # 转换数据类型
                        processed_row = {
                            "symbol": config.symbol,
                            "date": row_date,
                            "open": float(row.get('open', 0)),
                            "high": float(row.get('high', 0)),
                            "low": float(row.get('low', 0)),
                            "close": float(row.get('close', 0)),
                            "volume": int(float(row.get('volume', 0))),
                            "amount": float(row.get('amount', 0)),
                            "source": "local_backup",
                            "timestamp": datetime.combine(row_date, datetime.min.time())
                        }
                        data.append(processed_row)

        except Exception as e:
            self.logger.error(f"读取备份文件失败 {backup_file}: {e}")

        return data

    def _assess_data_quality(self, data: List[Dict[str, Any]], config: HistoricalDataConfig) -> Tuple[float, DataQualityLevel]:
        """评估备份数据质量"""
        if not data:
            return 0.0, DataQualityLevel.INVALID

        # 备份数据通常质量较高
        completeness = len(data) / max((config.end_date - config.start_date).days, 1)

        # 检查数据完整性
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        field_completeness = sum(
            1 for item in data
            if all(item.get(field) is not None for field in required_fields)
        ) / len(data)

        quality_score = (completeness * 0.6 + field_completeness * 0.4)

        if quality_score >= 0.98:
            level = DataQualityLevel.EXCELLENT
        elif quality_score >= 0.95:
            level = DataQualityLevel.GOOD
        elif quality_score >= 0.90:
            level = DataQualityLevel.ACCEPTABLE
        elif quality_score >= 0.70:
            level = DataQualityLevel.POOR
        else:
            level = DataQualityLevel.INVALID

        return quality_score, level


class HistoricalDataAcquisitionService:
    """历史数据采集服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化数据源适配器
        self.adapters: Dict[DataSourceType, DataSourceAdapter] = {}
        self._initialize_adapters()

        # 初始化存储和缓存
        self.timescale_storage = TimescaleStorage(config.get('timescale_config', {}))
        self.redis_cache = RedisCache(config.get('redis_config', {}))

        # 并发控制
        self.max_concurrent_batches = config.get('max_concurrent_batches', 3)
        self.semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        # 质量阈值
        self.quality_threshold = config.get('quality_threshold', 0.80)

    def _initialize_adapters(self):
        """初始化数据源适配器"""
        adapter_configs = self.config.get('adapters', {})

        # AKShare适配器
        if 'akshare' in adapter_configs:
            self.adapters[DataSourceType.AKSHARE] = AKShareAdapter(adapter_configs['akshare'])

        # Yahoo适配器
        if 'yahoo' in adapter_configs:
            self.adapters[DataSourceType.YAHOO] = YahooAdapter(adapter_configs['yahoo'])

        # 本地备份适配器
        if 'local_backup' in adapter_configs:
            self.adapters[DataSourceType.LOCAL_BACKUP] = LocalBackupAdapter(adapter_configs['local_backup'])

        self.logger.info(f"初始化了 {len(self.adapters)} 个数据源适配器: {list(self.adapters.keys())}")

    async def acquire_historical_data(self, config: HistoricalDataConfig) -> HistoricalDataBatch:
        """
        采集历史数据

        Args:
            config: 采集配置

        Returns:
            历史数据批次结果
        """
        batch = HistoricalDataBatch(
            batch_id=f"historical_{config.symbol}_{config.start_date.year}_{int(datetime.now().timestamp())}",
            year=config.start_date.year,
            symbol=config.symbol,
            config=config,
            status="processing"
        )

        try:
            # 验证数据类型
            if not self._validate_data_type(config.data_type):
                batch.status = "failed"
                batch.completed_at = datetime.now()
                self.logger.error(f"不支持的数据类型: {config.data_type}")
                return batch
            
            self.logger.info(f"开始采集历史数据批次: {batch.batch_id}")

            # 并行从多个数据源采集
            tasks = []
            sources_to_try = [config.data_source] + config.fallback_sources

            for source_type in sources_to_try:
                if source_type in self.adapters:
                    task = self._collect_from_source(source_type, config)
                    tasks.append(task)

            if not tasks:
                raise ValueError(f"没有可用的数据源适配器")

            # 执行采集任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"采集任务异常: {result}")
                elif isinstance(result, DataSourceResult):
                    batch.results.append(result)
                    if result.quality_score >= self.quality_threshold:
                        valid_results.append(result)

            # 选择最佳结果
            if valid_results:
                batch.best_result = max(valid_results, key=lambda r: r.quality_score)
                batch.status = "completed"
                self.logger.info(f"历史数据批次完成: {batch.batch_id}, 最佳质量: {batch.best_result.quality_level.value}")
            else:
                batch.status = "failed"
                batch.best_result = max(batch.results, key=lambda r: r.quality_score) if batch.results else None
                self.logger.warning(f"历史数据批次失败: {batch.batch_id}, 无符合质量阈值的结果")

            batch.completed_at = datetime.now()

        except Exception as e:
            batch.status = "failed"
            batch.completed_at = datetime.now()
            self.logger.error(f"历史数据采集异常: {e}")

        return batch

    async def _collect_from_source(self, source_type: DataSourceType, config: HistoricalDataConfig) -> DataSourceResult:
        """从指定数据源采集数据"""
        async with self.semaphore:
            adapter = self.adapters[source_type]
            return await adapter.collect_historical_data(config)

    async def acquire_yearly_data(self, symbol: str, year: int,
                                data_types: List[str] = None) -> List[HistoricalDataBatch]:
        """
        采集指定年份的历史数据

        Args:
            symbol: 标的代码
            year: 年份
            data_types: 数据类型列表

        Returns:
            历史数据批次列表
        """
        if data_types is None:
            data_types = ["stock"]

        batches = []

        for data_type in data_types:
            config = HistoricalDataConfig(
                data_source=DataSourceType.AKSHARE,  # 主要数据源
                symbol=symbol,
                start_date=datetime(year, 1, 1),
                end_date=datetime(year, 12, 31),
                data_type=data_type,
                priority=1,
                fallback_sources=[
                    DataSourceType.YAHOO,
                    DataSourceType.LOCAL_BACKUP
                ]
            )

            batch = await self.acquire_historical_data(config)
            batches.append(batch)

        return batches

    async def acquire_multi_year_data(self, symbol: str, start_year: int, end_year: int,
                                    data_types: List[str] = None) -> List[HistoricalDataBatch]:
        """
        采集多年历史数据

        Args:
            symbol: 标的代码
            start_year: 开始年份
            end_year: 结束年份
            data_types: 数据类型列表

        Returns:
            历史数据批次列表
        """
        all_batches = []

        # 并行采集各年数据
        tasks = [
            self.acquire_yearly_data(symbol, year, data_types)
            for year in range(start_year, end_year + 1)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"多年数据采集异常: {result}")
            else:
                all_batches.extend(result)

        self.logger.info(f"完成多年数据采集: {symbol}, {start_year}-{end_year}, 共{len(all_batches)}个批次")
        return all_batches

    async def store_batch_results(self, batches: List[HistoricalDataBatch]) -> Dict[str, Any]:
        """
        存储批次结果到TimescaleDB

        Args:
            batches: 历史数据批次列表

        Returns:
            存储统计信息
        """
        stats = {
            "total_batches": len(batches),
            "stored_batches": 0,
            "total_records": 0,
            "errors": []
        }

        for batch in batches:
            if batch.best_result and batch.best_result.data:
                try:
                    # 添加数据源标签
                    for record in batch.best_result.data:
                        record['data_source'] = batch.best_result.source.value
                        record['quality_score'] = batch.best_result.quality_score
                        record['batch_id'] = batch.batch_id

                    # 存储到TimescaleDB
                    await self.timescale_storage.store_historical_data(
                        batch.symbol,
                        batch.best_result.data,
                        batch.config.data_type
                    )

                    # 缓存热点数据
                    await self._cache_hot_data(batch.symbol, batch.best_result.data)

                    stats["stored_batches"] += 1
                    stats["total_records"] += len(batch.best_result.data)

                    self.logger.info(f"存储批次成功: {batch.batch_id}, {len(batch.best_result.data)}条记录")

                except Exception as e:
                    error_msg = f"存储批次失败 {batch.batch_id}: {e}"
                    stats["errors"].append(error_msg)
                    self.logger.error(error_msg)

        self.logger.info(f"历史数据存储完成: {stats['stored_batches']}/{stats['total_batches']} 批次, {stats['total_records']}条记录")
        return stats

    async def _cache_hot_data(self, symbol: str, data: List[Dict[str, Any]]):
        """缓存热点数据"""
        try:
            # 缓存最近一年的数据
            recent_data = sorted(data, key=lambda x: x['date'], reverse=True)[:252]  # 约一年交易日

            cache_key = f"historical:{symbol}:recent"
            await self.redis_cache.set_json(cache_key, recent_data, expire_seconds=3600)  # 1小时过期

        except Exception as e:
            self.logger.warning(f"缓存热点数据失败 {symbol}: {e}")

    def get_supported_symbols(self, source_type: DataSourceType = None) -> Set[str]:
        """获取支持的标的列表"""
        if source_type and source_type in self.adapters:
            return self.adapters[source_type].get_supported_symbols()

        # 合并所有数据源支持的标的
        all_symbols = set()
        for adapter in self.adapters.values():
            all_symbols.update(adapter.get_supported_symbols())

        return all_symbols

    def _validate_data_type(self, data_type: str) -> bool:
        """
        验证数据类型是否合法
        
        Args:
            data_type: 数据类型
            
        Returns:
            bool: 是否合法
        """
        valid_data_types = ["stock", "index", "fund", "bond", "futures", "fundamental"]
        return data_type in valid_data_types

    def get_data_quality_stats(self, batches: List[HistoricalDataBatch]) -> Dict[str, Any]:
        """获取数据质量统计"""
        stats = {
            "total_batches": len(batches),
            "quality_distribution": {
                "excellent": 0,
                "good": 0,
                "acceptable": 0,
                "poor": 0,
                "invalid": 0
            },
            "average_quality_score": 0.0,
            "best_sources": {},
            "collection_times": []
        }

        total_score = 0.0

        for batch in batches:
            if batch.best_result:
                quality_level = batch.best_result.quality_level.value
                stats["quality_distribution"][quality_level] += 1
                total_score += batch.best_result.quality_score

                # 统计最佳数据源
                source = batch.best_result.source.value
                stats["best_sources"][source] = stats["best_sources"].get(source, 0) + 1

                # 收集采集时间
                stats["collection_times"].append(batch.best_result.collection_time)

        if batches:
            stats["average_quality_score"] = total_score / len(batches)

        return stats

    async def validate_data_integrity(self, symbol: str, start_year: int, end_year: int) -> Dict[str, Any]:
        """验证数据完整性"""
        validation_results = {
            "symbol": symbol,
            "period": f"{start_year}-{end_year}",
            "total_years": end_year - start_year + 1,
            "years_with_data": 0,
            "total_records": 0,
            "average_daily_records": 0,
            "missing_dates": [],
            "quality_issues": [],
            "is_complete": False
        }

        try:
            # 从数据库查询数据统计
            stats = await self.timescale_storage.get_data_stats(symbol, start_year, end_year)

            validation_results.update(stats)

            # 计算完整性
            expected_trading_days = validation_results["total_years"] * 252  # 每年约252个交易日
            actual_records = validation_results["total_records"]

            completeness_ratio = actual_records / expected_trading_days
            validation_results["completeness_ratio"] = completeness_ratio
            validation_results["is_complete"] = completeness_ratio >= 0.85  # 85%作为完整阈值

        except Exception as e:
            self.logger.error(f"数据完整性验证失败 {symbol}: {e}")
            validation_results["error"] = str(e)

        return validation_results