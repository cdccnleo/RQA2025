"""
增量采集状态持久化管理器
记录和管理增量采集的状态、历史和统计信息
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import os

from src.infrastructure.logging.core.unified_logger import get_unified_logger
from src.core.orchestration.incremental_collection_strategy import CollectionWindow

logger = get_unified_logger(__name__)


@dataclass
class IncrementalCollectionRecord:
    """增量采集记录"""
    record_id: str
    source_id: str
    data_type: str
    collection_mode: str  # 'incremental', 'complement', 'full'
    priority: str
    start_date: datetime
    end_date: datetime
    requested_start: Optional[datetime] = None
    requested_end: Optional[datetime] = None
    actual_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    processing_time: float = 0.0
    data_quality_score: float = 0.0
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    error_message: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        # 将datetime对象转换为ISO字符串
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IncrementalCollectionRecord':
        """从字典创建对象"""
        # 将ISO字符串转换为datetime对象
        for key in ['start_date', 'end_date', 'requested_start', 'requested_end',
                   'created_at', 'started_at', 'completed_at']:
            if key in data and data[key] and isinstance(data[key], str):
                try:
                    data[key] = datetime.fromisoformat(data[key])
                except ValueError:
                    pass
        return cls(**data)


@dataclass
class CollectionStatistics:
    """采集统计信息"""
    source_id: str
    total_collections: int = 0
    successful_collections: int = 0
    failed_collections: int = 0
    total_records_processed: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    avg_data_quality_score: float = 0.0
    last_collection_time: Optional[datetime] = None
    last_successful_collection: Optional[datetime] = None
    consecutive_failures: int = 0
    mode_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.mode_distribution is None:
            self.mode_distribution = {}

    def update_with_record(self, record: IncrementalCollectionRecord):
        """使用新记录更新统计信息"""
        self.total_collections += 1
        self.total_records_processed += record.processed_records
        self.total_processing_time += record.processing_time

        if record.status == 'completed':
            self.successful_collections += 1
            self.last_successful_collection = record.completed_at
            self.consecutive_failures = 0
        elif record.status == 'failed':
            self.failed_collections += 1
            self.consecutive_failures += 1

        self.last_collection_time = record.completed_at or record.started_at

        # 更新平均值
        if self.total_collections > 0:
            self.avg_processing_time = self.total_processing_time / self.total_collections

        # 更新模式分布
        mode = record.collection_mode
        if mode not in self.mode_distribution:
            self.mode_distribution[mode] = 0
        self.mode_distribution[mode] += 1

        # 计算平均质量评分
        if record.data_quality_score > 0:
            # 简单的移动平均
            if self.avg_data_quality_score == 0:
                self.avg_data_quality_score = record.data_quality_score
            else:
                self.avg_data_quality_score = (self.avg_data_quality_score + record.data_quality_score) / 2

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollectionStatistics':
        """从字典创建对象"""
        for key in ['last_collection_time', 'last_successful_collection']:
            if key in data and data[key] and isinstance(data[key], str):
                try:
                    data[key] = datetime.fromisoformat(data[key])
                except ValueError:
                    pass
        return cls(**data)


class IncrementalCollectionPersistence:
    """增量采集状态持久化管理器"""

    def __init__(self, storage_dir: str = "data/incremental_collections"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self._records_cache: Dict[str, IncrementalCollectionRecord] = {}
        self._stats_cache: Dict[str, CollectionStatistics] = {}

        # 文件路径
        self.records_file = self.storage_dir / "collection_records.json"
        self.stats_file = self.storage_dir / "collection_statistics.json"

        # 加载现有数据
        self._load_data()

        logger.info(f"增量采集状态持久化管理器初始化完成，存储目录: {self.storage_dir}")

    def create_collection_record(self, source_id: str, data_type: str,
                               collection_window: CollectionWindow) -> str:
        """
        创建采集记录

        Args:
            source_id: 数据源ID
            data_type: 数据类型
            collection_window: 采集时间窗口

        Returns:
            记录ID
        """
        record_id = f"{source_id}_{int(datetime.now().timestamp() * 1000)}"

        record = IncrementalCollectionRecord(
            record_id=record_id,
            source_id=source_id,
            data_type=data_type,
            collection_mode=collection_window.mode,
            priority=collection_window.priority,
            start_date=collection_window.start_date,
            end_date=collection_window.end_date,
            status='pending'
        )

        self._records_cache[record_id] = record
        self._save_records()

        logger.debug(f"创建采集记录: {record_id} for {source_id}")
        return record_id

    def update_collection_record(self, record_id: str, updates: Dict[str, Any]):
        """
        更新采集记录

        Args:
            record_id: 记录ID
            updates: 更新字段
        """
        if record_id not in self._records_cache:
            logger.warning(f"记录不存在: {record_id}")
            return

        record = self._records_cache[record_id]

        # 更新字段
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)

        # 自动更新时间戳
        if 'status' in updates:
            if updates['status'] == 'running' and not record.started_at:
                record.started_at = datetime.now()
            elif updates['status'] in ['completed', 'failed'] and not record.completed_at:
                record.completed_at = datetime.now()

        self._save_records()
        logger.debug(f"更新采集记录: {record_id}, 状态: {record.status}")

    def get_collection_record(self, record_id: str) -> Optional[IncrementalCollectionRecord]:
        """获取采集记录"""
        return self._records_cache.get(record_id)

    def get_source_collection_records(self, source_id: str, limit: int = 50) -> List[IncrementalCollectionRecord]:
        """获取数据源的采集记录"""
        source_records = [
            record for record in self._records_cache.values()
            if record.source_id == source_id
        ]

        # 按创建时间倒序排列
        source_records.sort(key=lambda r: r.created_at, reverse=True)

        return source_records[:limit]

    def get_collection_statistics(self, source_id: str) -> CollectionStatistics:
        """获取数据源的采集统计信息"""
        if source_id not in self._stats_cache:
            self._stats_cache[source_id] = CollectionStatistics(source_id=source_id)

        return self._stats_cache[source_id]

    def update_collection_statistics(self, source_id: str, record: IncrementalCollectionRecord):
        """更新采集统计信息"""
        stats = self.get_collection_statistics(source_id)
        stats.update_with_record(record)

        self._save_statistics()
        logger.debug(f"更新采集统计: {source_id}")

    def get_failed_collections(self, source_id: Optional[str] = None,
                             hours: int = 24) -> List[IncrementalCollectionRecord]:
        """
        获取失败的采集记录

        Args:
            source_id: 数据源ID（可选，None表示所有数据源）
            hours: 时间范围（小时）

        Returns:
            失败的采集记录列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        failed_records = [
            record for record in self._records_cache.values()
            if record.status == 'failed' and record.created_at >= cutoff_time
        ]

        if source_id:
            failed_records = [r for r in failed_records if r.source_id == source_id]

        return failed_records

    def get_collection_trends(self, source_id: str, days: int = 7) -> Dict[str, Any]:
        """
        获取采集趋势分析

        Args:
            source_id: 数据源ID
            days: 分析天数

        Returns:
            趋势分析数据
        """
        cutoff_time = datetime.now() - timedelta(days=days)

        relevant_records = [
            record for record in self._records_cache.values()
            if record.source_id == source_id and record.created_at >= cutoff_time
        ]

        trends = {
            'period_days': days,
            'total_collections': len(relevant_records),
            'successful_collections': len([r for r in relevant_records if r.status == 'completed']),
            'failed_collections': len([r for r in relevant_records if r.status == 'failed']),
            'total_records_processed': sum(r.processed_records for r in relevant_records),
            'avg_processing_time': 0.0,
            'avg_data_quality': 0.0,
            'daily_stats': {}
        }

        # 计算平均值
        completed_records = [r for r in relevant_records if r.status == 'completed']
        if completed_records:
            trends['avg_processing_time'] = sum(r.processing_time for r in completed_records) / len(completed_records)
            quality_scores = [r.data_quality_score for r in completed_records if r.data_quality_score > 0]
            if quality_scores:
                trends['avg_data_quality'] = sum(quality_scores) / len(quality_scores)

        # 按日期统计
        for record in relevant_records:
            date_key = record.created_at.date().isoformat()
            if date_key not in trends['daily_stats']:
                trends['daily_stats'][date_key] = {
                    'collections': 0,
                    'successful': 0,
                    'failed': 0,
                    'records_processed': 0
                }

            daily = trends['daily_stats'][date_key]
            daily['collections'] += 1
            daily['records_processed'] += record.processed_records

            if record.status == 'completed':
                daily['successful'] += 1
            elif record.status == 'failed':
                daily['failed'] += 1

        return trends

    def cleanup_old_records(self, days_to_keep: int = 90):
        """
        清理旧的采集记录

        Args:
            days_to_keep: 保留天数

        Returns:
            清理的记录数量
        """
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)

        old_records = [
            record_id for record_id, record in self._records_cache.items()
            if record.created_at < cutoff_time
        ]

        for record_id in old_records:
            del self._records_cache[record_id]

        if old_records:
            self._save_records()
            logger.info(f"清理了 {len(old_records)} 条旧采集记录")

        return len(old_records)

    def _load_data(self):
        """加载持久化数据"""
        try:
            # 加载记录
            if self.records_file.exists():
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    records_data = json.load(f)
                    for record_data in records_data:
                        record = IncrementalCollectionRecord.from_dict(record_data)
                        self._records_cache[record.record_id] = record

            # 加载统计
            if self.stats_file.exists():
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                    for stat_data in stats_data:
                        stats = CollectionStatistics.from_dict(stat_data)
                        self._stats_cache[stats.source_id] = stats

            logger.info(f"加载了 {len(self._records_cache)} 条采集记录和 {len(self._stats_cache)} 条统计信息")

        except Exception as e:
            logger.warning(f"加载持久化数据失败: {e}")

    def _save_records(self):
        """保存采集记录"""
        try:
            records_data = [record.to_dict() for record in self._records_cache.values()]

            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(records_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"保存采集记录失败: {e}")

    def _save_statistics(self):
        """保存统计信息"""
        try:
            stats_data = [stats.to_dict() for stats in self._stats_cache.values()]

            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"保存统计信息失败: {e}")


# 全局实例
_persistence_manager = None


def get_incremental_collection_persistence() -> IncrementalCollectionPersistence:
    """获取增量采集状态持久化管理器实例"""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = IncrementalCollectionPersistence()
    return _persistence_manager


def record_collection_start(source_id: str, data_type: str, collection_window: CollectionWindow) -> str:
    """
    记录采集开始

    Returns:
        记录ID
    """
    persistence = get_incremental_collection_persistence()
    record_id = persistence.create_collection_record(source_id, data_type, collection_window)

    persistence.update_collection_record(record_id, {
        'status': 'running',
        'started_at': datetime.now()
    })

    return record_id


def record_collection_result(record_id: str, success: bool, result_data: Dict[str, Any]):
    """
    记录采集结果

    Args:
        record_id: 记录ID
        success: 是否成功
        result_data: 结果数据
    """
    persistence = get_incremental_collection_persistence()

    updates = {
        'status': 'completed' if success else 'failed',
        'completed_at': datetime.now()
    }

    # 添加结果数据
    if 'actual_records' in result_data:
        updates['actual_records'] = result_data['actual_records']
    if 'processed_records' in result_data:
        updates['processed_records'] = result_data['processed_records']
    if 'failed_records' in result_data:
        updates['failed_records'] = result_data['failed_records']
    if 'processing_time' in result_data:
        updates['processing_time'] = result_data['processing_time']
    if 'data_quality_score' in result_data:
        updates['data_quality_score'] = result_data['data_quality_score']
    if 'error' in result_data:
        updates['error_message'] = result_data['error']

    persistence.update_collection_record(record_id, updates)

    # 更新统计信息
    record = persistence.get_collection_record(record_id)
    if record:
        persistence.update_collection_statistics(record.source_id, record)