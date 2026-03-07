#!/usr/bin/env python3
"""
RQA2025 事件持久化组件

提供企业级的事件持久化功能，支持事件存储、检索、重放和清理。
确保系统重启后能够恢复事件状态，支持事件驱动架构的可靠性。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import threading
import logging
import os
from enum import Enum
import gzip

from ...foundation.base import ComponentStatus, ComponentHealth
from ...patterns.standard_interface_template import StandardComponent

logger = logging.getLogger(__name__)


class PersistenceMode(Enum):
    """持久化模式"""
    MEMORY = "memory"           # 内存模式（仅用于测试）
    FILE = "file"              # 文件模式
    DATABASE = "database"      # 数据库模式
    DISTRIBUTED = "distributed"  # 分布式模式


class EventStatus(Enum):
    """事件状态"""
    PENDING = "pending"         # 待处理
    PROCESSING = "processing"   # 处理中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 处理失败
    RETRY = "retry"            # 重试中
    EXPIRED = "expired"        # 已过期


@dataclass
class PersistedEvent:
    """持久化事件"""
    event_id: str
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    status: EventStatus = EventStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    processing_timeout: int = 300  # 5分钟
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'event_data': self.event_data,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'processing_timeout': self.processing_timeout,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'error_message': self.error_message,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistedEvent':
        """从字典创建"""
        return cls(
            event_id=data['event_id'],
            event_type=data['event_type'],
            event_data=data['event_data'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            status=EventStatus(data['status']),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            processing_timeout=data.get('processing_timeout', 300),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            processed_at=datetime.fromisoformat(
                data['processed_at']) if data.get('processed_at') else None,
            error_message=data.get('error_message'),
            metadata=data.get('metadata', {})
        )


class EventPersistenceStrategy(ABC):
    """事件持久化策略基类"""

    @abstractmethod
    def store_event(self, event: PersistedEvent) -> bool:
        """存储事件"""

    @abstractmethod
    def retrieve_event(self, event_id: str) -> Optional[PersistedEvent]:
        """检索事件"""

    @abstractmethod
    def update_event_status(self, event_id: str, status: EventStatus,
                            error_message: Optional[str] = None) -> bool:
        """更新事件状态"""

    @abstractmethod
    def get_pending_events(self, limit: int = 100) -> List[PersistedEvent]:
        """获取待处理事件"""

    @abstractmethod
    def get_failed_events(self, limit: int = 100) -> List[PersistedEvent]:
        """获取失败事件"""

    @abstractmethod
    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[PersistedEvent]:
        """根据事件类型获取事件"""

    @abstractmethod
    def cleanup_expired_events(self, max_age_days: int = 30) -> int:
        """清理过期事件"""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""


class MemoryEventPersistence(EventPersistenceStrategy):
    """内存事件持久化策略（仅用于测试）"""

    def __init__(self):
        self._events: Dict[str, PersistedEvent] = {}
        self._lock = threading.RLock()

    def store_event(self, event: PersistedEvent) -> bool:
        """存储事件"""
        with self._lock:
            self._events[event.event_id] = event
            return True

    def retrieve_event(self, event_id: str) -> Optional[PersistedEvent]:
        """检索事件"""
        with self._lock:
            return self._events.get(event_id)

    def update_event_status(self, event_id: str, status: EventStatus,
                            error_message: Optional[str] = None) -> bool:
        """更新事件状态"""
        with self._lock:
            if event_id in self._events:
                event = self._events[event_id]
                event.status = status
                event.updated_at = datetime.now()
                if error_message:
                    event.error_message = error_message
                if status == EventStatus.COMPLETED:
                    event.processed_at = datetime.now()
                return True
            return False

    def get_pending_events(self, limit: int = 100) -> List[PersistedEvent]:
        """获取待处理事件"""
        with self._lock:
            pending = [e for e in self._events.values()
                       if e.status in [EventStatus.PENDING, EventStatus.RETRY]]
            return sorted(pending, key=lambda e: e.created_at)[:limit]

    def get_failed_events(self, limit: int = 100) -> List[PersistedEvent]:
        """获取失败事件"""
        with self._lock:
            failed = [e for e in self._events.values() if e.status == EventStatus.FAILED]
            return sorted(failed, key=lambda e: e.updated_at, reverse=True)[:limit]

    def cleanup_expired_events(self, max_age_days: int = 30) -> int:
        """清理过期事件"""
        with self._lock:
            cutoff = datetime.now() - timedelta(days=max_age_days)
            expired_ids = [eid for eid, event in self._events.items()
                           if event.created_at < cutoff]

            for eid in expired_ids:
                del self._events[eid]

            return len(expired_ids)

    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[PersistedEvent]:
        """根据事件类型获取事件"""
        with self._lock:
            matching = [e for e in self._events.values() if e.event_type == event_type]
            return sorted(matching, key=lambda e: e.created_at, reverse=True)[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'total_events': len(self._events),
                'pending_events': len([e for e in self._events.values()
                                       if e.status in [EventStatus.PENDING, EventStatus.RETRY]]),
                'completed_events': len([e for e in self._events.values()
                                         if e.status == EventStatus.COMPLETED]),
                'failed_events': len([e for e in self._events.values()
                                      if e.status == EventStatus.FAILED])
            }


class FileEventPersistence(EventPersistenceStrategy):
    """文件事件持久化策略"""

    def __init__(self, storage_path: str = "./event_storage"):
        self.storage_path = storage_path
        self._ensure_storage_path()
        self._lock = threading.RLock()

        # 索引文件
        self._index_file = os.path.join(storage_path, "event_index.json")
        self._load_index()

    def _ensure_storage_path(self):
        """确保存储路径存在"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def _load_index(self):
        """加载索引"""
        if os.path.exists(self._index_file):
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"加载事件索引失败: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self):
        """保存索引"""
        try:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存事件索引失败: {e}")

    def _get_event_file_path(self, event_id: str) -> str:
        """获取事件文件路径"""
        return os.path.join(self.storage_path, f"{event_id}.json.gz")

    def store_event(self, event: PersistedEvent) -> bool:
        """存储事件"""
        try:
            with self._lock:
                file_path = self._get_event_file_path(event.event_id)

                # 压缩存储
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    json.dump(event.to_dict(), f, ensure_ascii=False, indent=2)

                # 更新索引
                self._index[event.event_id] = {
                    'event_type': event.event_type,
                    'status': event.status.value,
                    'created_at': event.created_at.isoformat(),
                    'file_path': file_path
                }
                self._save_index()

                return True

        except Exception as e:
            logger.error(f"存储事件失败 {event.event_id}: {e}")
            return False

    def retrieve_event(self, event_id: str) -> Optional[PersistedEvent]:
        """检索事件"""
        try:
            with self._lock:
                if event_id not in self._index:
                    return None

                file_path = self._get_event_file_path(event_id)
                if not os.path.exists(file_path):
                    # 索引存在但文件不存在，从索引中移除
                    del self._index[event_id]
                    self._save_index()
                    return None

                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)

                return PersistedEvent.from_dict(data)

        except Exception as e:
            logger.error(f"检索事件失败 {event_id}: {e}")
            return None

    def update_event_status(self, event_id: str, status: EventStatus,
                            error_message: Optional[str] = None) -> bool:
        """更新事件状态"""
        try:
            with self._lock:
                if event_id not in self._index:
                    return False

                # 读取现有事件
                event = self.retrieve_event(event_id)
                if not event:
                    return False

                # 更新状态
                event.status = status
                event.updated_at = datetime.now()
                if error_message:
                    event.error_message = error_message
                if status == EventStatus.COMPLETED:
                    event.processed_at = datetime.now()

                # 重新存储
                success = self.store_event(event)
                if success:
                    # 更新索引中的状态
                    self._index[event_id]['status'] = status.value
                    self._save_index()

                return success

        except Exception as e:
            logger.error(f"更新事件状态失败 {event_id}: {e}")
            return False

    def get_pending_events(self, limit: int = 100) -> List[PersistedEvent]:
        """获取待处理事件"""
        try:
            with self._lock:
                pending_ids = [eid for eid, info in self._index.items()
                               if info['status'] in [EventStatus.PENDING.value, EventStatus.RETRY.value]]

                events = []
                for event_id in pending_ids[:limit]:
                    event = self.retrieve_event(event_id)
                    if event:
                        events.append(event)

                return sorted(events, key=lambda e: e.created_at)

        except Exception as e:
            logger.error(f"获取待处理事件失败: {e}")
            return []

    def get_failed_events(self, limit: int = 100) -> List[PersistedEvent]:
        """获取失败事件"""
        try:
            with self._lock:
                failed_ids = [eid for eid, info in self._index.items()
                              if info['status'] == EventStatus.FAILED.value]

                events = []
                for event_id in failed_ids[:limit]:
                    event = self.retrieve_event(event_id)
                    if event:
                        events.append(event)

                return sorted(events, key=lambda e: e.updated_at, reverse=True)

        except Exception as e:
            logger.error(f"获取失败事件失败: {e}")
            return []

    def cleanup_expired_events(self, max_age_days: int = 30) -> int:
        """清理过期事件"""
        try:
            with self._lock:
                cutoff = datetime.now() - timedelta(days=max_age_days)
                expired_ids = []

                for event_id, info in list(self._index.items()):
                    created_at = datetime.fromisoformat(info['created_at'])
                    if created_at < cutoff:
                        expired_ids.append(event_id)

                # 删除文件和索引
                for event_id in expired_ids:
                    file_path = self._get_event_file_path(event_id)
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"删除事件文件失败 {event_id}: {e}")

                    del self._index[event_id]

                self._save_index()
                return len(expired_ids)

        except Exception as e:
            logger.error(f"清理过期事件失败: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            with self._lock:
                stats = {
                    'total_events': len(self._index),
                    'pending_events': 0,
                    'completed_events': 0,
                    'failed_events': 0,
                    'processing_events': 0,
                    'expired_events': 0
                }

                for info in self._index.values():
                    status = info['status']
                    if status == EventStatus.PENDING.value:
                        stats['pending_events'] += 1
                    elif status == EventStatus.COMPLETED.value:
                        stats['completed_events'] += 1
                    elif status == EventStatus.FAILED.value:
                        stats['failed_events'] += 1
                    elif status == EventStatus.PROCESSING.value:
                        stats['processing_events'] += 1
                    elif status == EventStatus.EXPIRED.value:
                        stats['expired_events'] += 1

                return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


class EventPersistence(StandardComponent):
    """事件持久化管理器"""

    def __init__(self, mode: PersistenceMode = PersistenceMode.FILE,
                 config: Optional[Dict[str, Any]] = None):
        """初始化事件持久化管理器

        Args:
            mode: 持久化模式
            config: 配置参数
        """
        super().__init__("EventPersistence", "1.0.0", "事件持久化管理器")

        self.mode = mode
        self.config = config or {}

        # 持久化策略
        self._strategy: Optional[EventPersistenceStrategy] = None

        # 统计信息
        self._stats = {
            'stored_events': 0,
            'retrieved_events': 0,
            'updated_events': 0,
            'cleaned_events': 0
        }

        # 清理配置
        self._cleanup_interval = self.config.get('cleanup_interval', 3600)  # 1小时
        self._max_age_days = self.config.get('max_age_days', 30)  # 30天

        # 压缩配置
        self._compression_enabled = self.config.get('compression_enabled', True)

        # 存储路径（用于文件模式）
        self._storage_path = self.config.get('storage_path', './event_storage')

        # 线程安全
        self._lock = threading.RLock()

    def initialize(self) -> bool:
        """初始化持久化管理器"""
        try:
            self.set_status(ComponentStatus.INITIALIZING)

            # 创建持久化策略
            if self.mode == PersistenceMode.MEMORY:
                self._strategy = MemoryEventPersistence()
            elif self.mode == PersistenceMode.FILE:
                storage_path = self.config.get('storage_path', './event_storage')
                self._strategy = FileEventPersistence(storage_path)
            elif self.mode == PersistenceMode.DATABASE:
                # TODO: 实现数据库持久化策略
                raise NotImplementedError("数据库持久化策略暂未实现")
            elif self.mode == PersistenceMode.DISTRIBUTED:
                # TODO: 实现分布式持久化策略
                raise NotImplementedError("分布式持久化策略暂未实现")
            else:
                raise ValueError(f"不支持的持久化模式: {self.mode}")

            self.set_status(ComponentStatus.INITIALIZED)
            self.set_health(ComponentHealth.HEALTHY)

            logger.info(f"事件持久化管理器初始化完成，模式: {self.mode.value}")
            return True

        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            self.set_health(ComponentHealth.UNHEALTHY)
            logger.error(f"事件持久化管理器初始化失败: {e}")
            return False

    def shutdown(self) -> bool:
        """关闭持久化管理器"""
        try:
            self.set_status(ComponentStatus.STOPPING)

            # 执行最终清理
            if self._strategy:
                cleaned = self._strategy.cleanup_expired_events(self.max_age_days)
                logger.info(f"关闭时清理了 {cleaned} 个过期事件")

            self.set_status(ComponentStatus.STOPPED)
            logger.info("事件持久化管理器已关闭")
            return True

        except Exception as e:
            logger.error(f"事件持久化管理器关闭失败: {e}")
            return False

    def store_event(self, event_id_or_event, event_type: str = None, event_data: Dict[str, Any] = None,
                    timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """存储事件"""
        # 处理重载：如果第一个参数是PersistedEvent对象
        if isinstance(event_id_or_event, PersistedEvent):
            event = event_id_or_event
            return self._store_persisted_event(event)
        else:
            # 原始参数格式
            event_id = event_id_or_event
            return self._store_by_params(event_id, event_type, event_data, timestamp, metadata)

    def _store_persisted_event(self, event: PersistedEvent) -> bool:
        """存储PersistedEvent对象"""
        try:
            with self._lock:
                if not self._strategy:
                    return False

                success = self._strategy.store_event(event)
                if success:
                    self._stats['stored_events'] += 1

                return success

        except Exception as e:
            logger.error(f"存储事件失败 {event.event_id}: {e}")
            return False

    def _store_by_params(self, event_id: str, event_type: str, event_data: Dict[str, Any],
                        timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """按参数存储事件"""
        try:
            with self._lock:
                if not self._strategy:
                    return False

                persisted_event = PersistedEvent(
                    event_id=event_id,
                    event_type=event_type,
                    event_data=event_data,
                    timestamp=timestamp or datetime.now(),
                    metadata=metadata or {}
                )

                success = self._strategy.store_event(persisted_event)
                if success:
                    self._stats['stored_events'] += 1

                return success
        except Exception as e:
            logger.error(f"按参数存储事件失败 {event_id}: {e}")
            return False

    def retrieve_event(self, event_id: str) -> Optional[PersistedEvent]:
        """检索事件"""
        try:
            with self._lock:
                if not self._strategy:
                    return None

                event = self._strategy.retrieve_event(event_id)
                if event:
                    self._stats['retrieved_events'] += 1

                return event

        except Exception as e:
            logger.error(f"检索事件失败 {event_id}: {e}")
            return None

    def update_event_status(self, event_id: str, status: EventStatus,
                            error_message: Optional[str] = None) -> bool:
        """更新事件状态"""
        try:
            with self._lock:
                if not self._strategy:
                    return False

                success = self._strategy.update_event_status(event_id, status, error_message)
                if success:
                    self._stats['updated_events'] += 1

                return success

        except Exception as e:
            logger.error(f"更新事件状态失败 {event_id}: {e}")
            return False

    def mark_event_processing(self, event_id: str) -> bool:
        """标记事件为处理中"""
        return self.update_event_status(event_id, EventStatus.PROCESSING)

    def mark_event_completed(self, event_id: str) -> bool:
        """标记事件为已完成"""
        return self.update_event_status(event_id, EventStatus.COMPLETED)

    def mark_event_failed(self, event_id: str, error_message: str) -> bool:
        """标记事件为失败"""
        return self.update_event_status(event_id, EventStatus.FAILED, error_message)

    def mark_event_retry(self, event_id: str) -> bool:
        """标记事件为重试"""
        try:
            with self._lock:
                event = self.retrieve_event(event_id)
                if event and event.retry_count < event.max_retries:
                    event.retry_count += 1
                    # 重新存储事件
                    success = self._strategy.store_event(event) if self._strategy else False
                    if success:
                        self.update_event_status(event_id, EventStatus.RETRY)
                    return success
                return False

        except Exception as e:
            logger.error(f"标记事件重试失败 {event_id}: {e}")
            return False

    def get_pending_events(self, limit: int = 100) -> List[PersistedEvent]:
        """获取待处理事件"""
        try:
            with self._lock:
                return self._strategy.get_pending_events(limit) if self._strategy else []

        except Exception as e:
            logger.error(f"获取待处理事件失败: {e}")
            return []

    def get_failed_events(self, limit: int = 100) -> List[PersistedEvent]:
        """获取失败事件"""
        try:
            with self._lock:
                return self._strategy.get_failed_events(limit) if self._strategy else []

        except Exception as e:
            logger.error(f"获取失败事件失败: {e}")
            return []

    def replay_events(self, event_types: Optional[List[str]] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> Iterator[PersistedEvent]:
        """重放事件"""
        # TODO: 实现事件重放功能
        # 这里需要根据时间范围和事件类型过滤事件
        logger.info("事件重放功能暂未实现")
        return iter([])

    def cleanup_expired_events(self, max_age_days: Optional[int] = None) -> int:
        """清理过期事件"""
        try:
            with self._lock:
                if not self._strategy:
                    return 0

                age_days = max_age_days or self.max_age_days
                cleaned = self._strategy.cleanup_expired_events(age_days)
                self._stats['cleaned_events'] += cleaned

                logger.info(f"清理了 {cleaned} 个过期事件")
                return cleaned

        except Exception as e:
            logger.error(f"清理过期事件失败: {e}")
            return 0

    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[PersistedEvent]:
        """根据事件类型获取事件"""
        try:
            with self._lock:
                if not self._strategy:
                    return []

                return self._strategy.get_events_by_type(event_type, limit)

        except Exception as e:
            logger.error(f"获取事件类型失败 {event_type}: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            with self._lock:
                strategy_stats = self._strategy.get_stats() if self._strategy else {}

                return {
                    **self._stats,
                    **strategy_stats,
                    'persistence_mode': self.mode.value,
                    'cleanup_interval': self.cleanup_interval,
                    'max_age_days': self.max_age_days
                }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def get_event_summary(self) -> Dict[str, Any]:
        """获取事件摘要"""
        try:
            with self._lock:
                stats = self.get_stats()

                return {
                    'total_events': stats.get('total_events', 0),
                    'pending_events': stats.get('pending_events', 0),
                    'processing_events': stats.get('processing_events', 0),
                    'completed_events': stats.get('completed_events', 0),
                    'failed_events': stats.get('failed_events', 0),
                    'expired_events': stats.get('expired_events', 0),
                    'storage_mode': self.mode.value,
                    'last_cleanup': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"获取事件摘要失败: {e}")
            return {}

    def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查（StandardComponent要求）"""
        try:
            stats = self.get_stats()

            health_status = {
                'component_name': self.service_name,
                'status': 'healthy',
                'total_events': stats.get('total_events', 0),
                'pending_events': stats.get('pending_events', 0),
                'failed_events': stats.get('failed_events', 0),
                'storage_mode': self.mode.value,
                'last_check': datetime.now().isoformat()
            }

            # 如果失败事件太多，认为不健康
            if stats.get('failed_events', 0) > stats.get('total_events', 1) * 0.1:
                health_status['status'] = 'warning'

            return health_status

        except Exception as e:
            logger.error(f"事件持久化健康检查失败: {e}")
            return {
                'component_name': self.service_name,
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
