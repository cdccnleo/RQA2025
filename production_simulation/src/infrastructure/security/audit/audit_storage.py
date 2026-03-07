#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 审计存储管理器

专门负责审计事件的存储、检索和持久化
从AuditLoggingManager中分离出来，提高代码组织性
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import gzip
import shutil


class AuditStorageManager:
    """审计存储管理器"""

    def __init__(self, storage_path: Optional[Path] = None, max_memory_events: int = 5000):
        self.storage_path = storage_path or Path("data/audit")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._memory_events: List['AuditEvent'] = []
        self._max_memory_events = max_memory_events
        self._lock = threading.RLock()

        # 存储配置
        self._archive_threshold = 1000  # 达到此数量时归档
        self._compression_enabled = True
        self._retention_days = 90  # 保留天数

        # 统计信息
        self._stats = {
            'total_events': 0,
            'archived_files': 0,
            'storage_size_mb': 0.0
        }

    def store_event(self, event: 'AuditEvent') -> None:
        """存储审计事件"""
        with self._lock:
            self._memory_events.append(event)
            self._stats['total_events'] += 1

            # 检查是否需要持久化
            if len(self._memory_events) >= self._archive_threshold:
                self._archive_events()

            # 检查是否需要清理内存
            if len(self._memory_events) > self._max_memory_events:
                self._cleanup_memory()

    def get_events(self, start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  event_type: Optional[str] = None,
                  limit: Optional[int] = None) -> List['AuditEvent']:
        """检索审计事件"""
        with self._lock:
            events = []

            # 从内存获取
            memory_events = self._filter_events(self._memory_events,
                                              start_time, end_time, event_type)
            events.extend(memory_events)

            # 从归档文件获取
            archived_events = self._get_archived_events(start_time, end_time, event_type)
            events.extend(archived_events)

            # 按时间排序（默认按时间升序，保持原始顺序）
            events.sort(key=lambda x: x.timestamp)

            if limit:
                events = events[:limit]

            return events

    def archive_old_events(self) -> int:
        """归档旧事件"""
        with self._lock:
            archived_count = 0

            # 归档当前内存中的事件
            if self._memory_events:
                archived_count += len(self._memory_events)
                self._archive_events()

            return archived_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        with self._lock:
            # 计算存储大小
            total_size = 0
            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

            self._stats['storage_size_mb'] = total_size / (1024 * 1024)
            self._stats['archived_files'] = len(list(self.storage_path.glob("*.json*")))

            return self._stats.copy()

    def cleanup_storage(self, days_to_keep: Optional[int] = None) -> int:
        """清理存储空间"""
        with self._lock:
            retention_days = days_to_keep or self._retention_days
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            deleted_count = 0
            for file_path in self.storage_path.glob("*.json*"):
                try:
                    # 从文件名解析日期
                    date_str = file_path.stem.split('_')[1]  # 格式: audit_20241026.json.gz
                    file_date = datetime.strptime(date_str, '%Y%m%d')

                    if file_date < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
                        logging.info(f"Cleaned up old audit file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to parse date from {file_path}: {e}")

            return deleted_count

    def _archive_events(self) -> None:
        """归档当前内存中的事件"""
        if not self._memory_events:
            return

        # 按日期分组事件
        events_by_date = defaultdict(list)
        for event in self._memory_events:
            date_key = event.timestamp.strftime('%Y%m%d')
            events_by_date[date_key].append(event)

        # 为每个日期创建归档文件
        for date_str, events in events_by_date.items():
            self._write_archive_file(date_str, events)

        # 清空内存
        self._memory_events.clear()

    def _write_archive_file(self, date_str: str, events: List['AuditEvent']) -> None:
        """写入归档文件"""
        filename = f"audit_{date_str}.json"
        if self._compression_enabled:
            filename += ".gz"

        file_path = self.storage_path / filename

        try:
            event_dicts = [event.to_dict() for event in events]

            if self._compression_enabled:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    json.dump(event_dicts, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(event_dicts, f, indent=2, ensure_ascii=False)

            logging.info(f"Archived {len(events)} events to {file_path}")

        except Exception as e:
            logging.error(f"Failed to archive events to {file_path}: {e}")

    def _filter_events(self, events: List['AuditEvent'],
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      event_type: Optional[str] = None) -> List['AuditEvent']:
        """过滤事件"""
        filtered_events = events.copy()

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type.value == event_type]

        return filtered_events

    def _get_archived_events(self, start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           event_type: Optional[str] = None) -> List['AuditEvent']:
        """从归档文件获取事件"""
        archived_events = []

        try:
            date_range, _ = self._get_date_range(start_time, end_time)

            for date_str in date_range:
                events = self._read_archive_file(date_str)
                if events:
                    filtered_events = self._filter_events(events, start_time, end_time, event_type)
                    archived_events.extend(filtered_events)

        except Exception as e:
            logging.error(f"Failed to read archived events: {e}")

        return archived_events

    def _read_archive_file(self, date_str: str) -> List['AuditEvent']:
        """读取归档文件"""
        events = []

        # 尝试压缩文件
        compressed_path = self.storage_path / f"audit_{date_str}.json.gz"
        if compressed_path.exists():
            try:
                with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                    event_dicts = json.load(f)
                    events = [self._dict_to_event(d) for d in event_dicts]
            except Exception as e:
                logging.error(f"Failed to read compressed archive {compressed_path}: {e}")

        # 尝试未压缩文件
        uncompressed_path = self.storage_path / f"audit_{date_str}.json"
        if not events and uncompressed_path.exists():
            try:
                with open(uncompressed_path, 'r', encoding='utf-8') as f:
                    event_dicts = json.load(f)
                    events = [self._dict_to_event(d) for d in event_dicts]
            except Exception as e:
                logging.error(f"Failed to read uncompressed archive {uncompressed_path}: {e}")

        return events

    def _get_date_range(self, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> tuple:
        """获取需要检查的日期范围，返回 (日期字符串列表, (开始时间, 结束时间))"""
        if not start_time and not end_time:
            start_time = datetime.now() - timedelta(days=7)

        if not start_time:
            start_time = datetime.now() - timedelta(days=30)
        if not end_time:
            end_time = datetime.now()

        dates = []
        current_date = start_time.date()
        end_date = end_time.date()

        while current_date <= end_date:
            dates.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)

        return dates, (start_time, end_time)

    def _get_expired_archive_files(self) -> List[Path]:
        """获取过期的归档文件"""
        expired_files = []
        cutoff_date = datetime.now() - timedelta(days=self._retention_days)

        for file_path in self.storage_path.glob("audit_*.json*"):
            try:
                name = file_path.name
                core = name.split('.')[0]  # audit_YYYYMMDD or audit_YYYY-MM-DD
                date_str = core.split('_', 1)[1]
                normalized = date_str.replace('-', '')
                file_date = datetime.strptime(normalized, '%Y%m%d')
                if file_date < cutoff_date:
                    expired_files.append(file_path)
            except Exception:
                continue

        return expired_files

    def _cleanup_memory(self) -> None:
        """清理内存中的旧事件"""
        # 保留最近的事件
        keep_count = self._max_memory_events // 2
        if len(self._memory_events) > keep_count:
            self._memory_events = self._memory_events[-keep_count:]

    @staticmethod
    def _dict_to_event(data: Dict[str, Any]) -> 'AuditEvent':
        """将字典转换为AuditEvent对象"""
        from src.infrastructure.security.audit.audit_events import AuditEvent, AuditEventType, AuditSeverity

        event_type = data.get('event_type')
        if isinstance(event_type, AuditEventType):
            parsed_event_type = event_type
        else:
            parsed_event_type = AuditEventType(str(event_type))

        severity = data.get('severity')
        if isinstance(severity, AuditSeverity):
            parsed_severity = severity
        else:
            parsed_severity = AuditSeverity(str(severity))

        timestamp_value = data.get('timestamp')
        if isinstance(timestamp_value, datetime):
            parsed_timestamp = timestamp_value
        else:
            parsed_timestamp = datetime.fromisoformat(str(timestamp_value))

        tags = data.get('tags', [])
        if isinstance(tags, set):
            tag_set = tags
        else:
            tag_set = set(tags)

        return AuditEvent(
            event_id=data['event_id'],
            event_type=parsed_event_type,
            severity=parsed_severity,
            timestamp=parsed_timestamp,
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            resource=data.get('resource'),
            action=data.get('action', ''),
            result=data.get('result', ''),
            details=data.get('details', {}),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            location=data.get('location'),
            risk_score=data.get('risk_score', 0.0),
            tags=tag_set
        )
