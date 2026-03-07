#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 访问控制组件 - 审计日志器

负责访问控制相关的审计日志记录和管理
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import threading

from .access_checker import AccessDecision, AccessRequest
from ...core.types import AuditEvent


class AuditLogger:
    """
    审计日志器

    负责访问控制相关的审计日志记录、查询和管理
    """

    def __init__(self, log_path: Optional[Path] = None, max_log_files: int = 30,
                 enable_async: bool = False):
        """
        初始化审计日志器

        Args:
            log_path: 日志存储路径
            max_log_files: 最大日志文件数量
            enable_async: 是否启用异步写入
        """
        self.log_path = log_path or Path("data/security/audit")
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.max_log_files = max_log_files
        self.enable_async = enable_async

        # 内存缓存最近的审计事件，避免并发写入导致查询不到
        self._recent_events: List[AuditEvent] = []
        self._recent_lock = threading.Lock()
        self._recent_max_events = 1000
        self.event_queue: List[AuditEvent] = []

        # 日志队列和线程（用于异步写入）
        self._log_queue = []
        self._log_lock = threading.Lock()
        self._log_thread = None
        self._stop_thread = False

        if self.enable_async:
            self._start_async_writer()

        logging.info("审计日志器初始化完成")

    def log_audit_event(self, event: AuditEvent) -> None:
        """记录外部传入的审计事件"""
        self._log_event(event)

    def log_access_check(self, request: AccessRequest, decision: AccessDecision,
                        details: Optional[Dict[str, Any]] = None):
        """
        记录访问检查事件

        Args:
            request: 访问请求
            decision: 访问决策
            details: 额外详情
        """
        event = AuditEvent(
            event_id=f"audit_{int(datetime.now().timestamp() * 1000000)}",
            timestamp=request.timestamp,
            user_id=request.user_id,
            action="access_check",
            resource=request.resource,
            permission=request.permission,
            decision=decision,
            details=details,
            ip_address=request.context.get('ip_address'),
            user_agent=request.context.get('user_agent')
        )

        self._log_event(event)

    def log_user_action(self, user_id: str, action: str, resource: str,
                       details: Optional[Dict[str, Any]] = None):
        """
        记录用户操作事件

        Args:
            user_id: 用户ID
            action: 操作类型
            resource: 资源标识
            details: 操作详情
        """
        event = AuditEvent(
            event_id=f"audit_{int(datetime.now().timestamp() * 1000000)}",
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            permission="",  # 用户操作可能不涉及权限
            decision=AccessDecision.ALLOW,  # 用户操作默认允许
            details=details
        )

        self._log_event(event)

    def log_security_event(self, event_type: str, severity: str, user_id: str,
                          description: str, details: Optional[Dict[str, Any]] = None):
        """
        记录安全事件

        Args:
            event_type: 事件类型
            severity: 严重程度
            user_id: 用户ID
            description: 事件描述
            details: 事件详情
        """
        event = AuditEvent(
            event_id=f"audit_{int(datetime.now().timestamp() * 1000000)}",
            timestamp=datetime.now(),
            user_id=user_id,
            action=f"security_{event_type}",
            resource="system",
            permission="",
            decision=AccessDecision.DENY if severity in ['high', 'critical'] else AccessDecision.ALLOW,
            details={
                'event_type': event_type,
                'severity': severity,
                'description': description,
                **(details or {})
            }
        )

        self._log_event(event)

    def query_audit_logs(self, user_id: Optional[str] = None,
                        action: Optional[str] = None,
                        resource: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        limit: int = 100) -> List[AuditEvent]:
        """
        查询审计日志

        Args:
            user_id: 用户ID过滤
            action: 操作类型过滤
            resource: 资源过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 结果数量限制

        Returns:
            审计事件列表
        """
        all_events = self._load_recent_logs()
        with self._recent_lock:
            recent_copy = list(self._recent_events)

        merged_events = all_events + recent_copy
        merged_events.sort(key=lambda x: x.timestamp, reverse=True)

        # 应用过滤器
        filtered_events = []
        for event in merged_events:
            if user_id and event.user_id != user_id:
                continue
            if action and action not in event.action:
                continue
            if resource and resource not in event.resource:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            filtered_events.append(event)

            if len(filtered_events) >= limit:
                break

        return filtered_events

    def get_audit_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        获取审计统计信息

        Args:
            days: 统计天数

        Returns:
            统计信息
        """
        start_time = datetime.now() - timedelta(days=days)
        events = self.query_audit_logs(start_time=start_time, limit=10000)

        # 初始化统计结果
        stats = self._init_statistics_structure(start_time, len(events))

        # 执行各项统计
        self._collect_action_statistics(events, stats)
        self._collect_user_statistics(events, stats)
        self._collect_decision_statistics(events, stats)
        self._collect_resource_statistics(events, stats)
        self._collect_hourly_statistics(events, stats)
        self._collect_security_events(events, stats)

        return stats

    def _init_statistics_structure(self, start_time: datetime, total_events: int) -> Dict[str, Any]:
        """
        初始化统计结果结构

        Args:
            start_time: 统计开始时间
            total_events: 总事件数

        Returns:
            初始化的统计结构
        """
        return {
            'total_events': total_events,
            'time_range': f"{start_time.date()} - {datetime.now().date()}",
            'events_by_action': {},
            'events_by_user': {},
            'events_by_decision': {},
            'security_events': [],
            'top_resources': {},
            'hourly_distribution': {}
        }

    def _collect_action_statistics(self, events: List[AuditEvent], stats: Dict[str, Any]):
        """
        收集操作类型统计

        Args:
            events: 审计事件列表
            stats: 统计结果字典
        """
        for event in events:
            stats['events_by_action'][event.action] = \
                stats['events_by_action'].get(event.action, 0) + 1

    def _collect_user_statistics(self, events: List[AuditEvent], stats: Dict[str, Any]):
        """
        收集用户统计

        Args:
            events: 审计事件列表
            stats: 统计结果字典
        """
        for event in events:
            stats['events_by_user'][event.user_id] = \
                stats['events_by_user'].get(event.user_id, 0) + 1

    def _collect_decision_statistics(self, events: List[AuditEvent], stats: Dict[str, Any]):
        """
        收集决策统计

        Args:
            events: 审计事件列表
            stats: 统计结果字典
        """
        for event in events:
            decision_value = event.decision.value
            stats['events_by_decision'][decision_value] = \
                stats['events_by_decision'].get(decision_value, 0) + 1

    def _collect_resource_statistics(self, events: List[AuditEvent], stats: Dict[str, Any]):
        """
        收集资源统计

        Args:
            events: 审计事件列表
            stats: 统计结果字典
        """
        for event in events:
            stats['top_resources'][event.resource] = \
                stats['top_resources'].get(event.resource, 0) + 1

    def _collect_hourly_statistics(self, events: List[AuditEvent], stats: Dict[str, Any]):
        """
        收集小时分布统计

        Args:
            events: 审计事件列表
            stats: 统计结果字典
        """
        for event in events:
            hour = event.timestamp.hour
            stats['hourly_distribution'][hour] = \
                stats['hourly_distribution'].get(hour, 0) + 1

    def _collect_security_events(self, events: List[AuditEvent], stats: Dict[str, Any]):
        """
        收集安全事件

        Args:
            events: 审计事件列表
            stats: 统计结果字典
        """
        for event in events:
            if 'security' in event.action:
                security_event = {
                    'timestamp': event.timestamp.isoformat(),
                    'user_id': event.user_id,
                    'action': event.action,
                    'severity': event.details.get('severity', 'unknown') if event.details else 'unknown'
                }
                stats['security_events'].append(security_event)

    def export_audit_logs(self, file_path: Path, user_id: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> bool:
        """
        导出审计日志

        Args:
            file_path: 导出文件路径
            user_id: 用户ID过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            是否导出成功
        """
        try:
            # 获取要导出的审计事件
            events = self._get_events_for_export(user_id, start_time, end_time)

            # 准备导出数据结构
            export_data = self._prepare_export_data(events, user_id, start_time, end_time)

            # 写入文件
            self._write_export_file(file_path, export_data, len(events))

            return True

        except Exception as e:
            logging.error(f"审计日志导出失败: {e}")
            return False

    def _get_events_for_export(self, user_id: Optional[str] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[AuditEvent]:
        """
        获取要导出的审计事件

        Args:
            user_id: 用户ID过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            审计事件列表
        """
        return self.query_audit_logs(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=100000  # 导出时允许更多记录
        )

    def _prepare_export_data(self, events: List[AuditEvent], user_id: Optional[str],
                           start_time: Optional[datetime],
                           end_time: Optional[datetime]) -> Dict[str, Any]:
        """
        准备导出数据结构

        Args:
            events: 审计事件列表
            user_id: 用户ID过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            导出数据字典
        """
        return {
            'export_time': datetime.now().isoformat(),
            'total_records': len(events),
            'filters': {
                'user_id': user_id,
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None
            },
            'events': [self._format_event_for_export(event) for event in events]
        }

    def _format_event_for_export(self, event: AuditEvent) -> Dict[str, Any]:
        """
        格式化单个事件用于导出

        Args:
            event: 审计事件

        Returns:
            格式化后的事件字典
        """
        return {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id,
            'action': event.action,
            'resource': event.resource,
            'permission': event.permission,
            'decision': event.decision.value,
            'details': event.details,
            'ip_address': event.ip_address,
            'user_agent': event.user_agent
        }

    def _write_export_file(self, file_path: Path, export_data: Dict[str, Any], record_count: int):
        """
        写入导出文件

        Args:
            file_path: 文件路径
            export_data: 导出数据
            record_count: 记录数量
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logging.info(f"审计日志导出成功: {file_path} ({record_count} 条记录)")

    def _log_event(self, event: AuditEvent):
        """
        记录审计事件

        Args:
            event: 审计事件
        """
        self.event_queue.append(event)
        if len(self.event_queue) > self._recent_max_events:
            self.event_queue = self.event_queue[-self._recent_max_events:]

        self._store_recent_event(event)

        if self.enable_async:
            with self._log_lock:
                self._log_queue.append(event)
        else:
            self._write_event_to_file(event)

    def _write_event_to_file(self, event: AuditEvent):
        """
        将事件写入文件

        Args:
            event: 审计事件
        """
        try:
            # 按日期创建日志文件
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.log_path / f"audit_{date_str}.jsonl"

            # 转换为JSON格式
            event_data = event.to_dict()
            decision_value = getattr(event.decision, "value", event.decision)
            event_data["decision"] = decision_value

            # 追加到文件
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data, ensure_ascii=False) + '\n')

            # 日志轮转
            self._rotate_logs_if_needed()

        except Exception as e:
            logging.error(f"写入审计日志失败: {e}")

    def _store_recent_event(self, event: AuditEvent):
        """将事件存入内存缓存，便于快速查询"""
        with self._recent_lock:
            self._recent_events.append(event)
            if len(self._recent_events) > self._recent_max_events:
                self._recent_events = self._recent_events[-self._recent_max_events:]

    def _rotate_logs_if_needed(self):
        """检查并执行日志轮转"""
        try:
            log_files = list(self.log_path.glob("audit_*.jsonl"))
            if len(log_files) > self.max_log_files:
                # 删除最旧的文件
                log_files.sort(key=lambda x: x.stat().st_mtime)
                files_to_remove = log_files[:len(log_files) - self.max_log_files]
                for file_path in files_to_remove:
                    file_path.unlink()
                    logging.info(f"删除过期审计日志: {file_path.name}")
        except Exception as e:
            logging.error(f"日志轮转失败: {e}")

    def _load_recent_logs(self, days: int = 30) -> List[AuditEvent]:
        """
        加载最近的日志

        Args:
            days: 加载天数

        Returns:
            审计事件列表
        """
        start_date = self._calculate_start_date(days)
        log_files = self._get_log_files()

        events = []
        for log_file in log_files:
            file_events = self._load_events_from_file(log_file, start_date)
            events.extend(file_events)

        return self._sort_events_by_time(events)

    def _calculate_start_date(self, days: int) -> datetime:
        """
        计算开始日期

        Args:
            days: 天数

        Returns:
            开始日期
        """
        return datetime.now() - timedelta(days=days)

    def _get_log_files(self) -> List[Path]:
        """
        获取日志文件列表

        Returns:
            排序的日志文件列表（最新的优先）
        """
        log_files = list(self.log_path.glob("audit_*.jsonl"))
        log_files.sort(reverse=True)  # 最新的文件优先
        return log_files

    def _load_events_from_file(self, log_file: Path, start_date: datetime) -> List[AuditEvent]:
        """
        从单个文件加载事件

        Args:
            log_file: 日志文件路径
            start_date: 开始日期

        Returns:
            审计事件列表
        """
        events = []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event_data = json.loads(line.strip())
                        event = self._parse_event_data(event_data, start_date)
                        if event:
                            events.append(event)

        except Exception as e:
            logging.error(f"读取日志文件失败 {log_file.name}: {e}")

        return events

    def _parse_event_data(self, event_data: Dict, start_date: datetime) -> Optional[AuditEvent]:
        """
        解析事件数据

        Args:
            event_data: 事件数据字典
            start_date: 开始日期

        Returns:
            审计事件对象，如果时间不符合要求则返回None
        """
        event_time = datetime.fromisoformat(event_data['timestamp'])
        if event_time < start_date:
            return None

        return AuditEvent(
            event_id=event_data['event_id'],
            timestamp=event_time,
            user_id=event_data['user_id'],
            action=event_data['action'],
            resource=event_data['resource'],
            permission=event_data['permission'],
            decision=AccessDecision(event_data['decision']),
            details=event_data.get('details', {}),
            ip_address=event_data.get('ip_address'),
            user_agent=event_data.get('user_agent')
        )

    def _sort_events_by_time(self, events: List[AuditEvent]) -> List[AuditEvent]:
        """
        按时间排序事件

        Args:
            events: 审计事件列表

        Returns:
            排序后的审计事件列表
        """
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events

    def _start_async_writer(self):
        """启动异步写入线程"""
        self._log_thread = threading.Thread(target=self._async_writer_loop, daemon=True)
        self._log_thread.start()

    def _async_writer_loop(self):
        """异步写入循环"""
        while not self._stop_thread:
            try:
                # 批量处理日志队列
                events_to_write = []
                with self._log_lock:
                    if self._log_queue:
                        events_to_write = self._log_queue[:100]  # 每次处理最多100条
                        del self._log_queue[:len(events_to_write)]

                # 写入事件
                for event in events_to_write:
                    self._write_event_to_file(event)

                # 如果队列为空，短暂休眠
                if not events_to_write:
                    import time
                    time.sleep(0.1)

            except Exception as e:
                logging.error(f"异步日志写入异常: {e}")

    def shutdown(self):
        """关闭审计日志器"""
        self._stop_thread = True
        if self._log_thread:
            self._log_thread.join(timeout=5)

        # 处理剩余的日志
        with self._log_lock:
            for event in self._log_queue:
                self._write_event_to_file(event)
            self._log_queue.clear()

        logging.info("审计日志器已关闭")
