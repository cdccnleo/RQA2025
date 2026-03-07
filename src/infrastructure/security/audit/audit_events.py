#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 审计事件管理器

专门负责审计事件的定义、创建和管理
从AuditLoggingManager中分离出来，提高代码组织性
"""

import logging
import uuid
import asyncio
import threading
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import defaultdict


class AuditEventType(Enum):
    """审计事件类型"""
    SECURITY = "security"      # 安全事件
    ACCESS = "access"         # 访问事件
    DATA_OPERATION = "data_operation"  # 数据操作
    CONFIG_CHANGE = "config_change"    # 配置变更
    USER_MANAGEMENT = "user_management"  # 用户管理
    SYSTEM_EVENT = "system_event"      # 系统事件
    COMPLIANCE = "compliance"   # 合规事件


class AuditSeverity(Enum):
    """审计事件严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    result: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    risk_score: float = 0.0
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'resource': self.resource,
            'action': self.action,
            'result': self.result,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'location': self.location,
            'risk_score': self.risk_score,
            'tags': list(self.tags)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """从字典创建事件"""
        return cls(
            event_id=data['event_id'],
            event_type=AuditEventType(data['event_type']),
            severity=AuditSeverity(data['severity']),
            timestamp=datetime.fromisoformat(data['timestamp']),
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
            tags=set(data.get('tags', []))
        )


class AuditEventBuilder:
    """审计事件构建器"""

    def __init__(self):
        self._event = None

    def new_event(self, event_type: AuditEventType, severity: AuditSeverity) -> 'AuditEventBuilder':
        """创建新的事件"""
        self._event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now()
        )
        return self

    def with_user(self, user_id: Optional[str]) -> 'AuditEventBuilder':
        """设置用户ID"""
        if self._event:
            self._event.user_id = user_id
        return self

    def with_session(self, session_id: Optional[str]) -> 'AuditEventBuilder':
        """设置会话ID"""
        if self._event:
            self._event.session_id = session_id
        return self

    def with_resource(self, resource: Optional[str]) -> 'AuditEventBuilder':
        """设置资源"""
        if self._event:
            self._event.resource = resource
        return self

    def with_action(self, action: str) -> 'AuditEventBuilder':
        """设置操作"""
        if self._event:
            self._event.action = action
        return self

    def with_result(self, result: str) -> 'AuditEventBuilder':
        """设置结果"""
        if self._event:
            self._event.result = result
        return self

    def with_details(self, details: Dict[str, Any]) -> 'AuditEventBuilder':
        """设置详细信息"""
        if self._event:
            self._event.details.update(details)
        return self

    def with_ip_address(self, ip_address: Optional[str]) -> 'AuditEventBuilder':
        """设置IP地址"""
        if self._event:
            self._event.ip_address = ip_address
        return self

    def with_user_agent(self, user_agent: Optional[str]) -> 'AuditEventBuilder':
        """设置用户代理"""
        if self._event:
            self._event.user_agent = user_agent
        return self

    def with_location(self, location: Optional[str]) -> 'AuditEventBuilder':
        """设置位置"""
        if self._event:
            self._event.location = location
        return self

    def with_risk_score(self, risk_score: float) -> 'AuditEventBuilder':
        """设置风险分数"""
        if self._event:
            self._event.risk_score = risk_score
        return self

    def with_tags(self, tags: Set[str]) -> 'AuditEventBuilder':
        """设置标签"""
        if self._event:
            self._event.tags.update(tags)
        return self

    def build(self) -> AuditEvent:
        """构建事件"""
        if not self._event:
            raise ValueError("No event to build. Call new_event() first.")
        return self._event

    def _generate_event_id(self) -> str:
        """生成事件ID"""
        return f"audit_{uuid.uuid4().hex[:12]}"


class AuditEventFilter:
    """审计事件过滤器"""

    def __init__(self):
        self._filters = {}

    def by_event_type(self, event_types: List[AuditEventType]) -> 'AuditEventFilter':
        """按事件类型过滤"""
        self._filters['event_types'] = event_types
        return self

    def by_severity(self, severities: List[AuditSeverity]) -> 'AuditEventFilter':
        """按严重程度过滤"""
        self._filters['severities'] = severities
        return self

    def by_user(self, user_ids: List[str]) -> 'AuditEventFilter':
        """按用户过滤"""
        self._filters['user_ids'] = user_ids
        return self

    def by_time_range(self, start_time: datetime, end_time: datetime) -> 'AuditEventFilter':
        """按时间范围过滤"""
        self._filters['start_time'] = start_time
        self._filters['end_time'] = end_time
        return self

    def by_resource(self, resources: List[str]) -> 'AuditEventFilter':
        """按资源过滤"""
        self._filters['resources'] = resources
        return self

    def by_risk_score(self, min_score: float, max_score: float) -> 'AuditEventFilter':
        """按风险分数过滤"""
        self._filters['min_risk_score'] = min_score
        self._filters['max_risk_score'] = max_score
        return self

    def matches(self, event: AuditEvent) -> bool:
        """检查事件是否匹配过滤条件"""
        # 事件类型过滤
        if 'event_types' in self._filters:
            if event.event_type not in self._filters['event_types']:
                return False

        # 严重程度过滤
        if 'severities' in self._filters:
            if event.severity not in self._filters['severities']:
                return False

        # 用户过滤
        if 'user_ids' in self._filters:
            if event.user_id not in self._filters['user_ids']:
                return False

        # 时间范围过滤
        if 'start_time' in self._filters and 'end_time' in self._filters:
            if not (self._filters['start_time'] <= event.timestamp <= self._filters['end_time']):
                return False

        # 资源过滤
        if 'resources' in self._filters:
            if event.resource not in self._filters['resources']:
                return False

        # 风险分数过滤
        if 'min_risk_score' in self._filters and 'max_risk_score' in self._filters:
            if not (self._filters['min_risk_score'] <= event.risk_score <= self._filters['max_risk_score']):
                return False

        return True


class AuditEventManager:
    """审计事件管理器"""

    def __init__(self, max_events: int = 10000):
        self._events: List[AuditEvent] = []
        self._max_events = max_events
        self._lock = threading.RLock()
        self._event_builder = AuditEventBuilder()

    def create_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: str = "",
        result: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        location: Optional[str] = None,
        risk_score: Optional[float] = None,
        tags: Optional[Set[str]] = None,
    ) -> AuditEvent:
        """创建审计事件"""
        with self._lock:
            event_builder = (
                self._event_builder
                .new_event(event_type, severity)
                .with_user(user_id)
                .with_session(session_id)
                .with_resource(resource)
                .with_action(action or "")
            )

            resolved_details = details.copy() if details else {}
            if result is None:
                if 'result' in resolved_details:
                    resolved_result = str(resolved_details['result'])
                elif 'success' in resolved_details:
                    resolved_result = "success" if resolved_details['success'] else "failure"
                else:
                    resolved_result = ""
            else:
                resolved_result = result

            if resolved_result:
                event_builder = event_builder.with_result(resolved_result)

            if resolved_details:
                event_builder = event_builder.with_details(resolved_details)

            if ip_address:
                event_builder = event_builder.with_ip_address(ip_address)
            if user_agent:
                event_builder = event_builder.with_user_agent(user_agent)
            if location:
                event_builder = event_builder.with_location(location)
            if risk_score is not None:
                event_builder = event_builder.with_risk_score(risk_score)
            if tags:
                event_builder = event_builder.with_tags(set(tags))

            event = event_builder.build()

            # 如果仍未设置 result，确保字段存在
            if not event.result and resolved_result:
                event.result = resolved_result

            self._add_event(event)
            return event

    def add_event(self, event: AuditEvent) -> None:
        """添加审计事件"""
        with self._lock:
            self._add_event(event)

    def get_events(self, filter_obj: Optional[AuditEventFilter] = None,
                  limit: Optional[int] = None) -> List[AuditEvent]:
        """获取审计事件"""
        with self._lock:
            events = self._events.copy()

            if filter_obj:
                events = [e for e in events if filter_obj.matches(e)]

            if limit:
                events = events[-limit:]

            return events

    async def get_events_async(self, filter_obj: Optional[AuditEventFilter] = None,
                              limit: Optional[int] = None) -> List[AuditEvent]:
        """异步获取审计事件"""
        # 在线程池中执行，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.get_events, filter_obj, limit
        )

    async def create_event_async(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: str = "",
        result: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        location: Optional[str] = None,
        risk_score: Optional[float] = None,
        tags: Optional[Set[str]] = None,
    ) -> AuditEvent:
        """异步创建审计事件"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.create_event,
            event_type,
            severity,
            user_id,
            session_id,
            resource,
            action,
            result,
            details,
            ip_address,
            user_agent,
            location,
            risk_score,
            tags,
        )

    def clear_events(self) -> None:
        """清除所有事件"""
        with self._lock:
            self._events.clear()

    def get_event_count(self) -> int:
        """获取事件总数"""
        with self._lock:
            return len(self._events)

    def get_events_by_type(self, event_type: AuditEventType) -> List[AuditEvent]:
        """按类型获取事件"""
        return self.get_events(AuditEventFilter().by_event_type([event_type]))

    def get_events_by_severity(self, severity: AuditSeverity) -> List[AuditEvent]:
        """按严重程度获取事件"""
        return self.get_events(AuditEventFilter().by_severity([severity]))

    def get_recent_events(self, hours: int = 24) -> List[AuditEvent]:
        """获取最近的事件"""
        start_time = datetime.now() - timedelta(hours=hours)
        end_time = datetime.now()
        return self.get_events(AuditEventFilter().by_time_range(start_time, end_time))

    def _add_event(self, event: AuditEvent) -> None:
        """添加事件到存储"""
        self._events.append(event)

        # 保持最大事件数量
        if len(self._events) > self._max_events:
            self._events.pop(0)

        logging.debug(f"Added audit event: {event.event_id} ({event.event_type.value})")
