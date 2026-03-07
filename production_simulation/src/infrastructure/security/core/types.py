#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 安全模块参数对象

提供参数对象模式，解决长参数列表问题
提高代码可读性和维护性
"""

from typing import Any, Dict, Iterable, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ========== 基础类型定义 ==========

class UserRole(Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    GUEST = "guest"


class Permission(Enum):
    """权限枚举"""
    # 交易权限
    TRADE_EXECUTE = "trade:execute"
    TRADE_CANCEL = "trade:cancel"
    ORDER_PLACE = "order:place"
    ORDER_CANCEL = "order:cancel"

    # 数据权限
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_EXPORT = "data:export"

    # 系统权限
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    USER_MANAGE = "user:manage"

    # 审计权限
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"


@dataclass
class User:
    """用户"""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: Set[UserRole] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    password_hash: Optional[str] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class UserSession:
    """用户会话"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return datetime.now() > self.expires_at


@dataclass
class AccessPolicy:
    """访问策略"""
    policy_id: str
    name: str
    description: str = ""
    resource_type: str = ""
    resource_pattern: str = ""
    permissions: Set[Any] = field(default_factory=set)
    roles: Set[Any] = field(default_factory=set)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    is_active: bool = True
    expiry_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化后处理，保留原始类型并构建字符串映射"""
        self.permissions = set(self.permissions or [])
        self.roles = set(self.roles or [])
        if self.metadata is None:
            self.metadata = {}

    def permission_values(self) -> Set[str]:
        return {self._value_of(p) for p in self.permissions}

    def role_values(self) -> Set[str]:
        return {self._value_of(r) for r in self.roles}

    @staticmethod
    def _value_of(value: Any) -> str:
        return getattr(value, "value", str(value))


# ========== 事件和审计类型 ==========

class EventType(Enum):
    """事件类型"""
    SECURITY = "security"
    ACCESS = "access"
    DATA_OPERATION = "data_operation"
    CONFIG_CHANGE = "config_change"
    USER_MANAGEMENT = "user_management"
    SYSTEM_EVENT = "system_event"
    COMPLIANCE = "compliance"
    AUTHENTICATION = "authentication"


class EventSeverity(Enum):
    """事件严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEventParams:
    """审计事件参数对象"""
    event_type: EventType
    severity: EventSeverity
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    risk_score: float = 0.0
    tags: Set[str] = field(default_factory=set)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    timestamp: datetime
    user_id: Optional[str] = None
    action: Optional[str] = None
    resource: Optional[str] = None
    permission: Optional[str] = None
    decision: Optional[Any] = None
    result: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    risk_score: float = 0.0
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "permission": self.permission,
            "decision": getattr(self.decision, "value", self.decision),
            "result": self.result,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "location": self.location,
            "risk_score": self.risk_score,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        decision = data.get("decision")

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            event_id=data["event_id"],
            timestamp=timestamp,
            user_id=data.get("user_id"),
            action=data.get("action"),
            resource=data.get("resource"),
            permission=data.get("permission"),
            decision=decision,
            result=data.get("result"),
            details=data.get("details", {}),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            location=data.get("location"),
            risk_score=data.get("risk_score", 0.0),
            tags=set(data.get("tags", [])),
        )


@dataclass
class UserCreationParams:
    """用户创建参数对象"""
    username: str
    email: Optional[str] = None
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: Optional[str] = None
    password: Optional[str] = None
    require_password_change: bool = False
    expiry_date: Optional[datetime] = None


@dataclass
class AccessCheckParams:
    """访问检查参数对象"""
    user_id: str
    resource: str
    permission: str
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    check_cache: bool = True
    include_inherited: bool = True
    policy_override: Optional[Dict[str, Any]] = None


@dataclass
class PolicyCreationParams:
    """策略创建参数对象"""
    name: str
    resource_type: str
    resource_pattern: str
    permissions: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    description: str = ""
    is_active: bool = True
    expiry_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryFilterParams:
    """查询过滤参数对象"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    user_ids: Set[str] = field(default_factory=set)
    event_types: Set[EventType] = field(default_factory=set)
    severities: Set[EventSeverity] = field(default_factory=set)
    resources: Set[str] = field(default_factory=set)
    actions: Set[str] = field(default_factory=set)
    results: Set[str] = field(default_factory=set)
    ip_addresses: Set[str] = field(default_factory=set)
    session_ids: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)
    min_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None
    min_severity: Optional[EventSeverity] = None
    max_severity: Optional[EventSeverity] = None
    tags: Set[str] = field(default_factory=set)
    include_tags: Set[str] = field(default_factory=set)
    exclude_tags: Set[str] = field(default_factory=set)
    event_id: Optional[str] = None
    location: Optional[str] = None
    session_id: Optional[str] = None
    tag: Optional[str] = None
    limit: Optional[int] = None
    offset: int = 0
    sort_by: str = "timestamp"
    sort_order: str = "desc"

    def __post_init__(self):
        """将可迭代参数统一转换为集合并校验取值"""
        self.user_ids = self._ensure_set(self.user_ids)
        self.event_types = self._ensure_set(self.event_types)
        self.severities = self._ensure_set(self.severities)
        self.resources = self._ensure_set(self.resources)
        self.actions = self._ensure_set(self.actions)
        self.results = self._ensure_set(self.results)
        self.ip_addresses = self._ensure_set(self.ip_addresses)
        self.session_ids = self._ensure_set(self.session_ids)
        self.locations = self._ensure_set(self.locations)
        self.tags = self._ensure_set(self.tags)
        self.include_tags = self._ensure_set(self.include_tags)
        self.exclude_tags = self._ensure_set(self.exclude_tags)

        if isinstance(self.min_severity, str):
            self.min_severity = EventSeverity(self.min_severity)
        if isinstance(self.max_severity, str):
            self.max_severity = EventSeverity(self.max_severity)

        if self.limit is not None and self.limit < 0:
            self.limit = None

        if self.sort_order not in {"asc", "desc"}:
            self.sort_order = "desc"

        if self.location:
            self.locations.add(self.location)
        if self.session_id:
            self.session_ids.add(self.session_id)
        if self.tag:
            self.tags.add(self.tag)

    @staticmethod
    def _ensure_set(value: Any) -> Set[Any]:
        """确保值为集合类型"""
        if value is None:
            return set()
        if isinstance(value, set):
            return value
        if isinstance(value, (list, tuple)):
            return set(value)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return set(value)
        return {value}


@dataclass
class ConfigOperationParams:
    """配置操作参数对象"""
    operation_type: str = "load"  # 'load', 'save', 'backup', 'restore'
    config_sections: Set[str] = field(default_factory=set)
    include_defaults: bool = True
    validate_before_save: bool = True
    create_backup: bool = True
    backup_suffix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    force_override: bool = False
    operation: Optional[str] = None
    config_type: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if isinstance(self.config_sections, str):
            self.config_sections = {self.config_sections}
        else:
            self.config_sections = set(self.config_sections)

        if self.config_type:
            self.config_sections = {self.config_type}

        if self.operation:
            self.operation_type = self.operation
        if not self.operation_type:
            self.operation_type = "load"



@dataclass
class AuthenticationParams:
    """认证参数对象"""
    username: str
    password: Optional[str] = None
    token: Optional[str] = None
    totp_code: Optional[str] = None
    session_timeout: int = 3600
    remember_me: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None


@dataclass
class ReportGenerationParams:
    """报告生成参数对象"""
    report_type: str = "security"  # 'security', 'compliance', 'access', 'audit'
    format: str = "json"  # 'json', 'csv', 'pdf', 'html'
    include_details: bool = True
    include_charts: bool = False
    time_range: Optional[Dict[str, datetime]] = None
    filters: QueryFilterParams = field(default_factory=QueryFilterParams)
    group_by: Set[str] = field(default_factory=set)
    aggregation: Dict[str, str] = field(default_factory=dict)
    custom_fields: Set[str] = field(default_factory=set)


@dataclass
class EncryptionParams:
    """加密参数对象"""
    algorithm: str = "AES256"
    key_size: int = 256
    mode: str = "GCM"
    key_id: Optional[str] = None
    iv: Optional[bytes] = None
    aad: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    enable_compression: bool = False
    enable_integrity_check: bool = True


@dataclass
class HealthCheckParams:
    """健康检查参数对象"""
    check_types: Set[str] = field(default_factory=set)  # 所有类型
    include_details: bool = True
    include_metrics: bool = True
    timeout: int = 30
    retry_count: int = 3
    threshold_warning: float = 0.8
    threshold_critical: float = 0.9
    context: Dict[str, Any] = field(default_factory=dict)
