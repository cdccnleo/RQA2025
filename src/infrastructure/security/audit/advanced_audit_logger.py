"""
完整审计日志系统模块

功能：
- 全面的操作日志记录
- 用户行为追踪
- 数据变更历史
- 合规性报告生成
- 日志分析与告警
- 不可篡改日志存储

技术栈：
- hashlib: 日志完整性校验
- json: 结构化日志
- sqlite: 本地日志存储
- asyncio: 异步日志写入

作者: Claude
创建日期: 2026-02-21
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from collections import deque
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """审计事件类型"""
    # 认证事件
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    MFA_VERIFICATION = "mfa_verification"
    
    # 数据访问事件
    DATA_READ = "data_read"
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # 系统管理事件
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    ROLE_ASSIGN = "role_assign"
    PERMISSION_CHANGE = "permission_change"
    
    # 策略事件
    STRATEGY_CREATE = "strategy_create"
    STRATEGY_UPDATE = "strategy_update"
    STRATEGY_DELETE = "strategy_delete"
    STRATEGY_EXECUTE = "strategy_execute"
    
    # 交易事件
    TRADE_EXECUTE = "trade_execute"
    TRADE_CANCEL = "trade_cancel"
    POSITION_UPDATE = "position_update"
    
    # 安全事件
    SECURITY_ALERT = "security_alert"
    ACCESS_DENIED = "access_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # 系统事件
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    BACKUP_CREATE = "backup_create"
    BACKUP_RESTORE = "backup_restore"


class AuditSeverity(Enum):
    """审计事件严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    status: str
    details: Dict[str, Any]
    severity: AuditSeverity
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    previous_hash: Optional[str] = None
    current_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'username': self.username,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'action': self.action,
            'status': self.status,
            'details': self.details,
            'severity': self.severity.value,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'previous_hash': self.previous_hash,
            'current_hash': self.current_hash
        }
    
    def calculate_hash(self) -> str:
        """计算事件哈希"""
        data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'action': self.action,
            'details': json.dumps(self.details, sort_keys=True)
        }
        if self.previous_hash:
            data['previous_hash'] = self.previous_hash
        
        hash_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()


@dataclass
class AuditQuery:
    """审计查询条件"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    user_id: Optional[str] = None
    event_types: Optional[List[AuditEventType]] = None
    severity: Optional[AuditSeverity] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    limit: int = 100
    offset: int = 0


class AuditStorage:
    """
    审计日志存储
    
    支持SQLite本地存储和链式哈希保证完整性
    """
    
    def __init__(self, db_path: str = "audit_logs.db"):
        """
        初始化存储
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
        self._last_hash: Optional[str] = None
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_db(self) -> None:
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                username TEXT,
                ip_address TEXT,
                user_agent TEXT,
                resource_type TEXT,
                resource_id TEXT,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT,
                severity TEXT NOT NULL,
                session_id TEXT,
                request_id TEXT,
                previous_hash TEXT,
                current_hash TEXT NOT NULL
            )
        ''')
        
        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_id ON audit_logs(user_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_event_type ON audit_logs(event_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_resource ON audit_logs(resource_type, resource_id)
        ''')
        
        conn.commit()
        
        # 加载最后一个哈希
        self._load_last_hash()
    
    def _load_last_hash(self) -> None:
        """加载最后一个事件的哈希"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT current_hash FROM audit_logs 
            ORDER BY timestamp DESC LIMIT 1
        ''')
        
        row = cursor.fetchone()
        if row:
            self._last_hash = row['current_hash']
    
    def store_event(self, event: AuditEvent) -> bool:
        """
        存储事件
        
        Args:
            event: 审计事件
            
        Returns:
            是否成功
        """
        try:
            # 设置链式哈希
            event.previous_hash = self._last_hash
            event.current_hash = event.calculate_hash()
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_logs (
                    event_id, event_type, timestamp, user_id, username,
                    ip_address, user_agent, resource_type, resource_id,
                    action, status, details, severity, session_id,
                    request_id, previous_hash, current_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.user_id,
                event.username,
                event.ip_address,
                event.user_agent,
                event.resource_type,
                event.resource_id,
                event.action,
                event.status,
                json.dumps(event.details),
                event.severity.value,
                event.session_id,
                event.request_id,
                event.previous_hash,
                event.current_hash
            ))
            
            conn.commit()
            self._last_hash = event.current_hash
            
            return True
        except Exception as e:
            logger.error(f"存储审计事件失败: {e}")
            return False
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """
        查询事件
        
        Args:
            query: 查询条件
            
        Returns:
            事件列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if query.start_time:
            conditions.append("timestamp >= ?")
            params.append(query.start_time.isoformat())
        
        if query.end_time:
            conditions.append("timestamp <= ?")
            params.append(query.end_time.isoformat())
        
        if query.user_id:
            conditions.append("user_id = ?")
            params.append(query.user_id)
        
        if query.event_types:
            event_types = [et.value for et in query.event_types]
            placeholders = ','.join('?' * len(event_types))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(event_types)
        
        if query.severity:
            conditions.append("severity = ?")
            params.append(query.severity.value)
        
        if query.resource_type:
            conditions.append("resource_type = ?")
            params.append(query.resource_type)
        
        if query.resource_id:
            conditions.append("resource_id = ?")
            params.append(query.resource_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor.execute(f'''
            SELECT * FROM audit_logs
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        ''', params + [query.limit, query.offset])
        
        rows = cursor.fetchall()
        events = []
        
        for row in rows:
            event = AuditEvent(
                event_id=row['event_id'],
                event_type=AuditEventType(row['event_type']),
                timestamp=datetime.fromisoformat(row['timestamp']),
                user_id=row['user_id'],
                username=row['username'],
                ip_address=row['ip_address'],
                user_agent=row['user_agent'],
                resource_type=row['resource_type'],
                resource_id=row['resource_id'],
                action=row['action'],
                status=row['status'],
                details=json.loads(row['details']) if row['details'] else {},
                severity=AuditSeverity(row['severity']),
                session_id=row['session_id'],
                request_id=row['request_id'],
                previous_hash=row['previous_hash'],
                current_hash=row['current_hash']
            )
            events.append(event)
        
        return events
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        验证日志完整性
        
        Returns:
            验证结果
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM audit_logs ORDER BY timestamp
        ''')
        
        rows = cursor.fetchall()
        errors = []
        previous_hash = None
        
        for i, row in enumerate(rows):
            # 验证链式哈希
            if row['previous_hash'] != previous_hash:
                errors.append({
                    'event_id': row['event_id'],
                    'index': i,
                    'error': '链式哈希不匹配',
                    'expected_previous': previous_hash,
                    'actual_previous': row['previous_hash']
                })
            
            # 验证当前哈希
            event_data = {
                'event_id': row['event_id'],
                'event_type': row['event_type'],
                'timestamp': row['timestamp'],
                'user_id': row['user_id'],
                'action': row['action'],
                'details': row['details'],
                'previous_hash': row['previous_hash']
            }
            
            expected_hash = hashlib.sha256(
                json.dumps(event_data, sort_keys=True).encode()
            ).hexdigest()
            
            if row['current_hash'] != expected_hash:
                errors.append({
                    'event_id': row['event_id'],
                    'index': i,
                    'error': '事件哈希不匹配',
                    'expected_hash': expected_hash,
                    'actual_hash': row['current_hash']
                })
            
            previous_hash = row['current_hash']
        
        return {
            'total_events': len(rows),
            'errors': errors,
            'is_valid': len(errors) == 0
        }


class AuditLogger:
    """
    审计日志主类
    
    提供完整的审计日志功能
    """
    
    def __init__(self, storage: Optional[AuditStorage] = None,
                 async_mode: bool = True):
        """
        初始化审计日志
        
        Args:
            storage: 存储实例
            async_mode: 是否异步模式
        """
        self.storage = storage or AuditStorage()
        self.async_mode = async_mode
        self.event_queue: deque = deque(maxlen=10000)
        self.alert_callbacks: List[Callable] = []
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        if async_mode:
            self._start_worker()
    
    def _start_worker(self) -> None:
        """启动异步工作线程"""
        self._running = True
        try:
            loop = asyncio.get_event_loop()
            self._worker_task = loop.create_task(self._process_queue())
        except RuntimeError:
            # 没有事件循环，创建新线程
            import threading
            self._worker_thread = threading.Thread(target=self._sync_process_queue)
            self._worker_thread.daemon = True
            self._worker_thread.start()
    
    async def _process_queue(self) -> None:
        """异步处理队列"""
        while self._running:
            while self.event_queue:
                event = self.event_queue.popleft()
                self.storage.store_event(event)
                
                # 检查是否需要告警
                if event.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
                    self._trigger_alert(event)
            
            await asyncio.sleep(0.1)
    
    def _sync_process_queue(self) -> None:
        """同步处理队列"""
        while self._running:
            while self.event_queue:
                event = self.event_queue.popleft()
                self.storage.store_event(event)
                
                if event.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
                    self._trigger_alert(event)
            
            time.sleep(0.1)
    
    def log(self, event_type: AuditEventType,
            action: str,
            user_id: Optional[str] = None,
            username: Optional[str] = None,
            status: str = "success",
            details: Optional[Dict[str, Any]] = None,
            severity: AuditSeverity = AuditSeverity.INFO,
            **kwargs) -> AuditEvent:
        """
        记录审计事件
        
        Args:
            event_type: 事件类型
            action: 操作
            user_id: 用户ID
            username: 用户名
            status: 状态
            details: 详细信息
            severity: 严重程度
            **kwargs: 其他参数
            
        Returns:
            审计事件
        """
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            username=username,
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent'),
            resource_type=kwargs.get('resource_type'),
            resource_id=kwargs.get('resource_id'),
            action=action,
            status=status,
            details=details or {},
            severity=severity,
            session_id=kwargs.get('session_id'),
            request_id=kwargs.get('request_id')
        )
        
        if self.async_mode:
            self.event_queue.append(event)
        else:
            self.storage.store_event(event)
            
            if severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
                self._trigger_alert(event)
        
        return event
    
    def _trigger_alert(self, event: AuditEvent) -> None:
        """触发告警"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
    
    def register_alert_callback(self, callback: Callable) -> None:
        """注册告警回调"""
        self.alert_callbacks.append(callback)
    
    def query(self, query: AuditQuery) -> List[AuditEvent]:
        """查询日志"""
        return self.storage.query_events(query)
    
    def get_user_activity(self, user_id: str,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[AuditEvent]:
        """
        获取用户活动日志
        
        Args:
            user_id: 用户ID
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量
            
        Returns:
            事件列表
        """
        query = AuditQuery(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        return self.query(query)
    
    def get_resource_history(self, resource_type: str, resource_id: str,
                            limit: int = 100) -> List[AuditEvent]:
        """
        获取资源变更历史
        
        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            limit: 限制数量
            
        Returns:
            事件列表
        """
        query = AuditQuery(
            resource_type=resource_type,
            resource_id=resource_id,
            limit=limit
        )
        return self.query(query)
    
    def generate_compliance_report(self, start_time: datetime,
                                  end_time: datetime) -> Dict[str, Any]:
        """
        生成合规性报告
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            合规性报告
        """
        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        events = self.query(query)
        
        # 统计
        event_type_counts = {}
        severity_counts = {}
        user_activity = {}
        
        for event in events:
            # 事件类型统计
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # 严重程度统计
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # 用户活动统计
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
        
        # 安全事件
        security_events = [
            e for e in events
            if e.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]
        ]
        
        return {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'event_type_distribution': event_type_counts,
                'severity_distribution': severity_counts,
                'active_users': len(user_activity),
                'security_incidents': len(security_events)
            },
            'security_events': [
                {
                    'event_id': e.event_id,
                    'timestamp': e.timestamp.isoformat(),
                    'type': e.event_type.value,
                    'severity': e.severity.value,
                    'user_id': e.user_id,
                    'action': e.action,
                    'details': e.details
                }
                for e in security_events[:50]
            ],
            'integrity_check': self.storage.verify_integrity()
        }
    
    def verify_integrity(self) -> Dict[str, Any]:
        """验证日志完整性"""
        return self.storage.verify_integrity()
    
    def export_logs(self, filepath: str, query: AuditQuery) -> bool:
        """
        导出日志
        
        Args:
            filepath: 文件路径
            query: 查询条件
            
        Returns:
            是否成功
        """
        try:
            events = self.query(query)
            data = {
                'export_time': datetime.now().isoformat(),
                'total_events': len(events),
                'events': [e.to_dict() for e in events]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"导出日志失败: {e}")
            return False
    
    def stop(self) -> None:
        """停止审计日志服务"""
        self._running = False
        
        # 处理剩余事件
        while self.event_queue:
            event = self.event_queue.popleft()
            self.storage.store_event(event)


# 便捷函数
def create_audit_logger(storage_path: Optional[str] = None,
                       async_mode: bool = True) -> AuditLogger:
    """
    创建审计日志实例
    
    Args:
        storage_path: 存储路径
        async_mode: 是否异步模式
        
    Returns:
        AuditLogger实例
    """
    storage = AuditStorage(storage_path) if storage_path else None
    return AuditLogger(storage, async_mode)


# 单例实例
_audit_logger_instance: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    获取审计日志单例
    
    Returns:
        AuditLogger实例
    """
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger()
    return _audit_logger_instance


# 装饰器
def audit_log(event_type: AuditEventType, action: str,
              resource_type: Optional[str] = None,
              get_resource_id: Optional[Callable] = None):
    """
    审计日志装饰器
    
    Args:
        event_type: 事件类型
        action: 操作
        resource_type: 资源类型
        get_resource_id: 获取资源ID的函数
    """
import functools
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_audit_logger()
            
            # 获取用户信息（假设从上下文获取）
            user_id = kwargs.get('user_id') or 'anonymous'
            username = kwargs.get('username') or 'Anonymous'
            
            # 获取资源ID
            resource_id = None
            if get_resource_id:
                try:
                    resource_id = get_resource_id(*args, **kwargs)
                except Exception:
                    pass
            
            try:
                result = func(*args, **kwargs)
                
                # 记录成功
                logger.log(
                    event_type=event_type,
                    action=action,
                    user_id=user_id,
                    username=username,
                    status="success",
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details={'result': 'success'}
                )
                
                return result
            except Exception as e:
                # 记录失败
                logger.log(
                    event_type=event_type,
                    action=action,
                    user_id=user_id,
                    username=username,
                    status="failed",
                    resource_type=resource_type,
                    resource_id=resource_id,
                    severity=AuditSeverity.ERROR,
                    details={'error': str(e)}
                )
                raise
        return wrapper
    return decorator
