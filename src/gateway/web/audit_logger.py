"""
审计日志系统

本模块实现策略执行监控的审计日志功能，满足量化交易系统合规要求：
- QTS-016: 操作日志记录
- QTS-017: 审计追踪

功能特性：
- 详细的操作日志记录
- 审计追踪链
- 日志分级与分类
- 持久化存储
- 查询与导出
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from contextvars import ContextVar
import hashlib
import uuid


class AuditLevel(Enum):
    """审计日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(Enum):
    """审计日志类别"""
    STRATEGY_EXECUTION = "strategy_execution"      # 策略执行
    TRADING = "trading"                            # 交易操作
    RISK_CONTROL = "risk_control"                  # 风险控制
    SIGNAL_PROCESSING = "signal_processing"        # 信号处理
    SYSTEM_OPERATION = "system_operation"          # 系统操作
    USER_ACTION = "user_action"                    # 用户操作
    ALERT_MANAGEMENT = "alert_management"          # 告警管理
    ALERT = "alert"                                # 告警相关
    CONFIGURATION = "configuration"                # 配置变更
    DATA_ACCESS = "data_access"                    # 数据访问


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    log_id: str
    timestamp: datetime
    level: AuditLevel
    category: AuditCategory
    action: str
    user_id: Optional[str]
    strategy_id: Optional[str]
    session_id: Optional[str]
    source_ip: Optional[str]
    request_id: Optional[str]
    message: str
    details: Dict[str, Any]
    result: str  # "success", "failure", "partial"
    error_code: Optional[str]
    error_message: Optional[str]
    duration_ms: Optional[float]
    related_logs: List[str]  # 关联的日志ID列表
    checksum: str  # 数据完整性校验


class AuditContext:
    """审计上下文管理器"""
    
    _context = ContextVar('audit_context', default=None)
    
    def __init__(self):
        self.session_id: Optional[str] = None
        self.request_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.source_ip: Optional[str] = None
        self.strategy_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
    
    @classmethod
    def get_current(cls) -> 'AuditContext':
        """获取当前上下文"""
        context = cls._context.get()
        if context is None:
            context = cls()
            cls._context.set(context)
        return context
    
    @classmethod
    def set_context(cls, **kwargs):
        """设置上下文"""
        context = cls.get_current()
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
        return context
    
    @classmethod
    def clear(cls):
        """清除上下文"""
        cls._context.set(None)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'request_id': self.request_id,
            'user_id': self.user_id,
            'source_ip': self.source_ip,
            'strategy_id': self.strategy_id,
            'start_time': self.start_time.isoformat() if self.start_time else None
        }


class AuditLogger:
    """
    审计日志记录器
    
    单例模式，提供全局审计日志功能
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        retention_days: int = 365,
        batch_size: int = 100,
        enable_console: bool = True
    ):
        if self._initialized:
            return
        
        self._initialized = True
        
        # 配置
        self.db_path = db_path or "data/audit_logs.db"
        self.log_dir = Path(log_dir or "logs/audit")
        self.retention_days = retention_days
        self.batch_size = batch_size
        self.enable_console = enable_console
        
        # 确保目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        # 内存缓冲区
        self._buffer: List[AuditLogEntry] = []
        self._buffer_lock = threading.Lock()
        
        # 回调函数
        self._callbacks: List[Callable[[AuditLogEntry], None]] = []
        
        # 初始化Python日志
        self._logger = logging.getLogger("audit")
        self._logger.setLevel(logging.DEBUG)
        
        if enable_console and not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
            ))
            self._logger.addHandler(handler)
        
        # 文件日志
        file_handler = logging.FileHandler(
            self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self._logger.addHandler(file_handler)
    
    def _init_database(self):
        """初始化审计日志数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    action TEXT NOT NULL,
                    user_id TEXT,
                    strategy_id TEXT,
                    session_id TEXT,
                    source_ip TEXT,
                    request_id TEXT,
                    message TEXT NOT NULL,
                    details TEXT,
                    result TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT,
                    duration_ms REAL,
                    related_logs TEXT,
                    checksum TEXT NOT NULL
                )
            """)
            
            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_logs(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_strategy 
                ON audit_logs(strategy_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_user 
                ON audit_logs(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_category 
                ON audit_logs(category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_request 
                ON audit_logs(request_id)
            """)
            
            conn.commit()
    
    def _generate_checksum(self, entry: AuditLogEntry) -> str:
        """生成日志条目校验和"""
        data = f"{entry.log_id}{entry.timestamp.isoformat()}{entry.action}{entry.message}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _validate_checksum(self, entry: AuditLogEntry) -> bool:
        """验证日志条目完整性"""
        expected = self._generate_checksum(entry)
        return entry.checksum == expected
    
    def log(
        self,
        level: AuditLevel,
        category: AuditCategory,
        action: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        result: str = "success",
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
        related_logs: Optional[List[str]] = None,
        **kwargs
    ) -> AuditLogEntry:
        """
        记录审计日志
        
        Args:
            level: 日志级别
            category: 日志类别
            action: 操作类型
            message: 日志消息
            details: 详细信息
            result: 操作结果
            error_code: 错误代码
            error_message: 错误消息
            duration_ms: 操作耗时
            related_logs: 关联日志ID
            **kwargs: 其他字段
        
        Returns:
            创建的日志条目
        """
        # 获取当前上下文
        context = AuditContext.get_current()
        
        # 计算耗时
        if duration_ms is None and context.start_time:
            duration_ms = (datetime.now() - context.start_time).total_seconds() * 1000
        
        # 创建日志条目
        entry = AuditLogEntry(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            category=category,
            action=action,
            user_id=kwargs.get('user_id', context.user_id),
            strategy_id=kwargs.get('strategy_id', context.strategy_id),
            session_id=kwargs.get('session_id', context.session_id),
            source_ip=kwargs.get('source_ip', context.source_ip),
            request_id=kwargs.get('request_id', context.request_id),
            message=message,
            details=details or {},
            result=result,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
            related_logs=related_logs or [],
            checksum=""  # 临时值，后续计算
        )
        
        # 计算校验和
        entry.checksum = self._generate_checksum(entry)
        
        # 添加到缓冲区
        with self._buffer_lock:
            self._buffer.append(entry)
            
            # 批量写入
            if len(self._buffer) >= self.batch_size:
                self._flush_buffer()
        
        # 输出到控制台日志
        self._output_to_logger(entry)
        
        # 触发回调
        self._trigger_callbacks(entry)
        
        return entry
    
    def _output_to_logger(self, entry: AuditLogEntry):
        """输出到Python日志"""
        log_data = {
            'log_id': entry.log_id,
            'category': entry.category.value,
            'action': entry.action,
            'user_id': entry.user_id,
            'strategy_id': entry.strategy_id,
            'result': entry.result,
            'duration_ms': entry.duration_ms
        }
        
        log_msg = f"[{entry.category.value}] {entry.action}: {entry.message} | {json.dumps(log_data, ensure_ascii=False)}"
        
        if entry.level == AuditLevel.DEBUG:
            self._logger.debug(log_msg)
        elif entry.level == AuditLevel.INFO:
            self._logger.info(log_msg)
        elif entry.level == AuditLevel.WARNING:
            self._logger.warning(log_msg)
        elif entry.level == AuditLevel.ERROR:
            self._logger.error(log_msg)
        elif entry.level == AuditLevel.CRITICAL:
            self._logger.critical(log_msg)
    
    def _flush_buffer(self):
        """将缓冲区写入数据库"""
        if not self._buffer:
            return
        
        entries = self._buffer[:]
        self._buffer.clear()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for entry in entries:
                    conn.execute("""
                        INSERT INTO audit_logs VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        entry.log_id,
                        entry.timestamp.isoformat(),
                        entry.level.value,
                        entry.category.value,
                        entry.action,
                        entry.user_id,
                        entry.strategy_id,
                        entry.session_id,
                        entry.source_ip,
                        entry.request_id,
                        entry.message,
                        json.dumps(entry.details, ensure_ascii=False),
                        entry.result,
                        entry.error_code,
                        entry.error_message,
                        entry.duration_ms,
                        json.dumps(entry.related_logs),
                        entry.checksum
                    ))
                conn.commit()
        except Exception as e:
            self._logger.error(f"Failed to write audit logs: {e}")
            # 重新放回缓冲区
            with self._buffer_lock:
                self._buffer.extend(entries)
    
    def _trigger_callbacks(self, entry: AuditLogEntry):
        """触发回调函数"""
        for callback in self._callbacks:
            try:
                callback(entry)
            except Exception as e:
                self._logger.error(f"Audit callback error: {e}")
    
    def add_callback(self, callback: Callable[[AuditLogEntry], None]):
        """添加回调函数"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[AuditLogEntry], None]):
        """移除回调函数"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def flush(self):
        """强制刷新缓冲区"""
        with self._buffer_lock:
            self._flush_buffer()
    
    def query_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[AuditLevel] = None,
        category: Optional[AuditCategory] = None,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLogEntry]:
        """
        查询审计日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            level: 日志级别
            category: 日志类别
            user_id: 用户ID
            strategy_id: 策略ID
            action: 操作类型
            result: 操作结果
            limit: 返回数量限制
            offset: 偏移量
        
        Returns:
            日志条目列表
        """
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        if level:
            conditions.append("level = ?")
            params.append(level.value)
        if category:
            conditions.append("category = ?")
            params.append(category.value)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if strategy_id:
            conditions.append("strategy_id = ?")
            params.append(strategy_id)
        if action:
            conditions.append("action = ?")
            params.append(action)
        if result:
            conditions.append("result = ?")
            params.append(result)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM audit_logs
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            rows = cursor.fetchall()
            
            entries = []
            for row in rows:
                entry = AuditLogEntry(
                    log_id=row['log_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    level=AuditLevel(row['level']),
                    category=AuditCategory(row['category']),
                    action=row['action'],
                    user_id=row['user_id'],
                    strategy_id=row['strategy_id'],
                    session_id=row['session_id'],
                    source_ip=row['source_ip'],
                    request_id=row['request_id'],
                    message=row['message'],
                    details=json.loads(row['details']) if row['details'] else {},
                    result=row['result'],
                    error_code=row['error_code'],
                    error_message=row['error_message'],
                    duration_ms=row['duration_ms'],
                    related_logs=json.loads(row['related_logs']) if row['related_logs'] else [],
                    checksum=row['checksum']
                )
                entries.append(entry)
            
            return entries
    
    def export_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        导出审计日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            output_path: 输出路径
            format: 导出格式 (json, csv)
        
        Returns:
            导出文件路径
        """
        logs = self.query_logs(start_time=start_time, end_time=end_time, limit=100000)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            data = [asdict(log) for log in logs]
            # 转换datetime为字符串
            for item in data:
                item['timestamp'] = item['timestamp'].isoformat()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        elif format.lower() == "csv":
            import csv
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if logs:
                    writer = csv.DictWriter(f, fieldnames=[
                        'log_id', 'timestamp', 'level', 'category', 'action',
                        'user_id', 'strategy_id', 'message', 'result',
                        'error_code', 'duration_ms', 'checksum'
                    ])
                    writer.writeheader()
                    
                    for log in logs:
                        writer.writerow({
                            'log_id': log.log_id,
                            'timestamp': log.timestamp.isoformat(),
                            'level': log.level.value,
                            'category': log.category.value,
                            'action': log.action,
                            'user_id': log.user_id or '',
                            'strategy_id': log.strategy_id or '',
                            'message': log.message,
                            'result': log.result,
                            'error_code': log.error_code or '',
                            'duration_ms': log.duration_ms or 0,
                            'checksum': log.checksum
                        })
        
        return str(output_file)
    
    def cleanup_old_logs(self, days: Optional[int] = None):
        """
        清理过期日志
        
        Args:
            days: 保留天数，默认使用初始化时的配置
        """
        retention = days or self.retention_days
        cutoff = datetime.now() - timedelta(days=retention)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM audit_logs WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            conn.commit()
        
        self._logger.info(f"Cleaned up audit logs older than {cutoff}")
    
    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        获取审计日志统计
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            统计信息字典
        """
        conditions = ["1=1"]
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        
        where_clause = " AND ".join(conditions)
        
        with sqlite3.connect(self.db_path) as conn:
            # 总日志数
            total = conn.execute(
                f"SELECT COUNT(*) FROM audit_logs WHERE {where_clause}",
                params
            ).fetchone()[0]
            
            # 按级别统计
            level_stats = conn.execute(f"""
                SELECT level, COUNT(*) FROM audit_logs
                WHERE {where_clause}
                GROUP BY level
            """, params).fetchall()
            
            # 按类别统计
            category_stats = conn.execute(f"""
                SELECT category, COUNT(*) FROM audit_logs
                WHERE {where_clause}
                GROUP BY category
            """, params).fetchall()
            
            # 按结果统计
            result_stats = conn.execute(f"""
                SELECT result, COUNT(*) FROM audit_logs
                WHERE {where_clause}
                GROUP BY result
            """, params).fetchall()
            
            # 平均耗时
            avg_duration = conn.execute(f"""
                SELECT AVG(duration_ms) FROM audit_logs
                WHERE {where_clause} AND duration_ms IS NOT NULL
            """, params).fetchone()[0]
            
            return {
                'total_logs': total,
                'by_level': {row[0]: row[1] for row in level_stats},
                'by_category': {row[0]: row[1] for row in category_stats},
                'by_result': {row[0]: row[1] for row in result_stats},
                'avg_duration_ms': avg_duration or 0
            }
    
    # 便捷方法
    def log_strategy_start(
        self,
        strategy_id: str,
        strategy_name: str,
        user_id: str,
        **kwargs
    ) -> AuditLogEntry:
        """记录策略启动"""
        return self.log(
            level=AuditLevel.INFO,
            category=AuditCategory.STRATEGY_EXECUTION,
            action="strategy_start",
            message=f"策略启动: {strategy_name}",
            details={'strategy_name': strategy_name, **kwargs},
            user_id=user_id,
            strategy_id=strategy_id
        )
    
    def log_strategy_stop(
        self,
        strategy_id: str,
        strategy_name: str,
        user_id: str,
        reason: str,
        **kwargs
    ) -> AuditLogEntry:
        """记录策略停止"""
        return self.log(
            level=AuditLevel.INFO,
            category=AuditCategory.STRATEGY_EXECUTION,
            action="strategy_stop",
            message=f"策略停止: {strategy_name}, 原因: {reason}",
            details={'strategy_name': strategy_name, 'reason': reason, **kwargs},
            user_id=user_id,
            strategy_id=strategy_id
        )
    
    def log_signal_generated(
        self,
        strategy_id: str,
        signal_id: str,
        symbol: str,
        direction: str,
        **kwargs
    ) -> AuditLogEntry:
        """记录信号生成"""
        return self.log(
            level=AuditLevel.INFO,
            category=AuditCategory.SIGNAL_PROCESSING,
            action="signal_generated",
            message=f"信号生成: {symbol} {direction}",
            details={
                'signal_id': signal_id,
                'symbol': symbol,
                'direction': direction,
                **kwargs
            },
            strategy_id=strategy_id
        )
    
    def log_risk_event(
        self,
        strategy_id: str,
        event_type: str,
        severity: str,
        description: str,
        **kwargs
    ) -> AuditLogEntry:
        """记录风险事件"""
        level = AuditLevel.WARNING if severity == "medium" else AuditLevel.ERROR
        
        return self.log(
            level=level,
            category=AuditCategory.RISK_CONTROL,
            action=f"risk_{event_type}",
            message=description,
            details={'severity': severity, **kwargs},
            strategy_id=strategy_id
        )
    
    def log_alert_action(
        self,
        alert_id: str,
        action: str,
        user_id: str,
        **kwargs
    ) -> AuditLogEntry:
        """记录告警操作"""
        return self.log(
            level=AuditLevel.INFO,
            category=AuditCategory.ALERT_MANAGEMENT,
            action=f"alert_{action}",
            message=f"告警操作: {action}",
            details={'alert_id': alert_id, **kwargs},
            user_id=user_id
        )
    
    def log_config_change(
        self,
        config_name: str,
        old_value: Any,
        new_value: Any,
        user_id: str,
        **kwargs
    ) -> AuditLogEntry:
        """记录配置变更"""
        return self.log(
            level=AuditLevel.INFO,
            category=AuditCategory.CONFIGURATION,
            action="config_change",
            message=f"配置变更: {config_name}",
            details={
                'config_name': config_name,
                'old_value': str(old_value),
                'new_value': str(new_value),
                **kwargs
            },
            user_id=user_id
        )
    
    def log_data_access(
        self,
        data_type: str,
        operation: str,
        user_id: str,
        records_count: int,
        **kwargs
    ) -> AuditLogEntry:
        """记录数据访问"""
        return self.log(
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_ACCESS,
            action=f"data_{operation}",
            message=f"数据访问: {data_type}, 记录数: {records_count}",
            details={
                'data_type': data_type,
                'operation': operation,
                'records_count': records_count,
                **kwargs
            },
            user_id=user_id
        )


# 全局审计日志记录器实例
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """获取全局审计日志记录器实例"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def set_audit_logger(logger: AuditLogger):
    """设置全局审计日志记录器实例"""
    global _audit_logger
    _audit_logger = logger


# 上下文管理器装饰器
def with_audit_context(**context_kwargs):
    """审计上下文装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = AuditContext.get_current()
            AuditContext.set_context(**context_kwargs)
            context.start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                AuditContext.clear()
        return wrapper
    return decorator
