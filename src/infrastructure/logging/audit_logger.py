"""
audit_logger 模块

提供 audit_logger 相关功能和接口。
"""

import logging

import uuid

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Dict, Any, Optional, List
"""
基础设施层 - 数据库审计日志器

提供专门的数据库操作审计功能。
通用AuditLogger已迁移到core模块，请使用core.AuditLogger。
"""


class OperationType(Enum):
    """操作类型枚举"""
    QUERY = "query"
    WRITE = "write"
    TRANSACTION = "transaction"
    CONNECTION = "connection"
    CONFIGURATION = "configuration"
    CREATE = WRITE
    READ = QUERY


class SecurityLevel(Enum):
    """安全级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditRecord:
    """审计记录"""
    id: str
    timestamp: datetime
    operation_type: OperationType
    user_id: Optional[str]
    session_id: Optional[str]
    sql: str
    params: Optional[Dict[str, Any]]
    result: Dict[str, Any]
    execution_time: float
    security_level: SecurityLevel
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class DatabaseAuditLogger:
    """
    数据库审计日志器

    专门用于记录数据库操作的审计日志，包括查询、写入、事务等操作。
    提供完整的安全审计追踪功能。
    """

    def __init__(self,
                 log_file: str = "logs/database_audit.log",
                 max_records: int = 10000,
                 enable_encryption: bool = False,
                 retention_days: int = 365):
        """
        初始化数据库审计日志器

        Args:
            log_file: 日志文件路径
            max_records: 最大记录数
            enable_encryption: 是否启用加密
            retention_days: 保留天数
        """
        self._log_file = log_file
        self._max_records = max_records
        self._enable_encryption = enable_encryption
        self._retention_days = retention_days

        self._records: List[AuditRecord] = []
        self._lock = Lock()
        self._logger = logging.getLogger('database_audit_logger')

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志处理器"""
        handler = logging.FileHandler(self._log_file)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def _log_operation(self, operation_type: OperationType, sql: str,
                       params: Optional[Dict[str, Any]], result: Dict[str, Any],
                       execution_time: float, user_id: Optional[str] = None,
                       session_id: Optional[str] = None, ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        记录操作到审计日志

        Returns:
            审计记录ID
        """
        record_id = str(uuid.uuid4())

        # 确定安全级别
        security_level = self._determine_security_level(sql, params)

        record = AuditRecord(
            id=record_id,
            timestamp=datetime.now(),
            operation_type=operation_type,
            user_id=user_id,
            session_id=session_id,
            sql=sql,
            params=params,
            result=result,
            execution_time=execution_time,
            security_level=security_level,
            ip_address=ip_address,
            user_agent=user_agent,
            additional_info=additional_info
        )

        with self._lock:
            self._records.append(record)

            # 限制记录数量
            if len(self._records) > self._max_records:
                self._records.pop(0)

            # 记录到日志文件
            self._logger.info(
                f"数据库操作审计 | ID: {record_id} | 类型: {operation_type.value} | "
                f"SQL: {sql[:100]}... | 执行时间: {execution_time:.3f}s | "
                f"安全级别: {security_level.value}"
            )

        return record_id

    def _determine_security_level(self, sql: str, params: Optional[Dict[str, Any]]) -> SecurityLevel:
        """确定操作的安全级别"""
        sql_lower = sql.lower().strip()

        # 高风险操作
        if any(keyword in sql_lower for keyword in ['drop', 'delete', 'truncate', 'alter']):
            return SecurityLevel.CRITICAL

        # 中风险操作
        if any(keyword in sql_lower for keyword in ['update', 'insert', 'create']):
            return SecurityLevel.HIGH

        # 包含敏感信息
        if params and any(key in str(params).lower() for key in ['password', 'key', 'token']):
            return SecurityLevel.HIGH

        # 默认低风险
        return SecurityLevel.LOW

    def log_database_operation(self, operation_type: str = "", sql: str = "",
                               params: Optional[Dict[str, Any]] = None,
                               result: Optional[Dict[str, Any]] = None,
                               execution_time: float = 0.0,
                               user_id: Optional[str] = None,
                               session_id: Optional[str] = None,
                               ip_address: Optional[str] = None,
                               user_agent: Optional[str] = None,
                               operation: Optional[str] = None,
                               **extra: Any) -> str:
        """
        记录数据库操作

        Args:
            operation_type: 操作类型
            sql: SQL语句
            params: 参数
            result: 结果
            execution_time: 执行时间
            user_id: 用户ID
            session_id: 会话ID
            ip_address: IP地址
            user_agent: 用户代理

        Returns:
            审计记录ID
        """
        selected_operation = operation or operation_type
        if not selected_operation:
            selected_operation = "query"
        normalized_operation = str(selected_operation).lower()
        alias_map = {
            "create": "write",
            "read": "query",
        }
        normalized_operation = alias_map.get(normalized_operation, normalized_operation)
        try:
            op_type = OperationType(normalized_operation)
        except ValueError:
            op_type = OperationType.CONFIGURATION

        audit_result = result or {"status": "ok"}
        additional_info = extra if extra else None

        return self._log_operation(
            op_type,
            sql,
            params or {},
            audit_result,
            execution_time,
            user_id,
            session_id,
            ip_address,
            user_agent,
            additional_info=additional_info,
        )

    def get_audit_records(self, user_id: Optional[str] = None,
                          operation_type: Optional[OperationType] = None,
                          limit: int = 100) -> List[AuditRecord]:
        """
        获取审计记录

        Args:
            user_id: 用户ID过滤
            operation_type: 操作类型过滤
            limit: 返回记录数量限制

        Returns:
            审计记录列表
        """
        with self._lock:
            records = self._records.copy()

        # 应用过滤
        if user_id:
            records = [r for r in records if r.user_id == user_id]

        if operation_type:
            records = [r for r in records if r.operation_type == operation_type]

        # 按时间倒序，返回最新的记录
        return sorted(records, key=lambda r: r.timestamp, reverse=True)[:limit]

    def clear_old_records(self, days: Optional[int] = None):
        """清理过期记录"""
        retention = days if days is not None else self._retention_days
        cutoff_date = datetime.now() - timedelta(days=retention)

        with self._lock:
            original_count = len(self._records)
            self._records = [r for r in self._records if r.timestamp > cutoff_date]

            removed_count = original_count - len(self._records)
            if removed_count > 0:
                self._logger.info(f"清理了 {removed_count} 条过期审计记录")

    def get_statistics(self) -> Dict[str, Any]:
        """获取审计统计信息"""
        with self._lock:
            records = self._records.copy()

        if not records:
            return {"total_records": 0}

        # 按操作类型统计
        operation_stats = {}
        for record in records:
            op_type = record.operation_type.value
            operation_stats[op_type] = operation_stats.get(op_type, 0) + 1

        # 按安全级别统计
        security_stats = {}
        for record in records:
            sec_level = record.security_level.value
            security_stats[sec_level] = security_stats.get(sec_level, 0) + 1

        # 时间范围
        timestamps = [r.timestamp for r in records]
        time_range = {
            "earliest": min(timestamps).isoformat(),
            "latest": max(timestamps).isoformat()
        }

        return {
            "total_records": len(records),
            "operation_types": operation_stats,
            "security_levels": security_stats,
            "time_range": time_range,
            "average_execution_time": sum(r.execution_time for r in records) / len(records)
        }

    def get_records_by_user(self, user_id: str) -> List[AuditRecord]:
        """按用户ID获取审计记录"""
        with self._lock:
            return [record for record in self._records if record.user_id == user_id]

    def get_records_by_operation_type(self, operation_type: OperationType) -> List[AuditRecord]:
        """按操作类型获取审计记录"""
        with self._lock:
            return [record for record in self._records if record.operation_type == operation_type]

    def get_records_in_time_range(self, start_time: datetime, end_time: datetime) -> List[AuditRecord]:
        """获取指定时间范围内的审计记录"""
        with self._lock:
            return [record for record in self._records
                    if start_time <= record.timestamp <= end_time]

    def get_all_records(self) -> List[AuditRecord]:
        """获取所有审计记录"""
        with self._lock:
            return self._records.copy()

    def clear_records(self) -> bool:
        """清空所有审计记录"""
        with self._lock:
            self._records.clear()
            return True

    def export_records(self) -> List[Dict[str, Any]]:
        """导出所有审计记录为字典列表"""
        with self._lock:
            return [{
                'id': record.id,
                'timestamp': record.timestamp.isoformat(),
                'operation_type': record.operation_type.value,
                'user_id': record.user_id,
                'session_id': record.session_id,
                'sql': record.sql,
                'params': record.params,
                'result': record.result,
                'execution_time': record.execution_time,
                'security_level': record.security_level.value,
                'ip_address': record.ip_address,
                'user_agent': record.user_agent,
                'additional_info': record.additional_info
            } for record in self._records]

    def import_records(self, records_data: List[Dict[str, Any]]) -> bool:
        """从字典列表导入审计记录"""
        try:
            with self._lock:
                for record_data in records_data:
                    # 手动创建AuditRecord对象
                    record = AuditRecord(
                        id=record_data['id'],
                        timestamp=datetime.fromisoformat(record_data['timestamp']),
                        operation_type=OperationType(record_data['operation_type']),
                        user_id=record_data['user_id'],
                        session_id=record_data['session_id'],
                        sql=record_data['sql'],
                        params=record_data['params'],
                        result=record_data['result'],
                        execution_time=record_data['execution_time'],
                        security_level=SecurityLevel(record_data['security_level']),
                        ip_address=record_data.get('ip_address'),
                        user_agent=record_data.get('user_agent'),
                        additional_info=record_data.get('additional_info')
                    )
                    self._records.append(record)
                return True
        except Exception as e:
            self._logger.error(f"导入审计记录失败: {e}")
            return False

# 注意：通用AuditLogger已迁移到core模块
# from src.infrastructure.logging.core import AuditLogger
