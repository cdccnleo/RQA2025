#!/usr/bin/env python3
"""
RQA2025 安全审计系统

实现交易操作审计日志和安全事件记录
"""

import logging
import hashlib
import hmac
import json
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
import os
from pathlib import Path

# 导入统一基础设施集成层
try:
    from src.integration import get_trading_layer_adapter
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = False


logger = logging.getLogger(__name__)


class AuditEventType(Enum):

    """审计事件类型枚举"""
    LOGIN = "login"                          # 用户登录
    LOGOUT = "logout"                       # 用户登出
    TRADE_EXECUTE = "trade_execute"         # 交易执行
    ORDER_PLACE = "order_place"             # 下单
    ORDER_CANCEL = "order_cancel"           # 撤单
    RISK_VIOLATION = "risk_violation"       # 风控违规
    CONFIG_CHANGE = "config_change"         # 配置变更
    SYSTEM_START = "system_start"           # 系统启动
    SYSTEM_STOP = "system_stop"            # 系统停止
    SECURITY_VIOLATION = "security_violation"  # 安全违规
    DATA_ACCESS = "data_access"             # 数据访问


class SecurityLevel(Enum):

    """安全级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:

    """审计事件"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: str
    action: str
    result: str  # "success" 或 "failure"
    details: Dict[str, Any]
    security_level: SecurityLevel
    risk_score: float
    hash_value: str  # 事件哈希，用于完整性验证


@dataclass
class SecurityEvent:

    """安全事件"""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: SecurityLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    description: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


class AuditLogger:

    """审计日志记录器"""

    def __init__(self, log_directory: str = "logs/audit"):

        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # 当前日志文件
        self.current_log_file = None
        self.current_date = None

        # 缓冲区
        self.buffer: List[AuditEvent] = []
        self.buffer_size = 100
        self.flush_interval = 30  # 30秒刷新一次

        # 加密密钥（用于事件完整性验证）
        self.hmac_key = os.getenv('AUDIT_HMAC_KEY', 'default_audit_key').encode()

        # 文件锁
        self.file_lock = threading.RLock()
        self.buffer_lock = threading.RLock()

        # 定期刷新线程
        self.flush_thread = threading.Thread(
            target=self._periodic_flush,
            name="audit_flush",
            daemon=True
        )
        self.flush_thread.start()

        logger.info(f"审计日志记录器初始化完成，日志目录: {log_directory}")

    def log_event(self, event_type: AuditEventType, user_id: Optional[str] = None,


                  session_id: Optional[str] = None, ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None, resource: str = "",
                  action: str = "", result: str = "success",
                  details: Dict[str, Any] = None, security_level: SecurityLevel = SecurityLevel.LOW):
        """记录审计事件"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            security_level=security_level,
            risk_score=self._calculate_risk_score(event_type, security_level, result),
            hash_value=""
        )

        # 计算事件哈希
        event.hash_value = self._calculate_event_hash(event)

        # 添加到缓冲区
        with self.buffer_lock:
            self.buffer.append(event)

            # 如果缓冲区满，立即刷新
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()

    def _calculate_risk_score(self, event_type: AuditEventType,


                              security_level: SecurityLevel, result: str) -> float:
        """计算风险评分"""
        base_score = {
            SecurityLevel.LOW: 1.0,
            SecurityLevel.MEDIUM: 5.0,
            SecurityLevel.HIGH: 10.0,
            SecurityLevel.CRITICAL: 20.0
        }[security_level]

        # 基于事件类型的调整
        type_multiplier = {
            AuditEventType.LOGIN: 1.0,
            AuditEventType.LOGOUT: 0.5,
            AuditEventType.TRADE_EXECUTE: 3.0,
            AuditEventType.ORDER_PLACE: 2.0,
            AuditEventType.ORDER_CANCEL: 1.0,
            AuditEventType.RISK_VIOLATION: 8.0,
            AuditEventType.CONFIG_CHANGE: 7.0,
            AuditEventType.SECURITY_VIOLATION: 15.0,
            AuditEventType.DATA_ACCESS: 4.0,
            AuditEventType.SYSTEM_START: 0.1,
            AuditEventType.SYSTEM_STOP: 0.1
        }.get(event_type, 1.0)

        # 失败事件风险更高
        failure_multiplier = 2.0 if result == "failure" else 1.0

        return base_score * type_multiplier * failure_multiplier

    def _calculate_event_hash(self, event: AuditEvent) -> str:
        """计算事件哈希值"""
        # 创建事件数据的规范化表示
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "resource": event.resource,
            "action": event.action,
            "result": event.result,
            "security_level": event.security_level.value,
            "details": json.dumps(event.details, sort_keys=True)
        }

        # 序列化并计算HMAC
        message = json.dumps(event_data, sort_keys=True).encode()
        hash_obj = hmac.new(self.hmac_key, message, hashlib.sha256)
        return hash_obj.hexdigest()

    def _flush_buffer(self):
        """刷新缓冲区到文件"""
        with self.buffer_lock:
            if not self.buffer:
                return

            events_to_write = self.buffer.copy()
            self.buffer.clear()

        # 确保使用正确的日志文件
        self._ensure_log_file()

        with self.file_lock:
            try:
                with open(self.current_log_file, 'a', encoding='utf - 8') as f:
                    for event in events_to_write:
                        # 序列化事件为JSON
                        event_dict = {
                            "event_id": event.event_id,
                            "event_type": event.event_type.value,
                            "timestamp": event.timestamp.isoformat(),
                            "user_id": event.user_id,
                            "session_id": event.session_id,
                            "ip_address": event.ip_address,
                            "user_agent": event.user_agent,
                            "resource": event.resource,
                            "action": event.action,
                            "result": event.result,
                            "details": event.details,
                            "security_level": event.security_level.value,
                            "risk_score": event.risk_score,
                            "hash_value": event.hash_value
                        }

                        f.write(json.dumps(event_dict, ensure_ascii=False) + '\n')

                logger.debug(f"已写入 {len(events_to_write)} 条审计事件到文件")

            except Exception as e:
                logger.error(f"写入审计日志失败: {e}")
                # 将事件重新放回缓冲区
                with self.buffer_lock:
                    self.buffer.extend(events_to_write)

    def _ensure_log_file(self):
        """确保使用正确的日志文件"""
        today = datetime.now().date()

        if self.current_date != today:
            self.current_date = today
            self.current_log_file = self.log_directory / f"audit_{today.isoformat()}.log"

            # 如果文件不存在，写入文件头
            if not self.current_log_file.exists():
                try:
                    with open(self.current_log_file, 'w', encoding='utf - 8') as f:
                        header = {
                            "log_type": "audit",
                            "version": "1.0",
                            "created_at": datetime.now().isoformat(),
                            "hmac_key_hint": hashlib.sha256(self.hmac_key).hexdigest()[:16]
                        }
                        f.write("# " + json.dumps(header, ensure_ascii=False) + '\n')
                except Exception as e:
                    logger.error(f"创建审计日志文件失败: {e}")

    def _periodic_flush(self):
        """定期刷新缓冲区"""
        while True:
            try:
                self._flush_buffer()
                time.sleep(self.flush_interval)
            except Exception as e:
                logger.error(f"定期刷新审计日志失败: {e}")
                time.sleep(10)  # 出错后等待10秒重试

    def verify_event_integrity(self, event_dict: Dict[str, Any]) -> bool:
        """验证事件完整性"""
        try:
            # 重新计算哈希
            temp_event = AuditEvent(
                event_id=event_dict["event_id"],
                event_type=AuditEventType(event_dict["event_type"]),
                timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                user_id=event_dict.get("user_id"),
                session_id=event_dict.get("session_id"),
                ip_address=event_dict.get("ip_address"),
                user_agent=event_dict.get("user_agent"),
                resource=event_dict["resource"],
                action=event_dict["action"],
                result=event_dict["result"],
                details=event_dict["details"],
                security_level=SecurityLevel(event_dict["security_level"]),
                risk_score=event_dict["risk_score"],
                hash_value=""  # 重新计算
            )

            calculated_hash = self._calculate_event_hash(temp_event)
            return calculated_hash == event_dict["hash_value"]

        except Exception as e:
            logger.error(f"验证事件完整性失败: {e}")
            return False

    def get_events_by_user(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """获取指定用户的审计事件"""
        start_date = datetime.now() - timedelta(days=days)
        events = []

        # 遍历日志文件
        for log_file in self.log_directory.glob("audit_*.log"):
            try:
                with open(log_file, 'r', encoding='utf - 8') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue

                        event_dict = json.loads(line.strip())
                        if (event_dict.get("user_id") == user_id
                                and datetime.fromisoformat(event_dict["timestamp"]) >= start_date):
                            events.append(event_dict)

            except Exception as e:
                logger.error(f"读取日志文件失败 {log_file}: {e}")

        return events

    def get_events_by_type(self, event_type: AuditEventType, days: int = 7) -> List[Dict[str, Any]]:
        """获取指定类型的审计事件"""
        start_date = datetime.now() - timedelta(days=days)
        events = []

        # 遍历日志文件
        for log_file in self.log_directory.glob("audit_*.log"):
            try:
                with open(log_file, 'r', encoding='utf - 8') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue

                        event_dict = json.loads(line.strip())
                        if (event_dict.get("event_type") == event_type.value
                                and datetime.fromisoformat(event_dict["timestamp"]) >= start_date):
                            events.append(event_dict)

            except Exception as e:
                logger.error(f"读取日志文件失败 {log_file}: {e}")

        return events

    def get_high_risk_events(self, risk_threshold: float = 10.0, days: int = 7) -> List[Dict[str, Any]]:
        """获取高风险审计事件"""
        start_date = datetime.now() - timedelta(days=days)
        events = []

        # 遍历日志文件
        for log_file in self.log_directory.glob("audit_*.log"):
            try:
                with open(log_file, 'r', encoding='utf - 8') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue

                        event_dict = json.loads(line.strip())
                        if (event_dict.get("risk_score", 0) >= risk_threshold
                                and datetime.fromisoformat(event_dict["timestamp"]) >= start_date):
                            events.append(event_dict)

            except Exception as e:
                logger.error(f"读取日志文件失败 {log_file}: {e}")

        return events


class SecurityMonitor:

    """安全监控器"""

    def __init__(self):

        self.security_events: List[SecurityEvent] = []
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_ips: Dict[str, int] = {}

        # 配置
        self.max_failed_logins = 5
        self.lockout_duration_minutes = 30
        self.suspicious_threshold = 10

        # 基础设施集成
        self._infrastructure_adapter = None
        if INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            try:
                self._infrastructure_adapter = get_trading_layer_adapter()
            except Exception as e:
                logger.warning(f"基础设施集成初始化失败: {e}")

        logger.info("安全监控器初始化完成")

    def record_security_event(self, event_type: str, severity: SecurityLevel,


                              source_ip: Optional[str] = None, user_id: Optional[str] = None,
                              description: str = "", details: Dict[str, Any] = None):
        """记录安全事件"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            details=details or {}
        )

        self.security_events.append(event)

        # 限制事件数量
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

        # 特殊处理登录失败
        if event_type == "failed_login":
            self._handle_failed_login(source_ip or "unknown", user_id)

        # 特殊处理可疑活动（避免递归调用）
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL] and event_type != "suspicious_ip":
            self._handle_suspicious_activity(source_ip, event_type)

        logger.warning(f"记录安全事件: {event_type} - {description}")

    def _handle_failed_login(self, ip_address: str, user_id: Optional[str]):
        """处理登录失败"""
        if ip_address not in self.failed_login_attempts:
            self.failed_login_attempts[ip_address] = []

        self.failed_login_attempts[ip_address].append(datetime.now())

        # 清理过期记录
        cutoff_time = datetime.now() - timedelta(minutes=self.lockout_duration_minutes)
        self.failed_login_attempts[ip_address] = [
            attempt for attempt in self.failed_login_attempts[ip_address]
            if attempt > cutoff_time
        ]

        # 检查是否超过阈值
        if len(self.failed_login_attempts[ip_address]) >= self.max_failed_logins:
            # 直接创建安全事件对象，而不是调用record_security_event（避免递归）
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                event_type="brute_force_attempt",
                severity=SecurityLevel.HIGH,
                source_ip=ip_address,
                user_id=user_id,
                description=f"检测到暴力破解尝试，IP: {ip_address}",
                details={"failed_attempts": len(self.failed_login_attempts[ip_address])}
            )

            self.security_events.append(event)

            # 限制事件数量
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]

            logger.warning(f"检测到暴力破解尝试，IP: {ip_address}")

    def _handle_suspicious_activity(self, ip_address: Optional[str], event_type: str):
        """处理可疑活动"""
        if not ip_address:
            return

        if ip_address not in self.suspicious_ips:
            self.suspicious_ips[ip_address] = 0

        self.suspicious_ips[ip_address] += 1

        # 检查是否超过阈值，但不记录新的事件（避免递归）
        if self.suspicious_ips[ip_address] >= self.suspicious_threshold:
            # 直接创建安全事件对象，而不是调用record_security_event
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                event_type="suspicious_ip",
                severity=SecurityLevel.CRITICAL,
                source_ip=ip_address,
                user_id=None,
                description=f"检测到可疑IP地址: {ip_address}",
                details={"suspicious_events": self.suspicious_ips[ip_address]}
            )

            self.security_events.append(event)

            # 限制事件数量
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]

            logger.warning(f"检测到可疑IP地址: {ip_address}")

    def is_ip_blocked(self, ip_address: str) -> bool:
        """检查IP是否被封禁"""
        if ip_address in self.failed_login_attempts:
            recent_attempts = [
                attempt for attempt in self.failed_login_attempts[ip_address]
                if datetime.now() - attempt < timedelta(minutes=self.lockout_duration_minutes)
            ]

            if len(recent_attempts) >= self.max_failed_logins:
                return True

        return False

    def get_security_events(self, severity_filter: Optional[SecurityLevel] = None,


                            hours: int = 24) -> List[SecurityEvent]:
        """获取安全事件"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        events = [
            event for event in self.security_events
            if event.timestamp >= cutoff_time
        ]

        if severity_filter:
            events = [event for event in events if event.severity == severity_filter]

        return events

    def resolve_security_event(self, event_id: str, notes: str = ""):
        """解决安全事件"""
        for event in self.security_events:
            if event.event_id == event_id:
                event.resolved = True
                event.resolved_at = datetime.now()
                event.resolution_notes = notes
                logger.info(f"解决安全事件: {event_id}")
                break


class AuditSystem:

    """审计系统主类"""

    def __init__(self, log_directory: str = "logs/audit"):

        self.audit_logger = AuditLogger(log_directory)
        self.security_monitor = SecurityMonitor()

        # 基础设施集成
        self._infrastructure_adapter = None
        if INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            try:
                self._infrastructure_adapter = get_trading_layer_adapter()
            except Exception as e:
                logger.warning(f"基础设施集成初始化失败: {e}")

        logger.info("审计系统初始化完成")

    def log_trade_execution(self, user_id: str, trade_details: Dict[str, Any],


                            result: str = "success", session_id: Optional[str] = None,
                            ip_address: Optional[str] = None):
        """记录交易执行"""
        security_level = SecurityLevel.HIGH if result == "failure" else SecurityLevel.MEDIUM

        self.audit_logger.log_event(
            event_type=AuditEventType.TRADE_EXECUTE,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            resource="trading_engine",
            action="execute_trade",
            result=result,
            details=trade_details,
            security_level=security_level
        )

    def log_order_operation(self, user_id: str, order_type: str, order_details: Dict[str, Any],


                            operation: str, result: str = "success", session_id: Optional[str] = None,
                            ip_address: Optional[str] = None):
        """记录订单操作"""
        event_type = AuditEventType.ORDER_PLACE if operation == "place" else AuditEventType.ORDER_CANCEL
        security_level = SecurityLevel.MEDIUM if result == "failure" else SecurityLevel.LOW

        self.audit_logger.log_event(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            resource="order_manager",
            action=f"{operation}_order",
            result=result,
            details=order_details,
            security_level=security_level
        )

    def log_security_event(self, event_type: str, severity: SecurityLevel,


                           source_ip: Optional[str] = None, user_id: Optional[str] = None,
                           description: str = "", details: Dict[str, Any] = None):
        """记录安全事件"""
        # 同时记录到审计日志和安全监控器
        self.audit_logger.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id=user_id,
            ip_address=source_ip,
            resource="security_system",
            action=event_type,
            result="detected",
            details=details or {},
            security_level=severity
        )

        self.security_monitor.record_security_event(
            event_type, severity, source_ip, user_id, description, details
        )

    def log_user_authentication(self, user_id: str, action: str, result: str = "success",


                                session_id: Optional[str] = None, ip_address: Optional[str] = None,
                                user_agent: Optional[str] = None):
        """记录用户认证"""
        event_type = AuditEventType.LOGIN if action == "login" else AuditEventType.LOGOUT
        security_level = SecurityLevel.HIGH if result == "failure" else SecurityLevel.LOW

        # 如果登录失败，记录到安全监控器
        if action == "login" and result == "failure":
            self.security_monitor.record_security_event(
                "failed_login",
                SecurityLevel.MEDIUM,
                ip_address,
                user_id,
                "用户登录失败",
                {"action": action, "result": result}
            )

        self.audit_logger.log_event(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="authentication",
            action=action,
            result=result,
            details={},
            security_level=security_level
        )

    def log_system_operation(self, operation: str, details: Dict[str, Any] = None,


                             user_id: Optional[str] = None, ip_address: Optional[str] = None):
        """记录系统操作"""
        event_type = (AuditEventType.SYSTEM_START if operation == "start"
                      else AuditEventType.SYSTEM_STOP)

        self.audit_logger.log_event(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            resource="system",
            action=operation,
            result="success",
            details=details or {},
            security_level=SecurityLevel.LOW
        )

    def check_security_status(self, ip_address: str) -> Dict[str, Any]:
        """检查安全状态"""
        return {
            "ip_blocked": self.security_monitor.is_ip_blocked(ip_address),
            "failed_login_attempts": len(self.security_monitor.failed_login_attempts.get(ip_address, [])),
            "suspicious_score": self.security_monitor.suspicious_ips.get(ip_address, 0)
        }

    def get_audit_report(self, user_id: Optional[str] = None, event_type: Optional[AuditEventType] = None,


                         days: int = 7) -> Dict[str, Any]:
        """生成审计报告"""
        report = {
            "report_period_days": days,
            "generated_at": datetime.now().isoformat(),
            "total_events": 0,
            "high_risk_events": [],
            "security_events": [],
            "user_activity": {},
            "system_health": {}
        }

        # 获取高风险事件
        report["high_risk_events"] = self.audit_logger.get_high_risk_events(days=days)

        # 获取安全事件
        report["security_events"] = [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "severity": event.severity.value,
                "description": event.description,
                "resolved": event.resolved
            }
            for event in self.security_monitor.get_security_events(hours=days * 24)
        ]

        # 用户活动统计
        if user_id:
            user_events = self.audit_logger.get_events_by_user(user_id, days)
            report["user_activity"] = {
                "user_id": user_id,
                "total_events": len(user_events),
                "event_types": {}
            }

            for event in user_events:
                event_type = event.get("event_type", "unknown")
                if event_type not in report["user_activity"]["event_types"]:
                    report["user_activity"]["event_types"][event_type] = 0
                report["user_activity"]["event_types"][event_type] += 1

        # 系统健康统计
        report["system_health"] = {
            "blocked_ips": len([ip for ip in self.security_monitor.failed_login_attempts.keys()
                                if self.security_monitor.is_ip_blocked(ip)]),
            "suspicious_ips": len(self.security_monitor.suspicious_ips),
            "total_security_events": len(self.security_monitor.security_events)
        }

        report["total_events"] = len(report["high_risk_events"]) + len(report["security_events"])

        return report

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_info = {
            'component': 'AuditSystem',
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'log_directory': str(self.audit_logger.log_directory),
            'buffer_size': len(self.audit_logger.buffer),
            'security_events_count': len(self.security_monitor.security_events),
            'blocked_ips_count': len([ip for ip in self.security_monitor.failed_login_attempts.keys()
                                      if self.security_monitor.is_ip_blocked(ip)]),
            'infrastructure_integration': INFRASTRUCTURE_INTEGRATION_AVAILABLE,
            'warnings': [],
            'critical_issues': []
        }

        # 检查日志目录
        if not self.audit_logger.log_directory.exists():
            health_info['critical_issues'].append("审计日志目录不存在")

        # 检查缓冲区大小
        if len(self.audit_logger.buffer) > self.audit_logger.buffer_size * 2:
            health_info['warnings'].append("审计日志缓冲区过大")

        # 检查安全事件
        critical_security_events = [
            event for event in self.security_monitor.security_events
            if not event.resolved and event.severity == SecurityLevel.CRITICAL
        ]
        if critical_security_events:
            health_info['critical_issues'].extend([
                f"未解决的关键安全事件: {event.description}" for event in critical_security_events
            ])

        # 总体状态评估
        if health_info['critical_issues']:
            health_info['status'] = 'critical'
        elif health_info['warnings']:
            health_info['status'] = 'warning'

        return health_info


# 全局审计系统实例
_audit_system = None
_audit_system_lock = threading.Lock()


def get_audit_system(log_directory: str = "logs/audit") -> AuditSystem:
    """获取全局审计系统实例"""
    global _audit_system

    if _audit_system is None:
        with _audit_system_lock:
            if _audit_system is None:
                _audit_system = AuditSystem(log_directory)

    return _audit_system


# 便捷函数

def audit_trade_execution(user_id: str, trade_details: Dict[str, Any], result: str = "success",


                          session_id: Optional[str] = None, ip_address: Optional[str] = None):
    """审计交易执行"""
    system = get_audit_system()
    system.log_trade_execution(user_id, trade_details, result, session_id, ip_address)


def audit_order_operation(user_id: str, order_type: str, order_details: Dict[str, Any],


                          operation: str, result: str = "success", session_id: Optional[str] = None,
                          ip_address: Optional[str] = None):
    """审计订单操作"""
    system = get_audit_system()
    system.log_order_operation(user_id, order_type, order_details,
                               operation, result, session_id, ip_address)


def audit_security_event(event_type: str, severity: SecurityLevel, source_ip: Optional[str] = None,


                         user_id: Optional[str] = None, description: str = "", details: Dict[str, Any] = None):
    """审计安全事件"""
    system = get_audit_system()
    system.log_security_event(event_type, severity, source_ip, user_id, description, details)


def check_security_status(ip_address: str) -> Dict[str, Any]:
    """检查安全状态"""
    system = get_audit_system()
    return system.check_security_status(ip_address)
