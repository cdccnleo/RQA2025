"""
数据源告警管理器
监控数据源健康状态并发送告警通知
符合架构设计：使用EventBus进行事件通知
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

import asyncpg

from src.gateway.web.datasource_health_checker import HealthStatus, HealthStatusEnum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """告警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertType(str, Enum):
    """告警类型"""
    CONNECTION_FAILURE = "connection_failure"
    RESPONSE_TIMEOUT = "response_timeout"
    UNAVAILABLE = "unavailable"
    QUALITY_DEGRADED = "quality_degraded"


@dataclass
class DataSourceAlert:
    """数据源告警数据类"""
    alert_id: Optional[int] = None
    source_id: str = ""
    alert_type: AlertType = AlertType.CONNECTION_FAILURE
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None


@dataclass
class AlertRule:
    """告警规则"""
    alert_type: AlertType
    condition: str  # 条件表达式
    severity: AlertSeverity
    cooldown_minutes: int  # 冷却时间（分钟）
    description: str


class AlertRuleEngine:
    """告警规则引擎"""
    
    # 预定义告警规则
    RULES = {
        AlertType.CONNECTION_FAILURE: AlertRule(
            alert_type=AlertType.CONNECTION_FAILURE,
            condition="consecutive_failures >= 3",
            severity=AlertSeverity.WARNING,
            cooldown_minutes=30,
            description="数据源连接失败超过3次"
        ),
        AlertType.RESPONSE_TIMEOUT: AlertRule(
            alert_type=AlertType.RESPONSE_TIMEOUT,
            condition="response_time_ms > 10000",
            severity=AlertSeverity.WARNING,
            cooldown_minutes=15,
            description="数据源响应时间超过10秒"
        ),
        AlertType.UNAVAILABLE: AlertRule(
            alert_type=AlertType.UNAVAILABLE,
            condition="consecutive_failures >= 5",
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=60,
            description="数据源连续5次检测失败，已自动禁用"
        ),
        AlertType.QUALITY_DEGRADED: AlertRule(
            alert_type=AlertType.QUALITY_DEGRADED,
            condition="status == 'unhealthy' and consecutive_failures >= 2",
            severity=AlertSeverity.INFO,
            cooldown_minutes=20,
            description="数据源质量下降"
        )
    }
    
    def evaluate(self, health_status: HealthStatus) -> Optional[DataSourceAlert]:
        """评估告警规则
        
        Args:
            health_status: 健康状态
            
        Returns:
            如果触发告警则返回告警对象，否则返回None
        """
        # 按优先级顺序评估规则（critical > warning > info）
        for alert_type in [AlertType.UNAVAILABLE, AlertType.CONNECTION_FAILURE, 
                          AlertType.RESPONSE_TIMEOUT, AlertType.QUALITY_DEGRADED]:
            rule = self.RULES[alert_type]
            if self._check_condition(rule.condition, health_status):
                return DataSourceAlert(
                    source_id=health_status.source_id,
                    alert_type=alert_type,
                    severity=rule.severity,
                    message=self._generate_message(alert_type, health_status),
                    details={
                        "response_time_ms": health_status.response_time_ms,
                        "consecutive_failures": health_status.consecutive_failures,
                        "error_message": health_status.message,
                        "check_time": health_status.check_time.isoformat()
                    }
                )
        return None
    
    def _check_condition(self, condition: str, health_status: HealthStatus) -> bool:
        """检查条件是否满足"""
        try:
            # 构建评估上下文
            context = {
                "consecutive_failures": health_status.consecutive_failures,
                "response_time_ms": health_status.response_time_ms,
                "status": health_status.status.value,
                "message": health_status.message
            }
            # 安全地评估条件表达式
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(f"评估告警条件失败: {condition}, 错误: {e}")
            return False
    
    def _generate_message(self, alert_type: AlertType, health_status: HealthStatus) -> str:
        """生成告警消息"""
        messages = {
            AlertType.CONNECTION_FAILURE: 
                f"数据源 {health_status.source_id} 连接失败，已连续失败 {health_status.consecutive_failures} 次",
            AlertType.RESPONSE_TIMEOUT: 
                f"数据源 {health_status.source_id} 响应超时，响应时间 {health_status.response_time_ms}ms",
            AlertType.UNAVAILABLE: 
                f"数据源 {health_status.source_id} 已不可用，已连续失败 {health_status.consecutive_failures} 次，已自动禁用",
            AlertType.QUALITY_DEGRADED: 
                f"数据源 {health_status.source_id} 质量下降，状态: {health_status.status.value}"
        }
        return messages.get(alert_type, f"数据源 {health_status.source_id} 异常")


class DataSourceAlertManager:
    """数据源告警管理器
    
    管理数据源告警的创建、查询、确认和解决
    """
    
    def __init__(self):
        self._db_pool: Optional[asyncpg.Pool] = None
        self._rule_engine = AlertRuleEngine()
        self._alert_history: Dict[str, datetime] = {}  # 记录上次告警时间，用于冷却
        self._notification_handlers: List[callable] = []
    
    async def _get_db_pool(self) -> asyncpg.Pool:
        """获取数据库连接池"""
        if self._db_pool is None:
            import os
            db_password = os.environ.get('DB_PASSWORD', 'SecurePass123!')
            self._db_pool = await asyncpg.create_pool(
                host="rqa2025-postgres",
                port=5432,
                database="rqa2025_prod",
                user="rqa2025_admin",
                password=db_password,
                min_size=2,
                max_size=5
            )
        return self._db_pool
    
    def register_notification_handler(self, handler: callable):
        """注册通知处理器
        
        Args:
            handler: 通知处理函数，接收 DataSourceAlert 参数
        """
        self._notification_handlers.append(handler)
    
    async def check_and_alert(self, health_status: HealthStatus) -> Optional[DataSourceAlert]:
        """检查健康状态并发送告警
        
        Args:
            health_status: 健康状态
            
        Returns:
            如果触发告警则返回告警对象，否则返回None
        """
        # 评估告警规则
        alert = self._rule_engine.evaluate(health_status)
        if alert is None:
            return None
        
        # 检查冷却期
        if not self._check_cooldown(alert):
            logger.debug(f"告警 {alert.alert_type.value} 处于冷却期，跳过")
            return None
        
        # 检查是否已存在相同类型的活跃告警
        existing_alert = await self._get_active_alert_by_type(alert.source_id, alert.alert_type)
        if existing_alert:
            logger.debug(f"数据源 {alert.source_id} 已存在 {alert.alert_type.value} 类型的活跃告警，跳过")
            return None
        
        # 保存告警
        alert_id = await self._save_alert(alert)
        alert.alert_id = alert_id
        
        # 更新冷却时间
        self._update_cooldown(alert)
        
        # 发送通知
        await self._send_notifications(alert)
        
        logger.info(f"创建告警: {alert.alert_type.value} - {alert.message}")
        return alert
    
    def _check_cooldown(self, alert: DataSourceAlert) -> bool:
        """检查告警是否处于冷却期"""
        key = f"{alert.source_id}:{alert.alert_type.value}"
        last_alert_time = self._alert_history.get(key)
        
        if last_alert_time is None:
            return True
        
        rule = self._rule_engine.RULES[alert.alert_type]
        cooldown_end = last_alert_time + timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.now() > cooldown_end
    
    def _update_cooldown(self, alert: DataSourceAlert):
        """更新告警冷却时间"""
        key = f"{alert.source_id}:{alert.alert_type.value}"
        self._alert_history[key] = datetime.now()
    
    async def _get_active_alert_by_type(self, source_id: str, alert_type: AlertType) -> Optional[DataSourceAlert]:
        """获取指定类型的活跃告警"""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT alert_id, source_id, alert_type, severity, message, details, status, created_at
                    FROM data_source_alerts
                    WHERE source_id = $1 AND alert_type = $2 AND status = 'active'
                    LIMIT 1
                    """,
                    source_id, alert_type.value
                )
                
                if row:
                    return DataSourceAlert(
                        alert_id=row['alert_id'],
                        source_id=row['source_id'],
                        alert_type=AlertType(row['alert_type']),
                        severity=AlertSeverity(row['severity']),
                        message=row['message'],
                        details=json.loads(row['details']) if row['details'] else {},
                        status=AlertStatus(row['status']),
                        created_at=row['created_at']
                    )
                return None
        except Exception as e:
            logger.error(f"获取活跃告警失败: {e}")
            return None
    
    async def _save_alert(self, alert: DataSourceAlert) -> int:
        """保存告警到数据库"""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO data_source_alerts 
                    (source_id, alert_type, severity, message, details, status, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING alert_id
                    """,
                    alert.source_id,
                    alert.alert_type.value,
                    alert.severity.value,
                    alert.message,
                    json.dumps(alert.details),
                    alert.status.value,
                    alert.created_at
                )
                return row['alert_id']
        except Exception as e:
            logger.error(f"保存告警失败: {e}")
            raise
    
    async def _send_notifications(self, alert: DataSourceAlert):
        """发送告警通知"""
        # 调用所有注册的通知处理器
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"通知处理器执行失败: {e}")
        
        # 通过EventBus广播告警事件
        try:
            from src.core.event_bus import get_event_bus
            event_bus = get_event_bus()
            await event_bus.publish("datasource_alert_created", {
                "alert_id": alert.alert_id,
                "source_id": alert.source_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "message": alert.message,
                "created_at": alert.created_at.isoformat()
            })
        except Exception as e:
            logger.error(f"EventBus广播告警事件失败: {e}")
    
    async def get_alerts(
        self,
        source_id: Optional[str] = None,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DataSourceAlert]:
        """获取告警列表"""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                query = """
                    SELECT alert_id, source_id, alert_type, severity, message, details, status, 
                           created_at, acknowledged_at, acknowledged_by, resolved_at, resolved_by, resolution_notes
                    FROM data_source_alerts
                    WHERE 1=1
                """
                params = []
                
                if source_id:
                    query += f" AND source_id = ${len(params) + 1}"
                    params.append(source_id)
                
                if status:
                    query += f" AND status = ${len(params) + 1}"
                    params.append(status.value)
                
                if severity:
                    query += f" AND severity = ${len(params) + 1}"
                    params.append(severity.value)
                
                query += " ORDER BY created_at DESC"
                query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
                params.extend([limit, offset])
                
                rows = await conn.fetch(query, *params)
                
                return [
                    DataSourceAlert(
                        alert_id=row['alert_id'],
                        source_id=row['source_id'],
                        alert_type=AlertType(row['alert_type']),
                        severity=AlertSeverity(row['severity']),
                        message=row['message'],
                        details=json.loads(row['details']) if row['details'] else {},
                        status=AlertStatus(row['status']),
                        created_at=row['created_at'],
                        acknowledged_at=row['acknowledged_at'],
                        acknowledged_by=row['acknowledged_by'],
                        resolved_at=row['resolved_at'],
                        resolved_by=row['resolved_by'],
                        resolution_notes=row['resolution_notes']
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"获取告警列表失败: {e}")
            return []
    
    async def get_active_alerts(self) -> List[DataSourceAlert]:
        """获取活跃告警"""
        return await self.get_alerts(status=AlertStatus.ACTIVE)
    
    async def acknowledge_alert(
        self, 
        alert_id: int, 
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """确认告警"""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE data_source_alerts
                    SET status = 'acknowledged',
                        acknowledged_at = $1,
                        acknowledged_by = $2,
                        resolution_notes = COALESCE($3, resolution_notes)
                    WHERE alert_id = $4 AND status = 'active'
                    """,
                    datetime.now(), acknowledged_by, notes, alert_id
                )
                
                if result == "UPDATE 1":
                    logger.info(f"告警 {alert_id} 已被 {acknowledged_by} 确认")
                    return True
                return False
        except Exception as e:
            logger.error(f"确认告警失败: {e}")
            return False
    
    async def resolve_alert(
        self, 
        alert_id: int, 
        resolved_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """解决告警"""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE data_source_alerts
                    SET status = 'resolved',
                        resolved_at = $1,
                        resolved_by = $2,
                        resolution_notes = $3
                    WHERE alert_id = $4 AND status IN ('active', 'acknowledged')
                    """,
                    datetime.now(), resolved_by, notes, alert_id
                )
                
                if result == "UPDATE 1":
                    logger.info(f"告警 {alert_id} 已被 {resolved_by} 解决")
                    return True
                return False
        except Exception as e:
            logger.error(f"解决告警失败: {e}")
            return False
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                # 按状态统计
                status_rows = await conn.fetch(
                    """
                    SELECT status, COUNT(*) as count
                    FROM data_source_alerts
                    GROUP BY status
                    """
                )
                
                # 按级别统计
                severity_rows = await conn.fetch(
                    """
                    SELECT severity, COUNT(*) as count
                    FROM data_source_alerts
                    WHERE status = 'active'
                    GROUP BY severity
                    """
                )
                
                # 按类型统计
                type_rows = await conn.fetch(
                    """
                    SELECT alert_type, COUNT(*) as count
                    FROM data_source_alerts
                    WHERE status = 'active'
                    GROUP BY alert_type
                    """
                )
                
                return {
                    "by_status": {row['status']: row['count'] for row in status_rows},
                    "by_severity": {row['severity']: row['count'] for row in severity_rows},
                    "by_type": {row['alert_type']: row['count'] for row in type_rows},
                    "total_active": sum(row['count'] for row in severity_rows)
                }
        except Exception as e:
            logger.error(f"获取告警统计失败: {e}")
            return {}


# 全局告警管理器实例
_alert_manager: Optional[DataSourceAlertManager] = None


def get_alert_manager() -> DataSourceAlertManager:
    """获取告警管理器实例（单例模式）"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = DataSourceAlertManager()
    return _alert_manager
