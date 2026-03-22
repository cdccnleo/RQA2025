# -*- coding: utf-8 -*-
"""
风险控制层持久化管理模块

实现风险控制层各组件的PostgreSQL数据库持久化，遵循PostgreSQL优先存储策略。
支持风险检查、告警记录、风险指标、风险规则的完整生命周期管理。

设计原则：
1. PostgreSQL优先存储，连接失败时降级到文件系统
2. 使用统一数据库配置模块获取连接参数
3. 支持连接重试机制
4. 线程安全操作
"""

import os
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_postgresql_config() -> Optional[Dict[str, str]]:
    """
    获取PostgreSQL配置（使用统一配置模块）
    
    Returns:
        数据库配置字典，获取失败时返回None
    """
    try:
        from src.infrastructure.persistence.database_config import get_db_config
        config = get_db_config()
        return config.to_dict()
    except Exception as e:
        logger.warning(f"获取数据库配置失败: {e}，使用环境变量")
        return {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": os.getenv("POSTGRES_PORT", "5432"),
            "database": os.getenv("POSTGRES_DB", "rqa2025_prod"),
            "user": os.getenv("POSTGRES_USER", "rqa2025_admin"),
            "password": os.getenv("POSTGRES_PASSWORD", "")
        }


def _retry_db_operation(func, max_retries: int = 3, delay: float = 1.0):
    """
    数据库操作重试装饰器
    
    Args:
        func: 要执行的函数
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    
    Returns:
        函数执行结果
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
    raise last_error


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class RiskCheckData:
    """风险检查数据结构"""
    check_id: str
    check_type: str
    risk_level: str
    passed: bool
    score: float = 0.0
    symbol: Optional[str] = None
    account_id: Optional[str] = None
    strategy_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertData:
    """告警数据结构"""
    alert_id: str
    alert_type: str
    alert_level: str
    title: str
    message: str
    status: str = "active"
    source: Optional[str] = None
    rule_id: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetricData:
    """风险指标数据结构"""
    metric_id: str
    metric_name: str
    metric_type: str
    value: float = 0.0
    threshold_low: float = 0.0
    threshold_medium: float = 0.0
    threshold_high: float = 0.0
    risk_level: str = "low"
    symbol: Optional[str] = None
    portfolio_id: Optional[str] = None
    calculation_method: Optional[str] = None
    confidence_level: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskRuleData:
    """风险规则数据结构"""
    rule_id: str
    rule_name: str
    rule_type: str
    risk_type: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    alert_level: str = "warning"
    enabled: bool = True
    cooldown_minutes: int = 30
    priority: int = 100
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 抽象接口定义
# ============================================================

class IRiskCheckPersistence(ABC):
    """风险检查持久化接口"""
    
    @abstractmethod
    def save_check(self, check: RiskCheckData) -> bool:
        """保存风险检查记录"""
        pass
    
    @abstractmethod
    def get_check(self, check_id: str) -> Optional[RiskCheckData]:
        """获取风险检查记录"""
        pass
    
    @abstractmethod
    def get_checks_by_symbol(self, symbol: str, limit: int = 100) -> List[RiskCheckData]:
        """按标的获取风险检查记录"""
        pass
    
    @abstractmethod
    def get_checks_by_level(self, risk_level: str, limit: int = 100) -> List[RiskCheckData]:
        """按风险等级获取检查记录"""
        pass


class IAlertPersistence(ABC):
    """告警持久化接口"""
    
    @abstractmethod
    def save_alert(self, alert: AlertData) -> bool:
        """保存告警记录"""
        pass
    
    @abstractmethod
    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """更新告警状态"""
        pass
    
    @abstractmethod
    def get_alert(self, alert_id: str) -> Optional[AlertData]:
        """获取告警记录"""
        pass
    
    @abstractmethod
    def get_active_alerts(self, limit: int = 100) -> List[AlertData]:
        """获取活跃告警"""
        pass
    
    @abstractmethod
    def get_alerts_by_level(self, level: str, limit: int = 100) -> List[AlertData]:
        """按级别获取告警"""
        pass


class IRiskMetricPersistence(ABC):
    """风险指标持久化接口"""
    
    @abstractmethod
    def save_metric(self, metric: RiskMetricData) -> bool:
        """保存风险指标"""
        pass
    
    @abstractmethod
    def get_metric(self, metric_id: str) -> Optional[RiskMetricData]:
        """获取风险指标"""
        pass
    
    @abstractmethod
    def get_metrics_by_name(self, metric_name: str, limit: int = 100) -> List[RiskMetricData]:
        """按名称获取指标历史"""
        pass
    
    @abstractmethod
    def get_latest_metrics(self, limit: int = 100) -> List[RiskMetricData]:
        """获取最新指标"""
        pass


class IRiskRulePersistence(ABC):
    """风险规则持久化接口"""
    
    @abstractmethod
    def save_rule(self, rule: RiskRuleData) -> bool:
        """保存风险规则"""
        pass
    
    @abstractmethod
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新风险规则"""
        pass
    
    @abstractmethod
    def get_rule(self, rule_id: str) -> Optional[RiskRuleData]:
        """获取风险规则"""
        pass
    
    @abstractmethod
    def get_all_rules(self) -> List[RiskRuleData]:
        """获取所有规则"""
        pass
    
    @abstractmethod
    def get_enabled_rules(self) -> List[RiskRuleData]:
        """获取启用的规则"""
        pass
    
    @abstractmethod
    def delete_rule(self, rule_id: str) -> bool:
        """删除规则"""
        pass


# ============================================================
# PostgreSQL持久化实现
# ============================================================

class RiskCheckPersistence(IRiskCheckPersistence):
    """
    风险检查持久化管理器
    
    实现PostgreSQL优先存储策略，支持自动降级到文件系统。
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化风险检查持久化管理器
        
        Args:
            storage_dir: 文件存储目录（降级时使用）
        """
        self._lock = threading.RLock()
        self._db_config = _get_postgresql_config()
        self._use_postgresql = self._test_db_connection()
        
        self._storage_dir = Path(storage_dir or os.path.join(
            os.getcwd(), "data", "risk", "checks"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, RiskCheckData] = {}
        self._max_cache_size = 10000
        
        if self._use_postgresql:
            logger.info("RiskCheckPersistence: 使用PostgreSQL存储")
        else:
            logger.warning("RiskCheckPersistence: PostgreSQL不可用，降级到文件系统存储")
    
    def _test_db_connection(self) -> bool:
        """测试数据库连接"""
        if not self._db_config or not self._db_config.get("password"):
            return False
        try:
            import psycopg2
            conn = psycopg2.connect(**self._db_config)
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"数据库连接测试失败: {e}")
            return False
    
    def _get_connection(self):
        """获取数据库连接"""
        import psycopg2
        return psycopg2.connect(**self._db_config)
    
    def save_check(self, check: RiskCheckData) -> bool:
        """
        保存风险检查记录
        
        Args:
            check: 风险检查数据
        
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._save_check_to_db(check)
                else:
                    return self._save_check_to_file(check)
            except Exception as e:
                logger.error(f"保存风险检查记录失败: {e}")
                return False
    
    def _save_check_to_db(self, check: RiskCheckData) -> bool:
        """保存风险检查记录到PostgreSQL"""
        def _do_save():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO risk_checks (
                            check_id, check_type, risk_level, passed, score,
                            symbol, account_id, strategy_id, details, recommendations,
                            created_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (check_id) DO UPDATE SET
                            risk_level = EXCLUDED.risk_level,
                            passed = EXCLUDED.passed,
                            score = EXCLUDED.score,
                            details = EXCLUDED.details,
                            recommendations = EXCLUDED.recommendations,
                            metadata = EXCLUDED.metadata
                    """, (
                        check.check_id, check.check_type, check.risk_level,
                        check.passed, check.score, check.symbol, check.account_id,
                        check.strategy_id, json.dumps(check.details),
                        json.dumps(check.recommendations), check.created_at,
                        json.dumps(check.metadata)
                    ))
                    conn.commit()
            return True
        
        try:
            return _retry_db_operation(_do_save)
        except Exception as e:
            logger.error(f"数据库保存风险检查记录失败: {e}")
            self._use_postgresql = False
            return self._save_check_to_file(check)
    
    def _save_check_to_file(self, check: RiskCheckData) -> bool:
        """保存风险检查记录到文件系统"""
        try:
            file_path = self._storage_dir / f"{check.check_id}.json"
            data = asdict(check)
            data['created_at'] = data['created_at'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._update_cache(check)
            return True
        except Exception as e:
            logger.error(f"文件保存风险检查记录失败: {e}")
            return False
    
    def get_check(self, check_id: str) -> Optional[RiskCheckData]:
        """
        获取风险检查记录
        
        Args:
            check_id: 检查ID
        
        Returns:
            风险检查数据，不存在返回None
        """
        with self._lock:
            if check_id in self._cache:
                return self._cache[check_id]
            
            try:
                if self._use_postgresql:
                    return self._get_check_from_db(check_id)
                else:
                    return self._get_check_from_file(check_id)
            except Exception as e:
                logger.error(f"获取风险检查记录失败: {e}")
                return None
    
    def _get_check_from_db(self, check_id: str) -> Optional[RiskCheckData]:
        """从数据库获取风险检查记录"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT check_id, check_type, risk_level, passed, score,
                               symbol, account_id, strategy_id, details, recommendations,
                               created_at, metadata
                        FROM risk_checks WHERE check_id = %s
                    """, (check_id,))
                    row = cur.fetchone()
                    if row:
                        return RiskCheckData(
                            check_id=row[0], check_type=row[1], risk_level=row[2],
                            passed=row[3], score=float(row[4]), symbol=row[5],
                            account_id=row[6], strategy_id=row[7],
                            details=row[8] if isinstance(row[8], dict) else json.loads(row[8] or '{}'),
                            recommendations=row[9] if isinstance(row[9], list) else json.loads(row[9] or '[]'),
                            created_at=row[10],
                            metadata=row[11] if isinstance(row[11], dict) else json.loads(row[11] or '{}')
                        )
                    return None
        
        result = _retry_db_operation(_do_get)
        if result:
            self._update_cache(result)
        return result
    
    def _get_check_from_file(self, check_id: str) -> Optional[RiskCheckData]:
        """从文件获取风险检查记录"""
        file_path = self._storage_dir / f"{check_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        check = RiskCheckData(**data)
        self._update_cache(check)
        return check
    
    def get_checks_by_symbol(self, symbol: str, limit: int = 100) -> List[RiskCheckData]:
        """
        按标的获取风险检查记录
        
        Args:
            symbol: 标的代码
            limit: 返回数量限制
        
        Returns:
            风险检查记录列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_checks_by_symbol_from_db(symbol, limit)
                else:
                    return self._get_checks_by_symbol_from_files(symbol, limit)
            except Exception as e:
                logger.error(f"按标的获取风险检查记录失败: {e}")
                return []
    
    def _get_checks_by_symbol_from_db(self, symbol: str, limit: int) -> List[RiskCheckData]:
        """从数据库按标的获取风险检查记录"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT check_id, check_type, risk_level, passed, score,
                               symbol, account_id, strategy_id, details, recommendations,
                               created_at, metadata
                        FROM risk_checks 
                        WHERE symbol = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (symbol, limit))
                    rows = cur.fetchall()
                    return [
                        RiskCheckData(
                            check_id=row[0], check_type=row[1], risk_level=row[2],
                            passed=row[3], score=float(row[4]), symbol=row[5],
                            account_id=row[6], strategy_id=row[7],
                            details=row[8] if isinstance(row[8], dict) else json.loads(row[8] or '{}'),
                            recommendations=row[9] if isinstance(row[9], list) else json.loads(row[9] or '[]'),
                            created_at=row[10],
                            metadata=row[11] if isinstance(row[11], dict) else json.loads(row[11] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_checks_by_symbol_from_files(self, symbol: str, limit: int) -> List[RiskCheckData]:
        """从文件按标的获取风险检查记录"""
        checks = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                check = self._get_check_from_file(file_path.stem)
                if check and check.symbol == symbol:
                    checks.append(check)
                    if len(checks) >= limit:
                        break
            except Exception:
                continue
        checks.sort(key=lambda x: x.created_at, reverse=True)
        return checks[:limit]
    
    def get_checks_by_level(self, risk_level: str, limit: int = 100) -> List[RiskCheckData]:
        """
        按风险等级获取检查记录
        
        Args:
            risk_level: 风险等级
            limit: 返回数量限制
        
        Returns:
            风险检查记录列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_checks_by_level_from_db(risk_level, limit)
                else:
                    return self._get_checks_by_level_from_files(risk_level, limit)
            except Exception as e:
                logger.error(f"按风险等级获取检查记录失败: {e}")
                return []
    
    def _get_checks_by_level_from_db(self, risk_level: str, limit: int) -> List[RiskCheckData]:
        """从数据库按风险等级获取检查记录"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT check_id, check_type, risk_level, passed, score,
                               symbol, account_id, strategy_id, details, recommendations,
                               created_at, metadata
                        FROM risk_checks 
                        WHERE risk_level = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (risk_level, limit))
                    rows = cur.fetchall()
                    return [
                        RiskCheckData(
                            check_id=row[0], check_type=row[1], risk_level=row[2],
                            passed=row[3], score=float(row[4]), symbol=row[5],
                            account_id=row[6], strategy_id=row[7],
                            details=row[8] if isinstance(row[8], dict) else json.loads(row[8] or '{}'),
                            recommendations=row[9] if isinstance(row[9], list) else json.loads(row[9] or '[]'),
                            created_at=row[10],
                            metadata=row[11] if isinstance(row[11], dict) else json.loads(row[11] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_checks_by_level_from_files(self, risk_level: str, limit: int) -> List[RiskCheckData]:
        """从文件按风险等级获取检查记录"""
        checks = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                check = self._get_check_from_file(file_path.stem)
                if check and check.risk_level == risk_level:
                    checks.append(check)
                    if len(checks) >= limit:
                        break
            except Exception:
                continue
        checks.sort(key=lambda x: x.created_at, reverse=True)
        return checks[:limit]
    
    def _update_cache(self, check: RiskCheckData):
        """更新缓存"""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[check.check_id] = check


class AlertPersistence(IAlertPersistence):
    """
    告警持久化管理器
    
    实现PostgreSQL优先存储策略，支持自动降级到文件系统。
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化告警持久化管理器
        
        Args:
            storage_dir: 文件存储目录（降级时使用）
        """
        self._lock = threading.RLock()
        self._db_config = _get_postgresql_config()
        self._use_postgresql = self._test_db_connection()
        
        self._storage_dir = Path(storage_dir or os.path.join(
            os.getcwd(), "data", "risk", "alerts"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, AlertData] = {}
        self._max_cache_size = 5000
        
        if self._use_postgresql:
            logger.info("AlertPersistence: 使用PostgreSQL存储")
        else:
            logger.warning("AlertPersistence: PostgreSQL不可用，降级到文件系统存储")
    
    def _test_db_connection(self) -> bool:
        """测试数据库连接"""
        if not self._db_config or not self._db_config.get("password"):
            return False
        try:
            import psycopg2
            conn = psycopg2.connect(**self._db_config)
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"数据库连接测试失败: {e}")
            return False
    
    def _get_connection(self):
        """获取数据库连接"""
        import psycopg2
        return psycopg2.connect(**self._db_config)
    
    def save_alert(self, alert: AlertData) -> bool:
        """
        保存告警记录
        
        Args:
            alert: 告警数据
        
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._save_alert_to_db(alert)
                else:
                    return self._save_alert_to_file(alert)
            except Exception as e:
                logger.error(f"保存告警记录失败: {e}")
                return False
    
    def _save_alert_to_db(self, alert: AlertData) -> bool:
        """保存告警记录到PostgreSQL"""
        def _do_save():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO risk_alerts (
                            alert_id, alert_type, alert_level, title, message,
                            status, source, rule_id, acknowledged_by, acknowledged_at,
                            resolved_by, resolved_at, details, created_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (alert_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            acknowledged_by = EXCLUDED.acknowledged_by,
                            acknowledged_at = EXCLUDED.acknowledged_at,
                            resolved_by = EXCLUDED.resolved_by,
                            resolved_at = EXCLUDED.resolved_at,
                            details = EXCLUDED.details,
                            metadata = EXCLUDED.metadata
                    """, (
                        alert.alert_id, alert.alert_type, alert.alert_level,
                        alert.title, alert.message, alert.status, alert.source,
                        alert.rule_id, alert.acknowledged_by, alert.acknowledged_at,
                        alert.resolved_by, alert.resolved_at, json.dumps(alert.details),
                        alert.created_at, json.dumps(alert.metadata)
                    ))
                    conn.commit()
            return True
        
        try:
            return _retry_db_operation(_do_save)
        except Exception as e:
            logger.error(f"数据库保存告警记录失败: {e}")
            self._use_postgresql = False
            return self._save_alert_to_file(alert)
    
    def _save_alert_to_file(self, alert: AlertData) -> bool:
        """保存告警记录到文件系统"""
        try:
            file_path = self._storage_dir / f"{alert.alert_id}.json"
            data = asdict(alert)
            data['created_at'] = data['created_at'].isoformat()
            if data.get('acknowledged_at'):
                data['acknowledged_at'] = data['acknowledged_at'].isoformat()
            if data.get('resolved_at'):
                data['resolved_at'] = data['resolved_at'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._update_cache(alert)
            return True
        except Exception as e:
            logger.error(f"文件保存告警记录失败: {e}")
            return False
    
    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新告警状态
        
        Args:
            alert_id: 告警ID
            updates: 更新字段字典
        
        Returns:
            是否更新成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._update_alert_in_db(alert_id, updates)
                else:
                    return self._update_alert_in_file(alert_id, updates)
            except Exception as e:
                logger.error(f"更新告警状态失败: {e}")
                return False
    
    def _update_alert_in_db(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """在数据库中更新告警"""
        def _do_update():
            set_clauses = []
            values = []
            for key, value in updates.items():
                if key in ['details', 'metadata']:
                    set_clauses.append(f"{key} = %s")
                    values.append(json.dumps(value))
                elif key in ['acknowledged_at', 'resolved_at']:
                    set_clauses.append(f"{key} = %s")
                    values.append(value)
                else:
                    set_clauses.append(f"{key} = %s")
                    values.append(value)
            values.append(alert_id)
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        UPDATE risk_alerts 
                        SET {', '.join(set_clauses)}
                        WHERE alert_id = %s
                    """, values)
                    conn.commit()
            return True
        
        return _retry_db_operation(_do_update)
    
    def _update_alert_in_file(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """在文件中更新告警"""
        alert = self.get_alert(alert_id)
        if not alert:
            return False
        
        for key, value in updates.items():
            if hasattr(alert, key):
                setattr(alert, key, value)
        
        return self._save_alert_to_file(alert)
    
    def get_alert(self, alert_id: str) -> Optional[AlertData]:
        """
        获取告警记录
        
        Args:
            alert_id: 告警ID
        
        Returns:
            告警数据，不存在返回None
        """
        with self._lock:
            if alert_id in self._cache:
                return self._cache[alert_id]
            
            try:
                if self._use_postgresql:
                    return self._get_alert_from_db(alert_id)
                else:
                    return self._get_alert_from_file(alert_id)
            except Exception as e:
                logger.error(f"获取告警记录失败: {e}")
                return None
    
    def _get_alert_from_db(self, alert_id: str) -> Optional[AlertData]:
        """从数据库获取告警记录"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT alert_id, alert_type, alert_level, title, message,
                               status, source, rule_id, acknowledged_by, acknowledged_at,
                               resolved_by, resolved_at, details, created_at, metadata
                        FROM risk_alerts WHERE alert_id = %s
                    """, (alert_id,))
                    row = cur.fetchone()
                    if row:
                        return AlertData(
                            alert_id=row[0], alert_type=row[1], alert_level=row[2],
                            title=row[3], message=row[4], status=row[5], source=row[6],
                            rule_id=row[7], acknowledged_by=row[8], acknowledged_at=row[9],
                            resolved_by=row[10], resolved_at=row[11],
                            details=row[12] if isinstance(row[12], dict) else json.loads(row[12] or '{}'),
                            created_at=row[13],
                            metadata=row[14] if isinstance(row[14], dict) else json.loads(row[14] or '{}')
                        )
                    return None
        
        result = _retry_db_operation(_do_get)
        if result:
            self._update_cache(result)
        return result
    
    def _get_alert_from_file(self, alert_id: str) -> Optional[AlertData]:
        """从文件获取告警记录"""
        file_path = self._storage_dir / f"{alert_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('acknowledged_at'):
            data['acknowledged_at'] = datetime.fromisoformat(data['acknowledged_at'])
        if data.get('resolved_at'):
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        
        alert = AlertData(**data)
        self._update_cache(alert)
        return alert
    
    def get_active_alerts(self, limit: int = 100) -> List[AlertData]:
        """
        获取活跃告警
        
        Args:
            limit: 返回数量限制
        
        Returns:
            活跃告警列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_active_alerts_from_db(limit)
                else:
                    return self._get_active_alerts_from_files(limit)
            except Exception as e:
                logger.error(f"获取活跃告警失败: {e}")
                return []
    
    def _get_active_alerts_from_db(self, limit: int) -> List[AlertData]:
        """从数据库获取活跃告警"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT alert_id, alert_type, alert_level, title, message,
                               status, source, rule_id, acknowledged_by, acknowledged_at,
                               resolved_by, resolved_at, details, created_at, metadata
                        FROM risk_alerts 
                        WHERE status = 'active'
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (limit,))
                    rows = cur.fetchall()
                    return [
                        AlertData(
                            alert_id=row[0], alert_type=row[1], alert_level=row[2],
                            title=row[3], message=row[4], status=row[5], source=row[6],
                            rule_id=row[7], acknowledged_by=row[8], acknowledged_at=row[9],
                            resolved_by=row[10], resolved_at=row[11],
                            details=row[12] if isinstance(row[12], dict) else json.loads(row[12] or '{}'),
                            created_at=row[13],
                            metadata=row[14] if isinstance(row[14], dict) else json.loads(row[14] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_active_alerts_from_files(self, limit: int) -> List[AlertData]:
        """从文件获取活跃告警"""
        alerts = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                alert = self._get_alert_from_file(file_path.stem)
                if alert and alert.status == 'active':
                    alerts.append(alert)
                    if len(alerts) >= limit:
                        break
            except Exception:
                continue
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        return alerts[:limit]
    
    def get_alerts_by_level(self, level: str, limit: int = 100) -> List[AlertData]:
        """
        按级别获取告警
        
        Args:
            level: 告警级别
            limit: 返回数量限制
        
        Returns:
            告警列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_alerts_by_level_from_db(level, limit)
                else:
                    return self._get_alerts_by_level_from_files(level, limit)
            except Exception as e:
                logger.error(f"按级别获取告警失败: {e}")
                return []
    
    def _get_alerts_by_level_from_db(self, level: str, limit: int) -> List[AlertData]:
        """从数据库按级别获取告警"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT alert_id, alert_type, alert_level, title, message,
                               status, source, rule_id, acknowledged_by, acknowledged_at,
                               resolved_by, resolved_at, details, created_at, metadata
                        FROM risk_alerts 
                        WHERE alert_level = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (level, limit))
                    rows = cur.fetchall()
                    return [
                        AlertData(
                            alert_id=row[0], alert_type=row[1], alert_level=row[2],
                            title=row[3], message=row[4], status=row[5], source=row[6],
                            rule_id=row[7], acknowledged_by=row[8], acknowledged_at=row[9],
                            resolved_by=row[10], resolved_at=row[11],
                            details=row[12] if isinstance(row[12], dict) else json.loads(row[12] or '{}'),
                            created_at=row[13],
                            metadata=row[14] if isinstance(row[14], dict) else json.loads(row[14] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_alerts_by_level_from_files(self, level: str, limit: int) -> List[AlertData]:
        """从文件按级别获取告警"""
        alerts = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                alert = self._get_alert_from_file(file_path.stem)
                if alert and alert.alert_level == level:
                    alerts.append(alert)
                    if len(alerts) >= limit:
                        break
            except Exception:
                continue
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        return alerts[:limit]
    
    def _update_cache(self, alert: AlertData):
        """更新缓存"""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[alert.alert_id] = alert


class RiskMetricPersistence(IRiskMetricPersistence):
    """
    风险指标持久化管理器
    
    实现PostgreSQL优先存储策略，支持自动降级到文件系统。
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化风险指标持久化管理器
        
        Args:
            storage_dir: 文件存储目录（降级时使用）
        """
        self._lock = threading.RLock()
        self._db_config = _get_postgresql_config()
        self._use_postgresql = self._test_db_connection()
        
        self._storage_dir = Path(storage_dir or os.path.join(
            os.getcwd(), "data", "risk", "metrics"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, RiskMetricData] = {}
        self._max_cache_size = 10000
        
        if self._use_postgresql:
            logger.info("RiskMetricPersistence: 使用PostgreSQL存储")
        else:
            logger.warning("RiskMetricPersistence: PostgreSQL不可用，降级到文件系统存储")
    
    def _test_db_connection(self) -> bool:
        """测试数据库连接"""
        if not self._db_config or not self._db_config.get("password"):
            return False
        try:
            import psycopg2
            conn = psycopg2.connect(**self._db_config)
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"数据库连接测试失败: {e}")
            return False
    
    def _get_connection(self):
        """获取数据库连接"""
        import psycopg2
        return psycopg2.connect(**self._db_config)
    
    def save_metric(self, metric: RiskMetricData) -> bool:
        """
        保存风险指标
        
        Args:
            metric: 风险指标数据
        
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._save_metric_to_db(metric)
                else:
                    return self._save_metric_to_file(metric)
            except Exception as e:
                logger.error(f"保存风险指标失败: {e}")
                return False
    
    def _save_metric_to_db(self, metric: RiskMetricData) -> bool:
        """保存风险指标到PostgreSQL"""
        def _do_save():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO risk_metrics (
                            metric_id, metric_name, metric_type, value,
                            threshold_low, threshold_medium, threshold_high, risk_level,
                            symbol, portfolio_id, calculation_method, confidence_level,
                            created_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        metric.metric_id, metric.metric_name, metric.metric_type,
                        metric.value, metric.threshold_low, metric.threshold_medium,
                        metric.threshold_high, metric.risk_level, metric.symbol,
                        metric.portfolio_id, metric.calculation_method, metric.confidence_level,
                        metric.created_at, json.dumps(metric.metadata)
                    ))
                    conn.commit()
            return True
        
        try:
            return _retry_db_operation(_do_save)
        except Exception as e:
            logger.error(f"数据库保存风险指标失败: {e}")
            self._use_postgresql = False
            return self._save_metric_to_file(metric)
    
    def _save_metric_to_file(self, metric: RiskMetricData) -> bool:
        """保存风险指标到文件系统"""
        try:
            file_path = self._storage_dir / f"{metric.metric_id}.json"
            data = asdict(metric)
            data['created_at'] = data['created_at'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._update_cache(metric)
            return True
        except Exception as e:
            logger.error(f"文件保存风险指标失败: {e}")
            return False
    
    def get_metric(self, metric_id: str) -> Optional[RiskMetricData]:
        """
        获取风险指标
        
        Args:
            metric_id: 指标ID
        
        Returns:
            风险指标数据，不存在返回None
        """
        with self._lock:
            if metric_id in self._cache:
                return self._cache[metric_id]
            
            try:
                if self._use_postgresql:
                    return self._get_metric_from_db(metric_id)
                else:
                    return self._get_metric_from_file(metric_id)
            except Exception as e:
                logger.error(f"获取风险指标失败: {e}")
                return None
    
    def _get_metric_from_db(self, metric_id: str) -> Optional[RiskMetricData]:
        """从数据库获取风险指标"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT metric_id, metric_name, metric_type, value,
                               threshold_low, threshold_medium, threshold_high, risk_level,
                               symbol, portfolio_id, calculation_method, confidence_level,
                               created_at, metadata
                        FROM risk_metrics WHERE metric_id = %s
                    """, (metric_id,))
                    row = cur.fetchone()
                    if row:
                        return RiskMetricData(
                            metric_id=row[0], metric_name=row[1], metric_type=row[2],
                            value=float(row[3]), threshold_low=float(row[4]),
                            threshold_medium=float(row[5]), threshold_high=float(row[6]),
                            risk_level=row[7], symbol=row[8], portfolio_id=row[9],
                            calculation_method=row[10], confidence_level=float(row[11]) if row[11] else None,
                            created_at=row[12],
                            metadata=row[13] if isinstance(row[13], dict) else json.loads(row[13] or '{}')
                        )
                    return None
        
        result = _retry_db_operation(_do_get)
        if result:
            self._update_cache(result)
        return result
    
    def _get_metric_from_file(self, metric_id: str) -> Optional[RiskMetricData]:
        """从文件获取风险指标"""
        file_path = self._storage_dir / f"{metric_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        metric = RiskMetricData(**data)
        self._update_cache(metric)
        return metric
    
    def get_metrics_by_name(self, metric_name: str, limit: int = 100) -> List[RiskMetricData]:
        """
        按名称获取指标历史
        
        Args:
            metric_name: 指标名称
            limit: 返回数量限制
        
        Returns:
            风险指标列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_metrics_by_name_from_db(metric_name, limit)
                else:
                    return self._get_metrics_by_name_from_files(metric_name, limit)
            except Exception as e:
                logger.error(f"按名称获取指标历史失败: {e}")
                return []
    
    def _get_metrics_by_name_from_db(self, metric_name: str, limit: int) -> List[RiskMetricData]:
        """从数据库按名称获取指标历史"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT metric_id, metric_name, metric_type, value,
                               threshold_low, threshold_medium, threshold_high, risk_level,
                               symbol, portfolio_id, calculation_method, confidence_level,
                               created_at, metadata
                        FROM risk_metrics 
                        WHERE metric_name = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (metric_name, limit))
                    rows = cur.fetchall()
                    return [
                        RiskMetricData(
                            metric_id=row[0], metric_name=row[1], metric_type=row[2],
                            value=float(row[3]), threshold_low=float(row[4]),
                            threshold_medium=float(row[5]), threshold_high=float(row[6]),
                            risk_level=row[7], symbol=row[8], portfolio_id=row[9],
                            calculation_method=row[10], confidence_level=float(row[11]) if row[11] else None,
                            created_at=row[12],
                            metadata=row[13] if isinstance(row[13], dict) else json.loads(row[13] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_metrics_by_name_from_files(self, metric_name: str, limit: int) -> List[RiskMetricData]:
        """从文件按名称获取指标历史"""
        metrics = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                metric = self._get_metric_from_file(file_path.stem)
                if metric and metric.metric_name == metric_name:
                    metrics.append(metric)
            except Exception:
                continue
        metrics.sort(key=lambda x: x.created_at, reverse=True)
        return metrics[:limit]
    
    def get_latest_metrics(self, limit: int = 100) -> List[RiskMetricData]:
        """
        获取最新指标
        
        Args:
            limit: 返回数量限制
        
        Returns:
            风险指标列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_latest_metrics_from_db(limit)
                else:
                    return self._get_latest_metrics_from_files(limit)
            except Exception as e:
                logger.error(f"获取最新指标失败: {e}")
                return []
    
    def _get_latest_metrics_from_db(self, limit: int) -> List[RiskMetricData]:
        """从数据库获取最新指标"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT ON (metric_name) metric_id, metric_name, metric_type, value,
                               threshold_low, threshold_medium, threshold_high, risk_level,
                               symbol, portfolio_id, calculation_method, confidence_level,
                               created_at, metadata
                        FROM risk_metrics 
                        ORDER BY metric_name, created_at DESC
                        LIMIT %s
                    """, (limit,))
                    rows = cur.fetchall()
                    return [
                        RiskMetricData(
                            metric_id=row[0], metric_name=row[1], metric_type=row[2],
                            value=float(row[3]), threshold_low=float(row[4]),
                            threshold_medium=float(row[5]), threshold_high=float(row[6]),
                            risk_level=row[7], symbol=row[8], portfolio_id=row[9],
                            calculation_method=row[10], confidence_level=float(row[11]) if row[11] else None,
                            created_at=row[12],
                            metadata=row[13] if isinstance(row[13], dict) else json.loads(row[13] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_latest_metrics_from_files(self, limit: int) -> List[RiskMetricData]:
        """从文件获取最新指标"""
        metrics = []
        seen_names = set()
        
        all_metrics = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                metric = self._get_metric_from_file(file_path.stem)
                if metric:
                    all_metrics.append(metric)
            except Exception:
                continue
        
        all_metrics.sort(key=lambda x: x.created_at, reverse=True)
        
        for metric in all_metrics:
            if metric.metric_name not in seen_names:
                metrics.append(metric)
                seen_names.add(metric.metric_name)
                if len(metrics) >= limit:
                    break
        
        return metrics
    
    def _update_cache(self, metric: RiskMetricData):
        """更新缓存"""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[metric.metric_id] = metric


class RiskRulePersistence(IRiskRulePersistence):
    """
    风险规则持久化管理器
    
    实现PostgreSQL优先存储策略，支持自动降级到文件系统。
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化风险规则持久化管理器
        
        Args:
            storage_dir: 文件存储目录（降级时使用）
        """
        self._lock = threading.RLock()
        self._db_config = _get_postgresql_config()
        self._use_postgresql = self._test_db_connection()
        
        self._storage_dir = Path(storage_dir or os.path.join(
            os.getcwd(), "data", "risk", "rules"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, RiskRuleData] = {}
        self._max_cache_size = 1000
        
        if self._use_postgresql:
            logger.info("RiskRulePersistence: 使用PostgreSQL存储")
        else:
            logger.warning("RiskRulePersistence: PostgreSQL不可用，降级到文件系统存储")
    
    def _test_db_connection(self) -> bool:
        """测试数据库连接"""
        if not self._db_config or not self._db_config.get("password"):
            return False
        try:
            import psycopg2
            conn = psycopg2.connect(**self._db_config)
            conn.close()
            return True
        except Exception as e:
            logger.warning(f"数据库连接测试失败: {e}")
            return False
    
    def _get_connection(self):
        """获取数据库连接"""
        import psycopg2
        return psycopg2.connect(**self._db_config)
    
    def save_rule(self, rule: RiskRuleData) -> bool:
        """
        保存风险规则
        
        Args:
            rule: 风险规则数据
        
        Returns:
            是否保存成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._save_rule_to_db(rule)
                else:
                    return self._save_rule_to_file(rule)
            except Exception as e:
                logger.error(f"保存风险规则失败: {e}")
                return False
    
    def _save_rule_to_db(self, rule: RiskRuleData) -> bool:
        """保存风险规则到PostgreSQL"""
        def _do_save():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO risk_rules (
                            rule_id, rule_name, rule_type, risk_type, conditions,
                            actions, alert_level, enabled, cooldown_minutes, priority,
                            description, created_at, updated_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (rule_id) DO UPDATE SET
                            rule_name = EXCLUDED.rule_name,
                            conditions = EXCLUDED.conditions,
                            actions = EXCLUDED.actions,
                            alert_level = EXCLUDED.alert_level,
                            enabled = EXCLUDED.enabled,
                            cooldown_minutes = EXCLUDED.cooldown_minutes,
                            priority = EXCLUDED.priority,
                            description = EXCLUDED.description,
                            updated_at = EXCLUDED.updated_at,
                            metadata = EXCLUDED.metadata
                    """, (
                        rule.rule_id, rule.rule_name, rule.rule_type, rule.risk_type,
                        json.dumps(rule.conditions), json.dumps(rule.actions),
                        rule.alert_level, rule.enabled, rule.cooldown_minutes, rule.priority,
                        rule.description, rule.created_at, rule.updated_at, json.dumps(rule.metadata)
                    ))
                    conn.commit()
            return True
        
        try:
            return _retry_db_operation(_do_save)
        except Exception as e:
            logger.error(f"数据库保存风险规则失败: {e}")
            self._use_postgresql = False
            return self._save_rule_to_file(rule)
    
    def _save_rule_to_file(self, rule: RiskRuleData) -> bool:
        """保存风险规则到文件系统"""
        try:
            file_path = self._storage_dir / f"{rule.rule_id}.json"
            data = asdict(rule)
            data['created_at'] = data['created_at'].isoformat()
            data['updated_at'] = data['updated_at'].isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._update_cache(rule)
            return True
        except Exception as e:
            logger.error(f"文件保存风险规则失败: {e}")
            return False
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新风险规则
        
        Args:
            rule_id: 规则ID
            updates: 更新字段字典
        
        Returns:
            是否更新成功
        """
        with self._lock:
            updates['updated_at'] = datetime.now()
            
            try:
                if self._use_postgresql:
                    return self._update_rule_in_db(rule_id, updates)
                else:
                    return self._update_rule_in_file(rule_id, updates)
            except Exception as e:
                logger.error(f"更新风险规则失败: {e}")
                return False
    
    def _update_rule_in_db(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """在数据库中更新风险规则"""
        def _do_update():
            set_clauses = []
            values = []
            for key, value in updates.items():
                if key in ['conditions', 'actions', 'metadata']:
                    set_clauses.append(f"{key} = %s")
                    values.append(json.dumps(value))
                else:
                    set_clauses.append(f"{key} = %s")
                    values.append(value)
            values.append(rule_id)
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        UPDATE risk_rules 
                        SET {', '.join(set_clauses)}
                        WHERE rule_id = %s
                    """, values)
                    conn.commit()
            return True
        
        return _retry_db_operation(_do_update)
    
    def _update_rule_in_file(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """在文件中更新风险规则"""
        rule = self.get_rule(rule_id)
        if not rule:
            return False
        
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        return self._save_rule_to_file(rule)
    
    def get_rule(self, rule_id: str) -> Optional[RiskRuleData]:
        """
        获取风险规则
        
        Args:
            rule_id: 规则ID
        
        Returns:
            风险规则数据，不存在返回None
        """
        with self._lock:
            if rule_id in self._cache:
                return self._cache[rule_id]
            
            try:
                if self._use_postgresql:
                    return self._get_rule_from_db(rule_id)
                else:
                    return self._get_rule_from_file(rule_id)
            except Exception as e:
                logger.error(f"获取风险规则失败: {e}")
                return None
    
    def _get_rule_from_db(self, rule_id: str) -> Optional[RiskRuleData]:
        """从数据库获取风险规则"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT rule_id, rule_name, rule_type, risk_type, conditions,
                               actions, alert_level, enabled, cooldown_minutes, priority,
                               description, created_at, updated_at, metadata
                        FROM risk_rules WHERE rule_id = %s
                    """, (rule_id,))
                    row = cur.fetchone()
                    if row:
                        return RiskRuleData(
                            rule_id=row[0], rule_name=row[1], rule_type=row[2],
                            risk_type=row[3],
                            conditions=row[4] if isinstance(row[4], dict) else json.loads(row[4] or '{}'),
                            actions=row[5] if isinstance(row[5], list) else json.loads(row[5] or '[]'),
                            alert_level=row[6], enabled=row[7], cooldown_minutes=row[8],
                            priority=row[9], description=row[10],
                            created_at=row[11], updated_at=row[12],
                            metadata=row[13] if isinstance(row[13], dict) else json.loads(row[13] or '{}')
                        )
                    return None
        
        result = _retry_db_operation(_do_get)
        if result:
            self._update_cache(result)
        return result
    
    def _get_rule_from_file(self, rule_id: str) -> Optional[RiskRuleData]:
        """从文件获取风险规则"""
        file_path = self._storage_dir / f"{rule_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        rule = RiskRuleData(**data)
        self._update_cache(rule)
        return rule
    
    def get_all_rules(self) -> List[RiskRuleData]:
        """
        获取所有规则
        
        Returns:
            风险规则列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_all_rules_from_db()
                else:
                    return self._get_all_rules_from_files()
            except Exception as e:
                logger.error(f"获取所有规则失败: {e}")
                return []
    
    def _get_all_rules_from_db(self) -> List[RiskRuleData]:
        """从数据库获取所有规则"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT rule_id, rule_name, rule_type, risk_type, conditions,
                               actions, alert_level, enabled, cooldown_minutes, priority,
                               description, created_at, updated_at, metadata
                        FROM risk_rules
                        ORDER BY priority, rule_id
                    """)
                    rows = cur.fetchall()
                    return [
                        RiskRuleData(
                            rule_id=row[0], rule_name=row[1], rule_type=row[2],
                            risk_type=row[3],
                            conditions=row[4] if isinstance(row[4], dict) else json.loads(row[4] or '{}'),
                            actions=row[5] if isinstance(row[5], list) else json.loads(row[5] or '[]'),
                            alert_level=row[6], enabled=row[7], cooldown_minutes=row[8],
                            priority=row[9], description=row[10],
                            created_at=row[11], updated_at=row[12],
                            metadata=row[13] if isinstance(row[13], dict) else json.loads(row[13] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_all_rules_from_files(self) -> List[RiskRuleData]:
        """从文件获取所有规则"""
        rules = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                rule = self._get_rule_from_file(file_path.stem)
                if rule:
                    rules.append(rule)
            except Exception:
                continue
        rules.sort(key=lambda x: (x.priority, x.rule_id))
        return rules
    
    def get_enabled_rules(self) -> List[RiskRuleData]:
        """
        获取启用的规则
        
        Returns:
            启用的风险规则列表
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._get_enabled_rules_from_db()
                else:
                    return self._get_enabled_rules_from_files()
            except Exception as e:
                logger.error(f"获取启用的规则失败: {e}")
                return []
    
    def _get_enabled_rules_from_db(self) -> List[RiskRuleData]:
        """从数据库获取启用的规则"""
        def _do_get():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT rule_id, rule_name, rule_type, risk_type, conditions,
                               actions, alert_level, enabled, cooldown_minutes, priority,
                               description, created_at, updated_at, metadata
                        FROM risk_rules
                        WHERE enabled = TRUE
                        ORDER BY priority, rule_id
                    """)
                    rows = cur.fetchall()
                    return [
                        RiskRuleData(
                            rule_id=row[0], rule_name=row[1], rule_type=row[2],
                            risk_type=row[3],
                            conditions=row[4] if isinstance(row[4], dict) else json.loads(row[4] or '{}'),
                            actions=row[5] if isinstance(row[5], list) else json.loads(row[5] or '[]'),
                            alert_level=row[6], enabled=row[7], cooldown_minutes=row[8],
                            priority=row[9], description=row[10],
                            created_at=row[11], updated_at=row[12],
                            metadata=row[13] if isinstance(row[13], dict) else json.loads(row[13] or '{}')
                        ) for row in rows
                    ]
        
        return _retry_db_operation(_do_get)
    
    def _get_enabled_rules_from_files(self) -> List[RiskRuleData]:
        """从文件获取启用的规则"""
        rules = []
        for file_path in self._storage_dir.glob("*.json"):
            try:
                rule = self._get_rule_from_file(file_path.stem)
                if rule and rule.enabled:
                    rules.append(rule)
            except Exception:
                continue
        rules.sort(key=lambda x: (x.priority, x.rule_id))
        return rules
    
    def delete_rule(self, rule_id: str) -> bool:
        """
        删除规则
        
        Args:
            rule_id: 规则ID
        
        Returns:
            是否删除成功
        """
        with self._lock:
            try:
                if self._use_postgresql:
                    return self._delete_rule_from_db(rule_id)
                else:
                    return self._delete_rule_from_file(rule_id)
            except Exception as e:
                logger.error(f"删除规则失败: {e}")
                return False
    
    def _delete_rule_from_db(self, rule_id: str) -> bool:
        """从数据库删除规则"""
        def _do_delete():
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM risk_rules WHERE rule_id = %s", (rule_id,))
                    conn.commit()
            return True
        
        return _retry_db_operation(_do_delete)
    
    def _delete_rule_from_file(self, rule_id: str) -> bool:
        """从文件删除规则"""
        file_path = self._storage_dir / f"{rule_id}.json"
        if file_path.exists():
            file_path.unlink()
        if rule_id in self._cache:
            del self._cache[rule_id]
        return True
    
    def _update_cache(self, rule: RiskRuleData):
        """更新缓存"""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[rule.rule_id] = rule
