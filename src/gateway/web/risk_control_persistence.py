"""
风险控制记录持久化模块
存储风险控制流程数据到文件系统或PostgreSQL
符合架构设计：使用统一日志系统
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# 数据存储目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data")
RISK_CONTROL_DIR = os.path.join(DATA_DIR, "risk_control")


def ensure_directories():
    """确保数据目录存在"""
    os.makedirs(RISK_CONTROL_DIR, exist_ok=True)


def save_risk_control_record(record: Dict[str, Any]) -> bool:
    """
    保存风险控制记录到持久化存储
    
    Args:
        record: 风险控制记录字典，必须包含record_id或timestamp字段
    
    Returns:
        是否成功保存
    """
    try:
        ensure_directories()
        
        record_id = record.get("record_id") or record.get("id") or f"risk_{int(time.time())}"
        
        filepath = os.path.join(RISK_CONTROL_DIR, f"{record_id}.json")
        
        # 添加保存时间戳
        record_data = record.copy()
        record_data["record_id"] = record_id
        record_data["saved_at"] = time.time()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"风险控制记录已保存: {record_id}")
        
        # 同时尝试保存到PostgreSQL（如果可用）
        try:
            _save_to_postgresql(record_data)
        except Exception as e:
            logger.debug(f"保存到PostgreSQL失败（使用文件系统）: {e}")
        
        return True
    except Exception as e:
        logger.error(f"保存风险控制记录失败: {e}")
        return False


def _save_to_postgresql(record: Dict[str, Any]) -> bool:
    """尝试保存风险控制记录到PostgreSQL"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # 确保表存在（包含6个步骤的数据字段）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_control_records (
                record_id VARCHAR(100) PRIMARY KEY,
                record_type VARCHAR(50) NOT NULL,
                realtime_monitoring JSONB,
                risk_assessment JSONB,
                risk_intercept JSONB,
                compliance_check JSONB,
                risk_report JSONB,
                alert_notify JSONB,
                timestamp BIGINT NOT NULL,
                saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 创建索引
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_control_timestamp ON risk_control_records(timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_control_type ON risk_control_records(record_type);")
        except Exception as e:
            logger.debug(f"创建索引可能已存在: {e}")
        
        # 插入或更新风险控制记录
        cursor.execute("""
            INSERT INTO risk_control_records (
                record_id, record_type, realtime_monitoring, risk_assessment,
                risk_intercept, compliance_check, risk_report, alert_notify, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (record_id) 
            DO UPDATE SET
                realtime_monitoring = EXCLUDED.realtime_monitoring,
                risk_assessment = EXCLUDED.risk_assessment,
                risk_intercept = EXCLUDED.risk_intercept,
                compliance_check = EXCLUDED.compliance_check,
                risk_report = EXCLUDED.risk_report,
                alert_notify = EXCLUDED.alert_notify
        """, (
            record.get("record_id"),
            record.get("record_type", "overview_monitor"),
            json.dumps(record.get("realtime_monitoring", {})),
            json.dumps(record.get("risk_assessment", {})),
            json.dumps(record.get("risk_intercept", {})),
            json.dumps(record.get("compliance_check", {})),
            json.dumps(record.get("risk_report", {})),
            json.dumps(record.get("alert_notify", {})),
            record.get("timestamp", int(time.time()))
        ))
        
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        logger.debug(f"风险控制记录已保存到PostgreSQL: {record.get('record_id')}")
        return True
    except Exception as e:
        logger.debug(f"保存到PostgreSQL失败: {e}")
        if conn:
            conn.rollback()
            return_db_connection(conn)
        return False


def load_risk_control_record(record_id: str) -> Optional[Dict[str, Any]]:
    """
    从持久化存储加载风险控制记录
    
    Args:
        record_id: 记录ID
    
    Returns:
        风险控制记录字典，如果不存在则返回None
    """
    try:
        # 优先从PostgreSQL加载
        try:
            record = _load_from_postgresql(record_id)
            if record:
                return record
        except Exception as e:
            logger.debug(f"从PostgreSQL加载失败: {e}")
        
        # 从文件系统加载
        filepath = os.path.join(RISK_CONTROL_DIR, f"{record_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    except Exception as e:
        logger.error(f"加载风险控制记录失败: {e}")
        return None


def _load_from_postgresql(record_id: str) -> Optional[Dict[str, Any]]:
    """从PostgreSQL加载风险控制记录"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT record_id, record_type, realtime_monitoring, risk_assessment,
                   risk_intercept, compliance_check, risk_report, alert_notify, timestamp
            FROM risk_control_records
            WHERE record_id = %s
        """, (record_id,))
        
        row = cursor.fetchone()
        cursor.close()
        return_db_connection(conn)
        
        if row:
            return {
                "record_id": row[0],
                "record_type": row[1],
                "realtime_monitoring": json.loads(row[2]) if row[2] else {},
                "risk_assessment": json.loads(row[3]) if row[3] else {},
                "risk_intercept": json.loads(row[4]) if row[4] else {},
                "compliance_check": json.loads(row[5]) if row[5] else {},
                "risk_report": json.loads(row[6]) if row[6] else {},
                "alert_notify": json.loads(row[7]) if row[7] else {},
                "timestamp": row[8]
            }
        
        return None
    except Exception as e:
        logger.debug(f"从PostgreSQL加载失败: {e}")
        if conn:
            return_db_connection(conn)
        return None


def list_risk_control_records(
    record_type: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    列出风险控制记录
    
    Args:
        record_type: 记录类型过滤器
        limit: 返回数量限制
        start_time: 开始时间戳
        end_time: 结束时间戳
    
    Returns:
        风险控制记录列表
    """
    try:
        # 优先从PostgreSQL加载
        try:
            records = _list_from_postgresql(record_type, limit, start_time, end_time)
            if records:
                return records
        except Exception as e:
            logger.debug(f"从PostgreSQL加载失败: {e}")
        
        # 从文件系统加载
        records = []
        if os.path.exists(RISK_CONTROL_DIR):
            files = sorted(os.listdir(RISK_CONTROL_DIR), reverse=True)
            for filename in files[:limit]:
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(RISK_CONTROL_DIR, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            record = json.load(f)
                            
                            # 应用过滤器
                            if record_type and record.get("record_type") != record_type:
                                continue
                            if start_time and record.get("timestamp", 0) < start_time:
                                continue
                            if end_time and record.get("timestamp", 0) > end_time:
                                continue
                            
                            records.append(record)
                    except Exception as e:
                        logger.debug(f"加载文件失败 {filename}: {e}")
        
        return records
    except Exception as e:
        logger.error(f"列出风险控制记录失败: {e}")
        return []


def _list_from_postgresql(
    record_type: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict[str, Any]]:
    """从PostgreSQL列出风险控制记录"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        # 构建查询
        query = """
            SELECT record_id, record_type, realtime_monitoring, risk_assessment,
                   risk_intercept, compliance_check, risk_report, alert_notify, timestamp
            FROM risk_control_records
            WHERE 1=1
        """
        params = []
        
        if record_type:
            query += " AND record_type = %s"
            params.append(record_type)
        
        if start_time:
            query += " AND timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= %s"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        return_db_connection(conn)
        
        records = []
        for row in rows:
            records.append({
                "record_id": row[0],
                "record_type": row[1],
                "realtime_monitoring": json.loads(row[2]) if row[2] else {},
                "risk_assessment": json.loads(row[3]) if row[3] else {},
                "risk_intercept": json.loads(row[4]) if row[4] else {},
                "compliance_check": json.loads(row[5]) if row[5] else {},
                "risk_report": json.loads(row[6]) if row[6] else {},
                "alert_notify": json.loads(row[7]) if row[7] else {},
                "timestamp": row[8]
            })
        
        return records
    except Exception as e:
        logger.debug(f"从PostgreSQL加载失败: {e}")
        if conn:
            return_db_connection(conn)
        return []


def get_latest_risk_control_record() -> Optional[Dict[str, Any]]:
    """
    获取最新的风险控制记录
    
    Returns:
        最新的风险控制记录，如果不存在则返回None
    """
    records = list_risk_control_records(limit=1)
    return records[0] if records else None

