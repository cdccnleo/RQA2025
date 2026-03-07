"""
交易执行记录持久化模块
存储交易执行流程数据到文件系统或PostgreSQL
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
TRADING_EXECUTION_DIR = os.path.join(DATA_DIR, "trading_execution")


def ensure_directories():
    """确保数据目录存在"""
    os.makedirs(TRADING_EXECUTION_DIR, exist_ok=True)


def save_execution_record(record: Dict[str, Any]) -> bool:
    """
    保存交易执行记录到持久化存储
    
    Args:
        record: 执行记录字典，必须包含record_id或timestamp字段
    
    Returns:
        是否成功保存
    """
    try:
        ensure_directories()
        
        record_id = record.get("record_id") or record.get("id") or f"exec_{int(time.time())}"
        
        filepath = os.path.join(TRADING_EXECUTION_DIR, f"{record_id}.json")
        
        # 添加保存时间戳
        record_data = record.copy()
        record_data["record_id"] = record_id
        record_data["saved_at"] = time.time()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"交易执行记录已保存: {record_id}")
        
        # 同时尝试保存到PostgreSQL（如果可用）
        try:
            _save_to_postgresql(record_data)
        except Exception as e:
            logger.debug(f"保存到PostgreSQL失败（使用文件系统）: {e}")
        
        return True
    except Exception as e:
        logger.error(f"保存执行记录失败: {e}")
        return False


def _save_to_postgresql(record: Dict[str, Any]) -> bool:
    """尝试保存执行记录到PostgreSQL"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # 确保表存在
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_execution_records (
                record_id VARCHAR(100) PRIMARY KEY,
                record_type VARCHAR(50) NOT NULL,
                market_monitoring JSONB,
                signal_generation JSONB,
                risk_check JSONB,
                order_generation JSONB,
                order_routing JSONB,
                execution JSONB,
                position_management JSONB,
                result_feedback JSONB,
                timestamp BIGINT NOT NULL,
                saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 创建索引
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_execution_timestamp ON trading_execution_records(timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_execution_type ON trading_execution_records(record_type);")
        except Exception as e:
            logger.debug(f"创建索引可能已存在: {e}")
        
        # 插入或更新执行记录
        cursor.execute("""
            INSERT INTO trading_execution_records (
                record_id, record_type, market_monitoring, signal_generation,
                risk_check, order_generation, order_routing, execution,
                position_management, result_feedback, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (record_id) 
            DO UPDATE SET
                market_monitoring = EXCLUDED.market_monitoring,
                signal_generation = EXCLUDED.signal_generation,
                risk_check = EXCLUDED.risk_check,
                order_generation = EXCLUDED.order_generation,
                order_routing = EXCLUDED.order_routing,
                execution = EXCLUDED.execution,
                position_management = EXCLUDED.position_management,
                result_feedback = EXCLUDED.result_feedback
        """, (
            record.get("record_id"),
            record.get("record_type", "flow_monitor"),
            json.dumps(record.get("market_monitoring", {})),
            json.dumps(record.get("signal_generation", {})),
            json.dumps(record.get("risk_check", {})),
            json.dumps(record.get("order_generation", {})),
            json.dumps(record.get("order_routing", {})),
            json.dumps(record.get("execution", {})),
            json.dumps(record.get("position_management", {})),
            json.dumps(record.get("result_feedback", {})),
            record.get("timestamp", int(time.time()))
        ))
        
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        logger.debug(f"执行记录已保存到PostgreSQL: {record.get('record_id')}")
        return True
    except Exception as e:
        logger.debug(f"保存到PostgreSQL失败: {e}")
        if conn:
            conn.rollback()
            return_db_connection(conn)
        return False


def load_execution_record(record_id: str) -> Optional[Dict[str, Any]]:
    """
    从持久化存储加载交易执行记录
    
    Args:
        record_id: 记录ID
    
    Returns:
        执行记录字典，如果不存在则返回None
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
        filepath = os.path.join(TRADING_EXECUTION_DIR, f"{record_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    except Exception as e:
        logger.error(f"加载执行记录失败: {e}")
        return None


def _load_from_postgresql(record_id: str) -> Optional[Dict[str, Any]]:
    """从PostgreSQL加载执行记录"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT record_id, record_type, market_monitoring, signal_generation,
                   risk_check, order_generation, order_routing, execution,
                   position_management, result_feedback, timestamp
            FROM trading_execution_records
            WHERE record_id = %s
        """, (record_id,))
        
        row = cursor.fetchone()
        cursor.close()
        return_db_connection(conn)
        
        if row:
            return {
                "record_id": row[0],
                "record_type": row[1],
                "market_monitoring": json.loads(row[2]) if row[2] else {},
                "signal_generation": json.loads(row[3]) if row[3] else {},
                "risk_check": json.loads(row[4]) if row[4] else {},
                "order_generation": json.loads(row[5]) if row[5] else {},
                "order_routing": json.loads(row[6]) if row[6] else {},
                "execution": json.loads(row[7]) if row[7] else {},
                "position_management": json.loads(row[8]) if row[8] else {},
                "result_feedback": json.loads(row[9]) if row[9] else {},
                "timestamp": row[10]
            }
        
        return None
    except Exception as e:
        logger.debug(f"从PostgreSQL加载失败: {e}")
        if conn:
            return_db_connection(conn)
        return None


def list_execution_records(
    record_type: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    列出交易执行记录
    
    Args:
        record_type: 记录类型过滤器
        limit: 返回数量限制
        start_time: 开始时间戳
        end_time: 结束时间戳
    
    Returns:
        执行记录列表
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
        if os.path.exists(TRADING_EXECUTION_DIR):
            for filename in os.listdir(TRADING_EXECUTION_DIR):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(TRADING_EXECUTION_DIR, filename)
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
        
        # 按时间戳排序
        records.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return records[:limit]
    except Exception as e:
        logger.error(f"列出执行记录失败: {e}")
        return []


def _list_from_postgresql(
    record_type: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict[str, Any]]:
    """从PostgreSQL列出执行记录"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        # 构建查询
        conditions = []
        params = []
        
        if record_type:
            conditions.append("record_type = %s")
            params.append(record_type)
        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        
        cursor.execute(f"""
            SELECT record_id, record_type, market_monitoring, signal_generation,
                   risk_check, order_generation, order_routing, execution,
                   position_management, result_feedback, timestamp
            FROM trading_execution_records
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT %s
        """, params)
        
        rows = cursor.fetchall()
        cursor.close()
        return_db_connection(conn)
        
        records = []
        for row in rows:
            records.append({
                "record_id": row[0],
                "record_type": row[1],
                "market_monitoring": json.loads(row[2]) if row[2] else {},
                "signal_generation": json.loads(row[3]) if row[3] else {},
                "risk_check": json.loads(row[4]) if row[4] else {},
                "order_generation": json.loads(row[5]) if row[5] else {},
                "order_routing": json.loads(row[6]) if row[6] else {},
                "execution": json.loads(row[7]) if row[7] else {},
                "position_management": json.loads(row[8]) if row[8] else {},
                "result_feedback": json.loads(row[9]) if row[9] else {},
                "timestamp": row[10]
            })
        
        return records
    except Exception as e:
        logger.debug(f"从PostgreSQL列出失败: {e}")
        if conn:
            return_db_connection(conn)
        return []


def get_latest_execution_record() -> Optional[Dict[str, Any]]:
    """
    获取最新的交易执行记录
    
    Returns:
        最新的执行记录，如果不存在则返回None
    """
    records = list_execution_records(limit=1)
    return records[0] if records else None

