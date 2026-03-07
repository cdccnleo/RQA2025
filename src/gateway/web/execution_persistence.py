"""
策略执行状态持久化模块
存储策略执行状态到文件系统或PostgreSQL
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
EXECUTION_STATES_DIR = os.path.join(DATA_DIR, "execution_states")


def ensure_directories():
    """确保数据目录存在"""
    os.makedirs(EXECUTION_STATES_DIR, exist_ok=True)


def save_execution_state(strategy_id: str, state: Dict[str, Any]) -> bool:
    """
    保存策略执行状态到持久化存储
    优先使用PostgreSQL存储，失败时回退到文件系统
    
    Args:
        strategy_id: 策略ID
        state: 执行状态字典
    
    Returns:
        是否成功保存
    """
    try:
        if not strategy_id:
            logger.error("策略ID为空，无法保存执行状态")
            return False
        
        # 添加保存时间戳
        state_data = state.copy()
        state_data["strategy_id"] = strategy_id
        state_data["saved_at"] = time.time()
        state_data["updated_at"] = time.time()
        
        # 优先保存到PostgreSQL
        try:
            if _save_to_postgresql(state_data):
                logger.info(f"策略执行状态已保存到PostgreSQL: {strategy_id}")
                return True
        except Exception as e:
            logger.warning(f"保存到PostgreSQL失败，回退到文件系统: {e}")
        
        # 回退到文件系统
        ensure_directories()
        filepath = os.path.join(EXECUTION_STATES_DIR, f"{strategy_id}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"策略执行状态已保存到文件系统: {strategy_id}")
        return True
    except Exception as e:
        logger.error(f"保存执行状态失败: {e}")
        return False


def _save_to_postgresql(state: Dict[str, Any]) -> bool:
    """尝试保存执行状态到PostgreSQL"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # 确保表存在
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_execution_states (
                strategy_id VARCHAR(100) PRIMARY KEY,
                status VARCHAR(20) NOT NULL,
                latency DECIMAL(10, 2),
                throughput DECIMAL(10, 2),
                signals_count INTEGER DEFAULT 0,
                positions_count INTEGER DEFAULT 0,
                metrics JSONB,
                saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 插入或更新执行状态
        cursor.execute("""
            INSERT INTO strategy_execution_states (
                strategy_id, status, latency, throughput, 
                signals_count, positions_count, metrics, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (strategy_id) 
            DO UPDATE SET
                status = EXCLUDED.status,
                latency = EXCLUDED.latency,
                throughput = EXCLUDED.throughput,
                signals_count = EXCLUDED.signals_count,
                positions_count = EXCLUDED.positions_count,
                metrics = EXCLUDED.metrics,
                updated_at = CURRENT_TIMESTAMP
        """, (
            state.get("strategy_id"),
            state.get("status", "unknown"),
            state.get("latency"),
            state.get("throughput"),
            state.get("signals_count", 0),
            state.get("positions_count", 0),
            json.dumps(state.get("metrics", {}))
        ))
        
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        logger.debug(f"执行状态已保存到PostgreSQL: {state.get('strategy_id')}")
        return True
    except Exception as e:
        logger.debug(f"保存到PostgreSQL失败: {e}")
        if conn:
            conn.rollback()
            return_db_connection(conn)
        return False


def load_execution_state(strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    从持久化存储加载策略执行状态
    
    Args:
        strategy_id: 策略ID
    
    Returns:
        执行状态字典，如果不存在则返回None
    """
    try:
        # 优先从PostgreSQL加载
        try:
            state = _load_from_postgresql(strategy_id)
            if state:
                return state
        except Exception as e:
            logger.debug(f"从PostgreSQL加载失败: {e}")
        
        # 从文件系统加载
        filepath = os.path.join(EXECUTION_STATES_DIR, f"{strategy_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    except Exception as e:
        logger.error(f"加载执行状态失败: {e}")
        return None


def _load_from_postgresql(strategy_id: str) -> Optional[Dict[str, Any]]:
    """从PostgreSQL加载执行状态"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT strategy_id, status, latency, throughput, 
                   signals_count, positions_count, metrics, updated_at
            FROM strategy_execution_states
            WHERE strategy_id = %s
        """, (strategy_id,))
        
        row = cursor.fetchone()
        cursor.close()
        return_db_connection(conn)
        
        if row:
            return {
                "strategy_id": row[0],
                "status": row[1],
                "latency": float(row[2]) if row[2] else None,
                "throughput": float(row[3]) if row[3] else None,
                "signals_count": row[4] or 0,
                "positions_count": row[5] or 0,
                "metrics": json.loads(row[6]) if row[6] else {},
                "updated_at": row[7].isoformat() if row[7] else None
            }
        
        return None
    except Exception as e:
        logger.debug(f"从PostgreSQL加载失败: {e}")
        if conn:
            return_db_connection(conn)
        return None


def list_execution_states(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    列出执行状态
    优先从PostgreSQL加载，失败时回退到文件系统
    
    Args:
        status: 状态过滤器
        limit: 返回数量限制
    
    Returns:
        执行状态列表
    """
    try:
        # 优先从PostgreSQL加载
        try:
            states = _list_from_postgresql(status, limit)
            if states:
                logger.info(f"从PostgreSQL加载执行状态成功，返回 {len(states)} 条记录")
                return states
        except Exception as e:
            logger.warning(f"从PostgreSQL加载失败，回退到文件系统: {e}")
        
        # 从文件系统加载
        states = []
        if os.path.exists(EXECUTION_STATES_DIR):
            for filename in os.listdir(EXECUTION_STATES_DIR):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(EXECUTION_STATES_DIR, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            state = json.load(f)
                            if not status or state.get("status") == status:
                                states.append(state)
                    except Exception as e:
                        logger.debug(f"加载文件失败 {filename}: {e}")
        
        # 按更新时间排序
        states.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        logger.info(f"从文件系统加载执行状态成功，返回 {min(len(states), limit)} 条记录")
        return states[:limit]
    except Exception as e:
        logger.error(f"列出执行状态失败: {e}")
        return []


def _list_from_postgresql(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """从PostgreSQL列出执行状态"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT strategy_id, status, latency, throughput, 
                       signals_count, positions_count, metrics, updated_at
                FROM strategy_execution_states
                WHERE status = %s
                ORDER BY updated_at DESC
                LIMIT %s
            """, (status, limit))
        else:
            cursor.execute("""
                SELECT strategy_id, status, latency, throughput, 
                       signals_count, positions_count, metrics, updated_at
                FROM strategy_execution_states
                ORDER BY updated_at DESC
                LIMIT %s
            """, (limit,))
        
        rows = cursor.fetchall()
        cursor.close()
        return_db_connection(conn)
        
        states = []
        for row in rows:
            states.append({
                "strategy_id": row[0],
                "status": row[1],
                "latency": float(row[2]) if row[2] else None,
                "throughput": float(row[3]) if row[3] else None,
                "signals_count": row[4] or 0,
                "positions_count": row[5] or 0,
                "metrics": json.loads(row[6]) if row[6] else {},
                "updated_at": row[7].isoformat() if row[7] else None
            })
        
        return states
    except Exception as e:
        logger.debug(f"从PostgreSQL列出失败: {e}")
        if conn:
            return_db_connection(conn)
        return []


def update_execution_state(strategy_id: str, updates: Dict[str, Any]) -> bool:
    """
    更新执行状态
    
    Args:
        strategy_id: 策略ID
        updates: 更新数据
    
    Returns:
        是否成功更新
    """
    try:
        # 加载现有状态
        state = load_execution_state(strategy_id)
        if not state:
            # 如果不存在，创建新状态
            state = {"strategy_id": strategy_id}
        
        # 更新状态
        state.update(updates)
        state["updated_at"] = time.time()
        
        # 保存
        return save_execution_state(strategy_id, state)
    except Exception as e:
        logger.error(f"更新执行状态失败: {e}")
        return False


def delete_execution_state(strategy_id: str) -> bool:
    """
    删除执行状态
    
    Args:
        strategy_id: 策略ID
    
    Returns:
        是否成功删除
    """
    try:
        # 从文件系统删除
        filepath = os.path.join(EXECUTION_STATES_DIR, f"{strategy_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # 从PostgreSQL删除
        try:
            _delete_from_postgresql(strategy_id)
        except Exception as e:
            logger.debug(f"从PostgreSQL删除失败: {e}")
        
        logger.info(f"执行状态已删除: {strategy_id}")
        return True
    except Exception as e:
        logger.error(f"删除执行状态失败: {e}")
        return False


def _delete_from_postgresql(strategy_id: str) -> bool:
    """从PostgreSQL删除执行状态"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        cursor.execute("DELETE FROM strategy_execution_states WHERE strategy_id = %s", (strategy_id,))
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        return True
    except Exception as e:
        logger.debug(f"从PostgreSQL删除失败: {e}")
        if conn:
            conn.rollback()
            return_db_connection(conn)
        return False

