"""
交易信号持久化模块
存储交易信号到文件系统或PostgreSQL
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# 数据存储目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data")
SIGNALS_DIR = os.path.join(DATA_DIR, "trading_signals")


def ensure_directories():
    """确保数据目录存在"""
    os.makedirs(SIGNALS_DIR, exist_ok=True)


def save_signal(signal: Dict[str, Any]) -> bool:
    """
    保存交易信号到持久化存储
    
    Args:
        signal: 信号字典，必须包含id或signal_id字段
    
    Returns:
        是否成功保存
    """
    try:
        ensure_directories()
        
        signal_id = signal.get("id") or signal.get("signal_id")
        if not signal_id:
            logger.error("信号缺少id或signal_id字段，无法保存")
            return False
        
        filepath = os.path.join(SIGNALS_DIR, f"{signal_id}.json")
        
        # 添加保存时间戳
        signal_data = signal.copy()
        signal_data["id"] = signal_id
        signal_data["saved_at"] = time.time()
        
        # 确保timestamp是整数
        if "timestamp" in signal_data:
            if isinstance(signal_data["timestamp"], datetime):
                signal_data["timestamp"] = int(signal_data["timestamp"].timestamp())
            elif isinstance(signal_data["timestamp"], str):
                try:
                    dt = datetime.fromisoformat(signal_data["timestamp"].replace("Z", "+00:00"))
                    signal_data["timestamp"] = int(dt.timestamp())
                except:
                    signal_data["timestamp"] = int(time.time())
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(signal_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"交易信号已保存: {signal_id}")
        
        # 同时尝试保存到PostgreSQL（如果可用）
        try:
            _save_to_postgresql(signal_data)
        except Exception as e:
            logger.debug(f"保存到PostgreSQL失败（使用文件系统）: {e}")
        
        return True
    except Exception as e:
        logger.error(f"保存交易信号失败: {e}")
        return False


def _save_to_postgresql(signal: Dict[str, Any]) -> bool:
    """尝试保存信号到PostgreSQL"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # 确保表存在
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                signal_id VARCHAR(100) PRIMARY KEY,
                strategy_id VARCHAR(100),
                symbol VARCHAR(20),
                signal_type VARCHAR(20) NOT NULL,
                strength DECIMAL(5, 2),
                price DECIMAL(15, 6),
                status VARCHAR(20),
                timestamp BIGINT NOT NULL,
                accuracy DECIMAL(5, 4),
                latency DECIMAL(10, 2),
                quality DECIMAL(5, 4),
                saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 创建索引
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy ON trading_signals(strategy_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON trading_signals(timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON trading_signals(signal_type);")
        except Exception as e:
            logger.debug(f"创建索引可能已存在: {e}")
        
        # 插入或更新信号
        cursor.execute("""
            INSERT INTO trading_signals (
                signal_id, strategy_id, symbol, signal_type, strength, price,
                status, timestamp, accuracy, latency, quality
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (signal_id) 
            DO UPDATE SET
                status = EXCLUDED.status,
                accuracy = EXCLUDED.accuracy,
                latency = EXCLUDED.latency,
                quality = EXCLUDED.quality
        """, (
            signal.get("id") or signal.get("signal_id"),
            signal.get("strategy_id"),
            signal.get("symbol"),
            signal.get("type") or signal.get("signal_type", "unknown"),
            signal.get("strength"),
            signal.get("price"),
            signal.get("status", "pending"),
            signal.get("timestamp", int(time.time())),
            signal.get("accuracy"),
            signal.get("latency"),
            signal.get("quality")
        ))
        
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        logger.debug(f"信号已保存到PostgreSQL: {signal.get('id')}")
        return True
    except Exception as e:
        logger.debug(f"保存到PostgreSQL失败: {e}")
        if conn:
            conn.rollback()
            return_db_connection(conn)
        return False


def load_signal(signal_id: str) -> Optional[Dict[str, Any]]:
    """
    从持久化存储加载交易信号
    
    Args:
        signal_id: 信号ID
    
    Returns:
        信号字典，如果不存在则返回None
    """
    try:
        # 优先从PostgreSQL加载
        try:
            signal = _load_from_postgresql(signal_id)
            if signal:
                return signal
        except Exception as e:
            logger.debug(f"从PostgreSQL加载失败: {e}")
        
        # 从文件系统加载
        filepath = os.path.join(SIGNALS_DIR, f"{signal_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    except Exception as e:
        logger.error(f"加载交易信号失败: {e}")
        return None


def _load_from_postgresql(signal_id: str) -> Optional[Dict[str, Any]]:
    """从PostgreSQL加载信号"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT signal_id, strategy_id, symbol, signal_type, strength, price,
                   status, timestamp, accuracy, latency, quality
            FROM trading_signals
            WHERE signal_id = %s
        """, (signal_id,))
        
        row = cursor.fetchone()
        cursor.close()
        return_db_connection(conn)
        
        if row:
            return {
                "id": row[0],
                "signal_id": row[0],
                "strategy_id": row[1],
                "symbol": row[2],
                "type": row[3],
                "signal_type": row[3],
                "strength": float(row[4]) if row[4] else None,
                "price": float(row[5]) if row[5] else None,
                "status": row[6],
                "timestamp": row[7],
                "accuracy": float(row[8]) if row[8] else None,
                "latency": float(row[9]) if row[9] else None,
                "quality": float(row[10]) if row[10] else None
            }
        
        return None
    except Exception as e:
        logger.debug(f"从PostgreSQL加载失败: {e}")
        if conn:
            return_db_connection(conn)
        return None


def list_signals(
    strategy_id: Optional[str] = None,
    signal_type: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    列出交易信号
    
    Args:
        strategy_id: 策略ID过滤器
        signal_type: 信号类型过滤器
        limit: 返回数量限制
        start_time: 开始时间戳
        end_time: 结束时间戳
    
    Returns:
        信号列表
    """
    try:
        # 优先从PostgreSQL加载
        try:
            signals = _list_from_postgresql(strategy_id, signal_type, limit, start_time, end_time)
            if signals:
                return signals
        except Exception as e:
            logger.debug(f"从PostgreSQL加载失败: {e}")
        
        # 从文件系统加载
        signals = []
        if os.path.exists(SIGNALS_DIR):
            for filename in os.listdir(SIGNALS_DIR):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(SIGNALS_DIR, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            signal = json.load(f)
                            
                            # 应用过滤器
                            if strategy_id and signal.get("strategy_id") != strategy_id:
                                continue
                            if signal_type and signal.get("type") != signal_type and signal.get("signal_type") != signal_type:
                                continue
                            if start_time and signal.get("timestamp", 0) < start_time:
                                continue
                            if end_time and signal.get("timestamp", 0) > end_time:
                                continue
                            
                            signals.append(signal)
                    except Exception as e:
                        logger.debug(f"加载文件失败 {filename}: {e}")
        
        # 按时间戳排序
        signals.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return signals[:limit]
    except Exception as e:
        logger.error(f"列出交易信号失败: {e}")
        return []


def _list_from_postgresql(
    strategy_id: Optional[str] = None,
    signal_type: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict[str, Any]]:
    """从PostgreSQL列出信号"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        # 构建查询
        conditions = []
        params = []
        
        if strategy_id:
            conditions.append("strategy_id = %s")
            params.append(strategy_id)
        if signal_type:
            conditions.append("signal_type = %s")
            params.append(signal_type)
        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        
        cursor.execute(f"""
            SELECT signal_id, strategy_id, symbol, signal_type, strength, price,
                   status, timestamp, accuracy, latency, quality
            FROM trading_signals
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT %s
        """, params)
        
        rows = cursor.fetchall()
        cursor.close()
        return_db_connection(conn)
        
        signals = []
        for row in rows:
            signals.append({
                "id": row[0],
                "signal_id": row[0],
                "strategy_id": row[1],
                "symbol": row[2],
                "type": row[3],
                "signal_type": row[3],
                "strength": float(row[4]) if row[4] else None,
                "price": float(row[5]) if row[5] else None,
                "status": row[6],
                "timestamp": row[7],
                "accuracy": float(row[8]) if row[8] else None,
                "latency": float(row[9]) if row[9] else None,
                "quality": float(row[10]) if row[10] else None
            })
        
        return signals
    except Exception as e:
        logger.debug(f"从PostgreSQL列出失败: {e}")
        if conn:
            return_db_connection(conn)
        return []


def update_signal(signal_id: str, updates: Dict[str, Any]) -> bool:
    """
    更新交易信号
    
    Args:
        signal_id: 信号ID
        updates: 更新数据
    
    Returns:
        是否成功更新
    """
    try:
        # 加载现有信号
        signal = load_signal(signal_id)
        if not signal:
            logger.warning(f"信号不存在: {signal_id}")
            return False
        
        # 更新信号
        signal.update(updates)
        
        # 保存
        return save_signal(signal)
    except Exception as e:
        logger.error(f"更新交易信号失败: {e}")
        return False


def delete_signal(signal_id: str) -> bool:
    """
    删除交易信号
    
    Args:
        signal_id: 信号ID
    
    Returns:
        是否成功删除
    """
    try:
        # 从文件系统删除
        filepath = os.path.join(SIGNALS_DIR, f"{signal_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # 从PostgreSQL删除
        try:
            _delete_from_postgresql(signal_id)
        except Exception as e:
            logger.debug(f"从PostgreSQL删除失败: {e}")
        
        logger.info(f"交易信号已删除: {signal_id}")
        return True
    except Exception as e:
        logger.error(f"删除交易信号失败: {e}")
        return False


def _delete_from_postgresql(signal_id: str) -> bool:
    """从PostgreSQL删除信号"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trading_signals WHERE signal_id = %s", (signal_id,))
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

