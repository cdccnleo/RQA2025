"""
订单路由决策持久化模块
存储订单路由决策到文件系统或PostgreSQL
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
ROUTING_DECISIONS_DIR = os.path.join(DATA_DIR, "routing_decisions")


def ensure_directories():
    """确保数据目录存在"""
    os.makedirs(ROUTING_DECISIONS_DIR, exist_ok=True)


def save_routing_decision(decision: Dict[str, Any]) -> bool:
    """
    保存订单路由决策到持久化存储
    
    Args:
        decision: 路由决策字典，必须包含order_id字段
    
    Returns:
        是否成功保存
    """
    try:
        ensure_directories()
        
        order_id = decision.get("order_id")
        if not order_id:
            logger.error("路由决策缺少order_id字段，无法保存")
            return False
        
        filepath = os.path.join(ROUTING_DECISIONS_DIR, f"{order_id}.json")
        
        # 添加保存时间戳
        decision_data = decision.copy()
        decision_data["saved_at"] = time.time()
        
        # 确保timestamp是整数
        if "timestamp" in decision_data:
            if isinstance(decision_data["timestamp"], datetime):
                decision_data["timestamp"] = int(decision_data["timestamp"].timestamp())
            elif isinstance(decision_data["timestamp"], str):
                try:
                    dt = datetime.fromisoformat(decision_data["timestamp"].replace("Z", "+00:00"))
                    decision_data["timestamp"] = int(dt.timestamp())
                except:
                    decision_data["timestamp"] = int(time.time())
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(decision_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"订单路由决策已保存: {order_id}")
        
        # 同时尝试保存到PostgreSQL（如果可用）
        try:
            _save_to_postgresql(decision_data)
        except Exception as e:
            logger.debug(f"保存到PostgreSQL失败（使用文件系统）: {e}")
        
        return True
    except Exception as e:
        logger.error(f"保存路由决策失败: {e}")
        return False


def _save_to_postgresql(decision: Dict[str, Any]) -> bool:
    """尝试保存路由决策到PostgreSQL"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # 确保表存在
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routing_decisions (
                order_id VARCHAR(100) PRIMARY KEY,
                routing_strategy VARCHAR(50),
                target_route VARCHAR(100),
                cost DECIMAL(10, 6),
                latency DECIMAL(10, 2),
                status VARCHAR(20) NOT NULL,
                failure_reason TEXT,
                timestamp BIGINT NOT NULL,
                saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 创建索引
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_routing_timestamp ON routing_decisions(timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_routing_status ON routing_decisions(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_routing_strategy ON routing_decisions(routing_strategy);")
        except Exception as e:
            logger.debug(f"创建索引可能已存在: {e}")
        
        # 插入或更新路由决策
        cursor.execute("""
            INSERT INTO routing_decisions (
                order_id, routing_strategy, target_route, cost, latency,
                status, failure_reason, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (order_id) 
            DO UPDATE SET
                status = EXCLUDED.status,
                cost = EXCLUDED.cost,
                latency = EXCLUDED.latency,
                failure_reason = EXCLUDED.failure_reason
        """, (
            decision.get("order_id"),
            decision.get("routing_strategy"),
            decision.get("target_route"),
            decision.get("cost"),
            decision.get("latency"),
            decision.get("status", "unknown"),
            decision.get("failure_reason"),
            decision.get("timestamp", int(time.time()))
        ))
        
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        logger.debug(f"路由决策已保存到PostgreSQL: {decision.get('order_id')}")
        return True
    except Exception as e:
        logger.debug(f"保存到PostgreSQL失败: {e}")
        if conn:
            conn.rollback()
            return_db_connection(conn)
        return False


def load_routing_decision(order_id: str) -> Optional[Dict[str, Any]]:
    """
    从持久化存储加载订单路由决策
    
    Args:
        order_id: 订单ID
    
    Returns:
        路由决策字典，如果不存在则返回None
    """
    try:
        # 优先从PostgreSQL加载
        try:
            decision = _load_from_postgresql(order_id)
            if decision:
                return decision
        except Exception as e:
            logger.debug(f"从PostgreSQL加载失败: {e}")
        
        # 从文件系统加载
        filepath = os.path.join(ROUTING_DECISIONS_DIR, f"{order_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    except Exception as e:
        logger.error(f"加载路由决策失败: {e}")
        return None


def _load_from_postgresql(order_id: str) -> Optional[Dict[str, Any]]:
    """从PostgreSQL加载路由决策"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT order_id, routing_strategy, target_route, cost, latency,
                   status, failure_reason, timestamp
            FROM routing_decisions
            WHERE order_id = %s
        """, (order_id,))
        
        row = cursor.fetchone()
        cursor.close()
        return_db_connection(conn)
        
        if row:
            return {
                "order_id": row[0],
                "routing_strategy": row[1],
                "target_route": row[2],
                "cost": float(row[3]) if row[3] else None,
                "latency": float(row[4]) if row[4] else None,
                "status": row[5],
                "failure_reason": row[6],
                "timestamp": row[7]
            }
        
        return None
    except Exception as e:
        logger.debug(f"从PostgreSQL加载失败: {e}")
        if conn:
            return_db_connection(conn)
        return None


def list_routing_decisions(
    status: Optional[str] = None,
    routing_strategy: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    列出订单路由决策
    
    Args:
        status: 状态过滤器
        routing_strategy: 路由策略过滤器
        limit: 返回数量限制
        start_time: 开始时间戳
        end_time: 结束时间戳
    
    Returns:
        路由决策列表
    """
    try:
        # 优先从PostgreSQL加载
        try:
            decisions = _list_from_postgresql(status, routing_strategy, limit, start_time, end_time)
            if decisions:
                return decisions
        except Exception as e:
            logger.debug(f"从PostgreSQL加载失败: {e}")
        
        # 从文件系统加载
        decisions = []
        if os.path.exists(ROUTING_DECISIONS_DIR):
            for filename in os.listdir(ROUTING_DECISIONS_DIR):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(ROUTING_DECISIONS_DIR, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            decision = json.load(f)
                            
                            # 应用过滤器
                            if status and decision.get("status") != status:
                                continue
                            if routing_strategy and decision.get("routing_strategy") != routing_strategy:
                                continue
                            if start_time and decision.get("timestamp", 0) < start_time:
                                continue
                            if end_time and decision.get("timestamp", 0) > end_time:
                                continue
                            
                            decisions.append(decision)
                    except Exception as e:
                        logger.debug(f"加载文件失败 {filename}: {e}")
        
        # 按时间戳排序
        decisions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return decisions[:limit]
    except Exception as e:
        logger.error(f"列出路由决策失败: {e}")
        return []


def _list_from_postgresql(
    status: Optional[str] = None,
    routing_strategy: Optional[str] = None,
    limit: int = 100,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> List[Dict[str, Any]]:
    """从PostgreSQL列出路由决策"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        # 构建查询
        conditions = []
        params = []
        
        if status:
            conditions.append("status = %s")
            params.append(status)
        if routing_strategy:
            conditions.append("routing_strategy = %s")
            params.append(routing_strategy)
        if start_time:
            conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= %s")
            params.append(end_time)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        
        cursor.execute(f"""
            SELECT order_id, routing_strategy, target_route, cost, latency,
                   status, failure_reason, timestamp
            FROM routing_decisions
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT %s
        """, params)
        
        rows = cursor.fetchall()
        cursor.close()
        return_db_connection(conn)
        
        decisions = []
        for row in rows:
            decisions.append({
                "order_id": row[0],
                "routing_strategy": row[1],
                "target_route": row[2],
                "cost": float(row[3]) if row[3] else None,
                "latency": float(row[4]) if row[4] else None,
                "status": row[5],
                "failure_reason": row[6],
                "timestamp": row[7]
            })
        
        return decisions
    except Exception as e:
        logger.debug(f"从PostgreSQL列出失败: {e}")
        if conn:
            return_db_connection(conn)
        return []


def update_routing_decision(order_id: str, updates: Dict[str, Any]) -> bool:
    """
    更新路由决策
    
    Args:
        order_id: 订单ID
        updates: 更新数据
    
    Returns:
        是否成功更新
    """
    try:
        # 加载现有决策
        decision = load_routing_decision(order_id)
        if not decision:
            logger.warning(f"路由决策不存在: {order_id}")
            return False
        
        # 更新决策
        decision.update(updates)
        
        # 保存
        return save_routing_decision(decision)
    except Exception as e:
        logger.error(f"更新路由决策失败: {e}")
        return False


def delete_routing_decision(order_id: str) -> bool:
    """
    删除路由决策
    
    Args:
        order_id: 订单ID
    
    Returns:
        是否成功删除
    """
    try:
        # 从文件系统删除
        filepath = os.path.join(ROUTING_DECISIONS_DIR, f"{order_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # 从PostgreSQL删除
        try:
            _delete_from_postgresql(order_id)
        except Exception as e:
            logger.debug(f"从PostgreSQL删除失败: {e}")
        
        logger.info(f"路由决策已删除: {order_id}")
        return True
    except Exception as e:
        logger.error(f"删除路由决策失败: {e}")
        return False


def _delete_from_postgresql(order_id: str) -> bool:
    """从PostgreSQL删除路由决策"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        cursor.execute("DELETE FROM routing_decisions WHERE order_id = %s", (order_id,))
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

