"""
策略数据持久化模块
存储优化结果、生命周期事件等数据
支持双写机制（文件系统 + PostgreSQL）
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
OPTIMIZATION_RESULTS_DIR = os.path.join(DATA_DIR, "optimization_results")
LIFECYCLE_EVENTS_DIR = os.path.join(DATA_DIR, "lifecycle_events")


def ensure_directories():
    """确保数据目录存在"""
    os.makedirs(OPTIMIZATION_RESULTS_DIR, exist_ok=True)
    os.makedirs(LIFECYCLE_EVENTS_DIR, exist_ok=True)


def _get_db_connection():
    """获取数据库连接"""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        # 从环境变量获取连接信息
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            conn = psycopg2.connect(database_url)
        else:
            # 使用默认连接配置
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'postgres'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
                user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
                password=os.getenv('POSTGRES_PASSWORD', 'SecurePass123!')
            )
        return conn
    except Exception as e:
        logger.warning(f"无法连接到PostgreSQL: {e}")
        return None


def _ensure_table_exists():
    """确保optimization_results表存在"""
    conn = _get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id SERIAL PRIMARY KEY,
                    task_id VARCHAR(255) UNIQUE NOT NULL,
                    strategy_id VARCHAR(255) NOT NULL,
                    strategy_name VARCHAR(500),
                    method VARCHAR(100) NOT NULL,
                    target VARCHAR(100) NOT NULL,
                    results JSONB NOT NULL DEFAULT '[]'::jsonb,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 创建索引
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimization_results_task_id 
                ON optimization_results(task_id);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimization_results_strategy_id 
                ON optimization_results(strategy_id);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimization_results_saved_at 
                ON optimization_results(saved_at DESC);
            """)
            
            conn.commit()
            logger.info("optimization_results表已创建或已存在")
            return True
    except Exception as e:
        logger.warning(f"创建表失败: {e}")
        return False
    finally:
        conn.close()


def _save_optimization_result_to_postgresql(result: Dict[str, Any]) -> bool:
    """
    保存优化结果到PostgreSQL
    
    Args:
        result: 优化结果字典
    
    Returns:
        是否成功保存
    """
    conn = _get_db_connection()
    if not conn:
        return False
    
    try:
        # 确保表存在
        _ensure_table_exists()
        
        with conn.cursor() as cur:
            # 转换时间戳
            completed_at = None
            if "completed_at" in result and result["completed_at"]:
                if isinstance(result["completed_at"], (int, float)):
                    completed_at = datetime.fromtimestamp(result["completed_at"])
                elif isinstance(result["completed_at"], str):
                    try:
                        completed_at = datetime.fromisoformat(result["completed_at"].replace("Z", "+00:00"))
                    except:
                        pass
            
            # 插入或更新数据
            cur.execute("""
                INSERT INTO optimization_results 
                (task_id, strategy_id, strategy_name, method, target, results, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (task_id) DO UPDATE SET
                    strategy_id = EXCLUDED.strategy_id,
                    strategy_name = EXCLUDED.strategy_name,
                    method = EXCLUDED.method,
                    target = EXCLUDED.target,
                    results = EXCLUDED.results,
                    completed_at = EXCLUDED.completed_at,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                result.get("task_id"),
                result.get("strategy_id"),
                result.get("strategy_name"),
                result.get("method"),
                result.get("target"),
                json.dumps(result.get("results", [])),
                completed_at
            ))
            
            conn.commit()
            logger.info(f"优化结果已保存到PostgreSQL: {result.get('task_id')}")
            return True
    except Exception as e:
        logger.error(f"保存优化结果到PostgreSQL失败: {e}")
        return False
    finally:
        conn.close()


def save_optimization_result(task_id: str, result: Dict[str, Any]) -> bool:
    """
    保存优化结果（双写机制：优先PostgreSQL，同时写入文件系统）
    
    Args:
        task_id: 任务ID
        result: 优化结果字典
    
    Returns:
        是否成功保存（PostgreSQL优先，文件系统作为备份）
    """
    try:
        ensure_directories()
        
        # 确保结果包含task_id
        result["task_id"] = task_id
        result["saved_at"] = time.time()
        
        # 1. 优先写入PostgreSQL（主存储）
        pg_success = False
        try:
            pg_success = _save_optimization_result_to_postgresql(result)
            if pg_success:
                logger.info(f"优化结果已保存到PostgreSQL: {task_id}")
        except Exception as e:
            logger.warning(f"保存到PostgreSQL失败: {e}")
        
        # 2. 同时写入文件系统（备份存储）
        try:
            filepath = os.path.join(OPTIMIZATION_RESULTS_DIR, f"{task_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"优化结果已保存到文件系统: {task_id}")
        except Exception as e:
            logger.error(f"保存到文件系统失败: {e}")
            # 如果PostgreSQL也失败了，则返回失败
            if not pg_success:
                return False
        
        return True
    except Exception as e:
        logger.error(f"保存优化结果失败: {e}")
        return False


def _load_optimization_result_from_postgresql(task_id: str) -> Optional[Dict[str, Any]]:
    """
    从PostgreSQL加载优化结果
    
    Args:
        task_id: 任务ID
    
    Returns:
        优化结果字典，如果不存在返回None
    """
    conn = _get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT task_id, strategy_id, strategy_name, method, target, 
                       results, completed_at, saved_at, updated_at
                FROM optimization_results
                WHERE task_id = %s
            """, (task_id,))
            
            row = cur.fetchone()
            if row:
                result = {
                    "task_id": row[0],
                    "strategy_id": row[1],
                    "strategy_name": row[2],
                    "method": row[3],
                    "target": row[4],
                    "results": row[5] if isinstance(row[5], list) else json.loads(row[5]) if row[5] else [],
                    "completed_at": row[6].timestamp() if row[6] else None,
                    "saved_at": row[7].timestamp() if row[7] else None,
                    "updated_at": row[8].timestamp() if row[8] else None
                }
                logger.info(f"从PostgreSQL加载优化结果: {task_id}")
                return result
            return None
    except Exception as e:
        logger.warning(f"从PostgreSQL加载优化结果失败: {e}")
        return None
    finally:
        conn.close()


def load_optimization_result(task_id: str) -> Optional[Dict[str, Any]]:
    """
    加载优化结果（优先从PostgreSQL，失败时回退到文件系统）
    
    Args:
        task_id: 任务ID
    
    Returns:
        优化结果字典，如果不存在返回None
    """
    # 1. 优先从PostgreSQL加载
    try:
        result = _load_optimization_result_from_postgresql(task_id)
        if result:
            return result
    except Exception as e:
        logger.warning(f"从PostgreSQL加载优化结果失败，回退到文件系统: {e}")
    
    # 2. 从文件系统加载（回退）
    try:
        filepath = os.path.join(OPTIMIZATION_RESULTS_DIR, f"{task_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"从文件系统加载优化结果失败: {e}")
    
    return None


def _list_optimization_results_from_postgresql(strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    从PostgreSQL列出优化结果
    
    Args:
        strategy_id: 可选的策略ID筛选
    
    Returns:
        优化结果列表
    """
    conn = _get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cur:
            if strategy_id:
                cur.execute("""
                    SELECT task_id, strategy_id, strategy_name, method, target, 
                           results, completed_at, saved_at, updated_at
                    FROM optimization_results
                    WHERE strategy_id = %s
                    ORDER BY saved_at DESC
                """, (strategy_id,))
            else:
                cur.execute("""
                    SELECT task_id, strategy_id, strategy_name, method, target, 
                           results, completed_at, saved_at, updated_at
                    FROM optimization_results
                    ORDER BY saved_at DESC
                """)
            
            results = []
            for row in cur.fetchall():
                result = {
                    "task_id": row[0],
                    "strategy_id": row[1],
                    "strategy_name": row[2],
                    "method": row[3],
                    "target": row[4],
                    "results": row[5] if isinstance(row[5], list) else json.loads(row[5]) if row[5] else [],
                    "completed_at": row[6].timestamp() if row[6] else None,
                    "saved_at": row[7].timestamp() if row[7] else None,
                    "updated_at": row[8].timestamp() if row[8] else None
                }
                results.append(result)
            
            logger.info(f"从PostgreSQL加载了 {len(results)} 条优化结果")
            return results
    except Exception as e:
        logger.warning(f"从PostgreSQL列出优化结果失败: {e}")
        return []
    finally:
        conn.close()


def list_optimization_results(strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    列出优化结果（优先从PostgreSQL，失败时回退到文件系统）
    
    Args:
        strategy_id: 可选的策略ID筛选
    
    Returns:
        优化结果列表
    """
    # 1. 尝试从PostgreSQL加载
    try:
        results = _list_optimization_results_from_postgresql(strategy_id)
        if results:
            logger.info(f"从PostgreSQL加载了 {len(results)} 条优化结果")
            return results
    except Exception as e:
        logger.warning(f"从PostgreSQL列出失败，回退到文件系统: {e}")
    
    # 2. 从文件系统加载（降级方案）
    try:
        ensure_directories()
        
        results = []
        for filename in os.listdir(OPTIMIZATION_RESULTS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(OPTIMIZATION_RESULTS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        if strategy_id is None or result.get('strategy_id') == strategy_id:
                            results.append(result)
                except Exception as e:
                    logger.warning(f"加载优化结果文件失败 {filename}: {e}")
        
        # 按时间排序
        results.sort(key=lambda x: x.get('saved_at', 0), reverse=True)
        logger.info(f"从文件系统加载了 {len(results)} 条优化结果")
        return results
    except Exception as e:
        logger.error(f"列出优化结果失败: {e}")
        return []


def _delete_optimization_result_from_postgresql(task_id: str) -> bool:
    """
    从PostgreSQL删除优化结果
    
    Args:
        task_id: 任务ID
    
    Returns:
        是否成功删除
    """
    conn = _get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM optimization_results
                WHERE task_id = %s
            """, (task_id,))
            
            conn.commit()
            logger.info(f"从PostgreSQL删除优化结果: {task_id}")
            return True
    except Exception as e:
        logger.warning(f"从PostgreSQL删除优化结果失败: {e}")
        return False
    finally:
        conn.close()


def delete_optimization_result(task_id: str) -> bool:
    """
    删除优化结果（双删除：文件系统 + PostgreSQL）
    
    Args:
        task_id: 任务ID
    
    Returns:
        是否成功删除
    """
    file_deleted = False
    db_deleted = False
    
    # 1. 删除文件
    try:
        filepath = os.path.join(OPTIMIZATION_RESULTS_DIR, f"{task_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"优化结果文件已删除: {task_id}")
            file_deleted = True
    except Exception as e:
        logger.error(f"删除优化结果文件失败: {e}")
    
    # 2. 删除PostgreSQL记录
    try:
        db_deleted = _delete_optimization_result_from_postgresql(task_id)
    except Exception as e:
        logger.warning(f"从PostgreSQL删除失败: {e}")
    
    return file_deleted or db_deleted


def save_lifecycle_event(strategy_id: str, event: Dict[str, Any]) -> bool:
    """保存生命周期事件"""
    try:
        ensure_directories()
        
        # 生成文件名
        timestamp = int(time.time())
        filename = f"{strategy_id}_{timestamp}.json"
        filepath = os.path.join(LIFECYCLE_EVENTS_DIR, filename)
        
        # 添加元数据
        event_data = {
            "strategy_id": strategy_id,
            "timestamp": timestamp,
            "event": event
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"生命周期事件已保存: {strategy_id}")
        return True
    except Exception as e:
        logger.error(f"保存生命周期事件失败: {e}")
        return False


def load_lifecycle_events(strategy_id: str) -> List[Dict[str, Any]]:
    """加载策略的生命周期事件"""
    try:
        ensure_directories()
        
        events = []
        for filename in os.listdir(LIFECYCLE_EVENTS_DIR):
            if filename.startswith(f"{strategy_id}_") and filename.endswith('.json'):
                filepath = os.path.join(LIFECYCLE_EVENTS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        event_data = json.load(f)
                        events.append(event_data)
                except Exception as e:
                    logger.warning(f"加载生命周期事件文件失败 {filename}: {e}")
        
        # 按时间排序
        events.sort(key=lambda x: x.get('timestamp', 0))
        return events
    except Exception as e:
        logger.error(f"加载生命周期事件失败: {e}")
        return []
