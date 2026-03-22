"""
特征选择任务持久化模块
存储特征选择任务到PostgreSQL，降级到文件系统
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# 使用统一日志系统
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 数据存储目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data")
FEATURE_SELECTION_TASKS_DIR = os.path.join(DATA_DIR, "feature_selection_tasks")


def ensure_directories():
    """确保数据目录存在"""
    os.makedirs(FEATURE_SELECTION_TASKS_DIR, exist_ok=True)


def save_selection_task(task: Dict[str, Any]) -> bool:
    """
    保存特征选择任务到持久化存储
    优先保存到PostgreSQL，降级到文件系统
    
    Args:
        task: 任务信息字典，必须包含task_id字段
    
    Returns:
        是否成功保存
    """
    try:
        task_id = task.get("task_id")
        if not task_id:
            logger.error("任务缺少task_id字段，无法保存")
            return False
        
        # 添加保存时间戳
        task_data = task.copy()
        task_data["saved_at"] = time.time()
        task_data["updated_at"] = time.time()
        
        # 优先保存到PostgreSQL
        pg_success = False
        try:
            pg_success = _save_to_postgresql(task_data)
            if pg_success:
                logger.info(f"✅ 特征选择任务已保存到PostgreSQL: {task_id}")
        except Exception as e:
            logger.warning(f"⚠️ 保存到PostgreSQL失败，将降级到文件系统: {e}")
        
        # 如果PostgreSQL保存失败，降级到文件系统
        if not pg_success:
            ensure_directories()
            filepath = os.path.join(FEATURE_SELECTION_TASKS_DIR, f"{task_id}.json")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 特征选择任务已保存到文件系统: {task_id}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 保存特征选择任务失败: {e}")
        return False


def _save_to_postgresql(task: Dict[str, Any]) -> bool:
    """尝试保存任务到PostgreSQL"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # 确保表存在
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_selection_tasks (
                task_id VARCHAR(100) PRIMARY KEY,
                task_type VARCHAR(50) NOT NULL DEFAULT 'feature_selection',
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                symbol VARCHAR(20),
                parent_task_id VARCHAR(100),
                selection_method VARCHAR(50),
                n_features INTEGER DEFAULT 10,
                auto_execute BOOLEAN DEFAULT TRUE,
                input_features JSONB,
                total_input_features INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_selection_tasks_status 
            ON feature_selection_tasks(status);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_selection_tasks_symbol 
            ON feature_selection_tasks(symbol);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_selection_tasks_created 
            ON feature_selection_tasks(created_at DESC);
        """)
        
        # 插入或更新任务
        cursor.execute("""
            INSERT INTO feature_selection_tasks (
                task_id, task_type, status, progress, symbol,
                parent_task_id, selection_method, n_features, auto_execute,
                input_features, total_input_features, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
            )
            ON CONFLICT (task_id) 
            DO UPDATE SET
                status = EXCLUDED.status,
                progress = EXCLUDED.progress,
                updated_at = CURRENT_TIMESTAMP
        """, (
            task.get("task_id"),
            task.get("task_type", "feature_selection"),
            task.get("status", "pending"),
            task.get("progress", 0),
            task.get("symbol"),
            task.get("parent_task_id") or task.get("source_task_id"),
            task.get("selection_method"),
            task.get("n_features", 10),
            task.get("auto_execute", True),
            json.dumps(task.get("input_features", [])),
            task.get("total_input_features", 0)
        ))
        
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        return True
    except Exception as e:
        logger.warning(f"保存到PostgreSQL失败: {e}")
        return False


def update_selection_task_status(
    task_id: str,
    status: str,
    progress: int = None,
    end_time: int = None,
    processing_time: float = None,
    total_input_features: int = None,
    total_selected_features: int = None,
    symbols_processed: int = None,
    error_message: str = None
) -> bool:
    """
    更新特征选择任务状态
    优先更新PostgreSQL，降级到文件系统
    """
    try:
        # 优先更新PostgreSQL
        pg_success = False
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                updates = ["status = %s", "updated_at = CURRENT_TIMESTAMP"]
                params = [status]
                
                if progress is not None:
                    updates.append("progress = %s")
                    params.append(progress)
                
                cursor.execute(f"""
                    UPDATE feature_selection_tasks
                    SET {', '.join(updates)}
                    WHERE task_id = %s
                """, params + [task_id])
                
                conn.commit()
                cursor.close()
                return_db_connection(conn)
                pg_success = True
        except Exception as e:
            logger.debug(f"更新PostgreSQL失败: {e}")
        
        # 如果PostgreSQL更新失败，更新文件系统
        if not pg_success:
            filepath = os.path.join(FEATURE_SELECTION_TASKS_DIR, f"{task_id}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    task = json.load(f)
                
                task["status"] = status
                task["updated_at"] = time.time()
                if progress is not None:
                    task["progress"] = progress
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(task, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"更新任务状态失败: {e}")
        return False


def list_selection_tasks(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    列出特征选择任务
    优先从PostgreSQL查询，当数据库连接成功但无数据时降级到文件系统
    
    Args:
        limit: 返回数量限制
        offset: 偏移量
        status: 状态筛选
        
    Returns:
        任务列表
    """
    tasks = []
    db_connected = False
    db_has_data = False
    
    try:
        # 优先从PostgreSQL查询
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if conn:
            db_connected = True
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT task_id, task_type, status, progress, symbol,
                           parent_task_id, selection_method, n_features,
                           created_at, updated_at
                    FROM feature_selection_tasks
                    WHERE status = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (status, limit, offset))
            else:
                cursor.execute("""
                    SELECT task_id, task_type, status, progress, symbol,
                           parent_task_id, selection_method, n_features,
                           created_at, updated_at
                    FROM feature_selection_tasks
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
            
            rows = cursor.fetchall()
            for row in rows:
                tasks.append({
                    "task_id": row[0],
                    "task_type": row[1],
                    "status": row[2],
                    "progress": row[3],
                    "symbol": row[4],
                    "parent_task_id": row[5],
                    "selection_method": row[6],
                    "n_features": row[7],
                    "created_at": row[8].isoformat() if row[8] else None,
                    "updated_at": row[9].isoformat() if row[9] else None
                })
            
            cursor.close()
            return_db_connection(conn)
            
            if tasks:
                db_has_data = True
                logger.info(f"✅ 从PostgreSQL查询到 {len(tasks)} 个特征选择任务")
                return tasks
            else:
                logger.info(f"📭 PostgreSQL连接成功但无数据，准备降级到文件系统")
        else:
            logger.warning(f"⚠️ 无法获取数据库连接，直接降级到文件系统")
            
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL查询异常: {e}，降级到文件系统")
    
    # 当数据库连接成功但无数据，或连接失败时，从文件系统查询
    if os.path.exists(FEATURE_SELECTION_TASKS_DIR):
        try:
            task_files = sorted(
                [f for f in os.listdir(FEATURE_SELECTION_TASKS_DIR) if f.endswith('.json')],
                reverse=True
            )
            
            file_tasks = []
            for filename in task_files[offset:offset+limit]:
                filepath = os.path.join(FEATURE_SELECTION_TASKS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        task = json.load(f)
                    
                    if status and task.get('status') != status:
                        continue
                    
                    file_tasks.append(task)
                except Exception as e:
                    logger.warning(f"⚠️ 读取任务文件失败 {filename}: {e}")
            
            if file_tasks:
                if db_connected and not db_has_data:
                    logger.info(f"✅ 数据库无数据，已从文件系统加载 {len(file_tasks)} 个任务（降级成功）")
                else:
                    logger.info(f"✅ 从文件系统查询到 {len(file_tasks)} 个特征选择任务")
            
            return file_tasks
            
        except Exception as e:
            logger.error(f"❌ 从文件系统查询任务失败: {e}")
    else:
        logger.warning(f"⚠️ 文件系统任务目录不存在: {FEATURE_SELECTION_TASKS_DIR}")
    
    return tasks


def get_selection_tasks_stats() -> Dict[str, Any]:
    """
    获取特征选择任务统计
    优先从PostgreSQL查询，当数据库连接成功但无数据时降级到文件系统
    
    Returns:
        统计信息
    """
    stats = {
        "total": 0,
        "by_status": {},
        "by_method": {}
    }
    
    db_connected = False
    db_has_data = False
    
    try:
        # 优先从PostgreSQL查询
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if conn:
            db_connected = True
            cursor = conn.cursor()
            
            # 总数
            cursor.execute("SELECT COUNT(*) FROM feature_selection_tasks")
            db_total = cursor.fetchone()[0]
            
            if db_total > 0:
                db_has_data = True
                stats["total"] = db_total
                
                # 按状态统计
                cursor.execute("""
                    SELECT status, COUNT(*) 
                    FROM feature_selection_tasks 
                    GROUP BY status
                """)
                for row in cursor.fetchall():
                    stats["by_status"][row[0]] = row[1]
                
                # 按方法统计
                cursor.execute("""
                    SELECT selection_method, COUNT(*) 
                    FROM feature_selection_tasks 
                    GROUP BY selection_method
                """)
                for row in cursor.fetchall():
                    stats["by_method"][row[0]] = row[1]
                
                cursor.close()
                return_db_connection(conn)
                
                logger.info(f"✅ 从PostgreSQL获取统计: 总计 {stats['total']} 个任务")
                return stats
            else:
                logger.info(f"📭 PostgreSQL连接成功但无数据，准备从文件系统获取统计")
                cursor.close()
                return_db_connection(conn)
        else:
            logger.warning(f"⚠️ 无法获取数据库连接，直接从文件系统获取统计")
            
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL统计查询异常: {e}，降级到文件系统")
    
    # 当数据库连接成功但无数据，或连接失败时，从文件系统统计
    if os.path.exists(FEATURE_SELECTION_TASKS_DIR):
        try:
            file_count = 0
            for filename in os.listdir(FEATURE_SELECTION_TASKS_DIR):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(FEATURE_SELECTION_TASKS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        task = json.load(f)
                    
                    stats["total"] += 1
                    file_count += 1
                    
                    status = task.get('status', 'unknown')
                    stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
                    
                    method = task.get('selection_method', 'unknown')
                    stats["by_method"][method] = stats["by_method"].get(method, 0) + 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ 读取任务文件失败 {filename}: {e}")
            
            if file_count > 0:
                if db_connected and not db_has_data:
                    logger.info(f"✅ 数据库无数据，已从文件系统获取统计: 总计 {stats['total']} 个任务（降级成功）")
                else:
                    logger.info(f"✅ 从文件系统获取统计: 总计 {stats['total']} 个任务")
                    
        except Exception as e:
            logger.error(f"❌ 从文件系统获取统计失败: {e}")
    else:
        logger.warning(f"⚠️ 文件系统任务目录不存在: {FEATURE_SELECTION_TASKS_DIR}")
    
    return stats


def get_selection_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    获取特征选择任务详情
    优先从PostgreSQL查询，降级到文件系统
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务详情，不存在返回None
    """
    try:
        # 优先从PostgreSQL查询
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT task_id, task_type, status, progress, symbol,
                           parent_task_id, selection_method, n_features,
                           auto_execute, input_features, total_input_features,
                           created_at, updated_at
                    FROM feature_selection_tasks
                    WHERE task_id = %s
                """, (task_id,))
                
                row = cursor.fetchone()
                if row:
                    task = {
                        "task_id": row[0],
                        "task_type": row[1],
                        "status": row[2],
                        "progress": row[3],
                        "symbol": row[4],
                        "parent_task_id": row[5],
                        "selection_method": row[6],
                        "n_features": row[7],
                        "auto_execute": row[8],
                        "input_features": row[9],
                        "total_input_features": row[10],
                        "created_at": row[11].isoformat() if row[11] else None,
                        "updated_at": row[12].isoformat() if row[12] else None
                    }
                    
                    cursor.close()
                    return_db_connection(conn)
                    return task
                
                cursor.close()
                return_db_connection(conn)
        except Exception as e:
            logger.debug(f"从PostgreSQL查询失败: {e}")
        
        # 如果PostgreSQL查询失败，从文件系统查询
        filepath = os.path.join(FEATURE_SELECTION_TASKS_DIR, f"{task_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
        
    except Exception as e:
        logger.error(f"获取任务详情失败: {e}")
        return None


def create_selection_task(
    symbol: str,
    features: List[str],
    source_task_id: str,
    selection_method: str = "importance",
    config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    创建特征选择任务
    
    Args:
        symbol: 股票代码
        features: 特征列表
        source_task_id: 源任务ID（特征提取任务ID）
        selection_method: 选择方法，默认"importance"
        config: 配置参数，包择n_features（选择特征数）、auto_execute（是否自动执行）等
    
    Returns:
        创建的任务信息字典，失败返回None
    """
    try:
        import uuid
        
        task_id = f"selection_{symbol}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        n_features = config.get("n_features", 10) if config else 10
        auto_execute = config.get("auto_execute", True) if config else True
        
        task = {
            "task_id": task_id,
            "task_type": "feature_selection",
            "status": "pending",
            "progress": 0,
            "symbol": symbol,
            "parent_task_id": source_task_id,
            "source_task_id": source_task_id,
            "selection_method": selection_method,
            "n_features": n_features,
            "auto_execute": auto_execute,
            "input_features": features,
            "total_input_features": len(features),
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # 保存任务
        if save_selection_task(task):
            logger.info(f"✅ 特征选择任务创建成功: {task_id}, 股票: {symbol}, 输入特征: {len(features)}")
            
            # 如果配置了自动执行，提交到调度器
            if auto_execute:
                try:
                    from src.core.orchestration.scheduler import get_unified_scheduler
                    scheduler = get_unified_scheduler()
                    if scheduler:
                        import asyncio
                        payload = {
                            "symbol": symbol,
                            "features": features,
                            "selection_method": selection_method,
                            "n_features": n_features,
                            "task_id": task_id
                        }
                        scheduler_task_id = asyncio.run(scheduler.submit_task(
                            task_type="feature_selection",
                            payload=payload,
                            priority=5
                        ))
                        logger.info(f"✅ 特征选择任务已提交到调度器: {scheduler_task_id}")
                        task["scheduler_task_id"] = scheduler_task_id
                except Exception as e:
                    logger.warning(f"⚠️ 提交特征选择任务到调度器失败: {e}")
            
            return task
        else:
            logger.error(f"❌ 保存特征选择任务失败: {task_id}")
            return None
            
    except Exception as e:
        logger.error(f"❌ 创建特征选择任务失败: {e}", exc_info=True)
        return False
