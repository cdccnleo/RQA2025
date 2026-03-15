"""
特征选择任务持久化模块
存储特征选择任务到PostgreSQL，支持任务状态跟踪和监控
符合架构设计：使用统一日志系统
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


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
    
    Args:
        task: 任务信息字典，必须包含task_id字段
    
    Returns:
        是否成功保存
    """
    try:
        ensure_directories()
        
        task_id = task.get("task_id")
        if not task_id:
            logger.error("任务缺少task_id字段，无法保存")
            return False
        
        filepath = os.path.join(FEATURE_SELECTION_TASKS_DIR, f"{task_id}.json")
        
        # 添加保存时间戳
        task_data = task.copy()
        task_data["saved_at"] = time.time()
        task_data["updated_at"] = time.time()
        
        # 转换时间戳
        if "start_time" in task_data:
            if isinstance(task_data["start_time"], datetime):
                task_data["start_time"] = int(task_data["start_time"].timestamp())
            elif isinstance(task_data["start_time"], str):
                try:
                    dt = datetime.fromisoformat(task_data["start_time"].replace("Z", "+00:00"))
                    task_data["start_time"] = int(dt.timestamp())
                except:
                    pass
        
        if "end_time" in task_data:
            if isinstance(task_data["end_time"], datetime):
                task_data["end_time"] = int(task_data["end_time"].timestamp())
            elif isinstance(task_data["end_time"], str):
                try:
                    dt = datetime.fromisoformat(task_data["end_time"].replace("Z", "+00:00"))
                    task_data["end_time"] = int(dt.timestamp())
                except:
                    pass
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        
        logger.info(f"特征选择任务已保存到文件: {task_id}")
        
        # 同时尝试保存到PostgreSQL
        try:
            _save_to_postgresql(task_data)
        except Exception as e:
            logger.debug(f"保存到PostgreSQL失败（使用文件系统）: {e}")
        
        return True
    except Exception as e:
        logger.error(f"保存特征选择任务失败: {e}")
        return False


def _save_to_postgresql(task: Dict[str, Any]) -> bool:
    """
    保存任务到PostgreSQL
    
    Args:
        task: 任务数据字典
    
    Returns:
        是否成功保存
    """
    try:
        from src.infrastructure.persistence.postgresql import get_postgresql_connection
        
        conn = get_postgresql_connection()
        if not conn:
            logger.debug("PostgreSQL连接不可用，跳过数据库保存")
            return False
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feature_selection_tasks (
                    task_id, task_type, status, progress,
                    symbols, selection_method, top_k, min_quality,
                    start_time, end_time, processing_time,
                    total_input_features, total_selected_features, symbols_processed,
                    error_message, parent_task_id, results,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (task_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    progress = EXCLUDED.progress,
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time,
                    processing_time = EXCLUDED.processing_time,
                    total_input_features = EXCLUDED.total_input_features,
                    total_selected_features = EXCLUDED.total_selected_features,
                    symbols_processed = EXCLUDED.symbols_processed,
                    error_message = EXCLUDED.error_message,
                    results = EXCLUDED.results,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                task.get("task_id"),
                task.get("task_type", "feature_selection"),
                task.get("status", "pending"),
                task.get("progress", 0),
                json.dumps(task.get("symbols", [])),
                task.get("selection_method", ""),
                task.get("top_k"),
                task.get("min_quality"),
                task.get("start_time"),
                task.get("end_time"),
                task.get("processing_time"),
                task.get("total_input_features", 0),
                task.get("total_selected_features", 0),
                task.get("symbols_processed", 0),
                task.get("error_message"),
                task.get("parent_task_id"),
                json.dumps(task.get("results", {}))
            ))
            
            conn.commit()
            logger.info(f"特征选择任务已保存到PostgreSQL: {task.get('task_id')}")
            return True
            
    except Exception as e:
        logger.error(f"保存到PostgreSQL失败: {e}")
        raise


def update_selection_task_status(
    task_id: str,
    status: str,
    progress: Optional[int] = None,
    **kwargs
) -> bool:
    """
    更新特征选择任务状态
    
    Args:
        task_id: 任务ID
        status: 新状态
        progress: 进度百分比
        **kwargs: 其他要更新的字段
    
    Returns:
        是否成功更新
    """
    try:
        # 先更新文件系统
        filepath = os.path.join(FEATURE_SELECTION_TASKS_DIR, f"{task_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                task = json.load(f)
            
            task["status"] = status
            task["updated_at"] = time.time()
            if progress is not None:
                task["progress"] = progress
            
            # 更新其他字段
            for key, value in kwargs.items():
                task[key] = value
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(task, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        
        # 更新PostgreSQL
        try:
            from src.infrastructure.persistence.postgresql import get_postgresql_connection
            
            conn = get_postgresql_connection()
            if conn:
                with conn.cursor() as cur:
                    # 构建动态更新SQL
                    update_fields = ["status = %s", "updated_at = CURRENT_TIMESTAMP"]
                    params = [status]
                    
                    if progress is not None:
                        update_fields.append("progress = %s")
                        params.append(progress)
                    
                    for key, value in kwargs.items():
                        if key in ['end_time', 'processing_time', 'total_input_features', 
                                   'total_selected_features', 'symbols_processed', 'error_message', 'results']:
                            update_fields.append(f"{key} = %s")
                            if isinstance(value, (dict, list)):
                                params.append(json.dumps(value))
                            else:
                                params.append(value)
                    
                    params.append(task_id)
                    
                    sql = f"""
                        UPDATE feature_selection_tasks 
                        SET {', '.join(update_fields)}
                        WHERE task_id = %s
                    """
                    
                    cur.execute(sql, params)
                    conn.commit()
                    logger.debug(f"任务状态已更新: {task_id} -> {status}")
        except Exception as e:
            logger.debug(f"更新PostgreSQL失败: {e}")
        
        return True
    except Exception as e:
        logger.error(f"更新任务状态失败: {e}")
        return False


def get_selection_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    获取单个特征选择任务
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务信息字典，不存在返回None
    """
    try:
        # 优先从PostgreSQL查询
        try:
            from src.infrastructure.persistence.postgresql import get_postgresql_connection
            
            conn = get_postgresql_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT * FROM feature_selection_tasks 
                        WHERE task_id = %s
                    """, (task_id,))
                    
                    row = cur.fetchone()
                    if row:
                        columns = [desc[0] for desc in cur.description]
                        task = dict(zip(columns, row))
                        
                        # 解析JSON字段
                        for key in ['symbols', 'results']:
                            if task.get(key) and isinstance(task[key], str):
                                try:
                                    task[key] = json.loads(task[key])
                                except:
                                    pass
                        
                        return task
        except Exception as e:
            logger.debug(f"从PostgreSQL查询失败: {e}")
        
        # 降级到文件系统
        filepath = os.path.join(FEATURE_SELECTION_TASKS_DIR, f"{task_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    except Exception as e:
        logger.error(f"获取任务失败: {e}")
        return None


def list_selection_tasks(
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    获取特征选择任务列表
    
    Args:
        status: 按状态过滤
        limit: 返回数量限制
        offset: 偏移量
    
    Returns:
        任务列表
    """
    try:
        # 优先从PostgreSQL查询
        try:
            from src.infrastructure.persistence.postgresql import get_postgresql_connection
            
            conn = get_postgresql_connection()
            if conn:
                with conn.cursor() as cur:
                    if status:
                        cur.execute("""
                            SELECT * FROM feature_selection_tasks 
                            WHERE status = %s
                            ORDER BY created_at DESC
                            LIMIT %s OFFSET %s
                        """, (status, limit, offset))
                    else:
                        cur.execute("""
                            SELECT * FROM feature_selection_tasks 
                            ORDER BY created_at DESC
                            LIMIT %s OFFSET %s
                        """, (limit, offset))
                    
                    rows = cur.fetchall()
                    if rows:
                        columns = [desc[0] for desc in cur.description]
                        tasks = []
                        for row in rows:
                            task = dict(zip(columns, row))
                            # 解析JSON字段
                            for key in ['symbols', 'results']:
                                if task.get(key) and isinstance(task[key], str):
                                    try:
                                        task[key] = json.loads(task[key])
                                    except:
                                        pass
                            tasks.append(task)
                        return tasks
        except Exception as e:
            logger.debug(f"从PostgreSQL查询失败: {e}")
        
        # 降级到文件系统
        ensure_directories()
        tasks = []
        
        for filename in sorted(os.listdir(FEATURE_SELECTION_TASKS_DIR), reverse=True):
            if filename.endswith('.json'):
                filepath = os.path.join(FEATURE_SELECTION_TASKS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        task = json.load(f)
                        if status is None or task.get('status') == status:
                            tasks.append(task)
                            if len(tasks) >= limit:
                                break
                except Exception as e:
                    logger.warning(f"读取任务文件失败 {filename}: {e}")
        
        return tasks[offset:offset+limit]
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        return []


def get_selection_tasks_stats() -> Dict[str, Any]:
    """
    获取特征选择任务统计
    
    Returns:
        统计信息字典
    """
    try:
        # 优先从PostgreSQL查询
        try:
            from src.infrastructure.persistence.postgresql import get_postgresql_connection
            
            conn = get_postgresql_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE status = 'running') as active,
                            COUNT(*) FILTER (WHERE status = 'completed') as completed,
                            COUNT(*) FILTER (WHERE status = 'failed') as failed
                        FROM feature_selection_tasks
                    """)
                    
                    row = cur.fetchone()
                    if row:
                        return {
                            "total_tasks": row[0],
                            "active_tasks": row[1],
                            "completed_tasks": row[2],
                            "failed_tasks": row[3]
                        }
        except Exception as e:
            logger.debug(f"从PostgreSQL查询统计失败: {e}")
        
        # 降级到文件系统
        tasks = list_selection_tasks(limit=10000)
        
        return {
            "total_tasks": len(tasks),
            "active_tasks": len([t for t in tasks if t.get('status') == 'running']),
            "completed_tasks": len([t for t in tasks if t.get('status') == 'completed']),
            "failed_tasks": len([t for t in tasks if t.get('status') == 'failed'])
        }
    except Exception as e:
        logger.error(f"获取任务统计失败: {e}")
        return {
            "total_tasks": 0,
            "active_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0
        }
