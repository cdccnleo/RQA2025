"""
模型训练任务持久化模块
存储训练任务到文件系统或PostgreSQL
符合架构设计：使用统一日志系统
"""

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
    import logging
    logger = logging.getLogger(__name__)

# 数据存储目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data")
TRAINING_JOBS_DIR = os.path.join(DATA_DIR, "training_jobs")


def ensure_directories():
    """确保数据目录存在"""
    os.makedirs(TRAINING_JOBS_DIR, exist_ok=True)


def save_training_job(job: Dict[str, Any]) -> bool:
    """
    保存训练任务到持久化存储
    
    Args:
        job: 任务信息字典，必须包含job_id字段
    
    Returns:
        是否成功保存
    """
    try:
        ensure_directories()
        
        job_id = job.get("job_id")
        if not job_id:
            logger.error("任务缺少job_id字段，无法保存")
            return False
        
        filepath = os.path.join(TRAINING_JOBS_DIR, f"{job_id}.json")
        
        # 添加保存时间戳
        job_data = job.copy()
        job_data["saved_at"] = time.time()
        job_data["updated_at"] = time.time()
        
        # 如果start_time是整数时间戳，保持不变；如果是datetime对象，转换为时间戳
        if "start_time" in job_data:
            if isinstance(job_data["start_time"], datetime):
                job_data["start_time"] = int(job_data["start_time"].timestamp())
            elif isinstance(job_data["start_time"], str):
                try:
                    dt = datetime.fromisoformat(job_data["start_time"].replace("Z", "+00:00"))
                    job_data["start_time"] = int(dt.timestamp())
                except:
                    pass
        
        # 如果end_time存在，同样处理
        if "end_time" in job_data and job_data["end_time"]:
            if isinstance(job_data["end_time"], datetime):
                job_data["end_time"] = int(job_data["end_time"].timestamp())
            elif isinstance(job_data["end_time"], str):
                try:
                    dt = datetime.fromisoformat(job_data["end_time"].replace("Z", "+00:00"))
                    job_data["end_time"] = int(dt.timestamp())
                except:
                    pass
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(job_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练任务已保存: {job_id}")
        
        # 同时尝试保存到PostgreSQL（如果可用）
        try:
            _save_to_postgresql(job_data)
        except Exception as e:
            logger.debug(f"保存到PostgreSQL失败（使用文件系统）: {e}")
        
        return True
    except Exception as e:
        logger.error(f"保存训练任务失败: {e}")
        return False


def _save_to_postgresql(job: Dict[str, Any]) -> bool:
    """尝试保存任务到PostgreSQL"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # 确保表存在
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_training_jobs (
                job_id VARCHAR(100) PRIMARY KEY,
                model_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL,
                progress INTEGER DEFAULT 0,
                accuracy DECIMAL(10, 6),
                loss DECIMAL(10, 6),
                start_time BIGINT,
                end_time BIGINT,
                training_time INTEGER DEFAULT 0,
                config JSONB,
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status 
            ON model_training_jobs(status);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_jobs_created 
            ON model_training_jobs(created_at DESC);
        """)
        
        # 插入或更新任务
        cursor.execute("""
            INSERT INTO model_training_jobs (
                job_id, model_type, status, progress, accuracy, loss,
                start_time, end_time, training_time, config, error_message, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
            )
            ON CONFLICT (job_id) 
            DO UPDATE SET
                status = EXCLUDED.status,
                progress = EXCLUDED.progress,
                accuracy = EXCLUDED.accuracy,
                loss = EXCLUDED.loss,
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time,
                training_time = EXCLUDED.training_time,
                config = EXCLUDED.config,
                error_message = EXCLUDED.error_message,
                updated_at = CURRENT_TIMESTAMP
        """, (
            job.get("job_id"),
            job.get("model_type", ""),
            job.get("status", "pending"),
            job.get("progress", 0),
            job.get("accuracy"),
            job.get("loss"),
            job.get("start_time"),
            job.get("end_time"),
            job.get("training_time", 0),
            json.dumps(job.get("config", {})),
            job.get("error_message")
        ))
        
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        logger.debug(f"训练任务已保存到PostgreSQL: {job.get('job_id')}")
        return True
    except Exception as e:
        logger.debug(f"保存到PostgreSQL失败: {e}")
        return False


def load_training_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    加载训练任务
    
    Args:
        job_id: 任务ID
    
    Returns:
        任务信息字典，如果不存在则返回None
    """
    try:
        # 优先从PostgreSQL加载
        job = _load_from_postgresql(job_id)
        if job:
            return job
        
        # 如果PostgreSQL没有，从文件系统加载
        filepath = os.path.join(TRAINING_JOBS_DIR, f"{job_id}.json")
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            job = json.load(f)
        
        # 转换时间戳为整数（如果存在）
        if "start_time" in job and isinstance(job["start_time"], str):
            try:
                dt = datetime.fromisoformat(job["start_time"].replace("Z", "+00:00"))
                job["start_time"] = int(dt.timestamp())
            except:
                pass
        
        if "end_time" in job and job["end_time"] and isinstance(job["end_time"], str):
            try:
                dt = datetime.fromisoformat(job["end_time"].replace("Z", "+00:00"))
                job["end_time"] = int(dt.timestamp())
            except:
                pass
        
        return job
    except Exception as e:
        logger.error(f"加载训练任务失败: {e}")
        return None


def _load_from_postgresql(job_id: str) -> Optional[Dict[str, Any]]:
    """从PostgreSQL加载任务"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT job_id, model_type, status, progress, accuracy, loss,
                   start_time, end_time, training_time, config, error_message,
                   created_at, updated_at
            FROM model_training_jobs
            WHERE job_id = %s
        """, (job_id,))
        
        row = cursor.fetchone()
        cursor.close()
        return_db_connection(conn)
        
        if not row:
            return None
        
        # 转换为字典
        job = {
            "job_id": row[0],
            "model_type": row[1],
            "status": row[2],
            "progress": row[3],
            "accuracy": float(row[4]) if row[4] else None,
            "loss": float(row[5]) if row[5] else None,
            "start_time": row[6],
            "end_time": row[7],
            "training_time": row[8],
            "config": json.loads(row[9]) if row[9] else {},
            "error_message": row[10],
            "created_at": int(row[11].timestamp()) if row[11] else None,
            "updated_at": int(row[12].timestamp()) if row[12] else None
        }
        
        return job
    except Exception as e:
        logger.debug(f"从PostgreSQL加载任务失败: {e}")
        return None


def list_training_jobs(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    列出训练任务
    
    Args:
        status: 任务状态过滤器（可选）
        limit: 返回的最大任务数
    
    Returns:
        任务列表
    """
    try:
        jobs = []
        
        # 优先从PostgreSQL加载
        try:
            pg_jobs = _list_from_postgresql(status, limit)
            if pg_jobs:
                jobs.extend(pg_jobs)
        except Exception as e:
            logger.debug(f"从PostgreSQL加载任务列表失败: {e}")
        
        # 如果PostgreSQL没有足够的数据，从文件系统补充
        if len(jobs) < limit:
            file_jobs = _list_from_filesystem(status, limit - len(jobs))
            # 合并任务，去重（按job_id）
            existing_ids = {j.get("job_id") for j in jobs}
            for job in file_jobs:
                if job.get("job_id") not in existing_ids:
                    jobs.append(job)
        
        # 按创建时间排序
        jobs.sort(key=lambda x: x.get("start_time", x.get("created_at", 0)), reverse=True)
        
        return jobs[:limit]
    except Exception as e:
        logger.error(f"列出训练任务失败: {e}")
        return []


def _list_from_postgresql(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """从PostgreSQL列出任务"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT job_id, model_type, status, progress, accuracy, loss,
                       start_time, end_time, training_time, config, error_message,
                       created_at, updated_at
                FROM model_training_jobs
                WHERE status = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (status, limit))
        else:
            cursor.execute("""
                SELECT job_id, model_type, status, progress, accuracy, loss,
                       start_time, end_time, training_time, config, error_message,
                       created_at, updated_at
                FROM model_training_jobs
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
        
        rows = cursor.fetchall()
        cursor.close()
        return_db_connection(conn)
        
        jobs = []
        for row in rows:
            job = {
                "job_id": row[0],
                "model_type": row[1],
                "status": row[2],
                "progress": row[3],
                "accuracy": float(row[4]) if row[4] else None,
                "loss": float(row[5]) if row[5] else None,
                "start_time": row[6],
                "end_time": row[7],
                "training_time": row[8],
                "config": json.loads(row[9]) if row[9] else {},
                "error_message": row[10],
                "created_at": int(row[11].timestamp()) if row[11] else None,
                "updated_at": int(row[12].timestamp()) if row[12] else None
            }
            jobs.append(job)
        
        return jobs
    except Exception as e:
        logger.debug(f"从PostgreSQL列出任务失败: {e}")
        return []


def _list_from_filesystem(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """从文件系统列出任务"""
    try:
        ensure_directories()
        
        jobs = []
        for filename in os.listdir(TRAINING_JOBS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(TRAINING_JOBS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        job = json.load(f)
                    
                    # 应用状态过滤器
                    if status and job.get("status") != status:
                        continue
                    
                    jobs.append(job)
                except Exception as e:
                    logger.warning(f"加载任务文件失败 {filename}: {e}")
        
        # 按保存时间排序
        jobs.sort(key=lambda x: x.get("saved_at", x.get("start_time", 0)), reverse=True)
        
        return jobs[:limit]
    except Exception as e:
        logger.error(f"从文件系统列出任务失败: {e}")
        return []


def update_training_job(job_id: str, updates: Dict[str, Any]) -> bool:
    """
    更新训练任务
    
    Args:
        job_id: 任务ID
        updates: 要更新的字段字典
    
    Returns:
        是否成功更新
    """
    try:
        # 加载现有任务
        job = load_training_job(job_id)
        if not job:
            logger.warning(f"任务不存在: {job_id}")
            return False
        
        # 更新字段
        job.update(updates)
        job["updated_at"] = time.time()
        
        # 保存更新后的任务
        return save_training_job(job)
    except Exception as e:
        logger.error(f"更新训练任务失败: {e}")
        return False


def delete_training_job(job_id: str) -> bool:
    """
    删除训练任务
    
    Args:
        job_id: 任务ID
    
    Returns:
        是否成功删除
    """
    try:
        # 从文件系统删除
        filepath = os.path.join(TRAINING_JOBS_DIR, f"{job_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # 从PostgreSQL删除
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM model_training_jobs WHERE job_id = %s", (job_id,))
                conn.commit()
                cursor.close()
                return_db_connection(conn)
        except Exception as e:
            logger.debug(f"从PostgreSQL删除任务失败: {e}")
        
        logger.info(f"训练任务已删除: {job_id}")
        return True
    except Exception as e:
        logger.error(f"删除训练任务失败: {e}")
        return False

