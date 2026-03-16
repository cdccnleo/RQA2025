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
                source_task_id VARCHAR(100),
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
                source_task_id, selection_method, n_features, auto_execute,
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
            task.get("source_task_id"),
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
        config: 配置参数，包括n_features（选择特征数）、auto_execute（是否自动执行）等
    
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
