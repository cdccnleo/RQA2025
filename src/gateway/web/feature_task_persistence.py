"""
特征工程任务持久化模块
存储特征提取任务到文件系统或PostgreSQL
符合架构设计：使用统一日志系统
"""

import json
import os
from typing import Dict, List, Any, Optional
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


# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 数据存储目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data")
FEATURE_TASKS_DIR = os.path.join(DATA_DIR, "feature_tasks")


def ensure_directories():
    """确保数据目录存在"""
    os.makedirs(FEATURE_TASKS_DIR, exist_ok=True)


def save_feature_task(task: Dict[str, Any]) -> bool:
    """
    保存特征提取任务到持久化存储
    
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
        
        filepath = os.path.join(FEATURE_TASKS_DIR, f"{task_id}.json")
        
        # 添加保存时间戳
        task_data = task.copy()
        task_data["saved_at"] = time.time()
        task_data["updated_at"] = time.time()
        
        # 如果start_time是整数时间戳，保持不变；如果是datetime对象，转换为时间戳
        if "start_time" in task_data:
            if isinstance(task_data["start_time"], datetime):
                task_data["start_time"] = int(task_data["start_time"].timestamp())
            elif isinstance(task_data["start_time"], str):
                try:
                    dt = datetime.fromisoformat(task_data["start_time"].replace("Z", "+00:00"))
                    task_data["start_time"] = int(dt.timestamp())
                except:
                    pass
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2, cls=NumpyJSONEncoder)
        
        logger.info(f"特征提取任务已保存: {task_id}")
        
        # 同时尝试保存到PostgreSQL（如果可用）
        try:
            _save_to_postgresql(task_data)
        except Exception as e:
            logger.debug(f"保存到PostgreSQL失败（使用文件系统）: {e}")
        
        return True
    except Exception as e:
        logger.error(f"保存特征任务失败: {e}")
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
            CREATE TABLE IF NOT EXISTS feature_engineering_tasks (
                task_id VARCHAR(100) PRIMARY KEY,
                task_type VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL,
                progress INTEGER DEFAULT 0,
                feature_count INTEGER DEFAULT 0,
                start_time BIGINT,
                end_time BIGINT,
                config JSONB,
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 创建索引 - 支持分页和筛选
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_tasks_status 
            ON feature_engineering_tasks(status);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_tasks_created 
            ON feature_engineering_tasks(created_at DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_tasks_type 
            ON feature_engineering_tasks(task_type);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_tasks_status_created 
            ON feature_engineering_tasks(status, created_at DESC);
        """)
        
        # 插入或更新任务
        cursor.execute("""
            INSERT INTO feature_engineering_tasks (
                task_id, task_type, status, progress, feature_count,
                start_time, end_time, config, error_message, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
            )
            ON CONFLICT (task_id) 
            DO UPDATE SET
                status = EXCLUDED.status,
                progress = EXCLUDED.progress,
                feature_count = EXCLUDED.feature_count,
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time,
                config = EXCLUDED.config,
                error_message = EXCLUDED.error_message,
                updated_at = CURRENT_TIMESTAMP
        """, (
            task.get("task_id"),
            task.get("task_type", ""),
            task.get("status", "pending"),
            task.get("progress", 0),
            task.get("feature_count", 0),
            task.get("start_time"),
            task.get("end_time"),
            json.dumps(task.get("config", {})),
            task.get("error_message")
        ))
        
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        logger.debug(f"特征任务已保存到PostgreSQL: {task.get('task_id')}")
        return True
    except Exception as e:
        logger.debug(f"保存到PostgreSQL失败: {e}")
        return False


def load_feature_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    加载特征提取任务
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务信息字典，如果不存在则返回None
    """
    try:
        # 优先从PostgreSQL加载
        task = _load_from_postgresql(task_id)
        if task:
            return task
        
        # 如果PostgreSQL没有，从文件系统加载
        filepath = os.path.join(FEATURE_TASKS_DIR, f"{task_id}.json")
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            task = json.load(f)
        
        # 转换时间戳为整数（如果存在）
        if "start_time" in task and isinstance(task["start_time"], str):
            try:
                dt = datetime.fromisoformat(task["start_time"].replace("Z", "+00:00"))
                task["start_time"] = int(dt.timestamp())
            except:
                pass
        
        return task
    except Exception as e:
        logger.error(f"加载特征任务失败: {e}")
        return None


def _load_from_postgresql(task_id: str) -> Optional[Dict[str, Any]]:
    """从PostgreSQL加载任务"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT task_id, task_type, status, progress, feature_count,
                   start_time, end_time, config, error_message,
                   created_at, updated_at
            FROM feature_engineering_tasks
            WHERE task_id = %s
        """, (task_id,))
        
        row = cursor.fetchone()
        cursor.close()
        return_db_connection(conn)
        
        if not row:
            return None
        
        # 转换为字典
        task = {
            "task_id": row[0],
            "task_type": row[1],
            "status": row[2],
            "progress": row[3],
            "feature_count": row[4],
            "start_time": row[5],
            "end_time": row[6],
            "config": json.loads(row[7]) if row[7] else {},
            "error_message": row[8],
            "created_at": int(row[9].timestamp()) if row[9] else None,
            "updated_at": int(row[10].timestamp()) if row[10] else None
        }
        
        return task
    except Exception as e:
        logger.debug(f"从PostgreSQL加载任务失败: {e}")
        return None


def list_feature_tasks(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    列出特征提取任务。以 PostgreSQL 为主数据源，与库中数据一致；
    PG 不可用或无数据时再回退到文件系统，并将文件系统中的任务同步到PostgreSQL。
    """
    try:
        tasks = []
        pg_available = False

        # 1. 优先从 PostgreSQL 加载（主数据源，保证与库一致）
        try:
            pg_tasks = _list_from_postgresql(status, limit)
            pg_available = True  # 标记PostgreSQL可用
            for task in pg_tasks:
                if task.get("task_id"):
                    tasks.append(task)
            logger.debug(f"从PostgreSQL加载了 {len(tasks)} 个任务")
        except Exception as e:
            logger.debug(f"从PostgreSQL加载任务列表失败: {e}")

        # 2. 如果PostgreSQL有数据，直接返回，保证与库一致
        if tasks:
            # 统一使用 created_at 进行排序，避免 start_time 和 created_at 类型不一致的问题
            tasks.sort(key=lambda x: x.get("created_at", 0) or 0, reverse=True)
            logger.debug(f"PostgreSQL有数据，返回 {len(tasks)} 个任务")
            return tasks[:limit]

        # 3. PG 无数据或不可用时，从文件系统加载（降级）
        if not pg_available or not tasks:
            try:
                file_tasks = _list_from_filesystem(status, limit)
                for task in file_tasks:
                    tid = task.get("task_id")
                    if tid:
                        tasks.append(task)
                        # 将文件系统中的任务同步到PostgreSQL
                        try:
                            sync_success = _save_to_postgresql(task)
                            if sync_success:
                                logger.debug(f"已将文件系统任务 {tid} 同步到PostgreSQL")
                            else:
                                logger.debug(f"同步任务到PostgreSQL失败: 数据库连接不可用")
                        except Exception as e:
                            logger.debug(f"同步任务到PostgreSQL失败: {e}")
                logger.debug(f"从文件系统加载并同步了 {len(tasks)} 个任务到PostgreSQL")
                tasks.sort(key=lambda x: x.get("start_time", x.get("created_at", 0)), reverse=True)
                return tasks[:limit]
            except Exception as e:
                logger.debug(f"从文件系统加载任务列表失败: {e}")

        logger.debug("没有找到特征提取任务")
        return []
    except Exception as e:
        logger.error(f"列出特征任务失败: {e}")
        return []


def _list_from_postgresql(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """从PostgreSQL列出任务"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            logger.warning("数据库连接失败，无法列出任务")
            return []
        
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT task_id, task_type, status, progress, feature_count,
                       start_time, end_time, config, error_message,
                       created_at, updated_at
                FROM feature_engineering_tasks
                WHERE status = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (status, limit))
        else:
            cursor.execute("""
                SELECT task_id, task_type, status, progress, feature_count,
                       start_time, end_time, config, error_message,
                       created_at, updated_at
                FROM feature_engineering_tasks
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
        
        rows = cursor.fetchall()
        logger.info(f"从PostgreSQL查询到 {len(rows)} 个任务")
        
        cursor.close()
        return_db_connection(conn)
        
        tasks = []
        for i, row in enumerate(rows):
            try:
                # 处理 config 字段，可能是字符串或字典
                config_value = row[7]
                if isinstance(config_value, str):
                    config = json.loads(config_value) if config_value else {}
                elif isinstance(config_value, dict):
                    config = config_value
                else:
                    config = {}
                
                task = {
                    "task_id": row[0],
                    "task_type": row[1],
                    "status": row[2],
                    "progress": row[3],
                    "feature_count": row[4],
                    "start_time": row[5],
                    "end_time": row[6],
                    "config": config,
                    "error_message": row[8],
                    "created_at": int(row[9].timestamp()) if row[9] else None,
                    "updated_at": int(row[10].timestamp()) if row[10] else None
                }
                tasks.append(task)
            except Exception as e:
                logger.error(f"处理第 {i+1} 个任务时出错: {e}, task_id={row[0] if row else 'N/A'}")
                # 继续处理下一个任务
                continue
        
        logger.info(f"成功处理 {len(tasks)} 个任务")
        return tasks
    except Exception as e:
        logger.error(f"从PostgreSQL列出任务失败: {e}", exc_info=True)
        return []


def _list_from_filesystem(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """从文件系统列出任务"""
    try:
        ensure_directories()
        
        tasks = []
        for filename in os.listdir(FEATURE_TASKS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(FEATURE_TASKS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        task = json.load(f)
                    
                    # 应用状态过滤器
                    if status and task.get("status") != status:
                        continue
                    
                    tasks.append(task)
                except Exception as e:
                    logger.warning(f"加载任务文件失败 {filename}: {e}")
        
        # 按保存时间排序
        tasks.sort(key=lambda x: x.get("saved_at", x.get("start_time", 0)), reverse=True)
        
        return tasks[:limit]
    except Exception as e:
        logger.error(f"从文件系统列出任务失败: {e}")
        return []


def update_feature_task(task_id: str, updates: Dict[str, Any]) -> bool:
    """
    更新特征提取任务
    
    Args:
        task_id: 任务ID
        updates: 要更新的字段字典
    
    Returns:
        是否成功更新
    """
    try:
        # 加载现有任务
        task = load_feature_task(task_id)
        if not task:
            logger.warning(f"任务不存在: {task_id}")
            return False
        
        # 更新字段
        task.update(updates)
        task["updated_at"] = time.time()
        
        # 保存更新后的任务
        return save_feature_task(task)
    except Exception as e:
        logger.error(f"更新特征任务失败: {e}")
        return False


def delete_feature_task(task_id: str) -> bool:
    """
    删除特征提取任务，同时删除关联的特征存储数据
    
    Args:
        task_id: 任务ID
    
    Returns:
        是否成功删除
    """
    try:
        # 从文件系统删除
        filepath = os.path.join(FEATURE_TASKS_DIR, f"{task_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # 从PostgreSQL删除任务
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM feature_engineering_tasks WHERE task_id = %s", (task_id,))
                conn.commit()
                cursor.close()
                return_db_connection(conn)
        except Exception as e:
            logger.debug(f"从PostgreSQL删除任务失败: {e}")
        
        # 🗑️ 同步删除feature_store表中的关联特征数据
        try:
            deleted_count = delete_features_from_store_by_task(task_id)
            if deleted_count > 0:
                logger.info(f"已同步删除 {deleted_count} 个关联特征数据，任务ID: {task_id}")
        except Exception as e:
            logger.warning(f"删除关联特征数据失败: {e}")
        
        logger.info(f"特征任务已删除: {task_id}")
        return True
    except Exception as e:
        logger.error(f"删除特征任务失败: {e}")
        return False


def _ensure_feature_store_table(conn) -> bool:
    """
    确保feature_store表存在，如果不存在则创建
    
    Args:
        conn: PostgreSQL数据库连接
    
    Returns:
        是否成功创建或表已存在
    """
    try:
        cursor = conn.cursor()
        
        # 创建特征存储表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_store (
                feature_id VARCHAR(200) PRIMARY KEY,
                task_id VARCHAR(100) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                feature_type VARCHAR(50),
                parameters JSONB,
                symbol VARCHAR(20),
                quality_score DECIMAL(5, 4),
                importance DECIMAL(5, 4),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_task_id 
            ON feature_store(task_id);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_symbol 
            ON feature_store(symbol);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_feature_type 
            ON feature_store(feature_type);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_created_at 
            ON feature_store(created_at DESC);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_task_feature 
            ON feature_store(task_id, feature_name);
        """)
        
        conn.commit()
        cursor.close()
        logger.debug("✅ feature_store表已创建或已存在")
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建feature_store表失败: {e}")
        return False


def save_features_to_store(task_id: str, features: List[str], symbol: str = None, 
                          feature_types: Dict[str, str] = None, 
                          quality_scores: Dict[str, float] = None) -> bool:
    """
    保存特征到特征存储表
    
    Args:
        task_id: 任务ID
        features: 特征名称列表
        symbol: 股票代码
        feature_types: 特征类型字典 {特征名: 类型}
        quality_scores: 特征质量评分字典 {特征名: 评分}
    
    Returns:
        是否成功保存
    """
    logger.info(f"💾 save_features_to_store 被调用，任务ID: {task_id}, 特征数量: {len(features)}, 股票代码: {symbol}")
    
    # 过滤基础价格特征（双重保障）
    basic_price_features = {'open', 'high', 'low', 'close', 'volume', 'amount', 'date', 'datetime', 'timestamp'}
    filtered_features = [f for f in features if f.lower() not in basic_price_features]
    filtered_count = len(features) - len(filtered_features)
    if filtered_count > 0:
        logger.info(f"📝 save_features_to_store 内部过滤了 {filtered_count} 个基础价格特征")
    
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            logger.error("❌ PostgreSQL连接不可用，无法保存特征到存储表")
            return False
        
        logger.info(f"✅ PostgreSQL连接成功")
        
        # 确保feature_store表存在
        if not _ensure_feature_store_table(conn):
            logger.error("❌ 无法创建feature_store表，保存操作取消")
            return False
        
        cursor = conn.cursor()
        logger.info(f"📝 开始插入 {len(filtered_features)} 个特征到数据库")
        
        # 解析特征名称，提取特征类型和参数
        import re
        
        inserted_count = 0
        for feature_name in filtered_features:
            # 生成特征ID
            feature_id = f"{task_id}_{feature_name}"
            
            # 解析特征类型和参数
            feature_type = None
            parameters = {}
            
            if feature_types and feature_name in feature_types:
                feature_type = feature_types[feature_name]
            else:
                # 尝试从特征名解析，如 SMA_5, EMA_10
                match = re.match(r'([A-Za-z]+)_(\d+)', feature_name)
                if match:
                    feature_type = match.group(1).upper()
                    parameters['period'] = int(match.group(2))
                else:
                    # 尝试其他格式，如 RSI, MACD
                    base_name = feature_name.split('_')[0] if '_' in feature_name else feature_name
                    feature_type = base_name.upper()
            
            # 获取质量评分
            quality_score = quality_scores.get(feature_name) if quality_scores else None
            
            # 插入或更新特征记录
            # 使用 psycopg2 的 JSONB 适配，直接传递 Python 字典
            import psycopg2.extras
            cursor.execute("""
                INSERT INTO feature_store 
                (feature_id, task_id, feature_name, feature_type, parameters, symbol, quality_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (feature_id) DO UPDATE SET
                    feature_type = EXCLUDED.feature_type,
                    parameters = EXCLUDED.parameters,
                    quality_score = EXCLUDED.quality_score,
                    updated_at = CURRENT_TIMESTAMP
            """, (feature_id, task_id, feature_name, feature_type, 
                  psycopg2.extras.Json(parameters) if parameters else None, 
                  symbol, quality_score))
            inserted_count += 1
        
        conn.commit()
        logger.info(f"✅ 数据库提交成功，插入/更新 {inserted_count} 个特征")
        cursor.close()
        return_db_connection(conn)
        
        logger.info(f"✅ 已保存 {len(features)} 个特征到特征存储表，任务ID: {task_id}")
        return True
        
    except Exception as e:
        logger.error(f"保存特征到存储表失败: {e}")
        return False


def get_features_from_store(task_id: str = None, symbol: str = None, 
                           feature_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    从特征存储表获取特征列表
    
    Args:
        task_id: 任务ID（可选）
        symbol: 股票代码（可选）
        feature_type: 特征类型（可选）
        limit: 返回数量限制
    
    Returns:
        特征列表
    """
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            logger.warning("PostgreSQL连接不可用，无法从存储表获取特征")
            return []
        
        cursor = conn.cursor()
        
        # 构建查询条件
        conditions = []
        params = []
        
        if task_id:
            conditions.append("task_id = %s")
            params.append(task_id)
        if symbol:
            conditions.append("symbol = %s")
            params.append(symbol)
        if feature_type:
            conditions.append("feature_type = %s")
            params.append(feature_type)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        cursor.execute(f"""
            SELECT feature_id, task_id, feature_name, feature_type, parameters, 
                   symbol, quality_score, importance, created_at, updated_at
            FROM feature_store
            {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
        """, params + [limit])
        
        rows = cursor.fetchall()
        cursor.close()
        return_db_connection(conn)
        
        features = []
        for row in rows:
            features.append({
                "feature_id": row[0],
                "task_id": row[1],
                "name": row[2],
                "feature_type": row[3],
                "parameters": row[4] if row[4] else {},
                "symbol": row[5],
                "quality_score": row[6],
                "importance": row[7],
                "created_at": row[8].timestamp() if row[8] else None,
                "updated_at": row[9].timestamp() if row[9] else None
            })
        
        return features
        
    except Exception as e:
        logger.error(f"从存储表获取特征失败: {e}")
        return []


def delete_features_from_store_by_task(task_id: str) -> int:
    """
    根据任务ID删除feature_store表中的特征数据
    
    Args:
        task_id: 任务ID
    
    Returns:
        删除的特征数量
    """
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            logger.warning("PostgreSQL连接不可用，无法删除特征存储数据")
            return 0
        
        cursor = conn.cursor()
        
        # 先查询有多少条记录
        cursor.execute("SELECT COUNT(*) FROM feature_store WHERE task_id = %s", (task_id,))
        count = cursor.fetchone()[0]
        
        # 删除关联的特征数据
        cursor.execute("DELETE FROM feature_store WHERE task_id = %s", (task_id,))
        conn.commit()
        cursor.close()
        return_db_connection(conn)
        
        logger.debug(f"已从feature_store删除 {count} 个特征数据，任务ID: {task_id}")
        return count
        
    except Exception as e:
        logger.error(f"从feature_store删除特征数据失败: {e}")
        return 0


def update_feature_task_status(task_id: str, status: str, feature_count: int = None, error_message: str = None) -> bool:
    """
    更新特征提取任务状态
    
    用于调度器在任务完成或失败时同步更新数据库状态
    
    Args:
        task_id: 任务ID
        status: 新状态 (completed/failed/running等)
        feature_count: 特征数量（可选）
        error_message: 错误信息（可选）
    
    Returns:
        是否成功更新
    """
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        
        conn = get_db_connection()
        if not conn:
            logger.warning("PostgreSQL连接不可用，无法更新任务状态")
            return False
        
        cursor = conn.cursor()
        
        # 构建更新SQL
        update_fields = ["status = %s", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]
        
        if feature_count is not None:
            update_fields.append("feature_count = %s")
            params.append(feature_count)
        
        if error_message is not None:
            update_fields.append("error_message = %s")
            params.append(error_message)
        
        # 如果状态是completed或failed，设置end_time
        if status in ["completed", "failed"]:
            update_fields.append("end_time = %s")
            params.append(int(time.time()))
        
        # 添加task_id到参数列表
        params.append(task_id)
        
        # 执行更新
        sql = f"""
            UPDATE feature_engineering_tasks 
            SET {', '.join(update_fields)}
            WHERE task_id = %s
        """
        
        cursor.execute(sql, tuple(params))
        conn.commit()
        
        updated_rows = cursor.rowcount
        cursor.close()
        return_db_connection(conn)
        
        if updated_rows > 0:
            logger.debug(f"任务状态已更新到数据库: {task_id} -> {status}")
            return True
        else:
            logger.warning(f"数据库中没有找到任务: {task_id}")
            return False
            
    except Exception as e:
        logger.error(f"更新任务状态到数据库失败: {e}")
        return False

