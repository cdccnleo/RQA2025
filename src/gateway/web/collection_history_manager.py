"""
采集历史记录管理器

管理数据采集历史记录的创建、查询和统计。
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class CollectionHistoryRecord:
    """采集历史记录数据类"""
    id: Optional[int] = None
    source_id: str = ""
    collection_time: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # success/failed/pending
    records_collected: Optional[int] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    task_id: Optional[str] = None
    collection_type: str = "scheduled"  # scheduled/manual


class CollectionHistoryManager:
    """采集历史记录管理器"""
    
    def __init__(self):
        self._db_pool: Optional[asyncpg.Pool] = None
    
    async def _get_db_pool(self) -> asyncpg.Pool:
        """获取数据库连接池"""
        if self._db_pool is None:
            import os
            db_password = os.environ.get('DB_PASSWORD', 'SecurePass123!')
            self._db_pool = await asyncpg.create_pool(
                host="rqa2025-postgres",
                port=5432,
                database="rqa2025_prod",
                user="rqa2025_admin",
                password=db_password,
                min_size=2,
                max_size=5
            )
        return self._db_pool
    
    async def create_record(
        self,
        source_id: str,
        status: str,
        collection_type: str = "scheduled",
        records_collected: Optional[int] = None,
        error_message: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_ms: Optional[int] = None,
        task_id: Optional[str] = None
    ) -> int:
        """
        创建采集历史记录
        
        Args:
            source_id: 数据源ID
            status: 采集状态 (success/failed/pending)
            collection_type: 采集类型 (scheduled/manual)
            records_collected: 采集记录数
            error_message: 错误信息
            start_time: 开始时间
            end_time: 结束时间
            duration_ms: 耗时（毫秒）
            task_id: 任务ID
            
        Returns:
            int: 记录ID
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO data_collection_history 
                    (source_id, collection_time, status, records_collected, 
                     error_message, start_time, end_time, duration_ms, task_id, collection_type)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                    """,
                    source_id,
                    datetime.now(),
                    status,
                    records_collected,
                    error_message,
                    start_time,
                    end_time,
                    duration_ms,
                    task_id,
                    collection_type
                )
                
                logger.info(f"✅ 创建采集历史记录: {source_id}, 状态: {status}, ID: {row['id']}")
                return row['id']
                
        except Exception as e:
            logger.error(f"❌ 创建采集历史记录失败: {e}")
            raise
    
    async def update_record(
        self,
        record_id: int,
        status: Optional[str] = None,
        records_collected: Optional[int] = None,
        error_message: Optional[str] = None,
        end_time: Optional[datetime] = None,
        duration_ms: Optional[int] = None
    ) -> bool:
        """
        更新采集历史记录
        
        Args:
            record_id: 记录ID
            status: 采集状态
            records_collected: 采集记录数
            error_message: 错误信息
            end_time: 结束时间
            duration_ms: 耗时（毫秒）
            
        Returns:
            bool: 是否成功更新
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                # 构建动态更新SQL
                updates = []
                params = []
                param_idx = 1
                
                if status is not None:
                    updates.append(f"status = ${param_idx}")
                    params.append(status)
                    param_idx += 1
                
                if records_collected is not None:
                    updates.append(f"records_collected = ${param_idx}")
                    params.append(records_collected)
                    param_idx += 1
                
                if error_message is not None:
                    updates.append(f"error_message = ${param_idx}")
                    params.append(error_message)
                    param_idx += 1
                
                if end_time is not None:
                    updates.append(f"end_time = ${param_idx}")
                    params.append(end_time)
                    param_idx += 1
                
                if duration_ms is not None:
                    updates.append(f"duration_ms = ${param_idx}")
                    params.append(duration_ms)
                    param_idx += 1
                
                if not updates:
                    return True
                
                query = f"""
                    UPDATE data_collection_history
                    SET {', '.join(updates)}
                    WHERE id = ${param_idx}
                """
                params.append(record_id)
                
                result = await conn.execute(query, *params)
                
                if result == "UPDATE 1":
                    logger.debug(f"✅ 更新采集历史记录: ID={record_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"❌ 更新采集历史记录失败: {e}")
            return False
    
    async def get_history(
        self,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        collection_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[CollectionHistoryRecord]:
        """
        获取采集历史记录
        
        Args:
            source_id: 数据源ID过滤
            status: 状态过滤
            collection_type: 采集类型过滤
            start_date: 开始日期
            end_date: 结束日期
            limit: 返回记录数限制
            offset: 偏移量
            
        Returns:
            List[CollectionHistoryRecord]: 采集历史记录列表
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                query = """
                    SELECT id, source_id, collection_time, status, records_collected,
                           error_message, start_time, end_time, duration_ms, task_id, collection_type
                    FROM data_collection_history
                    WHERE 1=1
                """
                params = []
                
                if source_id:
                    query += f" AND source_id = ${len(params) + 1}"
                    params.append(source_id)
                
                if status:
                    query += f" AND status = ${len(params) + 1}"
                    params.append(status)
                
                if collection_type:
                    query += f" AND collection_type = ${len(params) + 1}"
                    params.append(collection_type)
                
                if start_date:
                    query += f" AND collection_time >= ${len(params) + 1}"
                    params.append(start_date)
                
                if end_date:
                    query += f" AND collection_time <= ${len(params) + 1}"
                    params.append(end_date)
                
                query += " ORDER BY collection_time DESC"
                query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
                params.extend([limit, offset])
                
                rows = await conn.fetch(query, *params)
                
                return [
                    CollectionHistoryRecord(
                        id=row['id'],
                        source_id=row['source_id'],
                        collection_time=row['collection_time'],
                        status=row['status'],
                        records_collected=row['records_collected'],
                        error_message=row['error_message'],
                        start_time=row['start_time'],
                        end_time=row['end_time'],
                        duration_ms=row['duration_ms'],
                        task_id=row['task_id'],
                        collection_type=row['collection_type']
                    )
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"❌ 获取采集历史记录失败: {e}")
            return []
    
    async def get_stats(
        self,
        source_id: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        获取采集统计信息
        
        Args:
            source_id: 数据源ID过滤
            days: 统计天数
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                start_date = datetime.now() - timedelta(days=days)
                
                # 基础查询条件
                base_query = "FROM data_collection_history WHERE collection_time >= $1"
                params = [start_date]
                
                if source_id:
                    base_query += f" AND source_id = ${len(params) + 1}"
                    params.append(source_id)
                
                # 总采集次数
                total_row = await conn.fetchrow(
                    f"SELECT COUNT(*) as count {base_query}",
                    *params
                )
                
                # 成功次数
                success_row = await conn.fetchrow(
                    f"SELECT COUNT(*) as count {base_query} AND status = 'success'",
                    *params
                )
                
                # 失败次数
                failed_row = await conn.fetchrow(
                    f"SELECT COUNT(*) as count {base_query} AND status = 'failed'",
                    *params
                )
                
                # 平均采集记录数
                avg_records_row = await conn.fetchrow(
                    f"SELECT AVG(records_collected) as avg {base_query} AND status = 'success'",
                    *params
                )
                
                # 平均耗时
                avg_duration_row = await conn.fetchrow(
                    f"SELECT AVG(duration_ms) as avg {base_query} AND status = 'success'",
                    *params
                )
                
                total = total_row['count'] or 0
                success = success_row['count'] or 0
                failed = failed_row['count'] or 0
                
                return {
                    "total_collections": total,
                    "successful_collections": success,
                    "failed_collections": failed,
                    "success_rate": (success / total * 100) if total > 0 else 0,
                    "avg_records_collected": round(avg_records_row['avg'] or 0, 2),
                    "avg_duration_ms": round(avg_duration_row['avg'] or 0, 2),
                    "period_days": days
                }
                
        except Exception as e:
            logger.error(f"❌ 获取采集统计信息失败: {e}")
            return {
                "total_collections": 0,
                "successful_collections": 0,
                "failed_collections": 0,
                "success_rate": 0,
                "avg_records_collected": 0,
                "avg_duration_ms": 0,
                "period_days": days
            }


# 全局实例
_history_manager: Optional[CollectionHistoryManager] = None


def get_collection_history_manager() -> CollectionHistoryManager:
    """获取采集历史记录管理器实例（单例模式）"""
    global _history_manager
    if _history_manager is None:
        _history_manager = CollectionHistoryManager()
    return _history_manager
