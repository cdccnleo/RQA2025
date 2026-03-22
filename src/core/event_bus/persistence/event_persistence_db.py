#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件持久化数据库实现

提供基于PostgreSQL的事件持久化支持，实现事件的可靠存储和检索。
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .event_persistence import EventStatus, PersistedEvent

logger = logging.getLogger(__name__)


@dataclass
class DatabaseEventPersistenceConfig:
    """数据库持久化配置
    
    配置项优先从环境变量读取，确保安全性。
    密码必须从环境变量 POSTGRES_PASSWORD 读取，禁止硬编码。
    """
    host: str = "postgres"
    port: int = 5432
    database: str = "rqa2025_prod"
    user: str = "rqa2025_admin"
    password: str = None  # 密码必须从环境变量读取
    table_name: str = "event_bus_events"
    
    def __post_init__(self):
        """初始化后验证密码配置"""
        import os
        if self.password is None:
            self.password = os.getenv("POSTGRES_PASSWORD")
        if not self.password:
            raise ValueError(
                "数据库密码未设置！请设置环境变量 POSTGRES_PASSWORD。\n"
                "示例：set POSTGRES_PASSWORD=YourSecurePassword\n"
                "或：export POSTGRES_PASSWORD=YourSecurePassword"
            )
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """获取连接参数"""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password
        }


class DatabaseEventPersistence:
    """
    数据库事件持久化管理器
    
    使用PostgreSQL存储事件，支持：
    - 事件的持久化存储
    - 事件状态跟踪
    - 事件历史查询
    - 事件重放
    """
    
    def __init__(self, config: Optional[DatabaseEventPersistenceConfig] = None):
        """
        初始化数据库持久化管理器
        
        Args:
            config: 数据库配置，None则使用默认配置
        """
        self.config = config or DatabaseEventPersistenceConfig()
        self._connection = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        初始化数据库连接和表结构
        
        Returns:
            是否初始化成功
        """
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            # 建立连接
            self._connection = psycopg2.connect(**self.config.connection_params)
            
            # 创建事件表
            self._create_events_table()
            
            self._initialized = True
            logger.info("✅ 事件持久化数据库初始化成功")
            return True
            
        except ImportError:
            logger.error("❌ psycopg2未安装，无法使用数据库持久化")
            return False
        except Exception as e:
            logger.error(f"❌ 事件持久化数据库初始化失败: {e}")
            return False
    
    def _create_events_table(self) -> None:
        """创建事件表"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.table_name} (
            event_id VARCHAR(64) PRIMARY KEY,
            event_type VARCHAR(128) NOT NULL,
            data JSONB NOT NULL,
            source VARCHAR(256) DEFAULT 'unknown',
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            correlation_id VARCHAR(64),
            status VARCHAR(32) DEFAULT 'pending',
            retry_count INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_type 
            ON {self.config.table_name}(event_type);
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_status 
            ON {self.config.table_name}(status);
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_timestamp 
            ON {self.config.table_name}(timestamp);
        CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_correlation 
            ON {self.config.table_name}(correlation_id);
        """
        
        with self._connection.cursor() as cursor:
            cursor.execute(create_table_sql)
            self._connection.commit()
    
    def save_event(self, event: Any) -> bool:
        """
        保存事件到数据库
        
        Args:
            event: 事件对象
            
        Returns:
            是否保存成功
        """
        if not self._initialized or not self._connection:
            logger.warning("⚠️ 事件持久化未初始化")
            return False
        
        try:
            event_id = getattr(event, 'event_id', None)
            if not event_id:
                logger.warning("⚠️ 事件缺少event_id")
                return False
            
            event_data = {
                'event_id': event_id,
                'event_type': str(getattr(event, 'event_type', 'unknown')),
                'data': getattr(event, 'data', {}),
                'source': getattr(event, 'source', 'unknown'),
                'timestamp': datetime.fromtimestamp(
                    getattr(event, 'timestamp', datetime.now().timestamp())
                ),
                'correlation_id': getattr(event, 'correlation_id', None),
                'status': EventStatus.PENDING.value
            }
            
            insert_sql = f"""
            INSERT INTO {self.config.table_name} 
            (event_id, event_type, data, source, timestamp, correlation_id, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO UPDATE SET
                data = EXCLUDED.data,
                status = EXCLUDED.status,
                updated_at = CURRENT_TIMESTAMP
            """
            
            with self._connection.cursor() as cursor:
                cursor.execute(insert_sql, (
                    event_data['event_id'],
                    event_data['event_type'],
                    json.dumps(event_data['data']),
                    event_data['source'],
                    event_data['timestamp'],
                    event_data['correlation_id'],
                    event_data['status']
                ))
                self._connection.commit()
            
            logger.debug(f"✅ 事件已持久化: {event_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存事件失败: {e}")
            if self._connection:
                self._connection.rollback()
            return False
    
    def update_event_status(
        self, 
        event_id: str, 
        status: EventStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        更新事件状态
        
        Args:
            event_id: 事件ID
            status: 新状态
            error_message: 错误信息（可选）
            
        Returns:
            是否更新成功
        """
        if not self._initialized or not self._connection:
            return False
        
        try:
            update_sql = f"""
            UPDATE {self.config.table_name}
            SET status = %s, 
                updated_at = CURRENT_TIMESTAMP,
                error_message = COALESCE(%s, error_message)
            WHERE event_id = %s
            """
            
            with self._connection.cursor() as cursor:
                cursor.execute(update_sql, (status.value, error_message, event_id))
                self._connection.commit()
                
                if cursor.rowcount > 0:
                    logger.debug(f"✅ 事件状态已更新: {event_id} -> {status.value}")
                    return True
                else:
                    logger.warning(f"⚠️ 事件不存在: {event_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 更新事件状态失败: {e}")
            if self._connection:
                self._connection.rollback()
            return False
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        获取事件
        
        Args:
            event_id: 事件ID
            
        Returns:
            事件数据，未找到返回None
        """
        if not self._initialized or not self._connection:
            return None
        
        try:
            select_sql = f"""
            SELECT * FROM {self.config.table_name}
            WHERE event_id = %s
            """
            
            with self._connection.cursor() as cursor:
                cursor.execute(select_sql, (event_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_dict(row, cursor.description)
                return None
                
        except Exception as e:
            logger.error(f"❌ 获取事件失败: {e}")
            return None
    
    def get_events_by_type(
        self, 
        event_type: str,
        status: Optional[EventStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        按类型获取事件
        
        Args:
            event_type: 事件类型
            status: 事件状态过滤（可选）
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            事件列表
        """
        if not self._initialized or not self._connection:
            return []
        
        try:
            if status:
                select_sql = f"""
                SELECT * FROM {self.config.table_name}
                WHERE event_type = %s AND status = %s
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
                """
                params = (event_type, status.value, limit, offset)
            else:
                select_sql = f"""
                SELECT * FROM {self.config.table_name}
                WHERE event_type = %s
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
                """
                params = (event_type, limit, offset)
            
            with self._connection.cursor() as cursor:
                cursor.execute(select_sql, params)
                rows = cursor.fetchall()
                
                return [self._row_to_dict(row, cursor.description) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ 获取事件列表失败: {e}")
            return []
    
    def get_events_by_correlation_id(self, correlation_id: str) -> List[Dict[str, Any]]:
        """
        按关联ID获取事件
        
        Args:
            correlation_id: 关联ID
            
        Returns:
            事件列表
        """
        if not self._initialized or not self._connection:
            return []
        
        try:
            select_sql = f"""
            SELECT * FROM {self.config.table_name}
            WHERE correlation_id = %s
            ORDER BY timestamp ASC
            """
            
            with self._connection.cursor() as cursor:
                cursor.execute(select_sql, (correlation_id,))
                rows = cursor.fetchall()
                
                return [self._row_to_dict(row, cursor.description) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ 获取关联事件失败: {e}")
            return []
    
    def get_pending_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取待处理事件
        
        Args:
            limit: 返回数量限制
            
        Returns:
            待处理事件列表
        """
        return self.get_events_by_type(
            event_type='',  # 空字符串表示所有类型
            status=EventStatus.PENDING,
            limit=limit
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取持久化统计
        
        Returns:
            统计信息
        """
        if not self._initialized or not self._connection:
            return {"error": "未初始化"}
        
        try:
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_events,
                COUNT(*) FILTER (WHERE status = 'pending') as pending_events,
                COUNT(*) FILTER (WHERE status = 'processing') as processing_events,
                COUNT(*) FILTER (WHERE status = 'completed') as completed_events,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_events,
                COUNT(DISTINCT event_type) as event_types
            FROM {self.config.table_name}
            """
            
            with self._connection.cursor() as cursor:
                cursor.execute(stats_sql)
                row = cursor.fetchone()
                
                if row:
                    return {
                        "total_events": row[0],
                        "pending_events": row[1],
                        "processing_events": row[2],
                        "completed_events": row[3],
                        "failed_events": row[4],
                        "event_types": row[5]
                    }
                return {}
                
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def replay_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        handler: Optional[callable] = None
    ) -> int:
        """
        重放事件
        
        Args:
            event_type: 事件类型过滤（可选）
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            handler: 事件处理函数（可选）
            
        Returns:
            重放的事件数量
        """
        if not self._initialized or not self._connection:
            return 0
        
        try:
            conditions = ["status = 'completed'"]
            params = []
            
            if event_type:
                conditions.append("event_type = %s")
                params.append(event_type)
            
            if start_time:
                conditions.append("timestamp >= %s")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= %s")
                params.append(end_time)
            
            where_clause = " AND ".join(conditions)
            
            select_sql = f"""
            SELECT * FROM {self.config.table_name}
            WHERE {where_clause}
            ORDER BY timestamp ASC
            """
            
            with self._connection.cursor() as cursor:
                cursor.execute(select_sql, params)
                rows = cursor.fetchall()
                
                replay_count = 0
                for row in row:
                    event_data = self._row_to_dict(row, cursor.description)
                    
                    if handler:
                        try:
                            handler(event_data)
                            replay_count += 1
                        except Exception as e:
                            logger.error(f"❌ 重放事件失败: {e}")
                    else:
                        replay_count += 1
                
                logger.info(f"✅ 已重放 {replay_count} 个事件")
                return replay_count
                
        except Exception as e:
            logger.error(f"❌ 重放事件失败: {e}")
            return 0
    
    def cleanup_old_events(self, days: int = 30) -> int:
        """
        清理旧事件
        
        Args:
            days: 保留天数
            
        Returns:
            清理的事件数量
        """
        if not self._initialized or not self._connection:
            return 0
        
        try:
            delete_sql = f"""
            DELETE FROM {self.config.table_name}
            WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '%s days'
            AND status IN ('completed', 'failed')
            """
            
            with self._connection.cursor() as cursor:
                cursor.execute(delete_sql, (days,))
                deleted_count = cursor.rowcount
                self._connection.commit()
                
                logger.info(f"✅ 已清理 {deleted_count} 个旧事件")
                return deleted_count
                
        except Exception as e:
            logger.error(f"❌ 清理旧事件失败: {e}")
            if self._connection:
                self._connection.rollback()
            return 0
    
    def _row_to_dict(self, row: tuple, description: list) -> Dict[str, Any]:
        """
        将数据库行转换为字典
        
        Args:
            row: 数据库行
            description: 列描述
            
        Returns:
            字典格式的数据
        """
        result = {}
        for i, col in enumerate(description):
            col_name = col.name
            value = row[i]
            
            # 处理JSONB字段
            if col_name == 'data' and isinstance(value, str):
                try:
                    value = json.loads(value)
                except:
                    pass
            
            result[col_name] = value
        
        return result
    
    def shutdown(self) -> None:
        """关闭数据库连接"""
        if self._connection:
            self._connection.close()
            self._connection = None
            self._initialized = False
            logger.info("✅ 事件持久化数据库连接已关闭")


__all__ = [
    'DatabaseEventPersistenceConfig',
    'DatabaseEventPersistence'
]
