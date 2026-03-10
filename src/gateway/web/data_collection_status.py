"""
数据源采集状态管理
使用数据库记录每天的数据源采集状态，防止容器重启后重复采集
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, Any

from src.infrastructure.persistence.database import get_db_connection

logger = logging.getLogger(__name__)


def ensure_collection_status_table():
    """确保采集状态表存在"""
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("❌ 无法获取数据库连接")
            return False
        
        cursor = conn.cursor()
        
        # 创建采集状态表
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS data_source_collection_status (
                id SERIAL PRIMARY KEY,
                source_id VARCHAR(100) NOT NULL,
                collection_date DATE NOT NULL,
                has_submitted BOOLEAN DEFAULT FALSE,
                task_id VARCHAR(100),
                submitted_at TIMESTAMP,
                completed_at TIMESTAMP,
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, collection_date)
            )
        """
        
        cursor.execute(create_table_sql)
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ 数据源采集状态表已创建或已存在")
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建采集状态表失败: {e}")
        return False


def has_submitted_today(source_id: str) -> bool:
    """
    检查今天是否已经提交过采集任务
    
    Args:
        source_id: 数据源ID
        
    Returns:
        bool: 今天是否已经提交过
    """
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("❌ 无法获取数据库连接")
            return False
        
        cursor = conn.cursor()
        today = date.today()
        
        query = """
            SELECT has_submitted, status 
            FROM data_source_collection_status 
            WHERE source_id = %s AND collection_date = %s
        """
        
        cursor.execute(query, (source_id, today))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            has_submitted, status = result
            if has_submitted:
                logger.debug(f"数据源 {source_id} 今天({today})已经提交过采集任务，状态: {status}")
                return True
        
        return