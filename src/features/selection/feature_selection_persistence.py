#!/usr/bin/env python3
"""
特征选择任务数据持久化机制

实现要求：
1. 主存储：PostgreSQL数据库
2. 降级机制：数据库连接失败时自动降级到文件系统
3. 重试逻辑：最大3次重试，指数退避
4. 数据一致性：确保数据库和文件系统存储的数据完整性
5. 日志记录：记录存储方法、时间戳和操作状态
6. 文件系统规范：JSON格式，标准化命名（包含时间戳和任务ID）
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = Path(os.environ.get('DATA_DIR', '/app/data'))
FEATURE_SELECTION_TASKS_DIR = DATA_DIR / 'feature_selection_tasks'

# 确保目录存在
FEATURE_SELECTION_TASKS_DIR.mkdir(parents=True, exist_ok=True)


class FeatureSelectionPersistence:
    """
    特征选择任务持久化管理器
    
    实现PostgreSQL优先存储，数据库连接失败时降级到文件系统
    """
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delay_base = 1  # 基础延迟（秒）
        self.db_connection = None
        
    def _get_db_connection(self) -> Optional[Any]:
        """
        获取PostgreSQL数据库连接，带重试机制
        
        Returns:
            数据库连接对象或None（连接失败）
        """
        for attempt in range(self.max_retries):
            try:
                # 导入PostgreSQL连接模块
                from src.gateway.web.postgresql_persistence import get_db_connection
                
                conn = get_db_connection()
                if conn:
                    logger.info(f"✅ PostgreSQL连接成功（尝试 {attempt + 1}/{self.max_retries}）")
                    return conn
                else:
                    logger.warning(f"⚠️ PostgreSQL连接返回None（尝试 {attempt + 1}/{self.max_retries}）")
                    
            except Exception as e:
                logger.error(f"❌ PostgreSQL连接失败（尝试 {attempt + 1}/{self.max_retries}）: {e}")
            
            # 指数退避重试
            if attempt < self.max_retries - 1:
                delay = self.retry_delay_base * (