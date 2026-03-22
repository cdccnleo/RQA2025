#!/usr/bin/env python3
"""
特征选择任务数据持久化模块

功能：
1. 优先使用PostgreSQL数据库存储
2. 连接失败时自动降级到文件系统
3. 实现重试机制（最多3次，指数退避）
4. 完整的日志记录

作者：AI Assistant
日期：2026-03-21
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

# 数据目录配置
DATA_DIR = Path(os.environ.get('DATA_DIR', '/app/data'))
FEATURE_SELECTION_TASKS_DIR = DATA_DIR / 'feature_selection_tasks'

# 确保目录存在
FEATURE_SELECTION_TASKS_DIR.mkdir(parents=True, exist_ok=True)


class FeatureSelectionPersistence:
    """
    特征选择任务持久化管理器
    
    实现PostgreSQL优先存储，连接失败时降级到文件系统
    """
    
    def __init__(self):
        self._db_config = self._load_db_config()
        self._max_retries = 3
        self._retry_delay_base