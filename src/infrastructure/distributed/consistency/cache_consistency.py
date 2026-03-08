#!/usr/bin/env python3
"""
分布式缓存一致性管理器

此文件作为主入口，导入并导出各个模块的组件。

重构说明(2025-11-01):
- consistency_models.py: 数据模型和枚举
- consistency_manager.py: Raft一致性管理器
- cache_sync_manager.py: 分布式缓存管理器

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from pathlib import Path
import sys

# 添加项目根路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# 导入所有组件
from .consistency_models import (
    ConsistencyLevel,
    NodeStatus,
    OperationType,
    CacheEntry,
    LogEntry,
    NodeInfo
)
from .consistency_manager import ConsistencyManager
from .cache_sync_manager import DistributedCacheManager

logger = logging.getLogger(__name__)


# 导出所有组件
__all__ = [
    # 枚举
    'ConsistencyLevel',
    'NodeStatus',
    'OperationType',
    # 数据类
    'CacheEntry',
    'LogEntry',
    'NodeInfo',
    # 核心类
    'ConsistencyManager',
    'DistributedCacheManager'
]


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)
    
    # 创建节点
    nodes = [
        NodeInfo(node_id="node1", host="localhost", port=8001),
        NodeInfo(node_id="node2", host="localhost", port=8002),
        NodeInfo(node_id="node3", host="localhost", port=8003)
    ]
    
    # 创建分布式缓存管理器
    cache_manager = DistributedCacheManager(
        node_id="node1",
        nodes=nodes,
        consistency_level=ConsistencyLevel.STRONG
    )
    
    # 启动
    cache_manager.start()
    
    # 测试缓存操作
    cache_manager.set("test_key", "test_value", ttl=300)
    value = cache_manager.get("test_key")
    logger.info(f"缓存值: {value}")
    
    # 停止
    cache_manager.stop()
