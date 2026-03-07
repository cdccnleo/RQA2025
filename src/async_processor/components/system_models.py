"""
系统处理器数据模型

系统处理器相关的枚举。

从system_processor.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SystemMetricType(Enum):
    """System metric type enumeration"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"


__all__ = ['SystemMetricType']

