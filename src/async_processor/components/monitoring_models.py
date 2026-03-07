"""
监控数据模型

监控处理器相关的枚举和数据类。

从monitoring_processor.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from enum import Enum
from typing import Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MonitoringLevel(Enum):
    """Monitoring level enumeration"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


__all__ = ['MetricType', 'MonitoringLevel']

