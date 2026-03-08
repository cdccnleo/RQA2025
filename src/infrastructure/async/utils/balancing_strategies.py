"""
负载均衡策略

各种负载均衡算法实现。

从load_balancer.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import List, Any
from enum import Enum
import random

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"


__all__ = ['LoadBalancingStrategy']

