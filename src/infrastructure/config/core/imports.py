"""
imports 模块

提供 imports 相关功能和接口。
"""

import json
import logging
import os

# ==================== 类型注解导入 ====================
import hashlib
import threading
import time

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Dict, Any, Optional, List, Callable, Type, Union,
    TypeVar, Generic, Protocol, runtime_checkable,
    Tuple, Set, Iterable, Iterator
)
from ..config_exceptions import (
    ConfigError, ConfigLoadError, ConfigValidationError,
    ConfigTypeError, ConfigAccessError
)

# ==================== 常用类型别名 ====================
T = TypeVar('T')
ConfigDict = Dict[str, Any]
OptionalConfig = Optional[ConfigDict]
ConfigList = List[ConfigDict]

# ==================== 日志配置 ====================
logger = logging.getLogger(__name__)

# ==================== 导出列表 ====================

__all__ = [
    # 基础模块
    'logging', 'time', 'os', 'json', 'threading',
    'datetime', 'Path', 'defaultdict', 'lru_cache',
    'ABC', 'abstractmethod', 'dataclass', 'field', 'Enum',
    'hashlib',

    # 类型注解
    'Dict', 'Any', 'Optional', 'List', 'Callable', 'Type', 'Union',
    'TypeVar', 'Generic', 'Protocol', 'runtime_checkable',
    'Tuple', 'Set', 'Iterable', 'Iterator',

    # 异常类
    'ConfigError', 'ConfigLoadError', 'ConfigValidationError',
    'ConfigTypeError', 'ConfigAccessError',

    # 类型别名
    'T', 'ConfigDict', 'OptionalConfig', 'ConfigList',

    # 日志
    'logger'
]




