
# from .data_api import *  # TODO: Fix data_manager dependency
from .database_adapter import *
from .influxdb_adapter import *
from .postgresql_adapter import *
from .redis_adapter import *
from .sqlite_adapter import *
"""
RQA2025 基础设施层工具系统 - 数据适配器模块

本模块包含各种数据源适配器，用于连接和操作不同的数据库和外部服务。

包含的适配器:
- 数据库适配器 (PostgreSQL, Redis, SQLite, InfluxDB)
- API数据服务适配器
- 通用数据库适配器接口

作者: RQA2025 Team
创建日期: 2025年9月27日
"""

__all__ = [
    # 数据库适配器
    "DatabaseAdapter",
    "PostgreSQLAdapter",
    "RedisAdapter",
    "SQLiteAdapter",
    "InfluxDBAdapter",
    # API服务
    "DataAPI",
    # 通用接口
    "IDatabaseAdapter",
    "QueryResult",
    "WriteResult",
    "HealthCheckResult",
]
