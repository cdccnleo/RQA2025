# 基础设施层命名规范

## 概述

本文档定义了基础设施层的文件命名、类命名、函数命名等规范，确保代码的一致性和可维护性。

## 文件命名规范

### 1. 主要模块文件

#### 统一管理器文件
- `unified_manager.py` - 统一配置管理器
- `unified_database_manager.py` - 统一数据库管理器
- `unified_cache_manager.py` - 统一缓存管理器

#### 接口文件
- `icache_manager.py` - 缓存接口
- `idatabase_manager.py` - 数据库接口
- `iconfig_manager.py` - 配置接口

#### 适配器文件
- `postgresql_adapter.py` - PostgreSQL适配器
- `redis_adapter.py` - Redis适配器
- `influxdb_adapter.py` - InfluxDB适配器
- `sqlite_adapter.py` - SQLite适配器

#### 监控文件
- `automation_monitor.py` - 自动化运维监控
- `performance_monitor.py` - 性能监控
- `application_monitor.py` - 应用监控
- `system_monitor.py` - 系统监控

### 2. 服务文件

#### 配置服务
- `unified_service.py` - 统一配置服务
- `unified_hot_reload.py` - 统一热重载服务
- `unified_sync.py` - 统一同步服务

#### 数据库服务
- `connection_pool.py` - 连接池
- `health_check_manager.py` - 健康检查管理器
- `query_cache_manager.py` - 查询缓存管理器

### 3. 工具文件

#### 验证器
- `config_validator.py` - 配置验证器
- `database_config_validator.py` - 数据库配置验证器

#### 监控器
- `slow_query_monitor.py` - 慢查询监控器
- `data_consistency_manager.py` - 数据一致性管理器

### 4. 异常文件
- `error_exceptions.py` - 异常定义
- `error_handler.py` - 错误处理器

## 类命名规范

### 1. 管理器类
- `UnifiedConfigManager` - 统一配置管理器
- `UnifiedDatabaseManager` - 统一数据库管理器
- `UnifiedCacheManager` - 统一缓存管理器

### 2. 接口类
- `ICacheManager` - 缓存管理器接口
- `IDatabaseManager` - 数据库管理器接口
- `IConfigManager` - 配置管理器接口

### 3. 适配器类
- `PostgreSQLAdapter` - PostgreSQL适配器
- `RedisAdapter` - Redis适配器
- `InfluxDBAdapter` - InfluxDB适配器
- `SQLiteAdapter` - SQLite适配器

### 4. 监控类
- `AutomationMonitor` - 自动化运维监控
- `PerformanceMonitor` - 性能监控
- `ApplicationMonitor` - 应用监控
- `SystemMonitor` - 系统监控

### 5. 服务类
- `UnifiedConfigService` - 统一配置服务
- `UnifiedHotReload` - 统一热重载服务
- `UnifiedSync` - 统一同步服务

### 6. 验证器类
- `ConfigValidator` - 配置验证器
- `DatabaseConfigValidator` - 数据库配置验证器

### 7. 异常类
- `ConfigError` - 配置错误
- `ConfigValidationError` - 配置验证错误
- `ConfigLoadError` - 配置加载错误
- `DatabaseError` - 数据库错误

## 函数命名规范

### 1. 获取函数
- `get_config()` - 获取配置
- `get_database()` - 获取数据库
- `get_cache()` - 获取缓存

### 2. 设置函数
- `set_config()` - 设置配置
- `set_database()` - 设置数据库
- `set_cache()` - 设置缓存

### 3. 验证函数
- `validate_config()` - 验证配置
- `validate_database()` - 验证数据库
- `validate_cache()` - 验证缓存

### 4. 监控函数
- `monitor_performance()` - 监控性能
- `monitor_health()` - 监控健康状态
- `monitor_errors()` - 监控错误

### 5. 初始化函数
- `init_config()` - 初始化配置
- `init_database()` - 初始化数据库
- `init_cache()` - 初始化缓存

## 变量命名规范

### 1. 配置变量
- `config_manager` - 配置管理器实例
- `database_manager` - 数据库管理器实例
- `cache_manager` - 缓存管理器实例

### 2. 监控变量
- `performance_monitor` - 性能监控实例
- `health_checker` - 健康检查器实例
- `error_handler` - 错误处理器实例

### 3. 状态变量
- `is_initialized` - 是否已初始化
- `is_running` - 是否正在运行
- `is_healthy` - 是否健康

## 常量命名规范

### 1. 配置常量
- `DEFAULT_CONFIG_PATH` - 默认配置路径
- `DEFAULT_DATABASE_URL` - 默认数据库URL
- `DEFAULT_CACHE_SIZE` - 默认缓存大小

### 2. 监控常量
- `HEALTH_CHECK_INTERVAL` - 健康检查间隔
- `PERFORMANCE_MONITOR_INTERVAL` - 性能监控间隔
- `ERROR_REPORT_INTERVAL` - 错误报告间隔

## 模块导入规范

### 1. 标准库导入
```python
import os
import sys
import logging
import threading
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
```

### 2. 第三方库导入
```python
import requests
import psutil
from prometheus_client import CollectorRegistry, Gauge, Counter
```

### 3. 内部模块导入
```python
from .unified_manager import UnifiedConfigManager
from .unified_database_manager import UnifiedDatabaseManager
from .icache_manager import ICacheManager
```

## 文档规范

### 1. 文件头注释
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块名称
模块功能描述
"""

import os
import sys
```

### 2. 类文档
```python
class ClassName:
    """类功能描述
    
    Attributes:
        attr1: 属性1描述
        attr2: 属性2描述
    """
```

### 3. 函数文档
```python
def function_name(param1: str, param2: int) -> bool:
    """函数功能描述
    
    Args:
        param1: 参数1描述
        param2: 参数2描述
        
    Returns:
        返回值描述
        
    Raises:
        ExceptionType: 异常描述
    """
```

## 总结

遵循这些命名规范可以确保：
1. 代码的一致性和可读性
2. 模块职责的清晰性
3. 接口的统一性
4. 维护的便利性

所有新开发的代码都应该遵循这些规范，现有代码在重构时也应该逐步调整为符合规范。 