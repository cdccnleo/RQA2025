#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理层持久化模块

实现 PostgreSQL 优先存储策略，提供数据缓存、血缘追踪、质量管理的持久化支持。

使用方式:
    from src.data.persistence import DataPersistence, CachePersistence
    
    persistence = DataPersistence()
    persistence.save_cache_entry(key, data, metadata)
"""

from .data_persistence import (
    DataPersistence,
    CachePersistence,
    LineagePersistence,
    QualityPersistence,
    CompliancePersistence,
    get_data_persistence,
    get_cache_persistence,
    get_lineage_persistence,
    get_quality_persistence
)

__all__ = [
    'DataPersistence',
    'CachePersistence',
    'LineagePersistence',
    'QualityPersistence',
    'CompliancePersistence',
    'get_data_persistence',
    'get_cache_persistence',
    'get_lineage_persistence',
    'get_quality_persistence'
]
