#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征存储对外暴露模块

为了保持 `src.features` 命名空间的稳定性，对外提供 `FeatureStore`
等核心对象，内部仍复用 `core.feature_store` 的实现。
"""

from .core.feature_store import FeatureStore, StoreConfig, FeatureMetadata

__all__ = ["FeatureStore", "StoreConfig", "FeatureMetadata"]

