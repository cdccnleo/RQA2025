#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置存储服务测试Fixtures

提供ConfigStorageService测试需要的Mock存储后端
"""

from typing import Any, Dict, Optional
from unittest.mock import Mock

class MockConfigStorage:
    """Mock配置存储后端 - 符合IConfigStorage接口"""

    def __init__(self):
        self._data = {}
        self._files = {}  # 模拟文件存储

    def load(self, source: str) -> Dict[str, Any]:
        """从源加载配置"""
        if source in self._files:
            return self._files[source].copy()
        return {}

    def save(self, config: Dict[str, Any], target: str) -> bool:
        """保存配置到目标"""
        self._files[target] = config.copy()
        return True

    def delete(self, target: str) -> bool:
        """删除配置"""
        if target in self._files:
            del self._files[target]
            return True
        return False

    def exists(self, target: str) -> bool:
        """检查配置是否存在"""
        return target in self._files

    def list_configs(self) -> list:
        """列出所有配置"""
        return list(self._files.keys())

    def clear(self):
        """清空所有配置"""
        self._files.clear()


def create_mock_storage_backend(initial_data: Optional[Dict[str, Dict[str, Any]]] = None) -> MockConfigStorage:
    """创建Mock存储后端

    Args:
        initial_data: 初始数据，格式为 {source: config_dict}
    """
    storage = MockConfigStorage()
    if initial_data:
        for source, config in initial_data.items():
            storage.save(config, source)
    return storage


__all__ = ['MockConfigStorage', 'create_mock_storage_backend']
