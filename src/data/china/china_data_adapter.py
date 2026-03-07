"""
空壳中国数据适配器，待实现
"""

from __future__ import annotations


class ChinaDataAdapter:
    """空壳中国数据适配器，待实现"""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config_keys={list(self.config.keys())})"





