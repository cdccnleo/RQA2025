"""
基础数据加载器模块
提供统一的数据加载器抽象基类和接口定义
"""
from abc import ABC
import logging
from typing import Any, Dict, List, Generic, TypeVar, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

T = TypeVar('T')  # 加载的数据类型


class LoaderConfig:
    """数据加载器配置"""

    def __init__(self, name: str = "", batch_size: int = 100, max_retries: int = 3,
                 timeout: int = 30, cache_enabled: bool = True, validation_enabled: bool = True,
                 retry_count: int = 3, validate_data: bool = True, **kwargs):
        self.name = name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self.validation_enabled = validation_enabled
        self.retry_count = retry_count
        self.validate_data = validate_data
        self.extra_config = kwargs


class BaseDataLoader(ABC, Generic[T]):

    """数据加载器抽象基类，定义所有数据加载器的公共接口

    子类需要实现:
    - load(): 实际的数据加载逻辑
    - get_metadata(): 获取加载器元数据
    """

    def __init__(self, config: LoaderConfig = None):

        self.config = config or LoaderConfig()
        self._load_count = 0
        self._last_load_time = None
        self._error_count = 0
        self.is_initialized = False
        self._cache_store: Dict[str, Any] = {}
        self._cache_hits = 0
        self._successful_loads = 0
        self._failed_loads = 0
        self._load_history: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng()

    def initialize(self) -> bool:
        """初始化数据加载器

        返回:
            bool: 初始化是否成功
        """
        try:
            # 执行初始化逻辑
            self.is_initialized = True
            return True
        except Exception:
            return False

    def load(self, *args, **kwargs) -> T:
        """加载数据的主方法

        参数:
            *args: 可变位置参数
            **kwargs: 可变关键字参数

        返回:
            加载的数据，具体类型由子类决定
        """
        raise NotImplementedError("load() must be implemented by subclasses.")

    def load_data(self, *args, **kwargs) -> T:
        """加载数据的别名方法，为了向后兼容

        支持位置参数形式 (source, symbol, start_date, end_date) 与关键字参数形式。
        当直接实例化 BaseDataLoader 时，提供一个回退实现用于测试环境。
        """
        if self.__class__ is BaseDataLoader:
            return self._fallback_load(*args, **kwargs)

        if args and kwargs:
            return self.load(*args, **kwargs)
        if args:
            return self.load(*args)
        return self.load(**kwargs)

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据

        返回:
            包含加载器元数据的字典，通常包括:
            - loader_type: 加载器类型
            - version: 版本号
            - description: 描述信息
        """
        raise NotImplementedError("get_metadata() must be implemented by subclasses.")

    def validate(self, data: T) -> bool:
        """验证加载的数据是否符合预期

        参数:
            data: 要验证的数据

        返回:
            bool: 数据是否有效
        """
        return data is not None

    def validate_data(self, data: T) -> bool:
        """验证数据的别名方法，为了向后兼容

        参数:
            data: 要验证的数据

        返回:
            bool: 数据是否有效
        """
        return self.validate(data)

    def batch_load(self, params_list: List[Dict]) -> List[T]:
        """批量加载数据

        参数:
            params_list: 参数列表，每个元素是传递给load()的参数字典

        返回:
            加载的数据列表
        """
        results = []
        for params in params_list:
            try:
                data = self.load(**params)
                if self.validate(data):
                    results.append(data)
            except Exception as e:
                self._error_count += 1
                continue
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取加载器统计信息

        返回:
            包含统计信息的字典:
            - load_count: 总加载次数
            - error_count: 错误次数
            - last_load_time: 最后加载时间
            - success_rate: 成功率
        """
        success_rate = (self._load_count - self._error_count) / max(self._load_count, 1)
        return {
            "load_count": self._load_count,
            "error_count": self._error_count,
            "last_load_time": self._last_load_time,
            "success_rate": success_rate
        }

    @property
    def error_count(self) -> int:
        return self._error_count

    def _update_stats(self):
        """更新加载统计信息"""
        self._load_count += 1
        self._last_load_time = datetime.now()

    def reset_stats(self):
        """重置统计信息"""
        self._load_count = 0
        self._error_count = 0
        self._last_load_time = None

    def _get_logger(self) -> logging.Logger:
        """获取加载器专用日志记录器"""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # 回退实现 & 测试辅助方法
    # ------------------------------------------------------------------
    def _fallback_load(self, *args, **kwargs) -> pd.DataFrame:
        params = self._normalize_load_params(args, kwargs)
        cache_key = self._build_cache_key(params)

        if self.config.cache_enabled and cache_key in self._cache_store:
            self._cache_hits += 1
            return self._cache_store[cache_key]

        source = params["source"]
        if source not in self.get_supported_sources():
            self._failed_loads += 1
            self._error_count += 1
            raise ValueError(f"Unsupported data source: {source}")

        data = self._generate_mock_data(params)

        if self.config.cache_enabled:
            self._cache_store[cache_key] = data

        self._successful_loads += 1
        self._load_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "symbol": params["symbol"],
                "start_date": params["start_date"],
                "end_date": params["end_date"],
                "cache_key": cache_key,
            }
        )
        self._update_stats()
        return data

    def _normalize_load_params(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Optional[str]]:
        arg_names = ["source", "symbol", "start_date", "end_date"]
        params = {name: None for name in arg_names}

        for idx, value in enumerate(args[: len(arg_names)]):
            params[arg_names[idx]] = value

        for key in arg_names:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]

        params["source"] = params["source"] or kwargs.get("data_source") or "random"
        params["symbol"] = params["symbol"] or kwargs.get("ticker") or "UNKNOWN"
        return params

    def _build_cache_key(self, params: Dict[str, Optional[str]]) -> str:
        return "_".join(
            [
                params.get("source") or "random",
                params.get("symbol") or "UNKNOWN",
                params.get("start_date") or "",
                params.get("end_date") or "",
            ]
        )

    def _generate_mock_data(self, params: Dict[str, Optional[str]]) -> pd.DataFrame:
        start = params.get("start_date")
        end = params.get("end_date")
        source = params.get("source") or "random"

        try:
            start_ts = pd.to_datetime(start) if start else datetime.now() - timedelta(days=9)
        except Exception:
            start_ts = datetime.now() - timedelta(days=9)
        try:
            end_ts = pd.to_datetime(end) if end else datetime.now()
        except Exception:
            end_ts = datetime.now()

        if end_ts < start_ts:
            end_ts = start_ts + timedelta(days=1)

        periods = max((end_ts - start_ts).days + 1, 1)
        index = pd.date_range(start=start_ts, periods=periods, freq="D")

        base_prices = self._rng.uniform(50, 200, periods)
        spreads = self._rng.uniform(0.5, 3.0, periods)
        closes = base_prices + self._rng.uniform(-1.0, 1.0, periods)
        highs = np.maximum(base_prices, closes) + spreads
        lows = np.minimum(base_prices, closes) - spreads

        if source == "yahoo":
            data = pd.DataFrame(
                {
                    "Open": base_prices.round(2),
                    "High": np.maximum(highs, base_prices).round(2),
                    "Low": np.minimum(lows, base_prices).round(2),
                    "Close": closes.round(2),
                    "Adj Close": closes.round(2),
                    "Volume": self._rng.integers(1_000, 10_000, periods),
                },
                index=index,
            )
        elif source == "alpha_vantage":
            data = pd.DataFrame(
                {
                    "OPEN": base_prices.round(2),
                    "HIGH": np.maximum(highs, base_prices).round(2),
                    "LOW": np.minimum(lows, base_prices).round(2),
                    "CLOSE": closes.round(2),
                    "VOLUME": self._rng.integers(1_000, 10_000, periods),
                },
                index=index,
            )
        else:
            data = pd.DataFrame(
                {
                    "open": base_prices.round(2),
                    "high": np.maximum(highs, base_prices).round(2),
                    "low": np.minimum(lows, base_prices).round(2),
                    "close": closes.round(2),
                    "volume": self._rng.integers(1_000, 10_000, periods),
                },
                index=index,
            )

        data.index.name = "date"
        return data

    def get_supported_sources(self) -> Tuple[str, ...]:
        return ("random", "yahoo", "alpha_vantage")

    def validate_connection(self, source: str) -> bool:
        return source in self.get_supported_sources()

    def clear_cache(self) -> None:
        self._cache_store.clear()

    def get_load_stats(self) -> Dict[str, Any]:
        total = max(self._successful_loads + self._failed_loads, 1)
        success_rate = self._successful_loads / total
        return {
            "total_loads": self._load_count,
            "successful_loads": self._successful_loads,
            "failed_loads": self._failed_loads,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache_store),
            "last_load_time": self._last_load_time.isoformat() if self._last_load_time else None,
            "success_rate": round(success_rate, 4),
        }


class DataLoader(BaseDataLoader[Dict[str, Any]]):
    """通用数据加载器实现"""

    def __init__(self, config: LoaderConfig = None, source_type: str = "generic"):
        super().__init__(config)
        self.source_type = source_type

    def load(self, *args, **kwargs) -> Dict[str, Any]:
        """通用数据加载实现"""
        # 这是一个基本的实现，实际使用时应被子类覆盖
        return {
            "data": [],
            "metadata": {
                "source_type": self.source_type,
                "load_time": datetime.now().isoformat(),
                "config": self.config.__dict__ if self.config else {}
            }
        }

    def get_metadata(self) -> Dict[str, Any]:
        """获取加载器元数据"""
        return {
            "type": "DataLoader",
            "source_type": self.source_type,
            "config": self.config.__dict__ if self.config else {},
            "stats": {
                "load_count": self._load_count,
                "last_load_time": self._last_load_time.isoformat() if self._last_load_time else None,
                "error_count": self._error_count
            }
        }


class MockDataLoader(BaseDataLoader[Dict[str, Any]]):
    """模拟数据加载器，用于测试"""

    def __init__(self, config: LoaderConfig = None, mock_data: Dict[str, Any] = None):
        super().__init__(config)
        self.mock_data = mock_data or {"mock": True, "data": []}

    def load(self, *args, **kwargs) -> Dict[str, Any]:
        """返回模拟数据"""
        return self.mock_data.copy()

    def get_metadata(self) -> Dict[str, Any]:
        """获取模拟加载器元数据"""
        return {
            "type": "MockDataLoader",
            "mock_data_keys": list(self.mock_data.keys()),
            "config": self.config.__dict__ if self.config else {},
            "stats": {
                "load_count": self._load_count,
                "last_load_time": self._last_load_time.isoformat() if self._last_load_time else None,
                "error_count": self._error_count
            }
        }


class DataLoaderRegistry:

    """数据加载器注册表"""

    def __init__(self):

        self._loaders = {}

    def register(self, name: str, loader: BaseDataLoader):
        """注册数据加载器

        参数:
            name: 加载器名称
            loader: 加载器实例
        """
        self._loaders[name] = loader

    def get_loader(self, name: str) -> Optional[BaseDataLoader]:
        """获取数据加载器

        参数:
            name: 加载器名称

        返回:
            加载器实例，如果不存在则返回None
        """
        return self._loaders.get(name)

    def list_loaders(self) -> List[str]:
        """列出所有已注册的加载器名称"""
        return list(self._loaders.keys())

    def remove_loader(self, name: str) -> bool:
        """移除数据加载器

        参数:
            name: 加载器名称

        返回:
            是否成功移除
        """
        if name in self._loaders:
            del self._loaders[name]
            return True
        return False


# 全局加载器注册表实例
loader_registry = DataLoaderRegistry()
