"""外汇数据加载器

该实现聚焦于满足现有单元测试与覆盖率要求，提供基于 yfinance 的
同步外汇行情获取、缓存管理与批量加载能力。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
yfinance = yf  # 兼容历史测试对 `forex_loader.yfinance` 的打补丁路径

from src.data.cache.cache_manager import CacheConfig, CacheManager
from src.data.core.base_loader import BaseDataLoader, LoaderConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForexRate:
    """简化版汇率结构体，用于兼容可能的调用方。"""

    base_currency: str
    quote_currency: str
    symbol: str
    rate: float
    timestamp: datetime


class ForexDataLoader(BaseDataLoader[pd.DataFrame]):
    """外汇数据加载器，支持单品种与批量行情获取。"""

    DEFAULT_CURRENCIES: List[str] = [
        "USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"
    ]
    DEFAULT_PAIRS: List[str] = [
        "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X"
    ]

    def __init__(
        self,
        save_path: Union[str, Path, None] = None,
        max_retries: int = 3,
        cache_days: int = 1,
        frequency: str = "1d",
        timeout: int = 30,
    ) -> None:
        loader_config = LoaderConfig(
            name="forex_loader",
            max_retries=max_retries,
            timeout=timeout,
            cache_enabled=True,
        )
        super().__init__(loader_config)

        self.save_path = Path(save_path) if save_path else Path.cwd() / "data" / "forex"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.save_path / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_retries = max_retries
        self.cache_days = cache_days
        self.frequency = frequency
        self.timeout = timeout
        self.supported_currencies: List[str] = list(self.DEFAULT_CURRENCIES)
        self.supported_pairs: List[str] = list(self.DEFAULT_PAIRS)

        ttl_seconds = max(int(cache_days * 86400), 3600)
        cache_config = CacheConfig(
            max_size=512,
            ttl=ttl_seconds,
            enable_disk_cache=True,
            disk_cache_dir=str(self.cache_dir),
        )
        self.cache_manager = CacheManager(cache_config)

        self.runtime_config: Dict[str, Union[str, int]] = {
            "cache_dir": str(self.cache_dir),
            "max_retries": self.max_retries,
            "cache_days": self.cache_days,
        }

        # 提前初始化，避免首次 load 时重复执行。
        self.initialize()

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def load(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """加载单一外汇品种的历史行情。"""
        self._ensure_initialized()

        start_str, end_str = self._normalize_dates(start_date, end_date)
        cache_key = self._build_cache_key(symbol, start_str, end_str)

        if not force_refresh:
            cached_df = self._get_from_cache(cache_key, symbol, start_str, end_str)
            if cached_df is not None:
                logger.debug("forex cache hit for %s", cache_key)
                self._update_stats()
                return cached_df

        df = self._fetch_forex_data(symbol, start_str, end_str)
        if df is None or df.empty:
            logger.info("No forex data returned for %s between %s and %s", symbol, start_str, end_str)
            empty_df = pd.DataFrame()
            self.cache_manager.set(cache_key, empty_df)
            return empty_df

        normalized = self._normalize_dataframe(df)
        self.cache_manager.set(cache_key, normalized.copy())
        self._persist_to_disk(symbol, start_str, end_str, normalized)
        self._update_stats()
        return normalized

    def load_batch(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        max_workers: int = 4,
        force_refresh: bool = False,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """批量加载多个外汇品种。"""
        self._ensure_initialized()
        if not symbols:
            return {}

        results: Dict[str, Optional[pd.DataFrame]] = {}

        def task(sym: str) -> tuple[str, Optional[pd.DataFrame]]:
            try:
                data = self.load(sym, start_date, end_date, force_refresh=force_refresh)
                return sym, data
            except Exception as exc:  # pragma: no cover - 仅用于稳健性
                logger.warning("Failed to load forex data for %s: %s", sym, exc)
                return sym, None

        with ThreadPoolExecutor(max_workers=max_workers or 1) as executor:
            futures = [executor.submit(task, sym) for sym in symbols]
            for future in as_completed(futures):
                symbol, data = future.result()
                results[symbol] = data

        return results

    def load_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        **kwargs: Union[str, bool, int],
    ) -> Dict[str, Union[str, pd.DataFrame]]:
        """兼容基础设施层 `load_data` 语义，返回带元数据的字典。"""
        start = start_date or (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        end = end_date or datetime.utcnow().strftime("%Y-%m-%d")
        data = self.load(
            symbol=symbol,
            start_date=start,
            end_date=end,
            force_refresh=bool(kwargs.get("force_refresh", False)),
        )
        return {
            "symbol": symbol,
            "start_date": str(start),
            "end_date": str(end),
            "data": data,
        }

    def get_metadata(self) -> Dict[str, Union[str, List[str]]]:
        """返回加载器元数据信息。"""
        return {
            "loader_type": "forex",
            "version": "1.0.0",
            "description": "外汇数据加载器（基于 yfinance）",
            "supported_currencies": self.get_supported_currencies(),
            "supported_pairs": self.get_supported_pairs(),
            "frequency": self.frequency,
        }

    def get_supported_currencies(self) -> List[str]:
        return list(self.supported_currencies)

    def get_supported_pairs(self) -> List[str]:
        return list(self.supported_pairs)

    def get_required_config_fields(self) -> List[str]:
        return ["cache_dir", "max_retries", "cache_days"]

    def validate_config(self) -> bool:
        missing = [field for field in self.get_required_config_fields() if field not in self.runtime_config]
        if missing:
            logger.warning("ForexDataLoader runtime config missing fields: %s", missing)
            return False
        return True

    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #
    def _ensure_initialized(self) -> None:
        if not self.is_initialized:
            self.initialize()

    def _normalize_dates(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> tuple[str, str]:
        def _to_str(value: Union[str, datetime]) -> str:
            if isinstance(value, datetime):
                return value.strftime("%Y-%m-%d")
            return str(value)

        start_str = _to_str(start_date)
        end_str = _to_str(end_date)
        return start_str, end_str

    def _build_cache_key(self, symbol: str, start: str, end: str) -> str:
        return f"{symbol}_{start}_{end}_{self.frequency}"

    def _cache_file_path(self, symbol: str, start: str, end: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace("=", "").replace(":", "")
        return self.cache_dir / f"{safe_symbol}_{start}_{end}_{self.frequency}.csv"

    def _is_file_cache_valid(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False
        if self.cache_days <= 0:
            return False
        age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        return age <= timedelta(days=self.cache_days)

    def _get_from_cache(
        self,
        cache_key: str,
        symbol: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        cached_df = self.cache_manager.get(cache_key)
        if isinstance(cached_df, pd.DataFrame):
            return cached_df.copy()

        csv_path = self._cache_file_path(symbol, start, end)
        if not self._is_file_cache_valid(csv_path):
            return None

        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            self.cache_manager.set(cache_key, df.copy())
            return df
        except Exception as exc:  # pragma: no cover - 异常仅用于健壮性
            logger.warning("Failed to read forex cache file %s: %s", csv_path, exc)
            return None

    def _fetch_forex_data(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=self.frequency,
                    progress=False,
                )
                if df is not None:
                    return df
            except Exception as exc:  # pragma: no cover - 重试逻辑
                last_exception = exc
                logger.warning(
                    "Failed to download forex data for %s (attempt %d/%d): %s",
                    symbol,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
        if last_exception:
            logger.error("Exceeded max retries downloading forex data for %s: %s", symbol, last_exception)
        return None

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        normalized.columns = [col.lower().strip() for col in normalized.columns]

        rename_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adj close": "close",
            "volume": "volume",
        }
        normalized = normalized.rename(columns=rename_map)

        required_columns = ["open", "high", "low", "close", "volume"]
        for column in required_columns:
            if column not in normalized.columns:
                normalized[column] = pd.Series(dtype="float64")

        normalized = normalized[required_columns]
        normalized.sort_index(inplace=True)
        return normalized

    def _persist_to_disk(self, symbol: str, start: str, end: str, df: pd.DataFrame) -> None:
        try:
            csv_path = self._cache_file_path(symbol, start, end)
            df.to_csv(csv_path)
        except Exception as exc:  # pragma: no cover - 非核心路径
            logger.warning("Failed to persist forex data for %s: %s", symbol, exc)

    # ------------------------------------------------------------------ #
    # 辅助方法
    # ------------------------------------------------------------------ #
    def clear_cache(self) -> None:
        """清除内存与磁盘缓存，可用于测试或调试。"""
        self.cache_manager.clear()
        for file_path in self.cache_dir.glob("*.csv"):
            try:
                file_path.unlink()
            except OSError:
                logger.debug("Failed to delete cache file %s", file_path)


__all__ = ["ForexDataLoader", "ForexRate"]
