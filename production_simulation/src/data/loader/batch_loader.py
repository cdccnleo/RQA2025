"""批量数据加载器，实现对多个股票的并发抓取。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..core.base_loader import BaseDataLoader, LoaderConfig
from .stock_loader import StockDataLoader

logger = logging.getLogger(__name__)


class BatchDataLoader(BaseDataLoader[Dict[str, Optional[pd.DataFrame]]]):
    """对多个股票执行批量加载的轻量调度器。"""

    def __init__(
        self,
        save_path: Union[str, Path, None] = None,
        max_retries: int = 3,
        cache_days: int = 1,
        timeout: int = 30,
        max_workers: int = 4,
        stock_loader: Optional[StockDataLoader] = None,
    ):
        loader_config = LoaderConfig(
            name="batch_loader",
            max_retries=max_retries,
            timeout=timeout,
        )
        super().__init__(loader_config)

        self.save_path = Path(save_path) if save_path else Path.cwd() / "data" / "batch"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.max_retries = max_retries
        self.cache_days = cache_days
        self.timeout = timeout
        self.max_workers = max_workers or 4

        self.stock_loader = stock_loader or StockDataLoader(
            save_path=str(self.save_path),
            max_retries=max_retries,
            cache_days=cache_days,
            timeout=timeout,
        )

    def load_batch(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        adjust: str = "hfq",
        max_workers: Optional[int] = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """批量加载多只股票的数据。"""
        if not symbols:
            return {}

        def task(sym: str):
            try:
                return self.stock_loader.load(
                    symbol=sym,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
            except Exception as exc:  # pragma: no cover - 容错路径
                logger.warning("批量加载股票 %s 失败: %s", sym, exc)
                return None

        results: Dict[str, Optional[pd.DataFrame]] = {}
        worker_count = max_workers or self.max_workers

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(task, sym) for sym in symbols]
            for sym, future in zip(symbols, futures):
                value = future.result()
                results[sym] = value

        self._update_stats()
        return results

    def load(self, *args, **kwargs) -> Dict[str, Optional[pd.DataFrame]]:
        """兼容接口，转发到 load_batch。"""
        return self.load_batch(*args, **kwargs)

    def validate(self, data: Any) -> bool:
        """批量数据的基础校验。"""
        if not isinstance(data, dict):
            return False
        return all(value is None or isinstance(value, pd.DataFrame) for value in data.values())

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "loader_type": "BatchDataLoader",
            "supports_batch": True,
            "max_workers": self.max_workers,
            "timeout": self.timeout,
        }
