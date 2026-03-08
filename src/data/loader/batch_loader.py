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

# 导入标准数据采集器
from src.infrastructure.orchestration.standard_data_collector import get_standard_data_collector

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
    
    async def load_batch_standard(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_type: str = "daily",
        adjust: str = "qfq"
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        使用标准数据采集器批量加载多只股票的数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型 (daily/weekly/monthly)
            adjust: 复权方式 (qfq/hfq)
            
        Returns:
            标准格式的采集结果
        """
        if not symbols:
            return {}
        
        logger.info(f"📊 使用标准数据采集器批量采集: {len(symbols)} 个股票")
        
        # 获取标准数据采集器实例
        standard_collector = get_standard_data_collector()
        
        # 转换日期格式
        if isinstance(start_date, datetime):
            start_date_str = start_date.strftime("%Y%m%d")
        else:
            start_date_str = str(start_date).replace("-", "")
        
        if isinstance(end_date, datetime):
            end_date_str = end_date.strftime("%Y%m%d")
        else:
            end_date_str = str(end_date).replace("-", "")
        
        # 使用标准数据采集器的批量采集功能
        batch_results = await standard_collector.batch_collect_stock_data(
            symbols=symbols,
            start_date=start_date_str,
            end_date=end_date_str,
            data_type=data_type,
            adjust=adjust
        )
        
        # 格式化结果
        results = {}
        for result in batch_results:
            symbol = result.get("symbol")
            if symbol:
                results[symbol] = result
        
        # 统计结果
        success_count = sum(1 for r in batch_results if r.get("success"))
        fail_count = len(batch_results) - success_count
        
        logger.info(f"🎉 标准批量采集完成")
        logger.info(f"✅ 成功: {success_count}, ❌ 失败: {fail_count}")
        
        return results
    
    async def load_incremental_standard(
        self,
        symbols: List[str],
        days: int = 7,
        data_type: str = "daily",
        adjust: str = "qfq"
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        使用标准数据采集器执行增量数据采集
        
        Args:
            symbols: 股票代码列表
            days: 采集天数
            data_type: 数据类型 (daily/weekly/monthly)
            adjust: 复权方式 (qfq/hfq)
            
        Returns:
            标准格式的增量采集结果
        """
        if not symbols:
            return {}
        
        logger.info(f"🔄 使用标准数据采集器增量采集: {len(symbols)} 个股票")
        
        # 获取标准数据采集器实例
        standard_collector = get_standard_data_collector()
        
        # 使用标准数据采集器的增量采集功能
        incremental_results = await standard_collector.incremental_collect(
            symbols=symbols,
            days=days,
            data_type=data_type,
            adjust=adjust
        )
        
        # 格式化结果
        results = {}
        for result in incremental_results:
            symbol = result.get("symbol")
            if symbol:
                results[symbol] = result
        
        # 统计结果
        success_count = sum(1 for r in incremental_results if r.get("success"))
        fail_count = len(incremental_results) - success_count
        
        logger.info(f"🎉 标准增量采集完成")
        logger.info(f"✅ 成功: {success_count}, ❌ 失败: {fail_count}")
        
        return results
