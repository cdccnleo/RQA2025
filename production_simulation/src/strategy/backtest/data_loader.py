#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测数据加载器 - 增强版

提供完整的数据加载功能，支持多种数据源和数据类型，
包括股票数据、财务数据、新闻数据、指数数据等。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from io import StringIO

# 暂时注释掉这些导入，因为数据层模块可能不存在
# from src.data.stock import StockDataLoader
# from src.data.fundamental import FundamentalDataLoader
# from src.data.news import NewsDataLoader
# from src.data.index import IndexDataLoader

# 创建模拟的数据加载器类


class StockDataLoader:

    def load_ohlcv(self, symbol, start_date, end_date, frequency='1d', adjustment='qfq'):

        # 模拟数据加载
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        return data

    def load_tick_data(self, symbol, start_date, end_date):

        # 模拟tick数据
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start=start_date, end=end_date, freq='1min')
        data = pd.DataFrame({
            'price': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        return data


class FundamentalDataLoader:

    def load_financial_data(self, symbol, start_date, end_date):

        # 模拟财务数据
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start=start_date, end=end_date, freq='Q')
        data = pd.DataFrame({
            'revenue': np.random.randint(1000000, 10000000, len(dates)),
            'profit': np.random.randint(100000, 1000000, len(dates)),
            'assets': np.random.randint(10000000, 100000000, len(dates))
        }, index=dates)
        return data


class NewsDataLoader:

    def load_news_data(self, symbol, start_date, end_date):

        # 模拟新闻数据
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame({
            'title': [f'News {i}' for i in range(len(dates))],
            'content': [f'Content {i}' for i in range(len(dates))],
            'sentiment': np.random.choice(['positive', 'negative', 'neutral'], len(dates))
        }, index=dates)
        return data


class IndexDataLoader:

    def load_index_data(self, symbol, start_date, end_date):

        # 模拟指数数据
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 3000,
            'high': np.random.randn(len(dates)).cumsum() + 3010,
            'low': np.random.randn(len(dates)).cumsum() + 2990,
            'close': np.random.randn(len(dates)).cumsum() + 3000,
            'volume': np.random.randint(10000000, 100000000, len(dates))
        }, index=dates)
        return data


logger = logging.getLogger(__name__)


class EnhancedDataLoader:

    """
    增强版数据加载器

    支持多种数据类型的加载，包括股票数据、财务数据、新闻数据、指数数据等。
    提供并行加载、缓存机制、数据预处理等功能。
    """

    def __init__(self, cache_dir: str = "cache", max_workers: int = 4,


                 enable_cache: bool = True, config: Optional[Dict] = None):
        """
        初始化数据加载器

        Args:
            cache_dir: 缓存目录
            max_workers: 最大工作线程数
            enable_cache: 是否启用缓存
            config: 配置字典（向后兼容）
        """
        # 处理配置参数
        if config is not None:
            if isinstance(config, dict):
                self.config = config.copy()
                cache_dir = config.get('cache_dir', cache_dir)
                max_workers = config.get('max_workers', max_workers)
                enable_cache = config.get('enable_cache', enable_cache)
            else:
                # 如果config不是字典，使用默认值
                self.config = {
                    'cache_dir': "cache",
                    'max_workers': 4,
                    'enable_cache': True,
                    'timezone': 'Asia / Shanghai'
                }
                cache_dir = "cache"
                max_workers = 4
                enable_cache = True
        else:
            self.config = {
                'cache_dir': cache_dir,
                'max_workers': max_workers,
                'enable_cache': enable_cache,
                'timezone': 'Asia / Shanghai'
            }

        # 确保cache_dir是字符串
        if not isinstance(cache_dir, str):
            cache_dir = str(cache_dir) if cache_dir is not None else "cache"

        # 处理复杂的配置结构
        if isinstance(cache_dir, dict):
            # 如果cache_dir是字典，尝试提取实际的缓存目录
            if 'data' in cache_dir:
                # 从data配置中提取缓存目录
                data_config = cache_dir['data']
                if 'stock' in data_config and 'save_path' in data_config['stock']:
                    cache_dir = data_config['stock']['save_path']
                elif 'financial' in data_config and 'cache_dir' in data_config['financial']:
                    cache_dir = data_config['financial']['cache_dir']
                else:
                    cache_dir = "cache"
            else:
                cache_dir = "cache"

        # 确保cache_dir是有效的字符串路径
        if not isinstance(cache_dir, str) or not cache_dir or cache_dir == "":
            cache_dir = "cache"

        # 处理Windows路径问题
        cache_dir = cache_dir.replace('\\', '/')

        # 确保缓存目录存在
        try:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # 如果创建目录失败，使用默认缓存目录
            logger.warning(f"无法创建缓存目录 {cache_dir}，使用默认目录: {e}")
            self.cache_dir = Path("cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.enable_cache = enable_cache

        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化数据加载器
        self.stock_loader = StockDataLoader()
        self.fundamental_loader = FundamentalDataLoader()
        self.news_loader = NewsDataLoader()
        self.index_loader = IndexDataLoader()

        # 缓存字典
        self.cache = {}

        logger.info(
            f"EnhancedDataLoader initialized with cache_dir={cache_dir}, max_workers={max_workers}")

    def load_ohlcv(self, symbol: str, start_date: str, end_date: str,


                   frequency: str = '1d', adjustment: str = 'qfq') -> pd.DataFrame:
        """
        加载OHLCV数据（向后兼容方法）

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            frequency: 频率
            adjustment: 复权类型

        Returns:
            OHLCV数据
        """
        return self.load_ohlcv_data([symbol], start_date, end_date, frequency, adjustment).get(symbol, pd.DataFrame())

    def load_ohlcv_data(self, symbols: List[str], start_date: str, end_date: str,


                        frequency: str = '1d', adjustment: str = 'qfq') -> Dict[str, pd.DataFrame]:
        """
        并行加载多个股票的OHLCV数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            frequency: 频率
            adjustment: 复权类型

        Returns:
            股票代码到数据的映射
        """
        if not symbols:
            return {}

        # 检查缓存
        cache_key = self._generate_cache_key(symbols, start_date, end_date, frequency, adjustment)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded OHLCV data from cache for {len(symbols)} symbols")
            return cached_data

        # 并行加载数据
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._load_single_ohlcv, symbol, start_date, end_date, frequency, adjustment): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[symbol] = data
                except Exception as e:
                    logger.error(f"Failed to load data for {symbol}: {e}")

        # 保存到缓存
        if self.enable_cache and results:
            self._save_to_cache(cache_key, results)

        logger.info(f"Loaded OHLCV data for {len(results)} symbols")
        return results

    def _load_single_ohlcv(self, symbol: str, start_date: str, end_date: str,


                           frequency: str, adjustment: str) -> pd.DataFrame:
        """加载单个股票的OHLCV数据"""
        try:
            return self.stock_loader.load_ohlcv(symbol, start_date, end_date, frequency, adjustment)
        except Exception as e:
            logger.error(f"Error loading OHLCV data for {symbol}: {e}")
            return pd.DataFrame()

    def load_tick_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载tick数据（向后兼容方法）

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            tick数据
        """
        try:
            return self.stock_loader.load_tick_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Error loading tick data for {symbol}: {e}")
            return pd.DataFrame()

    def load_fundamental_data(self, symbols: List[str], start_date: str, end_date: str,


                              data_type: str = 'financial') -> Dict[str, pd.DataFrame]:
        """
        加载基本面数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型

        Returns:
            股票代码到数据的映射
        """
        results = {}
        for symbol in symbols:
            try:
                data = self.fundamental_loader.load_financial_data(symbol, start_date, end_date)
                results[symbol] = data
            except Exception as e:
                logger.error(f"Error loading fundamental data for {symbol}: {e}")

        return results

    def load_news_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        加载新闻数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股票代码到数据的映射
        """
        results = {}
        for symbol in symbols:
            try:
                data = self.news_loader.load_news_data(symbol, start_date, end_date)
                results[symbol] = data
            except Exception as e:
                logger.error(f"Error loading news data for {symbol}: {e}")

        return results

    def load_index_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        加载指数数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股票代码到数据的映射
        """
        results = {}
        for symbol in symbols:
            try:
                data = self.index_loader.load_index_data(symbol, start_date, end_date)
                results[symbol] = data
            except Exception as e:
                logger.error(f"Error loading index data for {symbol}: {e}")

        return results

    def load_universe(self, universe_type: str = 'stock') -> List[str]:
        """
        加载股票池

        Args:
            universe_type: 股票池类型

        Returns:
            股票代码列表
        """
        # 模拟股票池
        if universe_type == 'stock':
            return ['000001.SZ', '000002.SZ', '000858.SZ']
        elif universe_type == 'index':
            return ['000300.SH', '000905.SH', '000852.SH']
        else:
            return []

    def load_comprehensive_data(self, symbols: List[str], start_date: str, end_date: str,


                                data_types: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        加载综合数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_types: 数据类型列表

        Returns:
            综合数据字典
        """
        if data_types is None:
            data_types = ['ohlcv', 'fundamental', 'news']

        results = {}
        for symbol in symbols:
            results[symbol] = {}

        if 'ohlcv' in data_types:
            ohlcv_data = self.load_ohlcv_data([symbol], start_date, end_date)
            results[symbol]['ohlcv'] = ohlcv_data.get(symbol, pd.DataFrame())

        if 'fundamental' in data_types:
            fundamental_data = self.load_fundamental_data([symbol], start_date, end_date)
            results[symbol]['fundamental'] = fundamental_data.get(symbol, pd.DataFrame())

        if 'news' in data_types:
            news_data = self.load_news_data([symbol], start_date, end_date)
            results[symbol]['news'] = news_data.get(symbol, pd.DataFrame())

        return results

    def preprocess_data(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        数据预处理

        Args:
            data: 原始数据
            **kwargs: 预处理参数

        Returns:
            预处理后的数据
        """
        processed_data = {}

        for symbol, df in data.items():
            if df.empty:
                continue

            # 数据清洗
            df = self._clean_data(df)

            # 数据标准化
            df = self._normalize_data(df)

            # 处理缺失值
            df = self._handle_missing_values(df)

            processed_data[symbol] = df

        return processed_data

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 移除重复数据
        df = df.drop_duplicates()

        # 移除异常值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in ['volume']:  # 成交量不需要标准化
                continue

            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if not numeric_columns.empty:
            # 前向填充
            df[numeric_columns] = df[numeric_columns].ffill()
            # 后向填充
            df[numeric_columns] = df[numeric_columns].bfill()

        return df

    def get_data_metadata(self) -> Dict[str, Any]:
        """
        获取数据元信息

        Returns:
            元信息字典
        """
        return {
            'total_symbols': len(self.cache),
            'data_types': list(set([key.split('_')[0] for key in self.cache.keys()])),
            'time_range': {},
            'data_quality': {},
            'timezone': self.config.get('timezone', 'Asia / Shanghai'),
            'loader_type': 'EnhancedDataLoader'
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        cache_files = len(list(self.cache_dir.glob('*.json')))
        cache_size_mb = sum(f.stat().st_size for f in self.cache_dir.glob('*.json')) / (1024 * 1024)

        return {
            'cache_files': cache_files,
            'cache_size_mb': round(cache_size_mb, 2),
            'cache_size': f"{cache_size_mb:.2f}MB"
        }

    def clear_cache(self, pattern: str = None) -> None:
        """
        清理缓存

        Args:
            pattern: 缓存文件模式
        """
        if pattern:
            cache_files = list(self.cache_dir.glob(pattern))
        else:
            cache_files = list(self.cache_dir.glob('*.json'))

        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.info(f"Cleared cache file: {cache_file}")
            except Exception as e:
                logger.error(f"Failed to clear cache file {cache_file}: {e}")

        # 清空内存缓存
        self.cache.clear()

    def _generate_cache_key(self, symbols: List[str], start_date: str, end_date: str,


                            frequency: str, adjustment: str) -> str:
        """生成缓存键"""
        key_data = {
            'symbols': sorted(symbols),
            'start_date': start_date,
            'end_date': end_date,
            'frequency': frequency,
            'adjustment': adjustment
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, pd.DataFrame]]:
        """从缓存加载数据"""
        if not self.enable_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf - 8') as f:
                cached_data = json.load(f)

            # 转换回DataFrame
            results = {}
            for symbol, data_str in cached_data.items():
                df = pd.read_json(StringIO(data_str))
                df.index = pd.to_datetime(df.index)
                results[symbol] = df

            return results
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, pd.DataFrame]) -> None:
        """保存数据到缓存"""
        if not self.enable_cache:
            return

        try:
            # 转换DataFrame为JSON字符串
            cached_data = {}
            for symbol, df in data.items():
                cached_data[symbol] = df.to_json()

            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf - 8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)

            # 更新内存缓存
            self.cache[cache_key] = data

        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")


# 向后兼容的别名
BacktestDataLoader = EnhancedDataLoader
DataLoader = EnhancedDataLoader
