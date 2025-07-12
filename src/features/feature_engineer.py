import os
import schedule

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import time
import logging
from pathlib import Path
import pickle
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

class FeatureProcessorProtocol(Protocol):
    """特征处理器协议"""
    def register_feature(self, config) -> None:
        """注册特征配置"""
        ...
from .sentiment.sentiment_analyzer import SentimentAnalyzer
from typing import Protocol

class FeatureProcessorProtocol(Protocol):
    """特征处理器协议"""
    def register_feature(self, config) -> None:
        """注册特征配置"""
        ...
    
    def calculate(self, indicator: str, price_data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        ...

logger = logging.getLogger(__name__)

class FeatureCalculationError(Exception):
    """特征计算失败异常"""
    pass

class FeatureEngineer:
    """增强版特征工程引擎"""

    def __init__(
        self, 
        technical_processor: FeatureProcessorProtocol,
        cache_dir: str = "./feature_cache",
        max_retries: int = 3,
        fallback_enabled: bool = True
    ):
        """
        初始化特征工程

        Args:
            technical_processor: 技术指标处理器实例
            cache_dir: 特征缓存目录
            max_retries: 最大重试次数
            fallback_enabled: 是否启用降级模式
        """
        self.max_retries = max_retries
        self.fallback_enabled = fallback_enabled
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化处理器
        self.technical_processor = technical_processor
        self.sentiment_analyzer = SentimentAnalyzer()

        # 线程池用于并行计算
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 缓存元数据
        self.cache_metadata: Dict[str, Dict] = {}
        self._load_cache_metadata()

    def _load_cache_metadata(self) -> None:
        """加载缓存元数据"""
        metadata_file = self.cache_dir / "feature_cache_metadata.pkl"
        if metadata_file.exists():
            try:
                with open(metadata_file, "rb") as f:
                    self.cache_metadata = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load feature cache metadata: {e}")

    def _save_cache_metadata(self) -> None:
        """保存缓存元数据"""
        metadata_file = self.cache_dir / "feature_cache_metadata.pkl"
        try:
            with open(metadata_file, "wb") as f:
                pickle.dump(self.cache_metadata, f)
        except Exception as e:
            logger.error(f"Failed to save feature cache metadata: {e}")

    def _get_cache_key(self, func_name: str, symbol: str, *args, **kwargs) -> str:
        """
        生成特征缓存键

        Args:
            func_name: 方法名称
            symbol: 标的代码
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            str: 缓存键
        """
        params = {
            "func": func_name,
            "symbol": symbol,
            "args": args,
            "kwargs": kwargs
        }
        return hashlib.md5(pickle.dumps(params)).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """
        从缓存获取特征

        Args:
            key: 缓存键

        Returns:
            Optional[pd.DataFrame]: 缓存的特征数据，不存在返回None
        """
        if key not in self.cache_metadata:
            return None

        cache_info = self.cache_metadata[key]
        cache_file = self.cache_dir / f"{key}.pkl"

        # 检查缓存有效期
        if datetime.now() - cache_info["timestamp"] > timedelta(hours=cache_info["ttl"]):
            try:
                os.remove(cache_file)
                del self.cache_metadata[key]
                self._save_cache_metadata()
            except Exception as e:
                logger.error(f"Failed to remove expired feature cache: {e}")
            return None

        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load feature cache {key}: {e}")
            return None

    def _save_to_cache(self, key: str, features: pd.DataFrame, ttl: int = 24) -> None:
        """
        保存特征到缓存

        Args:
            key: 缓存键
            features: 特征数据
            ttl: 缓存有效期(小时)
        """
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(features, f)

            self.cache_metadata[key] = {
                "timestamp": datetime.now(),
                "ttl": ttl
            }
            self._save_cache_metadata()
        except Exception as e:
            logger.error(f"Failed to save feature cache {key}: {e}")

    def calculate_technical_features(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        indicators: List[str],
        use_cache: bool = True,
        fallback_strategy: str = "simple"
    ) -> pd.DataFrame:
        """
        计算技术指标特征(带重试和降级)

        Args:
            symbol: 标的代码
            price_data: 价格数据(包含open, high, low, close, volume)
            indicators: 技术指标列表
            use_cache: 是否使用缓存
            fallback_strategy: 降级策略(simple|none)

        Returns:
            pd.DataFrame: 技术指标特征

        Raises:
            FeatureCalculationError: 当所有重试都失败时抛出
        """
        # 缓存检查逻辑保持不变
        cache_key = self._generate_cache_key(symbol, indicators)
        if use_cache and cache_key in self.cache_metadata:
            cache_info = self.cache_metadata[cache_key]
            if datetime.now() - cache_info["timestamp"] < timedelta(hours=cache_info["ttl"]):
                return self._load_from_cache(cache_key)

        # 带重试的特征计算
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                features = self._calculate_technical_features(
                    symbol=symbol,
                    price_data=price_data,
                    indicators=indicators
                )
                
                if use_cache:
                    self._save_to_cache(cache_key, features)
                return features
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                
                if self.fallback_enabled:
                    logger.warning(
                        f"Feature calculation failed after {self.max_retries} attempts, "
                        f"falling back to {fallback_strategy} strategy"
                    )
                    return self._fallback_technical_features(
                        price_data=price_data,
                        strategy=fallback_strategy
                    )
                
                raise FeatureCalculationError(
                    f"Failed to calculate features for {symbol} after {self.max_retries} retries"
                ) from last_error
        cache_key = self._get_cache_key("calculate_technical_features", symbol, *indicators)

        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.debug(f"Using cached technical features for {symbol}")
                return cached.copy()

        # 计算技术指标
        features = pd.DataFrame(index=price_data.index)
        for indicator in indicators:
            try:
                indicator_features = self.technical_processor.calculate(
                    indicator,
                    price_data
                )
                features = pd.concat([features, indicator_features], axis=1)
            except Exception as e:
                logger.error(f"Failed to calculate {indicator} for {symbol}: {e}")
                continue

        if use_cache:
            self._save_to_cache(cache_key, features)

        return features

    def calculate_sentiment_features(
        self,
        text_data: pd.DataFrame,
        models: List[str] = ["BERT"],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        计算情感分析特征

        Args:
            text_data: 文本数据(包含text和timestamp列)
            models: 情感分析模型列表
            use_cache: 是否使用缓存

        Returns:
            pd.DataFrame: 情感特征
        """
        cache_key = self._get_cache_key("calculate_sentiment_features", "all", *models)

        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.debug("Using cached sentiment features")
                return cached.copy()

        # 计算情感特征
        features = pd.DataFrame(index=text_data.index)
        for model in models:
            try:
                model_features = self.sentiment_analyzer.analyze(
                    model,
                    text_data["text"]
                )
                features = pd.concat([features, model_features], axis=1)
            except Exception as e:
                logger.error(f"Failed to calculate sentiment with {model}: {e}")
                continue

        if use_cache:
            self._save_to_cache(cache_key, features, ttl=12)  # 情感特征缓存12小时

        return features

    def batch_calculate_technical_features(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        indicators: List[str],
        incremental: bool = False,
        incremental_since: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        批量计算技术指标特征

        Args:
            symbols: 标的代码列表
            price_data: 各标的的价格数据字典
            indicators: 技术指标列表
            incremental: 是否增量计算
            incremental_since: 增量计算起始时间

        Returns:
            Dict[str, pd.DataFrame]: 各标的的技术指标特征
        """
        if incremental:
            if incremental_since is None:
                incremental_since = datetime.now() - timedelta(days=1)
            logger.info(f"Running incremental feature calculation since {incremental_since}")
        futures = {}
        results = {}

        for symbol in symbols:
            if symbol not in price_data:
                logger.warning(f"No price data for {symbol}")
                continue

            # 增量计算处理
            if incremental:
                # 获取已有特征
                cache_key = self._get_cache_key("calculate_technical_features", symbol, *indicators)
                existing_features = self._get_from_cache(cache_key)
                
                if existing_features is not None:
                    # 过滤增量数据
                    new_data = price_data[symbol][price_data[symbol].index > incremental_since]
                    if len(new_data) == 0:
                        results[symbol] = existing_features
                        continue
                    
                    # 合并新旧数据
                    full_data = pd.concat([
                        existing_features,
                        price_data[symbol][price_data[symbol].index > incremental_since]
                    ])
                else:
                    full_data = price_data[symbol]
            else:
                full_data = price_data[symbol]

            future = self.executor.submit(
                self.calculate_technical_features,
                symbol=symbol,
                price_data=full_data,
                indicators=indicators
            )
            futures[symbol] = future

        for symbol, future in futures.items():
            try:
                results[symbol] = future.result()
            except Exception as e:
                logger.error(f"Failed to calculate features for {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results

    def schedule_batch_calculation(
        self,
        symbols: List[str],
        data_provider: callable,
        indicators: List[str],
        interval_minutes: int = 60
    ) -> None:
        """
        定时批量计算特征
        
        Args:
            symbols: 标的代码列表
            data_provider: 数据提供函数
            indicators: 技术指标列表
            interval_minutes: 定时间隔(分钟)
        """
        def calculation_job():
            try:
                price_data = data_provider(symbols)
                self.batch_calculate_technical_features(
                    symbols=symbols,
                    price_data=price_data,
                    indicators=indicators,
                    incremental=True
                )
                logger.info("Scheduled feature calculation completed")
            except Exception as e:
                logger.error(f"Scheduled feature calculation failed: {e}")

        # 创建定时任务
        schedule.every(interval_minutes).minutes.do(calculation_job)
        logger.info(f"Started scheduled feature calculation every {interval_minutes} minutes")

    def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        status = {
            "cache_health": {
                "total_entries": len(self.cache_metadata),
                "valid_entries": sum(
                    1 for info in self.cache_metadata.values()
                    if datetime.now() - info["timestamp"] < timedelta(hours=info["ttl"])
                )
            },
            "processor_health": {
                "technical": self.technical_processor.is_ready(),
                "sentiment": self.sentiment_analyzer.is_ready()
            },
            "thread_pool_health": {
                "active_threads": self.executor._work_queue.qsize(),
                "max_workers": self.executor._max_workers
            }
        }
        return status

    def clear_cache(self, older_than_days: int = 7) -> None:
        """
        清理过期特征缓存

        Args:
            older_than_days: 保留最近多少天的缓存
        """
        cutoff = datetime.now() - timedelta(days=older_than_days)
        keys_to_delete = []

        for key, info in self.cache_metadata.items():
            if info["timestamp"] < cutoff:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                os.remove(cache_file)
                del self.cache_metadata[key]
            except Exception as e:
                logger.error(f"Failed to delete feature cache {key}: {e}")

        self._save_cache_metadata()
        logger.info(f"Cleared {len(keys_to_delete)} expired feature cache entries")

    def _fallback_technical_features(
        self,
        price_data: pd.DataFrame,
        strategy: str = "simple"
    ) -> pd.DataFrame:
        """
        降级模式下的技术指标计算

        Args:
            price_data: 价格数据
            strategy: 降级策略(simple|none)

        Returns:
            pd.DataFrame: 降级特征数据
        """
        logger.warning(f"Using fallback strategy: {strategy}")
        
        if strategy == "simple":
            # 简单降级策略：仅计算基本指标
            features = pd.DataFrame(index=price_data.index)
            features["ma_5"] = price_data["close"].rolling(5).mean()
            features["ma_20"] = price_data["close"].rolling(20).mean()
            features["volatility"] = price_data["close"].pct_change().rolling(20).std()
            features["fallback"] = True  # 标记为降级数据
            return features
        
        # 无降级策略：返回空DataFrame但标记降级状态
        features = pd.DataFrame(index=price_data.index)
        features["fallback"] = True
        return features

class ASharesFeatureMixin:
    """A股特有特征混合类"""

    @staticmethod
    def calculate_limit_status(symbol: str, realtime_engine) -> int:
        """计算涨跌停状态特征"""
        status = realtime_engine.get_limit_status(symbol)
        return 1 if status == 'up' else -1 if status == 'down' else 0

    @staticmethod
    def calculate_margin_ratio(margin_data: pd.DataFrame) -> float:
        """计算融资融券余额比"""
        return (margin_data['margin_balance'] / margin_data['total_market_cap']).iloc[-1]