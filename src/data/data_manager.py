"""
数据管理器 - 新版实现
"""
import configparser
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Type
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import joblib

from src.infrastructure.utils import DataLoaderError
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.utils import validate_dates, fill_missing_values

from .interfaces import IDataModel
from .base_loader import BaseDataLoader
from .registry import DataRegistry
from .validator import DataValidator
from .cache import CacheManager, CacheStrategy

logger = get_logger(__name__)


class DataModel(IDataModel):
    """
    数据模型实现，用于封装数据和元数据
    """
    def __init__(self, data: pd.DataFrame, frequency: str, metadata: Dict[str, Any] = None):
        """
        初始化数据模型
        
        Args:
            data: 数据框
            frequency: 数据频率
            metadata: 元数据信息
        """
        self.data = data
        self._frequency = frequency
        self._metadata = metadata or {}
        self._metadata.update({
            'created_at': datetime.now().isoformat(),
            'data_shape': data.shape if data is not None else None,
            'data_columns': data.columns.tolist() if data is not None else None,
        })
    
    def validate(self) -> bool:
        """
        数据有效性验证
        
        Returns:
            bool: 数据是否有效
        """
        if self.data is None or self.data.empty:
            return False
        return True
    
    def get_frequency(self) -> str:
        """
        获取数据频率
        
        Returns:
            str: 数据频率
        """
        return self._frequency
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取元数据信息
        
        Returns:
            Dict[str, Any]: 元数据信息
        """
        return self._metadata


class DataManager:
    """
    数据管理器，负责协调数据加载、验证和缓存
    """
    def __init__(self, config_path: Optional[Union[str, Path]] = None, config_dict: Optional[dict] = None):
        """
        初始化数据管理器
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # 加载配置
        self.config = self._init_config(config_path, config_dict)
        
        # 初始化线程池
        max_workers = self.config.getint("General", "max_concurrent_workers", fallback=4)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.logger.info(f"初始化线程池完成，max_workers={max_workers}")
        
        # 初始化注册中心
        self.registry = DataRegistry()
        self.logger.info(f"注册中心初始化完成，当前注册状态: {self.registry.list_registered_loaders()}")
        
        # 初始化数据验证器
        self.validator = DataValidator()
        self.logger.info("数据验证器初始化完成")
        
        # 初始化数据质量监控器
        self.quality_monitor = DataQualityMonitor(self.validator)
        self.logger.info("数据质量监控器初始化完成")
        
        # 初始化缓存管理器
        cache_dir = Path(self.config.get("General", "cache_dir", fallback="cache"))
        cache_strategy = CacheStrategy({
            'max_cache_size': self.config.getint("General", "max_cache_size", fallback=1024 * 1024 * 1024),
            'cache_ttl': self.config.getint("General", "cache_ttl", fallback=86400),
        })
        self.cache_manager = CacheManager(cache_dir, cache_strategy)
        
        # 初始化加载器
        self._init_loaders()
        
        # 初始化数据血缘追踪
        self.data_lineage = {}
        
        self.logger.info("数据管理器初始化完成")
    
    def _init_config(self, config_path: Optional[Union[str, Path]], config_dict: Optional[dict]) -> configparser.ConfigParser:
        """
        初始化配置 - 简化版
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典
            
        Returns:
            configparser.ConfigParser: 配置对象
            
        Raises:
            DataLoaderError: 如果配置初始化失败
        """
        config = configparser.ConfigParser()
        
        # 加载默认配置
        config.read_dict({
            "General": {
                'max_concurrent_workers': '4',
                'cache_dir': 'cache',
                'max_cache_size': str(1024 * 1024 * 1024),
                'cache_ttl': '86400',
            },
            "Stock": {
                'save_path': 'data/stock',
                'max_retries': '3',
                'cache_days': '30',
                'frequency': 'daily',
                'adjust_type': 'none'
            },
            "News": {
                'save_path': 'data/news',
                'max_retries': '5',
                'cache_days': '7'
            },
            "Financial": {
                'save_path': 'data/fundamental',
                'max_retries': '3',
                'cache_days': '30'
            },
            "Index": {
                'save_path': 'data/index',
                'max_retries': '3',
                'cache_days': '30'
            },
            "Metadata": {
                'save_path': 'data/metadata'
            }
        })
        
        # 优先使用config_dict
        if config_dict:
            config.read_dict(config_dict)
        # 其次使用config_path
        elif config_path:
            path = Path(config_path)
            if path.exists():
                config.read(path)
        
        # 验证必要配置项
        required = {
            "General": ["max_concurrent_workers", "cache_dir"],
            "Stock": ["save_path", "max_retries"]
        }
        
        for section, keys in required.items():
            if not config.has_section(section):
                raise DataLoaderError(f"缺少必要配置节: {section}")
            for key in keys:
                if not config.has_option(section, key):
                    raise DataLoaderError(f"缺少必要配置项: {section}.{key}")
        
        return config
    
    def _init_loaders(self) -> None:
        """
        初始化数据加载器 - 重构版
        """
        # 基础适配器工厂
        def create_adapter(loader_class, source_type: str, requires_symbol: bool = True):
            class GenericLoaderAdapter(BaseDataLoader):
                def __init__(self, config: Dict[str, Any]):
                    super().__init__(config)
                    self.loader = loader_class.create_from_config(
                        config=config,
                        thread_pool=None
                    )
                
                def load(self, start_date: str, end_date: str, frequency: str, **kwargs) -> IDataModel:
                    if requires_symbol and 'symbol' not in kwargs:
                        raise ValueError(f"Symbol is required for {source_type} data loading")
                    
                    data = self.loader.load_data(
                        kwargs.get('symbol'), 
                        start_date, 
                        end_date
                    ) if requires_symbol else self.loader.load_data(start_date, end_date)
                    
                    metadata = {
                        'source': source_type,
                        'start_date': start_date,
                        'end_date': end_date,
                        **kwargs
                    }
                    if requires_symbol:
                        metadata['symbol'] = kwargs['symbol']
                    
                    return DataModel(data, frequency, metadata)
                
                def get_required_config_fields(self) -> list:
                    return ['save_path', 'max_retries', 'cache_days']
            
            return GenericLoaderAdapter

        # 动态注册加载器
        from src.data.loader import (
            StockDataLoader,
            IndexDataLoader,
            SentimentNewsLoader,
            FinancialDataLoader
        )

        loader_map = {
            'stock': (StockDataLoader, 'stock', True),
            'index': (IndexDataLoader, 'index', True),
            'news': (SentimentNewsLoader, 'news', False),
            'financial': (FinancialDataLoader, 'financial', True)
        }

        for name, (loader_class, source_type, requires_symbol) in loader_map.items():
            adapter_class = create_adapter(loader_class, source_type, requires_symbol)
            self.registry.register_class(name, adapter_class)
            self.registry.create_loader(name, dict(self.config[name.title()]))
        
        self.logger.info(f"数据加载器初始化完成，已注册: {list(loader_map.keys())}")
    
    def register_loader(self, name: str, loader: BaseDataLoader) -> None:
        """
        注册数据加载器
        
        Args:
            name: 加载器名称
            loader: 加载器实例
        """
        self.registry.register(name, loader)
        self.logger.info(f"注册加载器: {name}")
    
    def register_loader_class(self, name: str, loader_class: Type[BaseDataLoader]) -> None:
        """
        注册数据加载器类
        
        Args:
            name: 加载器名称
            loader_class: 加载器类
        """
        self.registry.register_class(name, loader_class)
        self.logger.info(f"注册加载器类: {name}")
    
    def load_data(self, data_type: str, start_date: str, end_date: str, frequency: str = "1d", **kwargs) -> IDataModel:
        """
        加载数据
        
        Args:
            data_type: 数据类型
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            **kwargs: 其他参数
            
        Returns:
            IDataModel: 数据模型
            
        Raises:
            DataLoaderError: 如果数据加载失败
        """
        try:
            # 验证日期
            validate_dates(start_date, end_date)
            
            # 生成缓存键
            cache_key = self._generate_cache_key(data_type, start_date, end_date, frequency, **kwargs)
            
            # 尝试从缓存加载
            cached_data = self.cache_manager.get_cached_data(cache_key)
            if cached_data is not None:
                self.logger.info(f"从缓存加载数据: {data_type}")
                return DataModel(cached_data, frequency, {
                    'source': data_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'from_cache': True,
                    **kwargs
                })
            
            # 获取加载器
            if not self.registry.is_registered(data_type):
                error_msg = f"无效的数据源类型: {data_type}"
                self.logger.error(error_msg)
                raise DataLoaderError(error_msg)
            
            loader = self.registry.get_loader(data_type)
            
            # 加载数据
            self.logger.info(f"从数据源加载数据: {data_type}")
            data_model = loader.load(start_date, end_date, frequency, **kwargs)
            
            # 验证数据
            if not data_model.validate():
                self.logger.warning(f"数据验证失败: {data_type}")
                return data_model
            
            # 跟踪数据质量指标
            try:
                self.quality_monitor.track_metrics(data_model.data)
                self.logger.debug(f"数据质量指标跟踪完成: {data_type}")
            except Exception as e:
                self.logger.warning(f"数据质量跟踪失败: {str(e)}")
            
            # 保存到缓存
            self.cache_manager.save_to_cache(cache_key, data_model)
            
            # 记录数据血缘
            self._record_data_lineage(data_type, data_model, start_date, end_date, **kwargs)
            
            return data_model
            
        except DataLoaderError as e:
            # 直接抛出已有的DataLoaderError
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
        except Exception as e:
            # 其他异常包装为DataLoaderError
            self.logger.error(f"数据加载失败: {data_type} ({start_date} to {end_date})")
            raise DataLoaderError(f"数据加载失败: {str(e)}") from e
    
    def load_multi_source(self, stock_symbols: List[str], index_symbols: List[str],
                          start: str, end: str, frequency: str = "1d") -> Dict[str, IDataModel]:
        """
        多源数据协同加载
        
        Args:
            stock_symbols: 股票代码列表
            index_symbols: 指数代码列表
            start: 开始日期
            end: 结束日期
            frequency: 数据频率
            
        Returns:
            Dict[str, IDataModel]: 数据模型字典
            
        Raises:
            DataLoaderError: 如果数据加载失败
        """
        try:
            validate_dates(start, end)
            
            # 加载各类数据
            market_data = self._load_stock_data(stock_symbols, start, end, frequency)
            index_data = self._load_index_data(index_symbols, start, end, frequency)
            news_data = self._load_news_data(start, end, frequency)
            financial_data = self._load_financial_data(stock_symbols, start, end, frequency)
            
            # 组合结果
            result = {
                "market": market_data,
                "index": index_data,
                "news": news_data,
                "fundamental": financial_data
            }
            
            # 记录数据版本信息
            self._record_data_version(result, start, end)
            
            return result
            
        except DataLoaderError as e:
            self.logger.error(f"多源数据加载失败: {str(e)}")
            raise DataLoaderError(f"多源数据加载失败: {str(e)}") from e
    
    def _load_stock_data(self, symbols: List[str], start: str, end: str, frequency: str) -> IDataModel:
        """
        加载股票数据
        
        Args:
            symbols: 股票代码列表
            start: 开始日期
            end: 结束日期
            frequency: 数据频率
            
        Returns:
            IDataModel: 股票数据模型
        """
        dfs = []
        for symbol in symbols:
            try:
                data_model = self.load_data('stock', start, end, frequency, symbol=symbol)
                if data_model.validate():
                    dfs.append(data_model.data)
                else:
                    self.logger.warning(f"股票数据无效: {symbol}")
            except Exception as e:
                self.logger.warning(f"加载股票数据失败: {symbol} - {str(e)}")
        
        if not dfs:
            self.logger.warning("所有股票数据均无效")
            return DataModel(pd.DataFrame(), frequency, {
                'source': 'stock',
                'start_date': start,
                'end_date': end,
                'symbols': symbols,
            })
        
        # 合并数据
        combined = pd.concat(dfs)
        
        # 数据透视
        pivoted = combined.pivot_table(
            index="date",
            columns="symbol",
            values=["open", "high", "low", "close", "volume"]
        )
        
        # 对齐数据
        aligned = self._align_data(pivoted, start, end)
        
        return DataModel(aligned, frequency, {
            'source': 'stock',
            'start_date': start,
            'end_date': end,
            'symbols': symbols,
        })
    
    def _load_index_data(self, symbols: List[str], start: str, end: str, frequency: str) -> IDataModel:
        """
        加载指数数据
        
        Args:
            symbols: 指数代码列表
            start: 开始日期
            end: 结束日期
            frequency: 数据频率
            
        Returns:
            IDataModel: 指数数据模型
        """
        dfs = []
        for symbol in symbols:
            try:
                data_model = self.load_data('index', start, end, frequency, symbol=symbol)
                if data_model.validate():
                    df = data_model.data
                    df["symbol"] = symbol  # 添加指数标识
                    dfs.append(df)
                else:
                    self.logger.warning(f"指数数据无效: {symbol}")
            except Exception as e:
                self.logger.warning(f"加载指数数据失败: {symbol} - {str(e)}")
        
        if not dfs:
            self.logger.warning("所有指数数据均无效")
            return DataModel(pd.DataFrame(), frequency, {
                'source': 'index',
                'start_date': start,
                'end_date': end,
                'symbols': symbols,
            })
        
        # 合并数据
        combined = pd.concat(dfs)
        
        # 数据透视
        pivoted = combined.pivot_table(
            index="date",
            columns="symbol",
            values=["open", "high", "low", "close", "volume"]
        )
        
        # 对齐数据
        aligned = self._align_data(pivoted, start, end)
        
        return DataModel(aligned, frequency, {
            'source': 'index',
            'start_date': start,
            'end_date': end,
            'symbols': symbols,
        })
    
    def _load_news_data(self, start: str, end: str, frequency: str) -> IDataModel:
        """
        加载新闻数据
        
        Args:
            start: 开始日期
            end: 结束日期
            frequency: 数据频率
            
        Returns:
            IDataModel: 新闻数据模型
        """
        try:
            data_model = self.load_data('news', start, end, frequency)
            if not data_model.validate():
                self.logger.warning("新闻数据无效")
                return DataModel(pd.DataFrame(), frequency, {
                    'source': 'news',
                    'start_date': start,
                    'end_date': end,
                })
            
            # 处理新闻数据
            news_df = data_model.data
            processed_news = (
                news_df
                .pipe(fill_missing_values, method="ffill")
                .drop_duplicates(["publish_time", "title"])
                .sort_values("publish_time")
            )
            
            return DataModel(processed_news, frequency, {
                'source': 'news',
                'start_date': start,
                'end_date': end,
            })
            
        except Exception as e:
            self.logger.warning(f"加载新闻数据失败: {str(e)}")
            return DataModel(pd.DataFrame(), frequency, {
                'source': 'news',
                'start_date': start,
                'end_date': end,
                'error': str(e),
            })
    
    def _load_financial_data(self, symbols: List[str], start: str, end: str, frequency: str) -> IDataModel:
        """
        加载财务数据
        
        Args:
            symbols: 股票代码列表
            start: 开始日期
            end: 结束日期
            frequency: 数据频率
            
        Returns:
            IDataModel: 财务数据模型
        """
        dfs = []
        for symbol in symbols:
            try:
                data_model = self.load_data('financial', start, end, frequency, symbol=symbol)
                if data_model.validate():
                    df = data_model.data
                    df["symbol"] = symbol  # 添加股票标识
                    dfs.append(df)
                else:
                    self.logger.warning(f"财务数据无效: {symbol}")
            except Exception as e:
                self.logger.warning(f"加载财务数据失败: {symbol} - {str(e)}")
        
        if not dfs:
            self.logger.warning("所有财务数据均无效")
            return DataModel(pd.DataFrame(), frequency, {
                'source': 'financial',
                'start_date': start,
                'end_date': end,
                'symbols': symbols,
            })
        
        # 合并数据
        combined = pd.concat(dfs).reset_index(drop=True)
        
        return DataModel(combined, frequency, {
            'source': 'financial',
            'start_date': start,
            'end_date': end,
            'symbols': symbols,
        })
    
    def _generate_cache_key(self, data_type: str, start_date: str, end_date: str, frequency: str, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            data_type: 数据类型
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            **kwargs: 其他参数
            
        Returns:
            str: 缓存键
        """
        # 构建缓存参数
        cache_params = {
            'data_type': data_type,
            'start_date': start_date,
            'end_date': end_date,
            'frequency': frequency,
            **kwargs
        }
        
        # 如果有symbols参数，确保它是排序的
        if 'symbols' in cache_params and isinstance(cache_params['symbols'], list):
            cache_params['symbols'] = sorted(cache_params['symbols'])
        
        # 如果有symbol参数，确保它是字符串
        if 'symbol' in cache_params:
            cache_params['symbol'] = str(cache_params['symbol'])
        
        # 使用缓存策略生成键
        return self.cache_manager.strategy.generate_cache_key(cache_params)
    
    def _record_data_lineage(self, data_type: str, data_model: IDataModel, start_date: str, end_date: str, **kwargs) -> None:
        """
        记录数据血缘
        
        Args:
            data_type: 数据类型
            data_model: 数据模型
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数
        """
        lineage_key = f"{data_type}_{start_date}_{end_date}"
        
        # 记录数据处理步骤
        self.data_lineage[lineage_key] = {
            'data_type': data_type,
            'start_date': start_date,
            'end_date': end_date,
            'processing_time': datetime.now().isoformat(),
            'metadata': data_model.get_metadata(),
            'parameters': kwargs
        }
    
    def _record_data_version(self, data_models: Dict[str, IDataModel], start: str, end: str) -> None:
        """
        记录数据版本信息
        
        Args:
            data_models: 数据模型字典
            start: 开始日期
            end: 结束日期
        """
        version_info = {
            "data_source_version": self._get_data_source_version(),
            "data_processing_params": self._get_data_processing_params(),
            "timestamp": datetime.now().isoformat(),
            "date_range": {"start": start, "end": end}
        }
        
        # 保存版本信息
        version_path = Path(self.config.get("Metadata", "save_path")) / "data_versions.pkl"
        os.makedirs(os.path.dirname(version_path), exist_ok=True)
        joblib.dump(version_info, version_path)
    
    def _get_data_source_version(self) -> Dict:
        """
        获取数据源版本信息
        
        Returns:
            Dict: 数据源版本信息
        """
        versions = {}
        for loader_name in self.registry.list_registered_loaders():
            loader = self.registry.get_loader(loader_name)
            if loader:
                versions[loader_name] = f"{loader.__class__.__name__}_{loader.metadata.get('version', 'v1.0')}"
        return versions
    
    def _get_data_processing_params(self) -> Dict:
        """
        获取数据处理参数
        
        Returns:
            Dict: 数据处理参数
        """
        return {
            "missing_value_handling": "ffill",
            "data_alignment": "daily",
            "cache_strategy": "time-based",
            "cache_days": self.config.getint("General", "cache_ttl") // 86400  # 转换为天
        }
    
    def _align_data(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """
        将数据对齐到指定日期范围
        
        Args:
            df: 输入数据，必须包含DatetimeIndex或可转换为DatetimeIndex的索引
            start: 起始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: 对齐到目标日期范围的DataFrame
            
        Raises:
            DataLoaderError: 如果输入数据无效或对齐失败
        """
        try:
            # 处理空数据情况
            if df.empty:
                self.logger.debug("输入数据为空，返回空DataFrame")
                try:
                    full_range = pd.date_range(start=start, end=end, freq="D")
                    return pd.DataFrame(index=full_range)
                except Exception as e:
                    raise DataLoaderError(f"无法生成日期范围: {str(e)}") from e
            
            # 确保索引是DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    raise DataLoaderError(f"无法将索引转换为DatetimeIndex: {str(e)}") from e
            
            # 生成完整日期范围
            try:
                full_range = pd.date_range(start=start, end=end, freq="D")
            except Exception as e:
                raise DataLoaderError(f"无效的日期范围参数: {str(e)}") from e
            
            # 处理少量数据情况（少于3个日期）
            if len(df.index) < 3:
                self.logger.warning(f"数据点不足({len(df.index)})，无法推断频率，使用简单对齐")
                try:
                    # 简单重新索引，不尝试推断频率
                    aligned = df.reindex(full_range)
                    self.logger.info(
                        f"简单数据对齐完成 - 原始行数: {len(df)}, 对齐后行数: {len(aligned)}, "
                        f"缺失值比例: {aligned.isna().mean().mean():.1%}"
                    )
                    return aligned
                except Exception as e:
                    raise DataLoaderError(f"简单对齐失败: {str(e)}") from e
            
            # 记录原始频率
            try:
                original_freq = pd.infer_freq(df.index)
                self.logger.debug(f"原始数据频率: {original_freq}")
            except ValueError:
                original_freq = "未知"
                self.logger.warning("无法推断数据频率，使用默认日频")
            
            # 处理不同频率的数据
            if original_freq == "H" or (isinstance(original_freq, str) and original_freq.startswith("H")):
                # 使用重采样替代分组
                aligned = df.resample("D").mean()
            else:
                # 其他情况按日重采样 - 使用最后一个值
                aligned = df.resample("D").last()
            
            # 重新索引但不填充缺失值，保留假期/非交易日
            aligned = aligned.reindex(full_range)
            
            # 记录对齐结果统计
            self.logger.info(
                f"数据对齐完成 - 原始行数: {len(df)}, 对齐后行数: {len(aligned)}, "
                f"缺失值比例: {aligned.isna().mean().mean():.1%}, "
                f"频率: {original_freq}"
            )
            
            return aligned
            
        except DataLoaderError:
            raise  # 重新抛出已知异常
        except Exception as e:
            error_msg = f"数据对齐失败: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoaderError(error_msg) from e
    
    def validate_all_configs(self) -> bool:
        """
        验证所有加载器配置
        
        Returns:
            bool: 所有配置是否有效
        """
        all_valid = True
        for loader_name in self.registry.list_registered_loaders():
            loader = self.registry.get_loader(loader_name)
            try:
                if not loader._validate_config():
                    self.logger.warning(f"加载器配置无效: {loader_name}")
                    all_valid = False
            except Exception as e:
                self.logger.error(f"验证加载器配置失败: {loader_name} - {str(e)}")
                all_valid = False
        
        return all_valid
    
    def track_data_lineage(self, data_type: str = None) -> Dict:
        """
        获取数据血缘信息
        
        Args:
            data_type: 数据类型，如果为None则返回所有数据源的血缘信息
            
        Returns:
            Dict: 数据血缘信息
        """
        if data_type:
            return {k: v for k, v in self.data_lineage.items() if v['data_type'] == data_type}
        return self.data_lineage
    
    def clean_expired_cache(self) -> int:
        """
        清理过期缓存
        
        Returns:
            int: 清理的缓存数量
        """
        return self.cache_manager.clean_expired_cache()
    
    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        return self.cache_manager.get_cache_stats()
    
    def shutdown(self) -> None:
        """
        关闭数据管理器
        """
        try:
            # 关闭线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
                self.logger.info("线程池已安全关闭")
            
            # 保存数据血缘信息
            lineage_path = Path(self.config.get("Metadata", "save_path")) / "data_lineage.pkl"
            os.makedirs(os.path.dirname(lineage_path), exist_ok=True)
            joblib.dump(self.data_lineage, lineage_path)
            
            self.logger.info("数据管理器已安全关闭")
        except Exception as e:
            self.logger.error(f"关闭过程中发生错误: {str(e)}")
            raise DataLoaderError(f"关闭失败: {str(e)}") from e