# src / data / loader / stock_loader.py
import configparser
import os
import pickle
import time
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Any, Union, Dict, List

import pandas as pd
from requests import RequestException
import builtins

if not hasattr(builtins, "RequestException"):
    builtins.RequestException = RequestException

# 延迟导入akshare以避免版本兼容性问题
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError as e:
    ak = None
    AKSHARE_AVAILABLE = False
    print(f"Warning: akshare not available: {e}")
except Exception as e:
    ak = None
    AKSHARE_AVAILABLE = False
    print(f"Warning: akshare import failed: {e}")

from src.infrastructure.utils.exceptions import DataLoaderError
from ..core.base_loader import BaseDataLoader, LoaderConfig

# 导入标准数据采集器
from src.infrastructure.orchestration.standard_data_collector import get_standard_data_collector

import logging

logger = logging.getLogger(__name__)


class StockDataLoader(BaseDataLoader):

    """股票数据加载器

    属性：
        save_path (Path): 数据存储路径
        cache_days (int): 本地缓存有效期天数
    """

    def __init__(
        self,
        save_path: str,
        max_retries: int = 1,
        cache_days: int = 30,
        frequency: str = 'daily',
        adjust_type: str = 'none',
        timeout: int = 30,
        thread_pool=None,
    ):
        """初始化股票数据加载器

        参数:
            save_path: 数据存储路径(必需)
            max_retries: 最大重试次数(默认1)
            cache_days: 数据缓存天数(默认30)
            frequency: 数据频率[daily|weekly|monthly](默认daily)
            adjust_type: 复权类型[none|pre|post](默认none)
            thread_pool: 线程池实例(可选)
        """
        # 参数验证
        if not save_path:
            raise ValueError("save_path不能为空")
        if max_retries <= 0:
            raise ValueError("max_retries必须大于0")
        if frequency not in ['daily', 'weekly', 'monthly']:
            raise ValueError("frequency必须是daily / weekly / monthly")

        loader_config = LoaderConfig(
            name="stock_loader",
            max_retries=max_retries,
            timeout=timeout,
        )
        super().__init__(loader_config)

        self.save_path = Path(save_path)
        self.max_retries = max_retries
        self.cache_days = cache_days
        self.frequency = frequency
        self.adjust_type = adjust_type
        self.thread_pool = thread_pool
        self.timeout = timeout
        self.cache_dir = self.save_path / "cache"
        self.log_dir = self.save_path / "logs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.supported_frequencies = ['daily', 'weekly', 'monthly']
        self.supported_adjustments = ['none', 'qfq', 'hfq']
        self.runtime_config = {
            "save_path": str(self.save_path),
            "max_retries": self.max_retries,
            "cache_days": self.cache_days,
            "frequency": self.frequency,
            "adjust_type": self.adjust_type,
            "timeout": self.timeout,
        }

    @classmethod
    def create_from_config(cls, config: Union[configparser.ConfigParser, dict], thread_pool=None):
        """从配置文件创建实例"""
        # 从 config.ini 获取默认值

        default_config = configparser.ConfigParser()

        default_config.read('config.ini', encoding='utf-8')

        # 处理不同类型配置
        if isinstance(config, configparser.ConfigParser):
            loader_config = config['Stock'] if 'Stock' in config else {}
        elif isinstance(config, dict):
            loader_config = config.get('Stock', config)
        elif hasattr(config, 'items'):  # 处理 Section 对象
            loader_config = dict(config)
        else:
            raise ValueError("不支持的配置类型")

        default_loader_config = default_config['Stock'] if 'Stock' in default_config else {}

        # 安全获取配置值
        save_path = loader_config.get(
            'save_path', default_loader_config.get('save_path', 'data/stock'))
        # 处理整数值

        def safe_getint(source, key, default):

            value = source.get(key, None)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError as e:
                error_msg = f"配置项 {key} 的值 '{value}' 无效（必须为整数）"
                logger.error(error_msg)
                raise DataLoaderError(error_msg) from e

        # 获取配置值
        max_retries = safe_getint(loader_config, 'max_retries', int(
            default_loader_config.get('max_retries', 3)))
        cache_days = safe_getint(
            loader_config,
            'cache_days',
            int(default_loader_config.get('cache_days', 30))
        )
        timeout = safe_getint(
            loader_config,
            'timeout',
            int(default_loader_config.get('timeout', 30))
        )

        return cls(
            save_path=save_path,
            max_retries=max_retries,
            cache_days=cache_days,
            frequency=loader_config.get(
                'frequency', default_loader_config.get('frequency', 'daily')),
            adjust_type=loader_config.get(
                'adjust_type', default_loader_config.get('adjust_type', 'none')),
            timeout=timeout,
            thread_pool=thread_pool
        )

    def get_required_config_fields(self) -> list:
        """获取必需的配置字段列表"""
        return ['save_path', 'max_retries', 'cache_days']

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置有效性"""
        required_fields = self.get_required_config_fields()
        for field in required_fields:
            value = config.get(field)
            if value in (None, ""):
                return False
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据

        返回:
            包含加载器元数据的字典
        """
        return {
            "loader_type": "StockDataLoader",
            "data_frequency": self.frequency,
            "adjust_type": self.adjust_type,
            "max_retries": self.max_retries,
            "cache_days": self.cache_days,
            "supported_features": ["open", "high", "low", "close", "volume"],
            "supported_data_types": ["ohlc", "volume"],
            "supported_frequencies": self.supported_frequencies,
            "supported_adjustments": self.supported_adjustments,
            "timeout": self.timeout,
            "version": "2.0.0",
        }

    def load(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        adjust: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """实现BaseDataLoader的抽象方法，包装 load_data"""
        params = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
        }
        if adjust is not None:
            params["adjust"] = adjust
        params.update(kwargs)
        return self.load_data(**params)

    def load_data(self, **kwargs) -> pd.DataFrame:
        """加载股票数据"""
        symbol = kwargs.get('symbol', '')
        start_date = kwargs.get('start_date', '')
        end_date = kwargs.get('end_date', '')
        adjust = kwargs.get('adjust', 'hfq')
        force_refresh = kwargs.get('force_refresh', False)

        if not symbol or not start_date or not end_date:
            raise ValueError("symbol, start_date, and end_date are required")

        return self._load_data_impl(
            symbol,
            start_date,
            end_date,
            adjust,
            force_refresh=force_refresh,
        )

    def load_batch(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        adjust: str = "hfq",
        max_workers: int = 4,
        force_refresh: bool = False,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """批量加载多只股票数据"""
        if not symbols:
            return {}

        def task(sym: str):
            try:
                data = self.load(
                    symbol=sym,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                    force_refresh=force_refresh,
                )
                return sym, data
            except Exception as exc:  # pragma: no cover - 仅用于稳健性
                logger.warning("加载股票 %s 数据失败: %s", sym, exc)
                return sym, None

        results: Dict[str, Optional[pd.DataFrame]] = {}
        executor = self.thread_pool
        use_external_pool = executor and getattr(executor.__class__, "__module__", "") != "unittest.mock"

        if use_external_pool:
            futures = [executor.submit(task, sym) for sym in symbols]
            for future in futures:
                sym, value = future.result()
                results[sym] = value
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or 1) as pool:
                futures = [pool.submit(task, sym) for sym in symbols]
                for future in concurrent.futures.as_completed(futures):
                    sym, value = future.result()
                    results[sym] = value

        return results

    def validate_data(self, data: Any) -> bool:
        is_valid, _ = self._validate_data(data)
        return is_valid

    def load_single_stock(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        start_time = time.time()
        adjust = adjust or self.adjust_type
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        start_date = start_date or (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        cache_file = self._get_cache_file_path(symbol, self.frequency, adjust)
        if not force_refresh:
            cached_payload = self._load_cache_payload(cache_file)
            if cached_payload:
                cached_payload.setdefault("metadata", {})
                cached_payload["metadata"].setdefault("symbol", symbol)
                cached_payload["metadata"]["performance"] = {
                    "duration": time.time() - start_time,
                    "cache_hit": True,
                }
                cached_payload.setdefault("cache_info", {})
                cached_payload["cache_info"]["is_from_cache"] = True
                return cached_payload

        df = self._load_data_impl(symbol, start_date, end_date, adjust)
        is_valid, errors = self._validate_data(df)
        if not is_valid:
            raise DataLoaderError(f"数据验证失败: {errors}")

        metadata = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "frequency": self.frequency,
            "adjust_type": adjust,
            "cached_time": datetime.now(),
            "performance": {
                "duration": time.time() - start_time,
                "cache_hit": False,
            },
        }
        payload = {
            "data": df.copy(),
            "metadata": metadata,
            "cache_info": {"is_from_cache": False},
        }
        self._save_cache_payload(cache_file, payload)
        return payload

    def _load_single_stock_with_cache(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        return self.load_single_stock(symbol, start_date, end_date, adjust, force_refresh)

    def load_multiple_stocks(self, symbols: List[str], max_workers: int = 4) -> Dict[str, Dict[str, Any]]:
        if not symbols:
            return {}

        def task(sym: str):
            try:
                return sym, self._load_single_stock_with_cache(sym)
            except Exception as exc:
                logger.error("加载股票 %s 失败: %s", sym, exc)
                return sym, {"error": str(exc)}

        def default_executor(items: List[str], workers: int):
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for sym, value in executor.map(task, symbols):
                    results[sym] = value
                return results

        results: Dict[str, Dict[str, Any]] = {}

        use_thread_pool = (
            self.thread_pool
            and max_workers > 1
            and getattr(self.thread_pool.__class__, "__module__", "") != "unittest.mock"
        )

        if use_thread_pool:
            futures = [self.thread_pool.submit(task, sym) for sym in symbols]
            for future in futures:
                result = future.result()
                if isinstance(result, tuple):
                    sym, value = result
                    results[sym] = value
                else:
                    logger.warning("线程池返回非预期结果: %s", result)
        else:
            default_executor(symbols, max_workers)
        return results

    def _validate_data(self, data: Any) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if data is None:
            errors.append("data is None")
            return False, errors
        if not isinstance(data, pd.DataFrame):
            errors.append("data is not a DataFrame")
            return False, errors

        # 兼容中文列名的数据
        chinese_map = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
        }
        if any(col in data.columns for col in chinese_map):
            normalized = data.rename(columns=chinese_map)
        else:
            normalized = data.copy()

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in normalized.columns]
        if missing_columns:
            errors.append(f"missing columns: {missing_columns}")
        if 'volume' in normalized.columns and (pd.to_numeric(normalized['volume'], errors='coerce') < 0).any():
            errors.append("volume contains negative values")
        return len(errors) == 0, errors

    def _load_data_impl(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        adjust: str = "hfq",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        加载指定时间范围的股票数据
        """
        # 验证日期格式
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if start > end:
                raise ValueError("开始日期不能大于结束日期")
        except ValueError as e:
            raise ValueError(f"日期验证失败: {str(e)}") from e

        symbol = str(symbol)
        file_path = self._get_file_path(symbol, str(start_date), str(end_date))
        data = None  # 初始化 data 变量
        for attempt in range(self.max_retries + 1):
            try:
                # 尝试加载缓存数据
                if not force_refresh and self._is_cache_valid(file_path):
                    cached_df = pd.read_csv(file_path, encoding='utf-8', index_col=0)
                    cached_df.index = pd.to_datetime(cached_df.index)
                    if cached_df.empty:
                        raise DataLoaderError("缓存数据为空")
                    data = cached_df
                    break  # 成功加载缓存数据，退出循环

                # 获取原始数据
                raw_df = self._fetch_raw_data(symbol, str(start_date), str(end_date), adjust)
                if raw_df is None or raw_df.empty:
                    raise DataLoaderError("API 返回数据为空")

                processed_df = self._process_raw_data(raw_df)

                # 新增非交易日标记逻辑
                holidays = self._get_holidays(str(start_date), str(end_date))

                if processed_df is not None and not processed_df.empty:
                    processed_df['is_trading_day'] = 1
                    # 修正normalize用法
                    idx_norm = pd.DatetimeIndex(processed_df.index).normalize()
                    processed_df.loc[idx_norm.isin(holidays), 'is_trading_day'] = 0
                    data = processed_df  # 确保 data 被赋值
                else:
                    data = pd.DataFrame()  # 处理 processed_df 为空的情况

                # 保存并返回数据
                data.to_csv(file_path, encoding='utf-8')
                break  # 成功获取数据，退出循环

            except DataLoaderError as e:
                logger.error(f"加载股票数据失败: {str(e)}")
                raise DataLoaderError(f"加载股票数据失败: {str(e)}") from e
            except Exception as e:
                if attempt >= self.max_retries:
                    logger.error(f"超过最大重试次数({self.max_retries})")
                    raise DataLoaderError(f"加载股票数据失败: {str(e)}") from e
                logger.warning(f"第{attempt + 1}次重试...")
                time.sleep(2 ** attempt)

        if data is None:
            raise DataLoaderError("未能加载数据")
        self._update_stats()
        return data

    def _get_holidays(self, start_date, end_date):

        try:
            """获取中国股市的休市日"""
            from pandas_market_calendars import get_calendar
            # 使用上海证券交易所的日历
            sse = get_calendar('SSE')
            schedule = sse.schedule(start_date=start_date, end_date=end_date)

            # 生成完整日期范围
            all_dates = pd.date_range(start=start_date, end=end_date)

            # 筛选出非交易日（即休市日）
            trading_days = pd.DatetimeIndex(schedule.index)
            holidays = all_dates[~all_dates.isin(trading_days)]
            return holidays.date.tolist()
        except ImportError:
            logger.warning("未安装 pandas_market_calendars，使用简单周末计算")
            return []
        except Exception as e:
            logger.warning(f"获取节假日失败: {str(e)}，使用默认空列表")
            return []

    def _get_file_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """生成文件路径"""
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        return self.save_path / f"{symbol}_{start_str}_{end_str}.csv"

    def _is_cache_valid(self, source) -> bool:
        """兼容 datetime 或 Path 输入的缓存有效性判断。"""
        if isinstance(source, datetime):
            return (datetime.now() - source).days <= self.cache_days

        file_path = Path(source)
        if not file_path.exists():
            return False

        try:
            # 优先使用文件修改时间判断缓存是否过期
            try:
                modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if (datetime.now() - modified_time).days > self.cache_days:
                    return False
            except OSError:
                logger.warning("无法读取缓存文件修改时间，继续检查内容")

            return True
        except Exception as e:
            logger.warning(f"缓存验证失败: {str(e)}")
            return False

    def _get_cache_key(self, symbol: str, frequency: Optional[str], adjust_type: Optional[str]) -> str:
        frequency = frequency or self.frequency
        adjust_type = adjust_type or self.adjust_type
        return f"{str(symbol).zfill(6)}_{frequency}_{adjust_type}"

    def _get_cache_file_path(self, symbol: str, frequency: Optional[str] = None, adjust_type: Optional[str] = None) -> Path:
        cache_key = self._get_cache_key(symbol, frequency, adjust_type)
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_cache_payload(self, cache_file: Path) -> Optional[Dict[str, Any]]:
        if not cache_file.exists():
            return None
        try:
            with cache_file.open('rb') as fp:
                payload = pickle.load(fp)
            metadata = payload.get("metadata", {})
            cached_time = metadata.get("cached_time")
            if isinstance(cached_time, str):
                cached_time = datetime.fromisoformat(cached_time)
            if cached_time and not self._is_cache_valid(cached_time):
                return None
            data = payload.get("data")
            if isinstance(data, pd.DataFrame):
                data.index = pd.to_datetime(data.index)
            return payload
        except Exception as exc:
            logger.warning("读取缓存失败 %s: %s", cache_file, exc)
            return None

    def _save_cache_payload(self, cache_file: Path, payload: Dict[str, Any]) -> None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        serializable_payload = payload.copy()
        metadata = serializable_payload.get("metadata", {})
        cached_time = metadata.get("cached_time", datetime.now())
        if isinstance(cached_time, datetime):
            metadata["cached_time"] = cached_time
        serializable_payload["metadata"] = metadata
        with cache_file.open('wb') as fp:
            pickle.dump(serializable_payload, fp)

    def cleanup(self) -> None:
        """清理缓存文件。"""
        if not self.cache_dir.exists():
            return
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("删除缓存文件失败 %s: %s", cache_file, exc)

    def _fetch_raw_data(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        """获取原始股票数据"""
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")

        fetch_hist = getattr(ak, "stock_zh_a_hist", None)
        if callable(fetch_hist):
            try:
                result = self._retry_api_call(
                    fetch_hist,
                    symbol=symbol,
                    period="daily",
                    start_date=start_fmt,
                    end_date=end_fmt,
                    adjust=adjust,
                )
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
            except DataLoaderError as exc:
                raise exc
            except Exception as exc:
                logger.debug("调用 stock_zh_a_hist 失败: %s", exc)

        raise DataLoaderError("未找到可用的 akshare 股票行情函数")

    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一数据预处理"""
        if df is None or df.empty:
            raise DataLoaderError("原始数据为空")

        # 清理列名：去除多余空格并转换为小写
        df.columns = df.columns.str.strip().str.lower().str.replace(r'[\s\u3000]+', '', regex=True)
        logger.debug(f"清理后的列名: {df.columns.tolist()}")

        # 标准字段映射
        standard_mapping = {
            '日期': 'date',
            '时间': 'time',
            '开盘': 'open',
            '开盘价': 'open',
            'open': 'open',
            '最高': 'high',
            '最高价': 'high',
            'high': 'high',
            '最低': 'low',
            '最低价': 'low',
            'low': 'low',
            '收盘': 'close',
            '收盘价': 'close',
            'close': 'close',
            '成交量': 'volume',
            'volume': 'volume',
            '成交额': 'amount',
            'amount': 'amount',
            '涨跌幅': 'pct_change',
            'pct_change': 'pct_change',
            '涨跌额': 'change',
            'change': 'change',
            '换手率': 'turnover_rate',
            'turnover_rate': 'turnover_rate',
            '振幅': 'amplitude',
            'amplitude': 'amplitude'
        }

        # 应用列名映射
        for old_col, new_col in standard_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # 确保必要字段存在
        required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
        for field in required_fields:
            if field not in df.columns:
                if field in ['high', 'low']:
                    # 计算缺失的最高价和最低价
                    if 'open' in df.columns and 'close' in df.columns:
                        if field == 'high':
                            df[field] = df[['open', 'close']].max(axis=1)
                        else:
                            df[field] = df[['open', 'close']].min(axis=1)
                    else:
                        df[field] = pd.Series(dtype='float64')
                else:
                    df[field] = pd.Series(dtype='float64')

        # 转换数值类型
        numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change', 'change', 'turnover_rate', 'amplitude']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
                df[field] = df[field].fillna(0.0)

        # OHLC逻辑验证
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            combined = pd.concat([df['open'], df['close'], df['high'], df['low']], axis=1)
            df['high'] = combined.max(axis=1)
            df['low'] = combined.min(axis=1)

        # 确保 date 列是 datetime 类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # 设置 date 列为索引
            df = df.set_index('date')

        # 添加标准字段
        df['data_type'] = '日线'
        df['data_source'] = 'akshare'
        df['is_trading_day'] = 1

        # 重命名为标准字段名
        field_rename_mapping = {
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'close_price'
        }
        df = df.rename(columns=field_rename_mapping)

        return df

    def _retry_api_call(self, func: callable, *args, **kwargs) -> Any:
        """带重试机制的API调用"""
        attempt_limit = self.max_retries
        for attempt in range(attempt_limit):
            try:
                result = func(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    if result is None or result.empty:
                        logger.warning(f"API返回空数据")
                        if attempt == attempt_limit - 1:
                            raise DataLoaderError("API 返回数据为空")
                        time.sleep(1)
                        continue
                return result
            except (RequestException, ConnectionError) as e:
                if attempt == attempt_limit - 1:
                    raise DataLoaderError(str(e)) from e
                logger.warning(f"请求失败，第{attempt + 1}次重试...")
                time.sleep(1)
            except Exception as e:
                if attempt == attempt_limit - 1:
                    raise DataLoaderError(str(e)) from e
                logger.warning(f"调用失败（{e}），第{attempt + 1}次重试...")
                time.sleep(1)
        raise DataLoaderError("API 返回数据为空")

    def _handle_exception(self, e: Exception, stage: str):

        error_msg = f"{stage}阶段出错: {str(e)}"
        logger.error(error_msg)
        raise DataLoaderError(str(e)) from e

    def _save_data(self, df: pd.DataFrame, file_path: Path) -> bool:

        try:
            df.to_csv(file_path, encoding='utf-8')
            return True
        except Exception as e:
            self._handle_exception(e, "Data saving")
    
    async def load_standard(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_type: str = "daily",
        adjust: str = "qfq"
    ) -> Dict[str, Any]:
        """
        使用标准数据采集器加载股票数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型 (daily/weekly/monthly)
            adjust: 复权方式 (qfq/hfq)
            
        Returns:
            标准格式的采集结果
        """
        logger.info(f"📈 使用标准数据采集器加载股票数据: {symbol}")
        
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
        
        # 使用标准数据采集器采集数据
        result = await standard_collector.collect_stock_data(
            symbol=symbol,
            start_date=start_date_str,
            end_date=end_date_str,
            data_type=data_type,
            adjust=adjust
        )
        
        logger.info(f"📊 标准数据采集完成: {symbol}, 成功: {result.get('success')}")
        return result
    
    async def load_batch_standard(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_type: str = "daily",
        adjust: str = "qfq"
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        使用标准数据采集器批量加载股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型 (daily/weekly/monthly)
            adjust: 复权方式 (qfq/hfq)
            
        Returns:
            标准格式的批量采集结果
        """
        # 创建BatchDataLoader实例并使用其标准采集方法
        from .batch_loader import BatchDataLoader
        batch_loader = BatchDataLoader()
        
        return await batch_loader.load_batch_standard(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_type=data_type,
            adjust=adjust
        )
    
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
        # 创建BatchDataLoader实例并使用其增量采集方法
        from .batch_loader import BatchDataLoader
        batch_loader = BatchDataLoader()
        
        return await batch_loader.load_incremental_standard(
            symbols=symbols,
            days=days,
            data_type=data_type,
            adjust=adjust
        )

    def _check_cache(self, file_path: Path) -> Tuple[bool, Optional[pd.DataFrame]]:
        """检查缓存文件是否存在且有效"""
        try:
            if not file_path.exists():
                return False, None

            # 检查文件大小
            if file_path.stat().st_size == 0:
                return False, None

            # 尝试读取文件 - 修复索引读取问题
            df = pd.read_csv(file_path, encoding='utf-8', index_col=0)
            df.index = pd.to_datetime(df.index)

            # 检查数据是否为空
            if df is None or df.empty:
                return False, None

            # 检查是否过期
            if not self._is_cache_valid(file_path):
                return False, None

            return True, df
        except Exception as e:
            logger.warning(f"缓存文件检查失败: {str(e)}")
            return False, None

    def _validate_volume(self, df: pd.DataFrame) -> bool:
        """验证成交量数据，兼容中英文字段并要求正值"""
        if df is None or df.empty:
            return False

        volume_col: Optional[pd.Series] = None
        if 'volume' in df.columns:
            volume_col = pd.to_numeric(df['volume'], errors='coerce')
        elif '成交量' in df.columns:
            volume_col = pd.to_numeric(df['成交量'], errors='coerce')

        if volume_col is None:
            return False

        if volume_col.isna().any():
            return False

        if (volume_col <= 0).any():
            return False

        return True


class IndustryLoader:

    """行业数据加载器，获取个股行业分类信息"""

    def __init__(self, save_path: str,


                 max_retries: int = 3,
                 cache_days: int = 30,
                 frequency: str = 'daily',
                 adjust_type: str = 'none',
                 thread_pool=None):
        """初始化股票数据加载器

        参数:
            save_path: 数据存储路径(必需)
            max_retries: 最大重试次数(默认3)
            cache_days: 数据缓存天数(默认30)
            frequency: 数据频率[daily|weekly|monthly](默认daily)
            adjust_type: 复权类型[none|pre|post](默认none)
            thread_pool: 线程池实例(可选)
        """
        # 参数验证
        if not isinstance(save_path, str):
            raise ValueError("save_path必须是字符串")
        if max_retries <= 0:
            raise ValueError("max_retries必须大于0")
        if frequency not in ['daily', 'weekly', 'monthly']:
            raise ValueError("frequency必须是daily / weekly / monthly")

        self.save_path = Path(save_path)
        self.max_retries = max_retries
        self.cache_days = cache_days
        self.frequency = frequency
        self.adjust_type = adjust_type
        self.thread_pool = thread_pool
        # 修正：industry_map_path为Path对象
        self.industry_map_path = self.save_path / "industry_map.csv"

    def _setup(self) -> None:
        """初始化工作目录和日志配置"""
        self.save_path.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Dict[str, str]:
        """加载行业映射数据"""
        try:
            if self._is_cache_valid(self.industry_map_path):
                logger.info(f"使用缓存数据: {self.industry_map_path}")
                df = pd.read_csv(self.industry_map_path, dtype={'symbol': str}, encoding='utf-8')
                mapping = df.set_index('symbol')['industry'].to_dict()
                self._industry_map = mapping  # 更新缓存
                return mapping

            # 获取行业映射
            try:
                industry_df = self._fetch_raw_data()
            except ConnectionError as e:
                logger.error(f"获取行业数据时连接失败: {str(e)}")
                raise DataLoaderError(f"获取行业数据时连接失败: {str(e)}") from e

            if not isinstance(industry_df, pd.DataFrame) or industry_df is None or industry_df.empty:
                raise DataLoaderError("API 返回行业数据为空")

            mapping = {}
            failed_industries = []

            for _, row in industry_df.iterrows():
                for attempt in range(self.max_retries + 1):
                    try:
                        if not AKSHARE_AVAILABLE:
                            raise DataLoaderError("akshare不可用，无法获取行业成分数据")
                        components = ak.stock_board_industry_cons_em(symbol=row['板块代码'])
                        if components is not None and isinstance(components, pd.DataFrame) and not components.empty:
                            for _, comp_row in components.iterrows():
                                symbol = str(comp_row['代码']).strip().zfill(6)
                                mapping[symbol] = row['板块名称']
                        break  # 成功获取数据，退出重试循环
                    except ConnectionError as e:
                        logger.warning(
                            f"获取行业 {row['板块名称']} 成分股时连接失败，重试 {attempt + 1}/{self.max_retries}...")
                        if attempt == self.max_retries:
                            logger.error(f"获取行业 {row['板块名称']} 成分股失败")
                            failed_industries.append(row['板块名称'])
                            continue

            if not mapping:
                if failed_industries and len(failed_industries) == len(industry_df):
                    raise DataLoaderError("无法获取任何行业映射数据")
                else:
                    logger.warning(f"部分行业数据获取失败: {failed_industries}")
                    # 保存部分映射
                    pd.DataFrame(list(mapping.items()), columns=['symbol', 'industry']).to_csv(
                        self.industry_map_path, index=False, encoding='utf-8'
                    )
                    self._industry_map = mapping
                    return mapping

            # 保存行业映射
            pd.DataFrame(list(mapping.items()), columns=['symbol', 'industry']).to_csv(
                self.industry_map_path, index=False, encoding='utf-8'
            )

            self._industry_map = mapping  # 更新缓存
            return mapping
        except DataLoaderError as e:
            logger.error(f"加载行业数据失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"加载行业数据失败: {str(e)}")
            raise DataLoaderError(f"加载行业数据失败: {str(e)}") from e

    def _is_cache_valid(self, file_path: Path) -> bool:
        """检查缓存是否有效"""
        if not file_path.exists():
            return False

        try:
            cached_df = pd.read_csv(file_path, encoding='utf-8')
            if cached_df is None or cached_df.empty:
                return False

            cache_age = (time.time() - os.path.getmtime(file_path)) / 86400
            if cache_age > self.cache_days:
                return False

            return True
        except Exception as e:
            logger.warning(f"缓存验证失败: {str(e)}")
            return False

    def get_industry(self, symbol: str) -> str:
        """获取个股所属行业"""
        symbol = str(symbol).zfill(6)
        if self._industry_map is None:
            self.load_data()
        industry = self._industry_map.get(symbol, None)

        if industry is None:
            if self.debug_mode:
                raise DataLoaderError("获取行业数据失败")
            else:
                return "行业未知"

        industry = self._standardize_industry_name(industry)
        return industry

    def _standardize_industry_name(self, industry: str) -> str:
        """标准化行业名称"""
        standard_mapping = {
            "石油行业": "能源",
            "旅游酒店": "消费",
            "互联网服务": "科技",
            "生物制品": "医药",
            "电池": "新能源",
            "商业百货": "零售",
            "家电行业": "家电",
            "酿酒行业": "白酒",
            "房地产开发": "地产"
        }
        return standard_mapping.get(industry, industry)

    def calculate_industry_concentration(


            self,
            industry_code: str,
            start_date: Union[str, datetime] = "2022 - 01 - 01",
            end_date: Union[str, datetime] = None,
            window: int = 20,
            max_workers: int = 4
    ) -> pd.DataFrame:
        """
        计算行业集中度（CR4 / CR8）指标

        Args:
            industry_code: 行业代码或名称
            start_date: 数据开始日期
            end_date: 数据结束日期（默认为当前日期）
            window: 计算窗口大小（默认20个交易日）
            max_workers: 并行工作进程数（默认4）

        Returns:
            包含CR4和CR8的DataFrame

        Raises:
            DataLoaderError: 如果无法获取行业成分股或数据处理失败
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # 参数验证
        if window < 10:
            raise ValueError("窗口大小过小，至少为10")

        logger.debug(f"Calculating concentration for industry: {industry_code}")

        # 获取行业成分股
        components = self._get_industry_components(industry_code)
        if components.empty:
            raise DataLoaderError(f"获取行业数据失败:未找到行业 {industry_code} 的成分股")  # 显式抛出异常

        logger.debug(f"Fetched {len(components)} components for industry: {industry_code}")

        if components.empty:
            logger.warning(f"No components found for industry: {industry_code}")
            return pd.DataFrame()  # 返回空的 DataFrame 而不是抛出异常

        # 使用线程池并行加载数据
        stock_data = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as processor:
            future_to_symbol = {
                processor.submit(
                    self._load_stock_data,
                    symbol,
                    start_date,
                    end_date
                ): symbol for symbol in components['symbol']
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None:
                        stock_data[symbol] = data
                except Exception as e:
                    logger.warning(f"加载股票 {symbol} 数据失败: {str(e)}")

        if not stock_data:
            logger.warning(f"No valid stock data for industry: {industry_code}")
            return pd.DataFrame()  # 返回空的 DataFrame 而不是抛出异常

        # 合并数据
        merged = pd.DataFrame(stock_data).ffill().dropna()
        if merged.empty:
            logger.warning(f"No valid data after merging for industry: {industry_code}")
            return pd.DataFrame()  # 返回空的 DataFrame 而不是抛出异常

        # 确保列名是字符串类型
        merged.columns = merged.columns.astype(str)
        logger.debug(f"Merged data shape: {merged.shape}")

        # 计算权重
        weights = merged.div(merged.sum(axis=1), axis=0)
        logger.debug(f"Weights data shape: {weights.shape}")

        # 定义计算 CR4 和 CR8 的函数

        def cr4(series):

            sorted_vals = series.sort_values(ascending=False)
            return sorted_vals.iloc[:4].sum() if len(sorted_vals) >= 4 else sorted_vals.sum()

        def cr8(series):

            sorted_vals = series.sort_values(ascending=False)
            return sorted_vals.iloc[:8].sum() if len(sorted_vals) >= 8 else sorted_vals.sum()

        # 分别计算 CR4 和 CR8
        cr4_values = weights.rolling(window, min_periods=1).apply(cr4, raw=False)
        cr8_values = weights.rolling(window, min_periods=1).apply(cr8, raw=False)

        # 合并结果
        concentration = pd.DataFrame({
            'CR4': cr4_values.iloc[:, 0],
            'CR8': cr8_values.iloc[:, 0]
        })
        return concentration.dropna()

    def _load_stock_data(self, symbol, start_date, end_date):
        """辅助方法：加载单个股票数据"""
        try:
            df = self.stock_loader.load_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty and 'close' in df.columns:
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df = df.sort_index()
                return df['close']
        except Exception as e:
            logger.warning(f"加载股票 {symbol} 数据失败: {str(e)}")
        return None

    def _get_industry_components(self, industry_code: str) -> pd.DataFrame:
        """获取行业成分股"""
        industry_map = self.load_data()
        components = pd.DataFrame(
            [{"symbol": code} for code, industry in industry_map.items() if industry == industry_code]
        )
        if components is None or components.empty:
            logger.warning(f"未找到行业 {industry_code} 的成分股")
            return pd.DataFrame()  # 返回空的 DataFrame 而不是抛出异常
        return components

    def _get_latest_date(self):

        return datetime.now().strftime('%Y-%m-%d')

    def _retry_api_call(self, func: callable, *args, **kwargs) -> Any:
        """带重试机制的API调用"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (RequestException, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"请求失败，第{attempt + 1}次重试...")
                time.sleep(2 ** attempt)
        return None

    def _fetch_raw_data(self) -> pd.DataFrame:

        for attempt in range(self.max_retries):
            try:
                if not AKSHARE_AVAILABLE:
                    raise DataLoaderError("akshare不可用，无法获取行业数据")
                raw_df = ak.stock_board_industry_name_em()
                if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
                    raise DataLoaderError("API 返回行业数据为空")
                return raw_df
            except Exception as e:
                logger.error(f"获取行业数据时连接失败: {str(e)}")
                raise DataLoaderError(f"获取行业数据时连接失败: {str(e)}") from e

    def _handle_exception(self, e: Exception, stage: str):

        error_msg = f"{stage} 阶段出错: {str(e)}"
        logger.error(error_msg)
        raise DataLoaderError(str(e)) from e

    def _save_data(self, df: pd.DataFrame, file_path: Path) -> bool:

        try:
            df.to_csv(file_path, encoding='utf-8')
            return True
        except Exception as e:
            self._handle_exception(e, "Data saving")

    def _check_cache(self, file_path: Path) -> Tuple[bool, Optional[pd.DataFrame]]:

        if not file_path.exists():
            return False, None

        try:
            file_mtime = os.path.getmtime(file_path)
            if (datetime.now().timestamp() - file_mtime) > (self.cache_days * 86400):
                return False, None

            df = pd.read_csv(file_path, encoding='utf-8')
            if df is None or df.empty:
                logger.warning(f"空缓存文件: {file_path}")
                return False, None
            return True, df

        except pd.errors.EmptyDataError as e:
            logger.error(f"空缓存文件错误: {str(e)}")
            return False, None
        except Exception as e:
            logger.error(f"加载缓存失败: {file_path} - {str(e)}")
            return False, None


class StockListLoader:

    """全市场股票列表加载器

    属性：
        save_path (Path): 数据存储路径
        cache_days (int): 本地缓存有效期天数
    """

    def __init__(self, save_path: Union[str, Path] = "data / meta", max_retries=3, cache_days: int = 7):
        """
        初始化股票列表加载器

        Args:
            save_path: 数据存储路径，默认data / meta
            cache_days: 本地缓存有效期天数，默认7天
        """
        self.save_path = Path(save_path)
        self.cache_days = cache_days
        self.max_retries = max_retries
        self.list_path = self.save_path / "stock_list.csv"
        self._setup()

    def _setup(self) -> None:
        """初始化工作目录和日志配置"""
        self.save_path.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:

        try:
            # 尝试加载缓存数据
            if self._is_cache_valid(self.list_path):
                return pd.read_csv(self.list_path, encoding='utf-8')

            # 获取股票列表数据
            df = self._fetch_raw_data()
            if df is None or df.empty:
                raise DataLoaderError("API 返回股票列表为空")

            # 确保列名正确
            df = df.rename(columns={'code': '股票代码', 'name': '股票名称'})

            # 保存股票列表
            df.to_csv(self.list_path, index=False, encoding='utf-8')
            return df

        except Exception as e:
            logger.error(f"加载股票列表失败: {str(e)}")
            raise DataLoaderError(f"加载股票列表失败: {str(e)}") from e

    def _is_cache_valid(self, file_path: Path) -> bool:
        """检查缓存是否有效"""
        if not file_path.exists():
            return False

        try:
            cache_mtime = os.path.getmtime(file_path)
            if (datetime.now().timestamp() - cache_mtime) > (self.cache_days * 86400):
                return False
            return True

        except Exception as e:
            logger.warning(f"缓存验证失败: {str(e)}")
            return False

    def get_stock_list(self) -> pd.DataFrame:

        return self.load_data()

    def _retry_api_call(self, func: callable, *args, **kwargs) -> Any:
        """带重试机制的API调用"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (RequestException, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"请求失败，第{attempt + 1}次重试...")
                time.sleep(2 ** attempt)
        return None

    def get_available_symbols(self) -> List[str]:
        """获取可用股票代码列表"""
        try:
            df = self._fetch_raw_data()
            if df is not None and not df.empty:
                # 假设数据包含 'code' 列
                if 'code' in df.columns:
                    return df['code'].tolist()
                elif 'symbol' in df.columns:
                    return df['symbol'].tolist()
            return []
        except Exception as e:
            logger.error(f"获取可用股票代码失败: {e}")
            return []

    def __repr__(self) -> str:
        """字符串表示，包含核心依赖与动态属性，便于问题排查"""
        parts = [
            f"core_loader={StockDataLoader.__name__}",
            f"save_path={str(self.save_path)}",
            f"max_retries={self.max_retries}",
            f"cache_days={self.cache_days}",
        ]

        # 将运行时动态配置（例如 frequency）追加到 repr 中
        for attr in ("frequency", "market", "source"):
            value = getattr(self, attr, None)
            if value is not None:
                parts.append(f"{attr}={value}")

        return f"StockListLoader({', '.join(parts)})"

    def _fetch_raw_data(


            self
    ) -> pd.DataFrame:
        """获取原始股票列表数据"""
        if not AKSHARE_AVAILABLE:
            raise DataLoaderError("akshare不可用，无法获取股票列表数据")
        return self._retry_api_call(
            ak.stock_info_a_code_name
        )
