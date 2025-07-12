# src/data/loader/stock_loader.py
import configparser
import os
import pandas as pd
from pathlib import Path
import akshare as ak
from requests import RequestException, Timeout
from numpy import dtype
from typing import Optional, Tuple, Any, Union, Dict
from src.infrastructure.utils.exceptions import DataLoaderError
from src.infrastructure.utils.logger import get_logger
from datetime import datetime
import time
import concurrent.futures
from urllib3.exceptions import MaxRetryError, NameResolutionError

logger = get_logger(__name__)


class StockDataLoader(BaseDataLoader):
    """股票数据加载器

    属性：
        save_path (Path): 数据存储路径
        cache_days (int): 本地缓存有效期天数
    """

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
        if not save_path:
            raise ValueError("save_path不能为空")
        if max_retries <= 0:
            raise ValueError("max_retries必须大于0")
        if frequency not in ['daily', 'weekly', 'monthly']:
            raise ValueError("frequency必须是daily/weekly/monthly")

        self.save_path = save_path
        self.max_retries = max_retries
        self.cache_days = cache_days
        self.frequency = frequency
        self.adjust_type = adjust_type
        self.thread_pool = thread_pool

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
            loader_config = config.get('Stock', {})
        elif hasattr(config, 'items'):  # 处理 Section 对象
            loader_config = dict(config)
        else:
            raise ValueError("不支持的配置类型")

        default_loader_config = default_config['Stock'] if 'Stock' in default_config else {}

        # 安全获取配置值
        save_path = loader_config.get('save_path', default_loader_config.get('save_path', 'data/stock'))

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
        max_retries = safe_getint(loader_config, 'max_retries',
                                  default_loader_config.getint('max_retries', 3))

        cache_days = safe_getint(
            loader_config,
            'cache_days',
            default_loader_config.getint('cache_days', 30) if hasattr(default_loader_config, 'getint') else 30
        )

        return cls(
            save_path=save_path,
            max_retries=max_retries,
            cache_days=cache_days,
            frequency=loader_config.get('frequency', default_loader_config.get('frequency', 'daily')),
            adjust_type=loader_config.get('adjust_type', default_loader_config.get('adjust_type', 'none')),
            thread_pool=thread_pool
        )

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
            "supported_features": ["open", "high", "low", "close", "volume"]
        }

    def load(self, *args, **kwargs) -> Any:
        """实现BaseDataLoader的抽象方法，包装load_data"""
        return self.load_data(*args, **kwargs)

    def load_data(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime],
                  adjust: str = "hfq") -> pd.DataFrame:
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
        file_path = self._get_file_path(symbol, start_date, end_date)
        data = None  # 初始化 data 变量
        for attempt in range(self.max_retries + 1):
            try:
                # 尝试加载缓存数据
                if self._is_cache_valid(file_path):
                    cached_df = pd.read_csv(file_path, encoding='utf-8', parse_dates=["date"], index_col="date")
                    if cached_df.empty:
                        raise DataLoaderError("缓存数据为空")
                    data = cached_df
                    break  # 成功加载缓存数据，退出循环

                # 获取原始数据
                raw_df = self._fetch_raw_data(symbol, start_date, end_date, adjust)
                if raw_df.empty:
                    raise DataLoaderError("API 返回数据为空")

                processed_df = self._process_raw_data(raw_df)

                # 新增非交易日标记逻辑
                holidays = self._get_holidays(start_date, end_date)

                if not processed_df.empty:
                    processed_df['is_trading_day'] = 1
                    processed_df.loc[processed_df.index.normalize().isin(holidays), 'is_trading_day'] = 0
                    data = processed_df  # 确保 data 被赋值
                else:
                    data = pd.DataFrame()  # 处理 processed_df 为空的情况

                # 保存并返回数据
                data.to_csv(file_path, index=False, encoding='utf-8')
                break  # 成功获取数据，退出循环

            except DataLoaderError as e:
                logger.error(f"加载股票数据失败: {str(e)}")
                raise DataLoaderError(f"加载股票数据失败: {str(e)}") from e
            except (ConnectionError, TimeoutError) as e:
                if attempt >= self.max_retries:
                    logger.error("超过最大重试次数(%s)", self.max_retries)
                    raise DataLoaderError(f"加载股票数据失败: {str(e)}") from e
                logger.warning("第%s次重试...", attempt + 1)
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"加载股票数据失败: {str(e)}")
                raise DataLoaderError(f"加载股票数据失败: {str(e)}") from e

        if data is None:
            raise DataLoaderError("未能加载数据")
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
        return self.save_path / f"{symbol}_{start_date}_{end_date}.csv"

    def _is_cache_valid(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False
        try:
            cached_df = pd.read_csv(file_path, encoding='utf-8', parse_dates=["date"], index_col="date")
            if cached_df.empty:
                return False
            if (datetime.now() - cached_df.index.max()).days > self.cache_days:
                return False
            return True
        except Exception as e:
            logger.warning(f"缓存验证失败: {str(e)}")
            return False

    def _fetch_raw_data(self, symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        """获取原始股票数据"""
        try:
            import re
            match = re.match(r"(\d+)", symbol)
            if not match:
                raise ValueError(f"无效股票代码: {symbol}")

            numeric_symbol = match.group(1)

            if not isinstance(numeric_symbol, str) or not numeric_symbol.isdigit():
                raise ValueError(f"无效股票代码: {symbol}")

            start_fmt = start_date.replace("-", "")
            end_fmt = end_date.replace("-", "")

            raw_df = self._retry_api_call(
                ak.stock_zh_a_hist,
                symbol=numeric_symbol,
                period="daily",
                start_date=start_fmt,
                end_date=end_fmt,
                adjust=adjust
            )
            return raw_df
        except ValueError as e:
            raise DataLoaderError(f"无效股票代码: {symbol}") from e

    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一数据预处理"""
        if df is None or df.empty:
            raise DataLoaderError("原始数据为空")

        # 清理列名：去除多余空格并转换为小写
        df.columns = df.columns.str.strip().str.lower().str.replace(r'[\s\u3000]+', '', regex=True)
        logger.debug(f"清理后的列名: {df.columns.tolist()}")

        # 列名映射（键为英文列名，值为中文列名）
        column_mapping = {
            'date': '日期',
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量'
        }

        # 将中文列名映射回英文列名
        df = df.rename(columns={v: k for k, v in column_mapping.items()})

        # 检查必要的英文列是否存在
        required_english_columns = list(column_mapping.keys())
        missing_cols = [col for col in required_english_columns if col not in df.columns]

        if missing_cols:
            logger.error(f"原始数据缺少必要列: {missing_cols}")
            raise DataLoaderError(f"原始数据缺少必要列: {missing_cols}")

        # 确保 date 列是 datetime 类型
        df['date'] = pd.to_datetime(df['date'])

        # 设置 date 列为索引
        df = df.set_index('date')

        # 添加交易日标记
        df['is_trading_day'] = 1

        return df

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

    def _handle_exception(self, e: Exception, stage: str):
        error_msg = f"{stage}阶段出错: {str(e)}"
        logger.error(error_msg)
        raise DataLoaderError(str(e)) from e

    def _save_data(self, df: pd.DataFrame, file_path: Path) -> bool:
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            self._handle_exception(e, "Data saving")

    def _check_cache(self, file_path: Path) -> Tuple[bool, Optional[pd.DataFrame]]:
        if not file_path.exists():
            return False, None

        try:
            file_mtime = os.path.getmtime(file_path)
            if (time.time() - file_mtime) > (self.cache_days * 86400):
                return False, None

            df = pd.read_csv(file_path, encoding='utf-8', dtype=dtype)  # 添加dtype参数
            if df.empty:
                logger.warning(f"空缓存文件: {file_path}")
                return False, None
            return True, df

        except pd.errors.EmptyDataError as e:
            logger.error(f"空缓存文件错误: {str(e)}")
            return False, None
        except Exception as e:
            logger.error(f"加载缓存失败: {file_path} - {str(e)}")
            return False, None


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
            raise ValueError("frequency必须是daily/weekly/monthly")

        self.save_path = save_path
        self.max_retries = max_retries
        self.cache_days = cache_days
        self.frequency = frequency
        self.adjust_type = adjust_type
        self.thread_pool = thread_pool

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

            if not isinstance(industry_df, pd.DataFrame) or industry_df.empty:
                raise DataLoaderError("API 返回行业数据为空")

            mapping = {}
            failed_industries = []

            for _, row in industry_df.iterrows():
                for attempt in range(self.max_retries + 1):
                    try:
                        components = ak.stock_board_industry_cons_em(symbol=row['板块代码'])
                        if components is not None and not components.empty:
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
            if cached_df.empty:
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
            start_date: Union[str, datetime] = "2022-01-01",
            end_date: Union[str, datetime] = None,
            window: int = 20,
            max_workers: int = 4
    ) -> pd.DataFrame:
        """
        计算行业集中度（CR4/CR8）指标

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
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
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
                raw_df = ak.stock_board_industry_name_em()
                if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
                    raise DataLoaderError("API 返回行业数据为空")
                return raw_df
            except (ConnectionError, Timeout, NameResolutionError) as e:
                if attempt == self.max_retries - 1:
                    raise DataLoaderError(f"获取行业数据失败: {str(e)}") from e
                logger.warning(f"获取行业数据失败，重试 {attempt + 1}/{self.max_retries}...")
                time.sleep(2 ** attempt)
        return pd.DataFrame()

    def _handle_exception(self, e: Exception, stage: str):
        error_msg = f"{stage} 阶段出错: {str(e)}"
        logger.error(error_msg)
        raise DataLoaderError(str(e)) from e

    def _save_data(self, df: pd.DataFrame, file_path: Path) -> bool:
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
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
            if df.empty:
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

    def __init__(self, save_path: Union[str, Path] = "data/meta", max_retries=3, cache_days: int = 7):
        """
        初始化股票列表加载器

        Args:
            save_path: 数据存储路径，默认data/meta
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
            if df.empty:
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

    def _fetch_raw_data(
            self
    ) -> pd.DataFrame:
        """获取原始股票列表数据"""
        return self._retry_api_call(
            ak.stock_info_a_code_name
        )
