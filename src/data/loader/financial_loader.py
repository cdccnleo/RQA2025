# src/data/loader/financial_loader.py
import configparser
from datetime import datetime
import pandas as pd
import os
from typing import Optional, Tuple, Any, Union, Dict
import akshare as ak
from pandas import DataFrame
from requests import RequestException
from src.infrastructure.utils.environment import is_production, is_development, is_testing
from src.infrastructure.utils.logger import get_logger
from pathlib import Path
import time
from src.infrastructure.utils.exceptions import DataLoaderError

logger = get_logger(__name__)  # 自动继承全局配置


class FinancialDataLoader(BaseDataLoader):
    """上市公司财务数据加载器，获取个股财务指标

    属性：
        save_path (Path): 数据存储路径
        cache_days (int): 本地缓存有效期天数
    """
    REQUIRED_COLUMNS = {'roe'}  # 定义必要列集合

    def __init__(self,
                 save_path: Union[str, Path] = "data/financial",
                 max_retries=3,
                 cache_days: int = 30,
                 raise_errors: Optional[bool] = None,
                 thread_pool=None):  # 添加 thread_pool 参数
        """
        初始化财务数据加载器

        Args:
            save_path: 数据存储路径，默认data/financial
            cache_days: 本地缓存有效期天数，默认30天
            raise_errors: 是否抛出异常 (None表示自动根据环境决定)
            thread_pool: 线程池实例
        """
        self.save_path = Path(save_path)
        self.max_retries = max_retries
        self.cache_days = cache_days
        self._setup()
        if raise_errors is None:
            self.raise_errors = not is_production()
        else:
            self.raise_errors = raise_errors
        self.thread_pool = thread_pool  # 存储线程池引用

    def _setup(self) -> None:
        """初始化工作目录和日志配置"""
        self.save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_from_config(cls, config: Union[configparser.ConfigParser, dict], thread_pool=None):
        """从配置文件创建实例"""
        # 从 config.ini 获取默认值
        default_config = configparser.ConfigParser()
        default_config.read('config.ini', encoding='utf-8')

        # 处理不同类型配置
        if isinstance(config, configparser.ConfigParser):
            loader_config = config['Financial'] if 'Financial' in config else {}  # 修改为对应的配置节名
        elif isinstance(config, dict):
            loader_config = config.get('Financial', {})
        elif hasattr(config, 'items'):  # 处理 Section 对象
            loader_config = dict(config)
        else:
            raise ValueError("不支持的配置类型")

        default_loader_config = default_config['Financial'] if 'Financial' in default_config else {}  # 修改为对应的配置节名

        # 安全获取配置值
        save_path = loader_config.get('save_path', default_loader_config.get('save_path', 'data/financial'))  # 修改默认路径

        # 处理整数值
        def safe_getint(source, key, default):
            value = source.get(key, None)
            if value is None:
                return default

            try:
                return int(value)
            except ValueError as e:
                raise DataLoaderError(f"配置项 {key} 的值无效（必须为整数）: {value}") from e

        max_retries = safe_getint(loader_config, 'max_retries',
                                  default_loader_config.getint('max_retries', 3) if hasattr(default_loader_config,
                                                                                            'getint') else 3)
        cache_days = safe_getint(loader_config, 'cache_days',
                                 default_loader_config.getint('cache_days', 30) if hasattr(default_loader_config,
                                                                                           'getint') else 30)

        return cls(
            save_path=save_path,
            max_retries=max_retries,
            cache_days=cache_days,
            thread_pool=thread_pool
        )

    def load_data(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """加载财务数据，优先使用有效缓存，否则获取新数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 加载的财务数据，date 为索引
        """
        file_path = self.save_path / f"{symbol}_financial.csv"

        try:
            cache_result = self._is_cache_valid(file_path, start_date, end_date)
            if isinstance(cache_result, tuple):
                valid, reason = cache_result
            else:
                valid = cache_result
                reason = "未知原因"

            if valid:
                return self._load_cache_data(file_path)
            else:
                logger.info(f"缓存无效: {reason}")
                return self._fetch_and_process_data(symbol, start_date, end_date)
        except Exception as e:
            return self._handle_error(e, "加载财务数据", symbol)

    def _handle_error(self, e: Exception, context: str, symbol: str = None) -> pd.DataFrame:
        # 不再截取错误消息，保留完整信息
        error_msg = str(e)
        full_context = f"{context}失败" + (f" (股票: {symbol})" if symbol else "")
        logger.error(f"{full_context}: {error_msg}", exc_info=True)

        if self.raise_errors:
            # 使用完整错误消息
            raise DataLoaderError(f"{full_context}: {error_msg}")
        return pd.DataFrame()

    def _is_cache_valid(self, file_path: Path, start_date: Union[str, datetime],
                        end_date: Union[str, datetime]) -> tuple[bool, str]:
        """统一返回元组 (是否有效, 原因)"""
        if not file_path.exists():
            return False, "缓存文件不存在"

        if not self._is_file_timely(file_path):
            return False, "缓存文件已过期"

        try:
            cached_df, validation_error = self._load_cache_data(file_path, validate=True)
            if cached_df.empty:
                return False, "缓存文件为空"

            # 处理验证错误
            if validation_error:
                return False, f"缓存数据验证失败: {validation_error}"

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            min_date = cached_df.index.min()
            max_date = cached_df.index.max()

            if min_date > start_dt or max_date < end_dt:
                return False, f"缓存日期范围不足: {min_date}~{max_date}"

            return True, "缓存验证通过"

        except Exception as e:
            return False, f"缓存验证失败: {str(e)}"

    def _is_file_timely(self, file_path: Path) -> bool:
        """检查文件是否在有效期内

        Args:
            file_path: 文件路径

        Returns:
            bool: 文件是否在有效期内
        """
        if not file_path.exists():
            return False

        file_mtime = os.path.getmtime(file_path)
        return (time.time() - file_mtime) <= (self.cache_days * 86400)

    def _load_cache_data(self, file_path: Path, validate=False) -> Union[
        DataFrame, tuple[DataFrame, Optional[str]], tuple[DataFrame, str]]:
        """加载缓存数据并进行基本验证"""
        try:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except Exception as e:
                # 捕获所有CSV解析错误
                raise DataLoaderError(f"加载缓存数据失败: {str(e)}") from e

            if df.empty:
                return pd.DataFrame(), "缓存文件为空"

            # 检查是否有未命名的索引列
            if 'Unnamed: 0' in df.columns:
                df = df.rename(columns={'Unnamed: 0': 'date'})

            # 检查是否有重复列名
            if any(df.columns.duplicated()):
                duplicate_cols = df.columns[df.columns.duplicated()].tolist()
                raise DataLoaderError(f"缓存文件包含重复列名: {duplicate_cols}")

            # 明确指定日期格式，处理无效日期
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            else:
                # 尝试使用第一列作为日期
                if len(df.columns) > 0:
                    df['date'] = pd.to_datetime(df[df.columns[0]], format='%Y-%m-%d', errors='coerce')
                else:
                    raise DataLoaderError("缓存文件中找不到日期列")

            # 检查无效日期
            invalid_dates = df[df['date'].isna()]
            if not invalid_dates.empty:
                line_numbers = invalid_dates.index.tolist()
                # 增强错误信息，包含具体行号
                invalid_lines = ", ".join(map(str, line_numbers))
                raise DataLoaderError(
                    f"缓存文件包含无效日期: 行 {invalid_lines}。请检查日期格式是否为YYYY-MM-DD。"
                )

            df.set_index('date', inplace=True)

            # 添加详细验证错误信息
            validation_error = None
            try:
                if validate and not self._validate_data(df):
                    validation_error = "缓存数据验证失败"
            except DataLoaderError as e:
                validation_error = str(e)

            return df, validation_error
        except DataLoaderError as e:
            raise e
        except Exception as e:
            return self._handle_error(e, "加载缓存数据")

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据是否包含必要的列和有效值"""
        if df.empty:
            error_msg = "数据验证失败: 数据为空"
            logger.error(error_msg)
            raise DataLoaderError(error_msg)

        # 检查必要列是否存在
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            error_msg = f"数据验证失败: 缺少必要列 {missing_cols}"
            logger.error(error_msg)
            raise DataLoaderError(error_msg)

        # 日期类型检查
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            error_msg = "数据验证失败: 索引不是日期类型"
            logger.error(error_msg)
            raise DataLoaderError(error_msg)

        # 验证所有关键指标
        validation_errors = []

        # ROE值检查
        if 'roe' in df.columns:
            invalid_roe = df[df['roe'] < -100]
            if not invalid_roe.empty:
                validation_errors.append("数据验证失败: 存在无效的ROE值")

        # 净利润增长率检查
        if 'net_profit_growth' in df.columns:
            invalid_growth = df[df['net_profit_growth'] < -100]
            if not invalid_growth.empty:
                # 获取第一个无效值及其日期
                first_invalid = invalid_growth.iloc[0]
                first_invalid_value = first_invalid['net_profit_growth']
                first_invalid_date = first_invalid.name.strftime('%Y-%m-%d')

                # 创建包含具体数值的错误消息
                validation_errors.append(
                    f"存在无效的净利润增长率 (日期: {first_invalid_date}, 值: {first_invalid_value})"
                )

        # 资产负债率检查
        if 'debt_asset_ratio' in df.columns:
            invalid_ratio = df[(df['debt_asset_ratio'] < 0) | (df['debt_asset_ratio'] > 100)]
            if not invalid_ratio.empty:
                validation_errors.append("数据验证失败: 存在无效的资产负债率")

        # 销售毛利率检查
        if 'gross_margin' in df.columns:
            invalid_margin = df[(df['gross_margin'] < 0) | (df['gross_margin'] > 100)]
            if not invalid_margin.empty:
                validation_errors.append("数据验证失败: 存在无效的销售毛利率")

        if validation_errors:
            error_msg = "; ".join(validation_errors)
            logger.error(error_msg)
            raise DataLoaderError(error_msg)

        return True

    def _fetch_and_process_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取并处理新数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 处理后的财务数据，date 为索引

        Raises:
            DataLoaderError: 数据获取或处理失败
        """
        try:
            raw_data = self._fetch_raw_data(symbol, start_date, end_date)

            # 检查是否获取到有效数据
            if raw_data.empty:
                error_msg = "无法获取财务数据"
                if self.raise_errors:
                    raise DataLoaderError(error_msg)
                logger.warning(error_msg)
                return pd.DataFrame()

            # 数据验证
            try:
                if not self._validate_data(raw_data):
                    if self.raise_errors:
                        raise DataLoaderError("数据验证失败")
                    return pd.DataFrame()
            except DataLoaderError as e:
                if self.raise_errors:
                    raise e
                return pd.DataFrame()

            # 验证数据范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            min_date = raw_data.index.min()
            max_date = raw_data.index.max()

            if min_date > end_dt or max_date < start_dt:
                # 创建更明确的错误消息
                error_msg = (
                    f"获取的财务数据范围与请求范围无交集: "
                    f"请求范围: {start_date} 到 {end_date}, "
                    f"实际获取: {min_date.strftime('%Y-%m-%d')} 到 {max_date.strftime('%Y-%m-%d')}"
                )
                if self.raise_errors:
                    raise DataLoaderError(error_msg)
                logger.warning(error_msg)
                return pd.DataFrame()

            # 保存数据
            file_path = self.save_path / f"{symbol}_financial.csv"
            self._save_data(raw_data, file_path)

            return raw_data

        except Exception as e:
            return self._handle_error(e, "获取和处理数据", symbol)

    def _validate_fetched_data(self, df: pd.DataFrame, start_date: str, end_date: str) -> bool:
        """验证获取的数据是否与请求的时间范围有交集"""
        if df.empty:
            return False

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        min_date = df.index.min()
        max_date = df.index.max()

        # 检查数据范围是否与请求的时间范围有交集
        if min_date > end_dt or max_date < start_dt:
            logger.debug(f"数据范围无交集: {min_date}~{max_date} vs {start_date}~{end_date}")
            return False

        return True

    def _fetch_raw_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从API获取原始财务数据（支持部分成功）"""
        # 验证日期格式
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise DataLoaderError(f"日期格式必须为YYYY-MM-DD: {start_date} 或 {end_date}")

        # 获取年份范围
        start_year = datetime.strptime(start_date, "%Y-%m-%d").year
        end_year = datetime.strptime(end_date, "%Y-%m-%d").year

        dfs = []
        errors = []  # 收集所有年份的错误信息
        for year in range(start_year, end_year + 1):
            try:
                # 确保调用 _retry_api_call
                df = self._retry_api_call(
                    self._fetch_financial_data,
                    symbol=symbol,
                    year=str(year)
                )

                if df is None or df.empty:
                    logger.warning(f"{year}年数据为空")
                    continue

                # 检查数据是否包含必要的列
                if not self._validate_basic_data(df):
                    error_msg = f"数据缺少必要列"
                    errors.append(error_msg)
                    logger.error(f"股票 {symbol} {year} 年数据验证失败: {error_msg}")
                    continue

                df = df.reset_index()
                dfs.append(df)

            except DataLoaderError as e:
                error_msg = str(e).split(":")[-1].strip()
                errors.append(error_msg)
                # 添加年份错误日志
                logger.error(f"股票 {symbol} {year} 年数据获取失败: {error_msg}")
            except Exception as e:
                # 捕获所有其他异常
                error_msg = f"未知错误: {str(e)}"
                errors.append(error_msg)
                logger.error(f"股票 {symbol} {year} 年数据获取异常: {error_msg}")

        if not dfs:
            # 所有年份都失败时返回空DataFrame
            logger.warning(f"未获取到任何有效数据: {symbol}")
            if self.raise_errors and errors:
                raise DataLoaderError(
                    f"获取财务数据失败: {errors[0] if errors else '未知原因'}"
                )
            return pd.DataFrame()

        combined_df = pd.concat(dfs)
        if not combined_df.empty:
            combined_df.set_index('date', inplace=True)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()

        # 在生产环境中，即使有错误也返回部分成功的数据
        if errors and not self.raise_errors:
            logger.warning(f"部分数据获取失败: {', '.join(errors)}")

        # 在测试环境中，如果有错误且配置为抛出错误，则抛出异常
        if self.raise_errors and errors:
            raise DataLoaderError(
                f"获取财务数据部分失败: {', '.join(errors)}"
            )

        return combined_df

    def _fetch_financial_data(self, symbol: str, year: str) -> pd.DataFrame:
        try:
            df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=year)
            if df is None or df.empty:
                logger.warning(f"未获取到股票 {symbol} {year} 年的财务数据")
                return pd.DataFrame()

            mapping = self._get_financial_column_mapping()
            df = df.rename(columns=mapping)

            # 检查必要列是否存在
            missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
            if missing_cols:
                # 创建更详细的错误消息
                missing_cols_str = ', '.join(missing_cols)
                raise DataLoaderError(
                    f"股票 {symbol} {year} 年财务数据缺少必要列: {missing_cols_str}。请检查数据源或列映射配置。"
                )

            # 处理日期列
            if 'date' not in df.columns:
                raise DataLoaderError(f"股票 {symbol} 的财务数据缺少日期列")

            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                # 增强错误消息
                logger.warning(f"股票 {symbol} 的日期转换失败: {str(e)}")
                raise DataLoaderError(f"股票 {symbol} 的日期格式无效") from e

            # 过滤无效日期
            df = df.dropna(subset=['date'])
            if df.empty:
                raise DataLoaderError("所有日期数据无效")

            return df.reset_index(drop=True)
        except Exception as e:
            error_msg = f"获取财务数据失败: {str(e).split(':')[-1].strip()}"
            logger.error(error_msg)
            raise DataLoaderError(error_msg) from e

    def _validate_basic_data(self, df: pd.DataFrame) -> bool:
        """验证基础数据完整性 - 严格检查必要列存在"""
        if df.empty:
            return False

        # 检查所有必要列是否存在
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            # 创建详细的错误消息
            missing_cols_str = ', '.join(missing_cols)
            raise DataLoaderError(
                f"财务数据验证失败: 缺少必要列 {missing_cols_str}。数据列: {', '.join(df.columns)}"
            )
        return True

    def _get_financial_column_mapping(self) -> Dict[str, str]:
        """获取财务列名映射"""
        return {
            "日期": "date",
            "净资产收益率(%)": "roe",
            "净利润增长率(%)": "net_profit_growth",
            "资产负债率(%)": "debt_asset_ratio",
            "销售毛利率(%)": "gross_margin"
        }

    def _retry_api_call(self, func: callable, *args, **kwargs) -> Any:
        symbol = kwargs.get('symbol', '未知股票')
        year = kwargs.get('year', '未知年份')

        # 完整的重试上下文
        retry_ctx = f"股票 {symbol} {year} 年"

        for attempt in range(self.max_retries + 1):  # 包含初始尝试
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"{retry_ctx} 第 {attempt + 1} 次重试成功")
                return result
            except (RequestException, ConnectionError) as e:
                # 记录详细的错误信息
                logger.error(
                    f"{retry_ctx} 第 {attempt + 1} 次尝试失败 (共 {self.max_retries} 次重试): {str(e)}"
                )

                if attempt == self.max_retries - 1:
                    error_msg = f"{retry_ctx} 数据获取失败: 所有重试均失败"
                    logger.error(error_msg)
                    raise DataLoaderError(error_msg) from e

                time.sleep(2 ** attempt)
            except Exception as e:
                error_msg = f"{retry_ctx} 数据获取异常: {str(e)}"
                logger.error(error_msg)
                raise DataLoaderError(error_msg) from e

        return None

    def _handle_exception(self, e: Exception, stage: str):
        error_msg = f"{stage} 阶段出错: {str(e)}"
        logger.error(error_msg)
        raise DataLoaderError(error_msg) from e

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据
        
        返回:
            包含加载器元数据的字典
        """
        return {
            "loader_type": "FinancialDataLoader",
            "cache_days": self.cache_days,
            "max_retries": self.max_retries,
            "supported_metrics": list(self.REQUIRED_COLUMNS),
            "raise_errors": self.raise_errors
        }

    def load(self, *args, **kwargs) -> Any:
        """实现BaseDataLoader的抽象方法，包装load_data"""
        return self.load_data(*args, **kwargs)

    def validate(self, data: Any) -> bool:
        """验证加载的数据是否符合预期
        
        参数:
            data: 要验证的数据
            
        返回:
            bool: 数据是否有效
        """
        if not isinstance(data, pd.DataFrame):
            return False
        return not data.empty and all(col in data.columns for col in self.REQUIRED_COLUMNS)

    def _save_data(self, df: pd.DataFrame, file_path: Path) -> bool:
        try:
            # 保存时包含索引（因为索引是日期）
            df.to_csv(file_path, index=True, encoding='utf-8')
            return True
        except Exception as e:
            self._handle_exception(e, "Data saving")
            return False
