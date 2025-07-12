# src/data/loader/index_loader.py
import configparser

import joblib
import pandas as pd
from pathlib import Path
import akshare as ak
from requests import RequestException
from typing import Any, Union
from src.infrastructure.utils.exceptions import DataLoaderError
from src.infrastructure.utils.logger import get_logger
import time
import os
from datetime import datetime

logger = get_logger(__name__)


class IndexDataLoader(BaseDataLoader):
    """指数数据加载器"""

    INDEX_MAPPING = {
        'HS300': '000300',
        'SZ50': '000016',
        'CY50': '399673',
        'KC50': '000688'
    }

    def __init__(self, save_path: Union[str, Path] = "data/index", max_retries=3, cache_days: int = 30,
                 thread_pool=None):  # 添加 thread_pool 参数
        """
        初始化指数数据加载器

        Args:
            save_path: 数据存储路径，默认data/index
            cache_days: 本地缓存有效期天数，默认30天
            thread_pool: 线程池实例
        """
        self.save_path = Path(save_path)
        self.max_retries = max_retries
        self.cache_days = cache_days
        self.thread_pool = thread_pool  # 存储线程池引用
        self._setup()

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
            loader_config = config['Index'] if 'Financial' in config else {}
        elif isinstance(config, dict):
            loader_config = config.get('Index', {})
        elif hasattr(config, 'items'):  # 处理 Section 对象
            loader_config = dict(config)
        else:
            raise ValueError("不支持的配置类型")

        default_loader_config = default_config['Index'] if 'Financial' in default_config else {}

        # 安全获取配置值
        save_path = loader_config.get('save_path', default_loader_config.get('save_path', 'data/index'))  # 修改默认路径

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

    def load_data(
            self,
            index_code: str,
            start_date: Union[str, datetime],
            end_date: Union[str, datetime],
            adjust: str = "hfq"
    ) -> pd.DataFrame:
        """
        加载指数历史数据

        Args:
            index_code: 指数代码 (hs300/sz50/csi500)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            adjust: 复权类型 (qfq/hfq)

        Returns:
            包含OHLC等字段的历史数据

        Raises:
            ValueError: 如果 start_date 大于 end_date
        """
        if index_code not in self.INDEX_MAPPING:
            raise ValueError(f"Invalid index code: {index_code}")

        # 校验 start_date 和 end_date 的大小关系
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if start_date > end_date:
            raise ValueError("开始日期不能大于结束日期")

        file_path = self._get_file_path(index_code, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        for attempt in range(self.max_retries + 1):
            try:
                # 尝试加载缓存数据
                if self._is_cache_valid(file_path):
                    return pd.read_csv(file_path, encoding='utf-8', parse_dates=["date"], index_col="date")

                # 获取原始数据
                index_symbol = self.INDEX_MAPPING.get(index_code, index_code)
                raw_df = self._fetch_raw_data(index_symbol, start_date.strftime("%Y-%m-%d"),
                                              end_date.strftime("%Y-%m-%d"))

                # 检查原始数据是否为空或None
                if raw_df is None or raw_df.empty:
                    raise DataLoaderError("API返回的数据为空")

                processed_df = self._process_raw_data(raw_df)

                # 合并缓存数据和新数据
                if file_path.exists():
                    cached_data = pd.read_csv(file_path, encoding='utf-8', parse_dates=["date"], index_col="date")
                    processed_df = self._merge_with_cache(file_path, processed_df)

                # 确保处理后的数据非空
                if processed_df.empty:
                    raise DataLoaderError("处理后的指数数据为空")

                # 保存数据
                self._save_data(processed_df, file_path)

                return processed_df

            except DataLoaderError as e:
                # 检查是否为API返回空数据错误，并允许重试
                if "API返回的数据为空" in str(e) and attempt < self.max_retries:
                    logger.warning("第%s次重试...", attempt + 1)
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise DataLoaderError(f"加载指数数据失败: {str(e)}") from e
            except (ConnectionError, TimeoutError) as e:
                if attempt >= self.max_retries:
                    logger.error("超过最大重试次数(%s)", self.max_retries)
                    raise DataLoaderError(f"超过最大重试次数({self.max_retries})") from e
                logger.warning("第%s次重试...", attempt + 1)
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"加载指数数据失败: {str(e)}")
                raise DataLoaderError(f"加载指数数据失败: {str(e)}") from e
        raise DataLoaderError("无法加载数据")

    def _get_file_path(self, index_code: str, start_date: str, end_date: str) -> Path:
        """生成文件路径"""
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        return self.save_path / f"{index_code}_{start}_{end}.csv"

    def _is_cache_valid(self, file_path: Path) -> bool:
        """检查缓存是否有效"""
        if not file_path.exists():
            return False

        file_mtime = os.path.getmtime(file_path)
        if (time.time() - file_mtime) > self.cache_days * 86400:
            return False

        try:
            # 检查缓存文件是否包含所有必要列
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                headers = first_line.split(',')
                required_columns = {'date', 'open', 'high', 'low', 'close', 'volume'}
                if not required_columns.issubset(headers):
                    logger.error("缓存文件缺少必要列")
                    return False

            # 尝试读取缓存数据并验证
            cached_df = pd.read_csv(file_path, encoding='utf-8', parse_dates=["date"], index_col="date")
            if cached_df.empty:
                logger.warning("缓存文件数据为空")
                return False
            return True
        except Exception as e:
            logger.error(f"缓存文件无效: {str(e)}")
            return False

    def _fetch_raw_data(self, index_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取原始指数数据"""
        for attempt in range(self.max_retries):
            try:
                raw_data = ak.index_zh_a_hist(symbol=index_symbol, period="daily", start_date=start_date,
                                              end_date=end_date)
                if raw_data is None or raw_data.empty:
                    raise DataLoaderError("API返回的数据为空或不存在")
                print(f"原始数据列名: {raw_data.columns.tolist()}")  # 添加调试信息
                return raw_data
            except (RequestException, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise DataLoaderError(f"超过最大重试次数({self.max_retries})") from e
                logger.warning(f"请求失败，第{attempt + 1}次重试...")
                time.sleep(2 ** attempt)
        raise DataLoaderError("无法获取数据")

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

        # 打印缺失列供调试
        logger.debug(f"缺失的列: {missing_cols}")

        if missing_cols:
            logger.error(f"原始数据缺少必要列: {missing_cols}")
            raise DataLoaderError(f"原始数据缺少必要列: {missing_cols}")

        # 将 'date' 列转换为 datetime 类型
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            logger.error(f"日期格式解析失败: {str(e)}")
            raise DataLoaderError("日期格式解析失败") from e

        # 按日聚合数据
        df.set_index("date", inplace=True)
        daily_data = df.resample("D").agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }).dropna()

        return daily_data

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

    def _merge_with_cache(self, file_path: Path, new_data: pd.DataFrame) -> pd.DataFrame:
        """将新获取的数据与缓存数据合并"""
        if not file_path.exists():
            return new_data  # 如果缓存文件不存在，直接返回新数据

        try:
            cached_data = pd.read_csv(file_path, encoding='utf-8', parse_dates=["date"], index_col="date")
            if not {'open', 'high', 'low', 'close', 'volume'}.issubset(cached_data.columns):
                logger.error("缓存数据缺少必要列")
                return new_data
        except Exception as e:
            logger.error(f"读取缓存文件失败: {str(e)}")
            return new_data  # 如果读取缓存失败，直接返回新数据

        # 确保新数据的索引是 datetime 类型
        new_data.index = pd.to_datetime(new_data.index)

        # 合并缓存数据和新数据
        combined = pd.concat([cached_data, new_data]).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]

        return combined

    def _handle_exception(self, e: Exception, stage: str):
        error_msg = f"{stage}阶段出错: {str(e)}"
        logger.error(error_msg)
        raise DataLoaderError(str(e)) from e

    def normalize_data(self, data, scaler_path=None, inverse=False):
        """
        数据标准化处理

        Args:
            data (pd.DataFrame): 输入数据框
            scaler_path (Path, optional): 标准化器存储路径
            inverse (bool, optional): 是否进行反标准化，默认为False

        Returns:
            pd.DataFrame: 标准化后的数据框
        """
        from sklearn.preprocessing import StandardScaler

        try:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("输入数据必须是 pandas DataFrame")

            if data.empty:
                raise ValueError("输入数据为空")

            # 检查必要列是否存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                raise DataLoaderError(f"数据缺少必要列: {missing_cols}")

            # 强制列顺序和名称与 required_columns 完全一致
            data = data[required_columns].copy()
            data.columns = required_columns

            # 提取数值进行标准化
            values = data.values

            if scaler_path and Path(scaler_path).exists():
                scaler = joblib.load(scaler_path)
            else:
                scaler = StandardScaler()
                scaler.fit(values)  # 使用数组进行训练
                if scaler_path:
                    joblib.dump(scaler, scaler_path)

            if inverse:
                normalized_values = scaler.inverse_transform(values)  # 反标准化
            else:
                normalized_values = scaler.transform(values)  # 标准化

            return pd.DataFrame(normalized_values, columns=data.columns, index=data.index)
        except Exception as e:
            logger.error(f"数据标准化失败: {str(e)}")
            raise DataLoaderError(f"数据标准化失败: {str(e)}") from e

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据
        
        返回:
            包含加载器元数据的字典
        """
        return {
            "loader_type": "IndexDataLoader",
            "cache_days": self.cache_days,
            "max_retries": self.max_retries,
            "supported_indices": list(self.INDEX_MAPPING.keys()),
            "thread_pool": self.thread_pool is not None
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
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        return not data.empty and required_columns.issubset(data.columns)

    def _save_data(self, df: pd.DataFrame, file_path: Path) -> bool:
        """保存数据到缓存文件"""
        try:
            # 确保存储路径存在
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # 创建保存副本
            save_df = df.copy()

            # 如果索引是日期类型，重置索引
            if isinstance(save_df.index, pd.DatetimeIndex):
                save_df.reset_index(inplace=True)
                save_df.rename(columns={"index": "date"}, inplace=True)

            # 强制列名格式化，去除多余空格并转换为小写
            save_df.columns = save_df.columns.str.strip().str.lower()

            # 确保包含所有必要列
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not set(required_columns).issubset(save_df.columns):
                logger.error(f"数据缺少必要列，现有列: {save_df.columns.tolist()}")
                return False

            # 确保日期列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(save_df['date']):
                try:
                    save_df['date'] = pd.to_datetime(save_df['date'])
                except Exception as e:
                    logger.error(f"日期格式转换失败: {str(e)}")
                    return False

            # 保存数据
            save_df.to_csv(file_path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            return False
