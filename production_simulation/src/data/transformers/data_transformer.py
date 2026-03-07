"""数据转换器"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class DataTransformer(ABC):

    """数据转换器基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化数据转换器

        Args:
            config: 转换器配置
        """
        self.config = config or {}

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """转换数据

        Args:
            data: 输入数据

        Returns:
            转换后的数据
        """

    def fit(self, data: Any) -> None:
        """拟合转换器

        Args:
            data: 训练数据
        """

    def fit_transform(self, data: Any) -> Any:
        """拟合并转换数据

        Args:
            data: 输入数据

        Returns:
            转换后的数据
        """
        self.fit(data)
        return self.transform(data)


class DataFrameTransformer(DataTransformer):

    """DataFrame转换器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)
        self.columns_to_drop = self.config.get('columns_to_drop', [])
        self.columns_to_keep = self.config.get('columns_to_keep', None)
        self.fill_method = self.config.get('fill_method', 'ffill')

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换DataFrame

        Args:
            data: 输入DataFrame

        Returns:
            转换后的DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("输入数据必须是pandas DataFrame")

        result = data.copy()

        # 删除指定列
        if self.columns_to_drop:
            result = result.drop(columns=self.columns_to_drop, errors='ignore')

        # 保留指定列
        if self.columns_to_keep:
            result = result[self.columns_to_keep]

        # 处理缺失值
        if self.fill_method == 'ffill':
            result = result.fillna(method='ffill')
        elif self.fill_method == 'bfill':
            result = result.fillna(method='bfill')
        elif self.fill_method == 'interpolate':
            result = result.interpolate()
        else:
            result = result.fillna(0)

        return result


class TimeSeriesTransformer(DataTransformer):

    """时间序列转换器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)
        self.resample_freq = self.config.get('resample_freq', None)
        self.fill_method = self.config.get('fill_method', 'ffill')

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换时间序列数据

        Args:
            data: 输入DataFrame

        Returns:
            转换后的DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("输入数据必须是pandas DataFrame")

        result = data.copy()

        # 设置时间索引
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date'])
            result = result.set_index('date')

        # 重采样
        if self.resample_freq and not result.empty:
            result = result.resample(self.resample_freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

        # 处理缺失值
        if self.fill_method == 'ffill':
            result = result.fillna(method='ffill')
        elif self.fill_method == 'bfill':
            result = result.fillna(method='bfill')
        elif self.fill_method == 'interpolate':
            result = result.interpolate()
        else:
            result = result.fillna(0)

        return result


class FeatureTransformer(DataTransformer):

    """特征转换器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)
        self.normalize = self.config.get('normalize', False)
        self.scale_features = self.config.get('scale_features', [])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换特征数据

        Args:
            data: 输入DataFrame

        Returns:
            转换后的DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("输入数据必须是pandas DataFrame")

        result = data.copy()
        self.normalize = self.config.get('normalize', self.normalize)
        self.scale_features = self.config.get('scale_features', self.scale_features)

        # 标准化数值列
        if self.normalize:
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in result.columns and result[col].std() > 0:
                    result[col] = (result[col] - result[col].mean()) / result[col].std()

        # 缩放指定特征
        if self.scale_features:
            for col in self.scale_features:
                if col in result.columns and result[col].std() > 0:
                    result[col] = (result[col] - result[col].min()) / \
                        (result[col].max() - result[col].min())

        return result


class NormalizationTransformer(DataTransformer):

    """标准化转换器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)
        self.method = self.config.get('method', 'zscore')  # zscore, minmax, robust

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化数据

        Args:
            data: 输入DataFrame

        Returns:
            标准化后的DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("输入数据必须是pandas DataFrame")

        result = data.copy()
        self.method = self.config.get('method', self.method)
        numeric_columns = result.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in result.columns and result[col].std() > 0:
                if self.method == 'zscore':
                    result[col] = (result[col] - result[col].mean()) / result[col].std()
                elif self.method == 'minmax':
                    result[col] = (result[col] - result[col].min()) / \
                        (result[col].max() - result[col].min())
                elif self.method == 'robust':
                    Q1 = result[col].quantile(0.25)
                    Q3 = result[col].quantile(0.75)
                    IQR = Q3 - Q1
                    result[col] = (result[col] - result[col].median()) / IQR

        return result


class MissingValueTransformer(DataTransformer):

    """缺失值处理转换器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)
        self.method = self.config.get('method', 'ffill')  # ffill, bfill, interpolate, drop, fill

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值

        Args:
            data: 输入DataFrame

        Returns:
            处理后的DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("输入数据必须是pandas DataFrame")

        result = data.copy()

        strategy = self.config.get('strategy', self.method)
        self.method = strategy  # 持久化最新策略，兼容旧逻辑

        if strategy == 'ffill':
            result = result.fillna(method='ffill')
        elif strategy == 'bfill':
            result = result.fillna(method='bfill')
        elif strategy == 'interpolate':
            result = result.interpolate()
        elif strategy == 'drop':
            result = result.dropna()
        elif strategy == 'fill':
            fill_value = self.config.get('fill_value', 0)
            result = result.fillna(fill_value)
        elif strategy == 'mean':
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                mean_value = result[col].mean()
                result[col] = result[col].fillna(mean_value)
        elif strategy == 'median':
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                median_value = result[col].median()
                result[col] = result[col].fillna(median_value)
        elif strategy == 'constant':
            fill_value = self.config.get('fill_value', 0)
            result = result.fillna(fill_value)
        else:
            # 默认回退到向前填充
            result = result.fillna(method='ffill')

        return result


class DateColumnTransformer(DataTransformer):

    """日期列转换器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)
        self.date_column = self.config.get('date_column', 'date')
        self.format = self.config.get('format', None)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换日期列

        Args:
            data: 输入DataFrame

        Returns:
            转换后的DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("输入数据必须是pandas DataFrame")

        result = data.copy()

        date_columns = self.config.get('date_columns')
        if not date_columns:
            date_columns = [self.date_column]
        elif isinstance(date_columns, str):
            date_columns = [date_columns]

        date_formats = self.config.get('date_formats', None)
        timezone = self.config.get('timezone')
        extract_features: List[str] = self.config.get('extract_features', [])

        for idx, column in enumerate(date_columns):
            if column not in result.columns:
                continue

            column_format = None
            if isinstance(date_formats, dict):
                column_format = date_formats.get(column, self.format)
            elif isinstance(date_formats, (list, tuple)):
                if idx < len(date_formats):
                    column_format = date_formats[idx]
                else:
                    column_format = self.format
            else:
                column_format = self.format

            try:
                converted = pd.to_datetime(result[column], format=column_format, errors='coerce')
            except Exception:
                converted = pd.to_datetime(result[column], errors='coerce')

            if timezone:
                try:
                    if converted.dt.tz is None:
                        converted = converted.dt.tz_localize(timezone)
                    else:
                        converted = converted.dt.tz_convert(timezone)
                except TypeError:
                    # 如果包含NaT导致 tz_localize 失败，则忽略时区处理
                    pass

            result[column] = converted

            for feature in extract_features:
                feature_column = f"{column}_{feature}"
                if feature == 'year':
                    result[feature_column] = result[column].dt.year
                elif feature == 'month':
                    result[feature_column] = result[column].dt.month
                elif feature == 'day':
                    result[feature_column] = result[column].dt.day
                elif feature == 'weekday':
                    result[feature_column] = result[column].dt.weekday
                elif feature == 'hour':
                    result[feature_column] = result[column].dt.hour
                elif feature == 'minute':
                    result[feature_column] = result[column].dt.minute
                elif feature == 'second':
                    result[feature_column] = result[column].dt.second

        return result
