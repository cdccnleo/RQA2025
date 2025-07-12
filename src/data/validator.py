"""
数据验证器
"""
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
from datetime import datetime, timedelta

from .interfaces import IDataModel

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """数据验证错误"""
    pass


class DataValidator:
    """
    数据验证器，负责验证数据模型的有效性
    """
    def __init__(self):
        """初始化数据验证器"""
        self._schema_registry: Dict[str, Dict[str, Any]] = {}
        logger.info("DataValidator initialized")

    def register_schema(self, model_type: str, schema: Dict[str, Any]) -> None:
        """
        注册数据模式

        Args:
            model_type: 数据模型类型
            schema: 数据模式定义
        """
        self._schema_registry[model_type] = schema
        logger.info(f"Registered schema for model type: {model_type}")

    def validate_schema(self, data: IDataModel) -> bool:
        """
        验证数据模式

        Args:
            data: 数据模型实例

        Returns:
            bool: 数据模式是否有效

        Raises:
            ValidationError: 验证失败时抛出
        """
        model_type = data.__class__.__name__
        if model_type not in self._schema_registry:
            raise ValidationError(f"No schema registered for model type: {model_type}")

        schema = self._schema_registry[model_type]

        # 获取数据模型的元数据
        metadata = data.get_metadata()

        # 验证必需字段
        required_fields = schema.get('required_fields', [])
        for field in required_fields:
            if field not in metadata:
                raise ValidationError(f"Missing required field: {field}")

        # 验证字段类型
        field_types = schema.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in metadata:
                actual_value = metadata[field]
                if not isinstance(actual_value, expected_type):
                    raise ValidationError(
                        f"Invalid type for field {field}: "
                        f"expected {expected_type}, got {type(actual_value)}"
                    )

        return True

    def validate_frequency(self, data: IDataModel, expected_freq: str) -> bool:
        """
        验证数据频率

        Args:
            data: 数据模型实例
            expected_freq: 期望的数据频率

        Returns:
            bool: 数据频率是否符合预期

        Raises:
            ValidationError: 验证失败时抛出
        """
        actual_freq = data.get_frequency()
        if actual_freq != expected_freq:
            raise ValidationError(
                f"Frequency mismatch: expected {expected_freq}, got {actual_freq}"
            )
        return True

    def validate_time_series(
        self,
        df: pd.DataFrame,
        date_column: str,
        freq: str,
        allow_gaps: bool = False
    ) -> bool:
        """
        验证时间序列数据的连续性

        Args:
            df: 数据框
            date_column: 日期列名
            freq: 期望的频率
            allow_gaps: 是否允许时间间隔

        Returns:
            bool: 时间序列是否有效

        Raises:
            ValidationError: 验证失败时抛出
        """
        if df.empty:
            raise ValidationError("Empty DataFrame")

        # 确保日期列存在
        if date_column not in df.columns:
            raise ValidationError(f"Date column '{date_column}' not found")

        # 转换日期列为datetime类型
        try:
            dates = pd.to_datetime(df[date_column])
        except Exception as e:
            raise ValidationError(f"Failed to parse dates: {e}")

        # 检查日期排序
        if not dates.is_monotonic_increasing:
            raise ValidationError("Dates are not in ascending order")

        if not allow_gaps:
            # 创建期望的日期范围
            date_range = pd.date_range(
                start=dates.min(),
                end=dates.max(),
                freq=freq
            )

            # 检查是否有缺失的日期
            missing_dates = date_range.difference(dates)
            if not missing_dates.empty:
                raise ValidationError(
                    f"Missing dates in time series: {missing_dates.tolist()}"
                )

        return True

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        rules: Dict[str, Any]
    ) -> List[str]:
        """
        验证数据质量

        Args:
            df: 数据框
            rules: 数据质量规则

        Returns:
            List[str]: 数据质量问题列表
        """
        issues = []

        # 检查空值
        if rules.get('check_nulls', True):
            null_counts = df.isnull().sum()
            columns_with_nulls = null_counts[null_counts > 0]
            if not columns_with_nulls.empty:
                for col, count in columns_with_nulls.items():
                    issues.append(f"Column '{col}' has {count} null values")

        # 检查重复行
        if rules.get('check_duplicates', True):
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                issues.append(f"Found {duplicate_count} duplicate rows")

        # 检查数值范围
        value_ranges = rules.get('value_ranges', {})
        for column, range_spec in value_ranges.items():
            if column in df.columns:
                min_val = range_spec.get('min')
                max_val = range_spec.get('max')

                if min_val is not None:
                    below_min = df[df[column] < min_val].shape[0]
                    if below_min > 0:
                        issues.append(
                            f"Column '{column}' has {below_min} values below {min_val}"
                        )

                if max_val is not None:
                    above_max = df[df[column] > max_val].shape[0]
                    if above_max > 0:
                        issues.append(
                            f"Column '{column}' has {above_max} values above {max_val}"
                        )

        return issues

    def validate_completeness(
        self,
        data: IDataModel,
        required_fields: List[str]
    ) -> bool:
        """
        验证数据完整性

        Args:
            data: 数据模型实例
            required_fields: 必需字段列表

        Returns:
            bool: 数据是否完整

        Raises:
            ValidationError: 验证失败时抛出
        """
        metadata = data.get_metadata()

        # 检查必需字段
        missing_fields = [
            field for field in required_fields
            if field not in metadata
        ]

        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing_fields)}"
            )

        return True

    def validate_ohlc_logic(self, df: pd.DataFrame) -> bool:
        """
        验证OHLC数据逻辑一致性

        Args:
            df: 包含OHLC数据的数据框

        Returns:
            bool: 数据是否逻辑一致

        Raises:
            ValidationError: 验证失败时抛出
        """
        if {'open', 'high', 'low', 'close'}.issubset(df.columns):
            # 最高价必须大于等于开盘价和收盘价中的较大者
            high_valid = df['high'] >= df[['open', 'close']].max(axis=1)

            # 最低价必须小于等于开盘价和收盘价中的较小者
            low_valid = df['low'] <= df[['open', 'close']].min(axis=1)

            # 检查是否存在逻辑错误
            valid = high_valid & low_valid

            if not valid.all():
                error_count = (~valid).sum()
                raise ValidationError(f"OHLC数据逻辑错误的行数: {error_count}")

        return True

    def validate_volume(self, df: pd.DataFrame) -> bool:
        """
        验证成交量数据

        Args:
            df: 包含成交量数据的数据框

        Returns:
            bool: 成交量数据是否有效

        Raises:
            ValidationError: 验证失败时抛出
        """
        if 'volume' not in df.columns:
            raise ValidationError("数据缺少必要的'volume'列")

        # 检查成交量是否包含负值
        negative_volume = df['volume'] < 0
        if negative_volume.any():
            raise ValidationError(f"发现负成交量的行数: {negative_volume.sum()}")

        return True
