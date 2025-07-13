"""
数据验证器
"""
from typing import Dict, Any, List, Optional, Callable
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .interfaces import IDataModel
# 避免循环导入，使用类型注解
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .data_manager import DataModel

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """数据验证错误"""
    pass


class ValidationResult:
    """验证结果"""
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []


class QualityReport:
    """质量报告"""
    def __init__(self, score: float, issues: List[str] = None, metrics: Dict[str, Any] = None):
        self.score = score
        self.issues = issues or []
        self.metrics = metrics or {}


class OutlierReport:
    """异常值报告"""
    def __init__(self, outlier_count: int, outlier_indices: List[int] = None, threshold: float = None):
        self.outlier_count = outlier_count
        self.outlier_indices = outlier_indices or []
        self.threshold = threshold


class ConsistencyReport:
    """一致性报告"""
    def __init__(self, is_consistent: bool, inconsistencies: List[str] = None):
        self.is_consistent = is_consistent
        self.inconsistencies = inconsistencies or []


class DataValidator:
    """
    数据验证器，负责验证数据模型的有效性
    """
    def __init__(self):
        """初始化数据验证器"""
        self._schema_registry: Dict[str, Dict[str, Any]] = {}
        self._custom_rules: List[Callable] = []
        logger.info("DataValidator initialized")

    def validate_data(self, data: pd.DataFrame) -> ValidationResult:
        """
        验证数据的基本有效性

        Args:
            data: 要验证的数据框

        Returns:
            ValidationResult: 验证结果
        """
        errors = []
        warnings = []

        if data is None:
            errors.append("数据为空")
            return ValidationResult(False, errors, warnings)

        if not isinstance(data, pd.DataFrame):
            errors.append("数据不是DataFrame类型")
            return ValidationResult(False, errors, warnings)

        if data.empty:
            warnings.append("数据框为空")

        # 检查是否有重复行
        if data.duplicated().any():
            warnings.append("数据包含重复行")

        # 检查是否有空值
        null_counts = data.isnull().sum()
        if null_counts.any():
            for col, count in null_counts[null_counts > 0].items():
                warnings.append(f"列 '{col}' 包含 {count} 个空值")

        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_quality(self, data: pd.DataFrame) -> QualityReport:
        """
        验证数据质量

        Args:
            data: 要验证的数据框

        Returns:
            QualityReport: 质量报告
        """
        issues = []
        metrics = {}

        if data is None or data.empty:
            return QualityReport(0.0, ["数据为空"], metrics)

        # 计算质量指标
        total_rows = len(data)
        total_cells = total_rows * len(data.columns)

        # 空值率
        null_count = data.isnull().sum().sum()
        null_rate = null_count / total_cells if total_cells > 0 else 0
        metrics['null_rate'] = null_rate

        # 重复行率
        duplicate_count = data.duplicated().sum()
        duplicate_rate = duplicate_count / total_rows if total_rows > 0 else 0
        metrics['duplicate_rate'] = duplicate_rate

        # 数据完整性评分
        completeness_score = 1 - null_rate
        uniqueness_score = 1 - duplicate_rate
        quality_score = (completeness_score + uniqueness_score) / 2

        # 记录问题
        if null_rate > 0.1:
            issues.append(f"空值率过高: {null_rate:.2%}")
        if duplicate_rate > 0.05:
            issues.append(f"重复行率过高: {duplicate_rate:.2%}")

        return QualityReport(quality_score, issues, metrics)

    def validate_data_model(self, model: 'DataModel') -> ValidationResult:
        """
        验证数据模型

        Args:
            model: 数据模型实例

        Returns:
            ValidationResult: 验证结果
        """
        errors = []
        warnings = []

        if model is None:
            errors.append("数据模型为空")
            return ValidationResult(False, errors, warnings)

        # 验证数据模型的基本属性
        if not hasattr(model, 'data'):
            errors.append("数据模型缺少data属性")
        elif model.data is None or model.data.empty:
            warnings.append("数据模型的数据为空")

        if not hasattr(model, 'frequency'):
            errors.append("数据模型缺少frequency属性")

        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_date_range(self, data: pd.DataFrame, date_col: str, start_date: str, end_date: str) -> bool:
        """
        验证日期范围

        Args:
            data: 数据框
            date_col: 日期列名
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            bool: 日期范围是否有效
        """
        if date_col not in data.columns:
            raise ValidationError(f"日期列 '{date_col}' 不存在")

        try:
            dates = pd.to_datetime(data[date_col])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # 检查日期范围
            if dates.min() < start_dt or dates.max() > end_dt:
                return False

            return True
        except Exception as e:
            raise ValidationError(f"日期验证失败: {e}")

    def validate_numeric_columns(self, data: pd.DataFrame, columns: List[str]) -> bool:
        """
        验证数值列

        Args:
            data: 数据框
            columns: 数值列名列表

        Returns:
            bool: 数值列是否有效
        """
        for col in columns:
            if col not in data.columns:
                raise ValidationError(f"列 '{col}' 不存在")

            # 检查是否为数值类型
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValidationError(f"列 '{col}' 不是数值类型")

            # 检查是否有无穷值
            if np.isinf(data[col]).any():
                raise ValidationError(f"列 '{col}' 包含无穷值")

        return True

    def validate_no_missing_values(self, data: pd.DataFrame) -> bool:
        """
        验证没有缺失值

        Args:
            data: 数据框

        Returns:
            bool: 是否没有缺失值
        """
        return not data.isnull().any().any()

    def validate_no_duplicates(self, data: pd.DataFrame) -> bool:
        """
        验证没有重复值

        Args:
            data: 数据框

        Returns:
            bool: 是否没有重复值
        """
        return not data.duplicated().any()

    def validate_outliers(self, data: pd.DataFrame, column: str, method: str = 'iqr') -> OutlierReport:
        """
        验证异常值

        Args:
            data: 数据框
            column: 列名
            method: 异常值检测方法 ('iqr' 或 'zscore')

        Returns:
            OutlierReport: 异常值报告
        """
        if column not in data.columns:
            raise ValidationError(f"列 '{column}' 不存在")

        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValidationError(f"列 '{column}' 不是数值类型")

        values = data[column].dropna()

        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = values[(values < lower_bound) | (values > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = values[z_scores > 3]
        else:
            raise ValidationError(f"不支持的异常值检测方法: {method}")

        outlier_indices = data[data[column].isin(outliers)].index.tolist()

        return OutlierReport(
            outlier_count=len(outliers),
            outlier_indices=outlier_indices,
            threshold=upper_bound if method == 'iqr' else 3
        )

    def validate_data_consistency(self, data: pd.DataFrame) -> ConsistencyReport:
        """
        验证数据一致性

        Args:
            data: 数据框

        Returns:
            ConsistencyReport: 一致性报告
        """
        inconsistencies = []

        # 检查OHLC逻辑一致性
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # 检查 high >= low
            invalid_high_low = data[data['high'] < data['low']]
            if not invalid_high_low.empty:
                inconsistencies.append(f"发现 {len(invalid_high_low)} 行 high < low")

            # 检查 high >= open, close
            invalid_high = data[(data['high'] < data['open']) | (data['high'] < data['close'])]
            if not invalid_high.empty:
                inconsistencies.append(f"发现 {len(invalid_high)} 行 high 不是最高价")

            # 检查 low <= open, close
            invalid_low = data[(data['low'] > data['open']) | (data['low'] > data['close'])]
            if not invalid_low.empty:
                inconsistencies.append(f"发现 {len(invalid_low)} 行 low 不是最低价")

        # 检查数值列的非负性
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['volume', 'amount'] and (data[col] < 0).any():
                inconsistencies.append(f"列 '{col}' 包含负值")

        return ConsistencyReport(len(inconsistencies) == 0, inconsistencies)

    def add_custom_rule(self, rule: Callable) -> None:
        """
        添加自定义验证规则

        Args:
            rule: 自定义验证函数
        """
        self._custom_rules.append(rule)
        logger.info("添加自定义验证规则")
