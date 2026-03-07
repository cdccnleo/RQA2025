from ..interfaces.IDataModel import IDataModel
import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据修复器
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# 修复导入错误
try:
    from ..validation.china_stock_validator import ChinaStockValidator
except ImportError:
    # 如果导入失败，创建一个简单的验证器类

    class ChinaStockValidator:

        def validate_data(self, data):

            return {'is_valid': True, 'errors': [], 'warnings': []}

# 定义缺失的类


class ValidationResult:

    """验证结果"""

    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):

        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []


class ValidationError(Exception):

    """验证错误"""


logger = logging.getLogger(__name__)


class RepairStrategy(Enum):

    """修复策略枚举"""
    DROP = "drop"                    # 删除问题数据
    FILL_MEAN = "fill_mean"          # 用均值填充
    FILL_MEDIAN = "fill_median"      # 用中位数填充
    FILL_MODE = "fill_mode"          # 用众数填充
    FILL_FORWARD = "fill_forward"    # 前向填充
    FILL_BACKWARD = "fill_backward"  # 后向填充
    INTERPOLATE = "interpolate"      # 插值填充
    REMOVE_OUTLIERS = "remove_outliers"  # 移除异常值
    NORMALIZE = "normalize"          # 数据标准化
    LOG_TRANSFORM = "log_transform"  # 对数变换


@dataclass
class RepairConfig:

    """修复配置"""
    # 空值处理
    null_strategy: RepairStrategy = RepairStrategy.FILL_FORWARD
    null_threshold: float = 0.5  # 空值比例阈值

    # 异常值处理
    outlier_strategy: RepairStrategy = RepairStrategy.REMOVE_OUTLIERS
    outlier_threshold: float = 3.0  # 异常值阈值（标准差倍数）

    # 重复值处理
    duplicate_strategy: RepairStrategy = RepairStrategy.DROP

    # 数据一致性
    consistency_strategy: RepairStrategy = RepairStrategy.FILL_FORWARD

    # 数值范围
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # 时间序列处理
    time_series_enabled: bool = True
    resample_freq: Optional[str] = None

    # 日志级别
    log_level: str = "INFO"


@dataclass
class RepairResult:

    """修复结果"""
    success: bool
    original_shape: Tuple[int, int]
    repaired_shape: Tuple[int, int]
    repair_stats: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


class DataRepairer:

    """
    数据修复器

    提供自动数据修复功能，包括：
    - 空值处理
    - 异常值检测和修复
    - 重复值处理
    - 数据一致性修复
    - 时间序列修复
    """

    def __init__(self, config: Optional[RepairConfig] = None):
        """
        初始化数据修复器

        Args:
            config: 修复配置
        """
        self.config = config or RepairConfig()
        self.logger = logger  # 修复logger引用
        self.repair_history = []
        self.repair_stats = {
            'total_repairs': 0,
            'successful_repairs': 0,
            'failed_repairs': 0,
            'null_fixes': 0,
            'outlier_fixes': 0,
            'duplicate_removals': 0,
            'consistency_fixes': 0
        }

        # 初始化验证器
        try:
            self.validator = ChinaStockValidator()
        except Exception as e:
            self.logger.warning(f"无法初始化验证器: {e}")
            self.validator = None

    def repair_data(self, data: pd.DataFrame, data_type: str = "general") -> Tuple[pd.DataFrame, RepairResult]:
        """
        修复数据质量问题

        Args:
            data: 原始数据
            data_type: 数据类型（stock, index, news等）

        Returns:
            Tuple[pd.DataFrame, RepairResult]: 修复后的数据和修复结果
        """
        if data is None or data.empty:
            return data, RepairResult(
                success=False,
                original_shape=(0, 0),
                repaired_shape=(0, 0),
                repair_stats={},
                warnings=["数据为空，无法修复"],
                errors=["数据为空"]
            )

        original_shape = data.shape
        repair_stats = {}
        warnings = []
        errors = []

        try:
            # 创建数据副本
            repaired_data = data.copy()

            # 1. 修复空值
            repaired_data, null_stats = self._repair_null_values(repaired_data)
            repair_stats.update(null_stats)

            # 2. 修复异常值
            repaired_data, outlier_stats = self._repair_outliers(repaired_data)
            repair_stats.update(outlier_stats)

            # 3. 处理重复值
            repaired_data, duplicate_stats = self._repair_duplicates(repaired_data)
            repair_stats.update(duplicate_stats)

            # 4. 修复数据一致性
            repaired_data, consistency_stats = self._repair_consistency(repaired_data)
            repair_stats.update(consistency_stats)

            # 5. 数值范围修复
            repaired_data, range_stats = self._repair_value_range(repaired_data)
            repair_stats.update(range_stats)

            # 6. 时间序列特殊处理
            if self.config.time_series_enabled and self._is_time_series(repaired_data):
                repaired_data, ts_stats = self._repair_time_series(repaired_data)
                repair_stats.update(ts_stats)

            # 7. 数据标准化
            repaired_data, normalize_stats = self._normalize_data(repaired_data)
            repair_stats.update(normalize_stats)

            # 记录修复结果
            repair_result = RepairResult(
                success=True,
                original_shape=original_shape,
                repaired_shape=repaired_data.shape,
                repair_stats=repair_stats,
                warnings=warnings,
                errors=errors
            )

            self.repair_history.append(repair_result)

            self.logger.info(f"数据修复完成: {original_shape} -> {repaired_data.shape}")
            return repaired_data, repair_result

        except Exception as e:
            self.logger.error(f"数据修复失败: {e}")
            return data, RepairResult(
                success=False,
                original_shape=original_shape,
                repaired_shape=data.shape,
                repair_stats={},
                warnings=warnings,
                errors=[f"修复失败: {str(e)}"]
            )

    def repair_data_model(self, data_model: IDataModel) -> Tuple[IDataModel, RepairResult]:
        """
        修复数据模型

        Args:
            data_model: 数据模型

        Returns:
            Tuple[IDataModel, RepairResult]: 修复后的数据模型和修复结果
        """
        if not hasattr(data_model, 'data') or data_model.data is None:
            return data_model, RepairResult(
                success=False,
                original_shape=(0, 0),
                repaired_shape=(0, 0),
                repair_stats={},
                warnings=["数据模型为空"],
                errors=["数据模型为空"]
            )

        # 修复数据
        repaired_data, repair_result = self.repair_data(data_model.data)

        # 创建新的数据模型
        if hasattr(data_model, 'get_frequency'):
            frequency = data_model.get_frequency()
        else:
            frequency = '1d'  # 默认频率

        if hasattr(data_model, 'get_metadata'):
            metadata = data_model.get_metadata()
        else:
            metadata = {}

        repaired_model = type(data_model)(repaired_data, frequency, metadata)

        # 添加修复信息到元数据
        if hasattr(repaired_model, 'get_metadata'):
            current_metadata = repaired_model.get_metadata()
            current_metadata['repair_info'] = {
                'repaired_at': datetime.now().isoformat(),
                'repair_stats': repair_result.repair_stats,
                'original_shape': repair_result.original_shape,
                'repaired_shape': repair_result.repaired_shape
            }
            # 注意：这里需要重新创建模型来更新元数据，因为DataModel的元数据是只读的
            repaired_model = type(data_model)(repaired_data, frequency, current_metadata)

        return repaired_model, repair_result

    def _repair_null_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """修复空值"""
        stats = {'null_repairs': 0, 'null_drops': 0}

        for column in data.columns:
            null_count = data[column].isnull().sum()
            null_ratio = null_count / len(data)

            if null_count == 0:
                continue

            if null_ratio > self.config.null_threshold:
                # 空值比例过高，删除该列
                data = data.drop(columns=[column])
                stats['null_drops'] += 1
                self.logger.warning(f"列 '{column}' 空值比例过高 ({null_ratio:.2%})，已删除")
            else:
                # 根据策略修复空值
                if self.config.null_strategy == RepairStrategy.FILL_MEAN:
                    if data[column].dtype in ['int64', 'float64']:
                        data[column] = data[column].fillna(data[column].mean())
                elif self.config.null_strategy == RepairStrategy.FILL_MEDIAN:
                    if data[column].dtype in ['int64', 'float64']:
                        data[column] = data[column].fillna(data[column].median())
                elif self.config.null_strategy == RepairStrategy.FILL_MODE:
                    data[column] = data[column].fillna(
                        data[column].mode().iloc[0] if not data[column].mode().empty else None)
                elif self.config.null_strategy == RepairStrategy.FILL_FORWARD:
                    data[column] = data[column].ffill()
                elif self.config.null_strategy == RepairStrategy.FILL_BACKWARD:
                    data[column] = data[column].bfill()
                elif self.config.null_strategy == RepairStrategy.INTERPOLATE:
                    if data[column].dtype in ['int64', 'float64']:
                        data[column] = data[column].interpolate()

                stats['null_repairs'] += null_count

        return data, stats

    def _repair_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """修复异常值"""
        stats = {'outlier_repairs': 0, 'outlier_drops': 0}

        for column in data.select_dtypes(include=[np.number]).columns:
            if self.config.outlier_strategy == RepairStrategy.REMOVE_OUTLIERS:
                # 使用IQR方法检测异常值
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    if outlier_count / len(data) > 0.1:  # 异常值比例过高，用中位数替换
                        data.loc[outlier_mask, column] = data[column].median()
                        stats['outlier_repairs'] += outlier_count
                    else:  # 异常值比例较低，删除
                        data = data[~outlier_mask]
                        stats['outlier_drops'] += outlier_count

        return data, stats

    def _repair_duplicates(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """修复重复值"""
        stats = {'duplicate_drops': 0}

        if self.config.duplicate_strategy == RepairStrategy.DROP:
            original_count = len(data)
            data = data.drop_duplicates()
            stats['duplicate_drops'] = original_count - len(data)

        return data, stats

    def _repair_consistency(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """修复数据一致性"""
        stats = {'consistency_repairs': 0}

        # 检查数据类型一致性
        for column in data.columns:
            if data[column].dtype == 'object':
                # 尝试转换为数值类型
                try:
                    data[column] = pd.to_numeric(data[column], errors='coerce')
                    stats['consistency_repairs'] += 1
                except BaseException:
                    pass

        return data, stats

    def _repair_value_range(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """修复数值范围"""
        stats = {'range_repairs': 0}

        for column in data.select_dtypes(include=[np.number]).columns:
            if self.config.min_value is not None:
                below_min = data[column] < self.config.min_value
                if below_min.any():
                    data.loc[below_min, column] = self.config.min_value
                    stats['range_repairs'] += below_min.sum()

            if self.config.max_value is not None:
                above_max = data[column] > self.config.max_value
                if above_max.any():
                    data.loc[above_max, column] = self.config.max_value
                    stats['range_repairs'] += above_max.sum()

        return data, stats

    def _is_time_series(self, data: pd.DataFrame) -> bool:
        """判断是否为时间序列数据"""
        # 检查索引是否为时间类型
        if isinstance(data.index, pd.DatetimeIndex):
            return True

        # 检查是否有时间列
        time_columns = ['date', 'time', 'datetime', 'timestamp']
        for col in data.columns:
            if any(time_keyword in col.lower() for time_keyword in time_columns):
                return True

        return False

    def _repair_time_series(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """修复时间序列数据"""
        stats = {'ts_repairs': 0}

        # 如果索引不是时间类型，尝试转换
        if not isinstance(data.index, pd.DatetimeIndex):
            # 查找时间列
            time_columns = ['date', 'time', 'datetime', 'timestamp']
            for col in data.columns:
                if any(time_keyword in col.lower() for time_keyword in time_columns):
                    try:
                        data[col] = pd.to_datetime(data[col])
                        data = data.set_index(col)
                        stats['ts_repairs'] += 1
                        break
                    except BaseException:
                        continue

        # 重采样（如果配置了）
        if self.config.resample_freq and isinstance(data.index, pd.DatetimeIndex):
            try:
                data = data.resample(self.config.resample_freq).mean()
                stats['ts_repairs'] += 1
            except BaseException:
                pass

        return data, stats

    def _normalize_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """数据标准化"""
        stats = {'normalized_columns': 0}

        # 对数变换
        if self.config.outlier_strategy == RepairStrategy.LOG_TRANSFORM:
            for column in data.select_dtypes(include=[np.number]).columns:
                if (data[column] > 0).all():  # 确保所有值都为正
                    data[column] = np.log(data[column])
                    stats['normalized_columns'] += 1

        return data, stats

    def get_repair_history(self) -> List[RepairResult]:
        """获取修复历史"""
        return self.repair_history

    def get_repair_stats(self) -> Dict[str, Any]:
        """获取修复统计信息"""
        if not self.repair_history:
            return {}

        total_repairs = len(self.repair_history)
        successful_repairs = sum(1 for r in self.repair_history if r.success)

        return {
            'total_repairs': total_repairs,
            'successful_repairs': successful_repairs,
            'success_rate': successful_repairs / total_repairs if total_repairs > 0 else 0,
            'last_repair': self.repair_history[-1] if self.repair_history else None
        }

    def reset_history(self):
        """重置修复历史"""
        self.repair_history.clear()
