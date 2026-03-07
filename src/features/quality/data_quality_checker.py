"""
数据质量检查工具
用于评估特征数据的质量指标
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float  # 数据完整性 (0-1)
    stability: float     # 数据稳定性 (0-1)
    calculation_success: float  # 计算成功率 (0-1)
    time_coverage: float  # 时间跨度覆盖率 (0-1)
    overall_factor: float  # 综合质量因子 (0-1)
    details: Dict  # 详细信息


class DataQualityChecker:
    """数据质量检查器"""
    
    # 质量因子权重配置
    QUALITY_FACTOR_WEIGHTS = {
        'completeness': 0.30,
        'stability': 0.25,
        'calculation_success': 0.25,
        'time_coverage': 0.20
    }
    
    # 阈值配置
    THRESHOLDS = {
        'completeness': 0.95,  # 缺失率 < 5%
        'stability': 0.95,     # 异常值比例 < 5%
        'calculation_success': 0.99,  # 成功率 > 99%
        'time_coverage': 0.90  # 覆盖率 > 90%
    }
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = timedelta(minutes=5)
    
    def check_data_quality(
        self,
        data: pd.Series,
        expected_start_date: Optional[datetime] = None,
        expected_end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> DataQualityMetrics:
        """
        检查数据质量
        
        Args:
            data: 特征数据 (pandas Series)
            expected_start_date: 预期开始日期
            expected_end_date: 预期结束日期
            use_cache: 是否使用缓存
        
        Returns:
            数据质量指标
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(data, expected_start_date, expected_end_date)
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                logger.debug(f"使用缓存的数据质量指标: {cache_key}")
                return cached_result
        
        # 计算各项指标
        completeness = self._calculate_completeness(data)
        stability = self._calculate_stability(data)
        calculation_success = self._calculate_calculation_success(data)
        time_coverage = self._calculate_time_coverage(
            data, expected_start_date, expected_end_date
        )
        
        # 计算综合质量因子
        overall_factor = self._calculate_overall_factor(
            completeness, stability, calculation_success, time_coverage
        )
        
        # 创建质量指标对象
        metrics = DataQualityMetrics(
            completeness=completeness,
            stability=stability,
            calculation_success=calculation_success,
            time_coverage=time_coverage,
            overall_factor=overall_factor,
            details={
                'missing_count': data.isna().sum(),
                'total_count': len(data),
                'outlier_count': self._count_outliers(data),
                'valid_count': data.notna().sum(),
                'checked_at': datetime.now().isoformat()
            }
        )
        
        # 更新缓存
        if use_cache:
            self._cache[cache_key] = (metrics, datetime.now())
        
        logger.info(f"数据质量检查完成: completeness={completeness:.3f}, "
                   f"stability={stability:.3f}, overall={overall_factor:.3f}")
        
        return metrics
    
    def _calculate_completeness(self, data: pd.Series) -> float:
        """
        计算数据完整性
        
        公式: 1 - (缺失值数量 / 总数量)
        """
        if len(data) == 0:
            return 0.0
        
        missing_ratio = data.isna().sum() / len(data)
        
        # 如果缺失率超过阈值，进行惩罚
        threshold = 1 - self.THRESHOLDS['completeness']  # 0.05
        if missing_ratio <= threshold:
            return 1.0
        else:
            # 线性惩罚
            return max(0.0, 1.0 - (missing_ratio - threshold) / (1 - threshold))
    
    def _calculate_stability(self, data: pd.Series) -> float:
        """
        计算数据稳定性
        
        基于异常值比例评估稳定性
        """
        valid_data = data.dropna()
        
        if len(valid_data) == 0:
            return 0.0
        
        # 使用IQR方法检测异常值
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
        outlier_ratio = len(outliers) / len(valid_data)
        
        # 如果异常值比例超过阈值，进行惩罚
        threshold = 1 - self.THRESHOLDS['stability']  # 0.05
        if outlier_ratio <= threshold:
            return 1.0
        else:
            # 线性惩罚
            return max(0.0, 1.0 - (outlier_ratio - threshold) / (1 - threshold))
    
    def _calculate_calculation_success(self, data: pd.Series) -> float:
        """
        计算计算成功率
        
        基于有效值比例评估计算成功率
        """
        if len(data) == 0:
            return 0.0
        
        valid_ratio = data.notna().sum() / len(data)
        
        # 如果成功率低于阈值，进行惩罚
        threshold = self.THRESHOLDS['calculation_success']  # 0.99
        if valid_ratio >= threshold:
            return 1.0
        else:
            # 线性惩罚
            return max(0.0, valid_ratio / threshold)
    
    def _calculate_time_coverage(
        self,
        data: pd.Series,
        expected_start: Optional[datetime],
        expected_end: Optional[datetime]
    ) -> float:
        """
        计算时间跨度覆盖率
        
        评估数据覆盖的时间范围是否符合预期
        """
        if len(data) == 0:
            return 0.0
        
        if expected_start is None or expected_end is None:
            # 如果没有预期时间范围，假设覆盖率100%
            return 1.0
        
        if not isinstance(data.index, pd.DatetimeIndex):
            # 如果索引不是时间类型，无法计算时间覆盖率
            return 1.0
        
        # 获取实际数据的时间范围
        actual_start = data.index.min()
        actual_end = data.index.max()
        
        # 计算预期时间跨度（天数）
        expected_days = (expected_end - expected_start).days
        if expected_days <= 0:
            return 1.0
        
        # 计算实际覆盖的时间跨度
        actual_start = max(actual_start, expected_start)
        actual_end = min(actual_end, expected_end)
        actual_days = max(0, (actual_end - actual_start).days)
        
        coverage = actual_days / expected_days
        
        # 如果覆盖率低于阈值，进行惩罚
        threshold = self.THRESHOLDS['time_coverage']  # 0.90
        if coverage >= threshold:
            return 1.0
        else:
            # 线性惩罚
            return max(0.0, coverage / threshold)
    
    def _calculate_overall_factor(
        self,
        completeness: float,
        stability: float,
        calculation_success: float,
        time_coverage: float
    ) -> float:
        """
        计算综合质量因子
        
        使用加权平均计算综合质量因子
        """
        weights = self.QUALITY_FACTOR_WEIGHTS
        
        overall = (
            completeness * weights['completeness'] +
            stability * weights['stability'] +
            calculation_success * weights['calculation_success'] +
            time_coverage * weights['time_coverage']
        )
        
        return round(overall, 3)
    
    def _count_outliers(self, data: pd.Series) -> int:
        """统计异常值数量"""
        valid_data = data.dropna()
        
        if len(valid_data) == 0:
            return 0
        
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
        return len(outliers)
    
    def _generate_cache_key(
        self,
        data: pd.Series,
        expected_start: Optional[datetime],
        expected_end: Optional[datetime]
    ) -> str:
        """生成缓存键"""
        # 使用数据的哈希值作为缓存键的一部分
        data_hash = hash(data.values.tobytes()) if len(data) > 0 else 0
        return f"{data_hash}_{expected_start}_{expected_end}"
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.info("数据质量检查缓存已清除")


# 全局数据质量检查器实例
_quality_checker: Optional[DataQualityChecker] = None


def get_quality_checker() -> DataQualityChecker:
    """获取全局数据质量检查器实例"""
    global _quality_checker
    if _quality_checker is None:
        _quality_checker = DataQualityChecker()
    return _quality_checker


def check_feature_quality(
    data: pd.Series,
    expected_start_date: Optional[datetime] = None,
    expected_end_date: Optional[datetime] = None
) -> DataQualityMetrics:
    """
    检查特征数据质量的便捷函数
    
    Args:
        data: 特征数据
        expected_start_date: 预期开始日期
        expected_end_date: 预期结束日期
    
    Returns:
        数据质量指标
    """
    checker = get_quality_checker()
    return checker.check_data_quality(data, expected_start_date, expected_end_date)
