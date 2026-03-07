#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能数据预处理流水线

功能：
- 异常值自动检测和修复
- 缺失值智能填充
- 数据标准化和归一化
- 预处理质量评估

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """异常值检测方法"""
    ZSCORE = "zscore"           # Z-score方法
    IQR = "iqr"                 # 四分位距方法
    ISOLATION_FOREST = "isolation_forest"  # 孤立森林
    MAD = "mad"                 # 中位数绝对偏差


class ImputationMethod(Enum):
    """缺失值填充方法"""
    MEAN = "mean"               # 均值填充
    MEDIAN = "median"           # 中位数填充
    MODE = "mode"               # 众数填充
    FORWARD_FILL = "ffill"      # 前向填充
    BACKWARD_FILL = "bfill"     # 后向填充
    INTERPOLATION = "interpolate"  # 插值填充
    KNN = "knn"                 # K近邻填充
    ITERATIVE = "iterative"     # 迭代填充


class ScalingMethod(Enum):
    """数据缩放方法"""
    STANDARD = "standard"       # 标准化（Z-score）
    MINMAX = "minmax"           # 归一化（Min-Max）
    ROBUST = "robust"           # 稳健缩放
    LOG = "log"                 # 对数变换


@dataclass
class PreprocessingResult:
    """预处理结果"""
    df: pd.DataFrame
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    outliers_detected: int
    outliers_replaced: int
    missing_values_before: int
    missing_values_after: int
    columns_processed: List[str]
    processing_steps: List[str]
    quality_score: float
    timestamp: datetime


@dataclass
class ColumnStats:
    """列统计信息"""
    column: str
    dtype: str
    missing_count: int
    missing_percent: float
    outlier_count: int
    outlier_percent: float
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


class IntelligentPreprocessingPipeline:
    """
    智能数据预处理流水线
    
    提供全自动的数据预处理功能：
    - 智能异常值检测和修复
    - 智能缺失值填充
    - 数据标准化和归一化
    - 预处理质量评估
    """
    
    def __init__(
        self,
        outlier_method: OutlierMethod = OutlierMethod.IQR,
        outlier_threshold: float = 3.0,
        imputation_method: ImputationMethod = ImputationMethod.ITERATIVE,
        scaling_method: ScalingMethod = ScalingMethod.STANDARD
    ):
        """
        初始化预处理流水线
        
        Args:
            outlier_method: 异常值检测方法
            outlier_threshold: 异常值阈值
            imputation_method: 缺失值填充方法
            scaling_method: 数据缩放方法
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.imputation_method = imputation_method
        self.scaling_method = scaling_method
        
        # 统计信息
        self.column_stats: Dict[str, ColumnStats] = {}
        self.processing_history: List[Dict] = []
        
        logger.info(f"智能预处理流水线初始化完成，异常值方法: {outlier_method.value}")
    
    async def process(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        auto_detect: bool = True
    ) -> PreprocessingResult:
        """
        执行完整的数据预处理流程
        
        Args:
            df: 原始DataFrame
            numeric_columns: 数值列列表（可选）
            auto_detect: 是否自动检测数值列
            
        Returns:
            预处理结果
        """
        start_time = datetime.now()
        original_shape = df.shape
        steps = []
        
        # 复制数据
        processed_df = df.copy()
        
        # 1. 自动检测数值列
        if auto_detect or numeric_columns is None:
            numeric_columns = self._detect_numeric_columns(processed_df)
        
        logger.info(f"检测到 {len(numeric_columns)} 个数值列: {numeric_columns}")
        
        # 2. 计算初始统计信息
        missing_before = processed_df.isnull().sum().sum()
        self._calculate_column_stats(processed_df, numeric_columns)
        
        # 3. 异常值检测和修复
        outliers_detected = 0
        outliers_replaced = 0
        
        for col in numeric_columns:
            detected, replaced = await self._handle_outliers(processed_df, col)
            outliers_detected += detected
            outliers_replaced += replaced
        
        if outliers_detected > 0:
            steps.append(f"异常值处理: 检测 {outliers_detected} 个，修复 {outliers_replaced} 个")
        
        # 4. 缺失值填充
        missing_after = 0
        if missing_before > 0:
            for col in numeric_columns:
                filled = await self._impute_missing_values(processed_df, col)
                if filled > 0:
                    steps.append(f"列 {col}: 填充 {filled} 个缺失值")
            
            missing_after = processed_df.isnull().sum().sum()
        
        # 5. 数据标准化/归一化
        for col in numeric_columns:
            await self._scale_column(processed_df, col)
        
        steps.append(f"数据缩放: {self.scaling_method.value}")
        
        # 6. 计算质量分数
        quality_score = self._calculate_quality_score(
            processed_df, numeric_columns, outliers_detected, missing_before, missing_after
        )
        
        # 7. 记录处理历史
        final_shape = processed_df.shape
        
        result = PreprocessingResult(
            df=processed_df,
            original_shape=original_shape,
            final_shape=final_shape,
            outliers_detected=outliers_detected,
            outliers_replaced=outliers_replaced,
            missing_values_before=missing_before,
            missing_values_after=missing_after,
            columns_processed=numeric_columns,
            processing_steps=steps,
            quality_score=quality_score,
            timestamp=datetime.now()
        )
        
        self.processing_history.append({
            'timestamp': start_time,
            'original_shape': original_shape,
            'final_shape': final_shape,
            'quality_score': quality_score,
            'steps': steps
        })
        
        logger.info(f"数据预处理完成，质量分数: {quality_score:.2f}")
        
        return result
    
    def _detect_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        自动检测数值列
        
        Args:
            df: DataFrame
            
        Returns:
            数值列列表
        """
        numeric_columns = []
        
        for col in df.columns:
            # 检查是否为数值类型
            if pd.api.types.is_numeric_dtype(df[col]):
                # 排除ID列和时间戳列
                if not any(keyword in col.lower() for keyword in ['id', 'code', 'timestamp', 'date', 'time']):
                    numeric_columns.append(col)
        
        return numeric_columns
    
    def _calculate_column_stats(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ):
        """
        计算列统计信息
        
        Args:
            df: DataFrame
            columns: 列列表
        """
        for col in columns:
            if col not in df.columns:
                continue
            
            series = df[col]
            missing_count = series.isnull().sum()
            missing_percent = missing_count / len(series) * 100 if len(series) > 0 else 0
            
            # 检测异常值
            outlier_count = self._count_outliers(series)
            outlier_percent = outlier_count / len(series) * 100 if len(series) > 0 else 0
            
            # 计算基本统计量
            mean = series.mean() if pd.api.types.is_numeric_dtype(series) else None
            std = series.std() if pd.api.types.is_numeric_dtype(series) else None
            min_val = series.min() if pd.api.types.is_numeric_dtype(series) else None
            max_val = series.max() if pd.api.types.is_numeric_dtype(series) else None
            
            self.column_stats[col] = ColumnStats(
                column=col,
                dtype=str(series.dtype),
                missing_count=missing_count,
                missing_percent=round(missing_percent, 2),
                outlier_count=outlier_count,
                outlier_percent=round(outlier_percent, 2),
                mean=mean,
                std=std,
                min=min_val,
                max=max_val
            )
    
    def _count_outliers(self, series: pd.Series) -> int:
        """
        计算异常值数量
        
        Args:
            series: 数据序列
            
        Returns:
            异常值数量
        """
        if not pd.api.types.is_numeric_dtype(series):
            return 0
        
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return 0
        
        if self.outlier_method == OutlierMethod.ZSCORE:
            z_scores = np.abs(stats.zscore(series_clean))
            return (z_scores > self.outlier_threshold).sum()
        
        elif self.outlier_method == OutlierMethod.IQR:
            Q1 = series_clean.quantile(0.25)
            Q3 = series_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((series_clean < lower_bound) | (series_clean > upper_bound)).sum()
        
        elif self.outlier_method == OutlierMethod.MAD:
            median = series_clean.median()
            mad = np.median(np.abs(series_clean - median))
            modified_z_scores = 0.6745 * (series_clean - median) / mad
            return (np.abs(modified_z_scores) > self.outlier_threshold).sum()
        
        return 0
    
    async def _handle_outliers(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Tuple[int, int]:
        """
        处理异常值
        
        Args:
            df: DataFrame
            column: 列名
            
        Returns:
            (检测到的异常值数量, 修复的异常值数量)
        """
        if column not in df.columns:
            return 0, 0
        
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            return 0, 0
        
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return 0, 0
        
        # 检测异常值
        outlier_mask = pd.Series(False, index=series.index)
        
        if self.outlier_method == OutlierMethod.ZSCORE:
            z_scores = np.abs(stats.zscore(series_clean))
            outlier_indices = series_clean[z_scores > self.outlier_threshold].index
            outlier_mask.loc[outlier_indices] = True
        
        elif self.outlier_method == OutlierMethod.IQR:
            Q1 = series_clean.quantile(0.25)
            Q3 = series_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (series < lower_bound) | (series > upper_bound)
        
        elif self.outlier_method == OutlierMethod.MAD:
            median = series_clean.median()
            mad = np.median(np.abs(series_clean - median))
            modified_z_scores = 0.6745 * (series_clean - median) / mad
            outlier_indices = series_clean[np.abs(modified_z_scores) > self.outlier_threshold].index
            outlier_mask.loc[outlier_indices] = True
        
        detected = outlier_mask.sum()
        
        if detected == 0:
            return 0, 0
        
        # 修复异常值（使用边界值替换）
        if self.outlier_method == OutlierMethod.IQR:
            Q1 = series_clean.quantile(0.25)
            Q3 = series_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df.loc[outlier_mask & (series < lower_bound), column] = lower_bound
            df.loc[outlier_mask & (series > upper_bound), column] = upper_bound
        else:
            # 使用中位数替换
            median = series_clean.median()
            df.loc[outlier_mask, column] = median
        
        return detected, detected
    
    async def _impute_missing_values(
        self,
        df: pd.DataFrame,
        column: str
    ) -> int:
        """
        填充缺失值
        
        Args:
            df: DataFrame
            column: 列名
            
        Returns:
            填充的缺失值数量
        """
        if column not in df.columns:
            return 0
        
        series = df[column]
        missing_mask = series.isnull()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            return 0
        
        if self.imputation_method == ImputationMethod.MEAN:
            fill_value = series.mean()
            df.loc[missing_mask, column] = fill_value
        
        elif self.imputation_method == ImputationMethod.MEDIAN:
            fill_value = series.median()
            df.loc[missing_mask, column] = fill_value
        
        elif self.imputation_method == ImputationMethod.MODE:
            fill_value = series.mode()[0] if not series.mode().empty else series.dropna().iloc[0]
            df.loc[missing_mask, column] = fill_value
        
        elif self.imputation_method == ImputationMethod.FORWARD_FILL:
            df[column] = series.fillna(method='ffill')
        
        elif self.imputation_method == ImputationMethod.BACKWARD_FILL:
            df[column] = series.fillna(method='bfill')
        
        elif self.imputation_method == ImputationMethod.INTERPOLATION:
            df[column] = series.interpolate(method='linear')
            # 处理边界缺失值
            df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
        
        elif self.imputation_method == ImputationMethod.KNN:
            try:
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                df[[column]] = imputer.fit_transform(df[[column]])
            except ImportError:
                logger.warning("scikit-learn未安装，使用中位数填充")
                df.loc[missing_mask, column] = series.median()
        
        elif self.imputation_method == ImputationMethod.ITERATIVE:
            try:
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                imputer = IterativeImputer(max_iter=10, random_state=42)
                df[[column]] = imputer.fit_transform(df[[column]])
            except ImportError:
                logger.warning("scikit-learn未安装，使用插值填充")
                df[column] = series.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        
        return missing_count
    
    async def _scale_column(
        self,
        df: pd.DataFrame,
        column: str
    ):
        """
        缩放列数据
        
        Args:
            df: DataFrame
            column: 列名
        """
        if column not in df.columns:
            return
        
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            return
        
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return
        
        if self.scaling_method == ScalingMethod.STANDARD:
            # Z-score标准化
            mean = series_clean.mean()
            std = series_clean.std()
            if std > 0:
                df[column] = (series - mean) / std
        
        elif self.scaling_method == ScalingMethod.MINMAX:
            # Min-Max归一化
            min_val = series_clean.min()
            max_val = series_clean.max()
            if max_val > min_val:
                df[column] = (series - min_val) / (max_val - min_val)
        
        elif self.scaling_method == ScalingMethod.ROBUST:
            # 稳健缩放
            median = series_clean.median()
            q1 = series_clean.quantile(0.25)
            q3 = series_clean.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df[column] = (series - median) / iqr
        
        elif self.scaling_method == ScalingMethod.LOG:
            # 对数变换
            min_val = series_clean.min()
            if min_val <= 0:
                # 平移数据使其为正
                series = series - min_val + 1
            df[column] = np.log(series)
    
    def _calculate_quality_score(
        self,
        df: pd.DataFrame,
        columns: List[str],
        outliers_detected: int,
        missing_before: int,
        missing_after: int
    ) -> float:
        """
        计算数据质量分数
        
        Args:
            df: DataFrame
            columns: 处理的列
            outliers_detected: 检测到的异常值数量
            missing_before: 处理前的缺失值数量
            missing_after: 处理后的缺失值数量
            
        Returns:
            质量分数（0-100）
        """
        total_cells = len(df) * len(columns)
        
        if total_cells == 0:
            return 100.0
        
        # 缺失值分数（40%权重）
        missing_score = (1 - missing_after / total_cells) * 40 if total_cells > 0 else 40
        
        # 异常值分数（30%权重）
        outlier_score = (1 - outliers_detected / total_cells) * 30 if total_cells > 0 else 30
        
        # 一致性分数（30%权重）
        consistency_score = 30
        for col in columns:
            if col in df.columns:
                series = df[col]
                # 检查数据类型一致性
                if series.dtype == 'object':
                    consistency_score -= 2
                # 检查数值范围
                if pd.api.types.is_numeric_dtype(series):
                    if series.min() < -1e6 or series.max() > 1e6:
                        consistency_score -= 1
        
        consistency_score = max(0, consistency_score)
        
        total_score = missing_score + outlier_score + consistency_score
        
        return round(min(total_score, 100.0), 2)
    
    def get_column_stats(self) -> Dict[str, ColumnStats]:
        """获取列统计信息"""
        return self.column_stats
    
    def get_processing_history(self) -> List[Dict]:
        """获取处理历史"""
        return self.processing_history
    
    def generate_report(self) -> str:
        """
        生成预处理报告
        
        Returns:
            报告字符串
        """
        if not self.processing_history:
            return "暂无预处理记录"
        
        latest = self.processing_history[-1]
        
        report = f"""
数据预处理报告
================
处理时间: {latest['timestamp']}
原始数据形状: {latest['original_shape']}
处理后数据形状: {latest['final_shape']}
质量分数: {latest['quality_score']:.2f}/100

处理步骤:
"""
        for i, step in enumerate(latest['steps'], 1):
            report += f"  {i}. {step}\n"
        
        report += "\n列统计信息:\n"
        for col, stats in self.column_stats.items():
            report += f"\n  {col}:\n"
            report += f"    数据类型: {stats.dtype}\n"
            report += f"    缺失值: {stats.missing_count} ({stats.missing_percent}%)\n"
            report += f"    异常值: {stats.outlier_count} ({stats.outlier_percent}%)\n"
            if stats.mean is not None:
                report += f"    均值: {stats.mean:.4f}, 标准差: {stats.std:.4f}\n"
                report += f"    范围: [{stats.min:.4f}, {stats.max:.4f}]\n"
        
        return report


# 全局流水线实例
_pipeline_instance: Optional[IntelligentPreprocessingPipeline] = None


def get_preprocessing_pipeline(
    outlier_method: OutlierMethod = OutlierMethod.IQR,
    imputation_method: ImputationMethod = ImputationMethod.ITERATIVE,
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
) -> IntelligentPreprocessingPipeline:
    """
    获取预处理流水线实例（单例模式）
    
    Args:
        outlier_method: 异常值检测方法
        imputation_method: 缺失值填充方法
        scaling_method: 数据缩放方法
        
    Returns:
        IntelligentPreprocessingPipeline实例
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = IntelligentPreprocessingPipeline(
            outlier_method=outlier_method,
            imputation_method=imputation_method,
            scaling_method=scaling_method
        )
    
    return _pipeline_instance
