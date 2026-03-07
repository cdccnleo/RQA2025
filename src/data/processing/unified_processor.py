"""
统一数据处理器 - 重构版本
"""
# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

from typing import Dict, Any
from src.infrastructure.logging import get_infrastructure_logger
import pandas as pd
import numpy as np
from datetime import datetime

from ..interfaces import IDataProcessor, IDataModel


logger = get_infrastructure_logger('__name__')


class UnifiedDataProcessor(IDataProcessor):

    """
    统一数据处理器，负责数据的清洗、转换和标准化
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据处理器

        Args:
            config: 处理器配置
        """
        self.config = config or {}
        self.processing_steps = []
        self.processing_info = {
            'processor_type': self.__class__.__name__,
            'created_at': datetime.now().isoformat(),
            'steps': []
        }
        logger.info("UnifiedDataProcessor initialized")

    def process(self, data: IDataModel, **kwargs) -> IDataModel:
        """
        处理数据

        Args:
            data: 输入数据模型
            **kwargs: 处理参数

        Returns:
            IDataModel: 处理后的数据模型
        """
        if data is None or not data.validate():
            raise ValueError("输入数据无效")

        # 记录处理开始
        self.processing_info['steps'].append({
            'step': 'start',
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.data.shape if data.data is not None else None
        })

        # 执行数据处理步骤
        processed_data = self._execute_processing_pipeline(data, **kwargs)

        # 记录处理完成
        self.processing_info['steps'].append({
            'step': 'complete',
            'timestamp': datetime.now().isoformat(),
            'data_shape': processed_data.data.shape if processed_data.data is not None else None
        })

        logger.info(f"Data processing completed: {len(self.processing_info['steps'])} steps")
        return processed_data

    def process(self, data: IDataModel, **kwargs) -> IDataModel:
        """
        执行数据处理管道

        Args:
            data: 输入数据模型
            **kwargs: 处理参数

        Returns:
            IDataModel: 处理后的数据模型
        """
        df = data.data.copy()

        # 1. 数据清洗
        df = self._clean_data(df, **kwargs)

        # 2. 数据标准化
        df = self._normalize_data(df, **kwargs)

        # 3. 数据对齐
        df = self._align_data(df, **kwargs)

        # 4. 数据验证
        df = self._validate_processed_data(df, **kwargs)

        # 创建新的数据模型
        processed_model = type(data)(
            data=df,
            frequency=data.get_frequency(),
            metadata={
                **data.get_metadata(),
                'processed_at': datetime.now().isoformat(),
                'processor': self.__class__.__name__,
                'processing_steps': len(self.processing_info['steps'])
            }
        )

        return processed_model

    def _clean_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        数据清洗

        Args:
            df: 数据框
            **kwargs: 清洗参数

        Returns:
            pd.DataFrame: 清洗后的数据框
        """
        # 记录清洗步骤
        self.processing_info['steps'].append({
            'step': 'clean_data',
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape
        })

        # 1. 删除重复行
        df = df.drop_duplicates()

        # 2. 处理空值
        fill_method = kwargs.get('fill_method', 'forward')
        if fill_method == 'forward':
            df = df.fillna(method='ffill')
        elif fill_method == 'backward':
            df = df.fillna(method='bfill')
        elif fill_method == 'interpolate':
            df = df.interpolate()
        else:
            df = df.fillna(0)

        # 3. 处理异常值
        outlier_method = kwargs.get('outlier_method', 'iqr')
        if outlier_method == 'iqr':
            df = self._remove_outliers_iqr(df)
        elif outlier_method == 'zscore':
            df = self._remove_outliers_zscore(df)

        # 记录清洗结果
        self.processing_info['steps'][-1]['cleaned_shape'] = df.shape
        # 计算删除的行数（使用当前步骤的original_shape）
        current_step = self.processing_info['steps'][-1]
        if 'original_shape' in current_step:
            self.processing_info['steps'][-1]['removed_rows'] = current_step['original_shape'][0] - df.shape[0]
        else:
            self.processing_info['steps'][-1]['removed_rows'] = 0

        return df

    def _normalize_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        数据标准化

        Args:
            df: 数据框
            **kwargs: 标准化参数

        Returns:
            pd.DataFrame: 标准化后的数据框
        """
        # 记录标准化步骤
        self.processing_info['steps'].append({
            'step': 'normalize_data',
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape
        })

        # 1. 数值列标准化
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            normalize_method = kwargs.get('normalize_method', 'minmax')
            if normalize_method == 'minmax':
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / \
                    (df[numeric_cols].max() - df[numeric_cols].min())
            elif normalize_method == 'zscore':
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()
                                    ) / df[numeric_cols].std()
            elif normalize_method == 'robust':
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].median()) / \
                    (df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25))

        # 2. 日期列标准化
        date_cols = df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])

        # 记录标准化结果
        self.processing_info['steps'][-1]['normalized_shape'] = df.shape

        return df

    def _align_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        数据对齐

        Args:
            df: 数据框
            **kwargs: 对齐参数

        Returns:
            pd.DataFrame: 对齐后的数据框
        """
        # 记录对齐步骤
        self.processing_info['steps'].append({
            'step': 'align_data',
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape
        })

        # 1. 索引对齐
        if 'index_col' in kwargs:
            df = df.set_index(kwargs['index_col'])

        # 2. 时间对齐
        if 'time_col' in kwargs:
            df = df.sort_values(kwargs['time_col'])
            df = df.set_index(kwargs['time_col'])

        # 3. 列对齐
        if 'required_columns' in kwargs:
            missing_cols = set(kwargs['required_columns']) - set(df.columns)
            for col in missing_cols:
                df[col] = np.nan

        # 记录对齐结果
        self.processing_info['steps'][-1]['aligned_shape'] = df.shape

        return df

    def _validate_processed_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        验证处理后的数据

        Args:
            df: 数据框
            **kwargs: 验证参数

        Returns:
            pd.DataFrame: 验证后的数据框
        """
        # 记录验证步骤
        self.processing_info['steps'].append({
            'step': 'validate_processed_data',
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape
        })

        # 1. 检查数据完整性
        if df.empty:
            raise ValueError("处理后的数据为空")

        # 2. 检查数据类型
        expected_dtypes = kwargs.get('expected_dtypes', {})
        for col, dtype in expected_dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        # 3. 检查数据范围
        value_ranges = kwargs.get('value_ranges', {})
        for col, (min_val, max_val) in value_ranges.items():
            if col in df.columns:
                df[col] = df[col].clip(min_val, max_val)

        # 记录验证结果
        self.processing_info['steps'][-1]['validated_shape'] = df.shape

        return df

    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用IQR方法移除异常值

        Args:
            df: 数据框

        Returns:
            pd.DataFrame: 移除异常值后的数据框
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)

        return df

    def _remove_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        使用Z - score方法移除异常值

        Args:
            df: 数据框
            threshold: Z - score阈值

        Returns:
            pd.DataFrame: 移除异常值后的数据框
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df[col] = df[col].mask(z_scores > threshold, df[col].median())

        return df

    def get_processing_info(self) -> Dict[str, Any]:
        """
        获取处理信息

        Returns:
            Dict[str, Any]: 处理信息，包含处理步骤、参数等
        """
        return self.processing_info

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息

        Returns:
            Dict[str, Any]: 处理统计信息
        """
        if not self.processing_info['steps']:
            return {}

        total_steps = len(self.processing_info['steps'])
        start_step = next(
            (step for step in self.processing_info['steps'] if step['step'] == 'start'), None)
        complete_step = next(
            (step for step in self.processing_info['steps'] if step['step'] == 'complete'), None)

        if start_step and complete_step:
            start_time = datetime.fromisoformat(start_step['timestamp'])
            complete_time = datetime.fromisoformat(complete_step['timestamp'])
            processing_time = (complete_time - start_time).total_seconds()
        else:
            processing_time = 0

        return {
            'total_steps': total_steps,
            'processing_time_seconds': processing_time,
            'steps': self.processing_info['steps']
        }
