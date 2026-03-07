#!/usr/bin/env python3
"""
RQA2025数据预处理器
提供高级数据预处理和清洗功能
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class DataPreprocessor:

    """数据预处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.preprocessing_stats = {
            'total_processed': 0,
            'missing_values_handled': 0,
            'outliers_handled': 0,
            'duplicates_removed': 0
        }

    def preprocess(self, data: pd.DataFrame,


                   steps: Optional[List[str]] = None) -> pd.DataFrame:
        """
        执行数据预处理

        Args:
            data: 原始数据
            steps: 预处理步骤列表

        Returns:
            预处理后的数据
        """
        if data is None or data.empty:
            logger.warning("输入数据为空")
            return data

        if steps is None:
            steps = ['validate', 'clean', 'normalize']

        processed_data = data.copy()
        original_shape = processed_data.shape

        for step in steps:
            try:
                if step == 'validate':
                    processed_data = self._validate_data(processed_data)
                elif step == 'clean':
                    processed_data = self._clean_data(processed_data)
                elif step == 'normalize':
                    processed_data = self._normalize_data(processed_data)

                logger.info(f"预处理步骤 {step} 完成")

            except Exception as e:
                logger.error(f"预处理步骤 {step} 失败: {e}")
                continue

        final_shape = processed_data.shape
        logger.info(f"数据预处理完成: {original_shape} -> {final_shape}")
        self.preprocessing_stats['total_processed'] += 1

        return processed_data

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """验证数据结构"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']

        # 检查必需列
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必需列: {missing_columns}")

        # 检查数据类型
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

        numeric_columns = ['open', 'high', 'low', 'close']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        if 'volume' in data.columns:
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

        # 删除无效行
        data = data.dropna(subset=numeric_columns)

        return data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        original_count = len(data)

        # 1. 处理重复数据
        duplicates_before = len(data)
        data = data.drop_duplicates(subset=['timestamp'])
        duplicates_after = len(data)
        duplicates_removed = duplicates_before - duplicates_after

        if duplicates_removed > 0:
            logger.info(f"移除重复数据: {duplicates_removed} 条")
            self.preprocessing_stats['duplicates_removed'] += duplicates_removed

        # 2. 处理缺失值
        missing_before = data.isnull().sum().sum()
        data = self._handle_missing_values(data)
        missing_after = data.isnull().sum().sum()
        missing_handled = missing_before - missing_after

        if missing_handled > 0:
            logger.info(f"处理缺失值: {missing_handled} 个")
            self.preprocessing_stats['missing_values_handled'] += missing_handled

        # 3. 处理异常值
        data = self._handle_outliers(data)

        # 4. 排序和重置索引
        data = data.sort_values('timestamp').reset_index(drop=True)

        final_count = len(data)
        if original_count != final_count:
            logger.info(f"数据清理: {original_count} -> {final_count}")

        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 价格数据的向前填充
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')

        # 成交量填充为0
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)

        return data

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        outliers_handled = 0

        for col in numeric_columns:
            if col in data.columns:
                # 使用IQR方法检测异常值
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)

                if outlier_mask.sum() > 0:
                    median_value = data[col].median()
                    data.loc[outlier_mask, col] = median_value
                    outliers_handled += outlier_mask.sum()

        if outliers_handled > 0:
            logger.info(f"处理异常值: {outliers_handled} 个")
            self.preprocessing_stats['outliers_handled'] += outliers_handled

        return data

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化数据"""
        # 价格数据标准化 (可选)
        if self.config.get('normalize_prices', False):
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    if 'close' in data.columns and col != 'close':
                        data[col] = data[col] / data['close']

        if self.config.get('normalize_volume', False) and 'volume' in data.columns:
            data['volume'] = np.log1p(data['volume'])

        return data


class DataQualityMonitor:

    """数据质量监控器"""

    def __init__(self):

        self.quality_metrics = {
            'total_records': 0,
            'missing_values': 0,
            'duplicates': 0,
            'outliers': 0,
            'data_quality_score': 0.0
        }

    def assess_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """评估数据质量"""
        metrics = {
            'total_records': len(data),
            'missing_values': data.isnull().sum().sum(),
            'duplicates': len(data) - len(data.drop_duplicates()),
            'completeness': 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'timestamp_monotonic': data['timestamp'].is_monotonic_increasing
        }

        # 计算数据质量评分
        completeness_score = metrics['completeness'] * 100
        monotonic_score = 100 if metrics['timestamp_monotonic'] else 0
        duplicate_penalty = min(metrics['duplicates'] / len(data) * 100, 20) if len(data) > 0 else 0

        quality_score = (completeness_score + monotonic_score - duplicate_penalty)
        metrics['data_quality_score'] = max(0, min(100, quality_score))

        self.quality_metrics.update(metrics)

        return metrics

    def generate_quality_report(self, data: pd.DataFrame) -> str:
        """生成质量报告"""
        metrics = self.assess_quality(data)

        report = f"""
            数据质量评估报告
==================
总记录数: {metrics['total_records']}
缺失值数量: {metrics['missing_values']}
重复记录数: {metrics['duplicates']}
数据完整性: {metrics['completeness']:.1%}
时间序列单调性: {'✓' if metrics['timestamp_monotonic'] else '✗'}
数据质量评分: {metrics['data_quality_score']:.1f}/100

质量等级: {'优秀' if metrics['data_quality_score'] >= 90 else
           '良好' if metrics['data_quality_score'] >= 80 else
           '一般' if metrics['data_quality_score'] >= 60 else '较差'}
               """

        return report.strip()
