from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import time

from src.features.technical.technical_processor import TechnicalProcessor
from src.features.sentiment.sentiment_analyzer import SentimentAnalyzer
from src.features.feature_metadata import FeatureMetadata
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """特征工程处理器，用于生成和管理特征"""

    def __init__(
            self,
            stock_code: str,
            metadata: Optional[FeatureMetadata] = None
    ):
        """
        初始化特征工程处理器

        Args:
            stock_code: 股票代码
            metadata: 特征元数据对象（可选）
        """
        self.stock_code = stock_code
        self.technical_processor = TechnicalProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_metadata = metadata or FeatureMetadata()
        self.logger = logger

    def _validate_stock_data(self, data: pd.DataFrame) -> None:
        """
        验证股票数据的有效性

        Args:
            data: 股票数据DataFrame
        """
        if data.empty:
            raise ValueError("输入数据为空")

        # 检查必要的列
        required_columns = ['close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺失必要价格列: {', '.join(missing_columns)}")

        # 检查数据类型
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"列 {col} 不是数值类型")

        # 检查数据有效性
        if (data[['close', 'high', 'low']] < 0).any().any():
            raise ValueError("检测到负值价格")
        if (data['volume'] < 0).any():
            raise ValueError("检测到负值交易量")

        # 检查价格逻辑
        if (data['high'] < data['low']).any():
            raise ValueError("价格高低值逻辑错误")
        if ((data['close'] > data['high']) | (data['close'] < data['low'])).any():
            raise ValueError("收盘价超出高低价范围")

        # 检查NaN值
        if data[required_columns].isna().any().any():
            raise ValueError("价格数据包含 NaN 值")

        # 检查索引
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("索引不是时间戳类型")

        # 检查重复日期
        if not data.index.is_unique:
            raise ValueError("索引存在重复日期")

        # 检查未来日期
        current_time = pd.Timestamp.now()
        if data.index.max() > current_time:
            raise ValueError("检测到未来日期数据")

        # 检查索引排序
        if not data.index.is_monotonic_increasing:
            raise ValueError("索引非单调递增")

    def generate_technical_features(
            self,
            stock_data: pd.DataFrame,
            indicators: List[str] = None,
            params: Dict = None
    ) -> pd.DataFrame:
        """
        生成技术指标特征

        Args:
            stock_data: 股票数据DataFrame
            indicators: 要计算的技术指标列表
            params: 技术指标参数

        Returns:
            技术指标特征DataFrame
        """
        # 验证数据
        self._validate_stock_data(stock_data)

        # 默认指标和参数
        if indicators is None:
            indicators = ["ma", "rsi", "macd", "bollinger"]
        if params is None:
            params = {
                "ma": {"windows": [5, 10, 20, 30, 60]},
                "rsi": {"window": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"window": 20, "num_std": 2}
            }

        # 更新元数据中的特征参数
        self.feature_metadata.update_feature_params({
            "technical_indicators": indicators,
            "technical_params": params
        })

        # 生成技术指标
        try:
            features = self.technical_processor.calculate_indicators(
                data=stock_data,
                indicators=indicators,
                params=params
            )
            self.logger.info(f"技术指标特征生成完成，特征列名: {features.columns.tolist()}")
            return features

        except Exception as e:
            self.logger.error(f"生成技术指标特征失败: {str(e)}")
            raise

    def generate_sentiment_features(
            self,
            news_data: pd.DataFrame,
            text_col: str = "content",
            date_col: str = "date",
            output_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        生成情感分析特征

        Args:
            news_data: 新闻数据DataFrame
            text_col: 文本列名
            date_col: 日期列名
            output_cols: 输出列名列表

        Returns:
            情感分析特征DataFrame
        """
        try:
            # 更新元数据中的特征参数
            self.feature_metadata.update_feature_params({
                "sentiment_text_col": text_col,
                "sentiment_date_col": date_col,
                "sentiment_output_cols": output_cols
            })

            features = self.sentiment_analyzer.generate_features(
                news_data=news_data,
                text_col=text_col,
                date_col=date_col,
                output_cols=output_cols
            )
            self.logger.info(f"情感分析特征生成完成，特征列名: {features.columns.tolist()}")
            return features

        except Exception as e:
            self.logger.error(f"生成情感分析特征失败: {str(e)}")
            raise

    def merge_features(
            self,
            stock_data: pd.DataFrame,
            technical_features: pd.DataFrame,
            sentiment_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        合并所有特征

        Args:
            stock_data: 原始股票数据
            technical_features: 技术指标特征
            sentiment_features: 情感分析特征（可选）

        Returns:
            合并后的特征DataFrame
        """
        try:
            # 验证索引
            if not technical_features.index.equals(stock_data.index):
                raise ValueError("技术指标特征索引不匹配")
            if sentiment_features is not None and not sentiment_features.index.equals(stock_data.index):
                raise ValueError("情感分析特征索引不匹配")

            # 合并特征
            result = pd.concat([
                stock_data,
                technical_features,
                sentiment_features if sentiment_features is not None else pd.DataFrame(index=stock_data.index)
            ], axis=1)

            # 更新元数据
            self.feature_metadata.update_feature_columns(result.columns.tolist())

            self.logger.info(f"特征合并完成，最终特征数: {len(result.columns)}")
            return result

        except Exception as e:
            self.logger.error(f"特征合并失败: {str(e)}")
            raise

    def save_metadata(self, path: str) -> None:
        """
        保存特征元数据到文件

        Args:
            path: 元数据文件路径
        """
        try:
            self.feature_metadata.save_metadata(path)
            self.logger.info(f"特征元数据保存成功: {path}")
        except Exception as e:
            self.logger.error(f"保存特征元数据失败: {str(e)}")
            raise

    def load_metadata(self, path: str) -> None:
        """
        从文件加载特征元数据

        Args:
            path: 元数据文件路径
        """
        try:
            self.feature_metadata = FeatureMetadata(metadata_path=path)
            self.logger.info(f"特征元数据加载成功: {path}")
        except Exception as e:
            self.logger.error(f"加载特征元数据失败: {str(e)}")
            raise
