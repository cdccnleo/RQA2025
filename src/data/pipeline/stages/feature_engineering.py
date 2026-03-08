"""
特征工程阶段模块

负责特征计算、选择、标准化和存储
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from .base import PipelineStage
from ..exceptions import StageExecutionException
from ..config import StageConfig


class FeatureEngineeringStage(PipelineStage):
    """
    特征工程阶段
    
    功能：
    - 技术指标特征计算
    - 统计特征提取
    - 特征选择
    - 特征标准化
    - 特征存储
    """
    
    def __init__(self, config: Optional[StageConfig] = None):
        super().__init__("feature_engineering", config)
        self._features_df: Optional[pd.DataFrame] = None
        self._selected_features: List[str] = []
        self._feature_stats: Dict[str, Any] = {}
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行特征工程
        
        Args:
            context: 包含processed_data的上下文
            
        Returns:
            包含features, selected_features的输出
        """
        self.logger.info("开始特征工程阶段")
        
        # 获取输入数据
        processed_data = context.get("processed_data")
        if processed_data is None:
            raise StageExecutionException(
                message="缺少processed_data输入",
                stage_name=self.name
            )
        
        if isinstance(processed_data, dict):
            df = pd.DataFrame(processed_data)
        else:
            df = processed_data.copy()
        
        self.logger.info(f"输入数据: {df.shape}")
        
        # 1. 计算技术指标特征
        self.logger.info("计算技术指标特征")
        features_df = self._calculate_technical_features(df)
        
        # 2. 计算统计特征
        self.logger.info("计算统计特征")
        features_df = self._calculate_statistical_features(features_df)
        
        # 3. 特征选择
        selection_method = self.config.config.get("feature_selection", "variance")
        if selection_method:
            self.logger.info(f"执行特征选择: {selection_method}")
            features_df, selected_features = self._select_features(features_df, selection_method)
            self._selected_features = selected_features
        
        # 4. 特征标准化
        standardization = self.config.config.get("standardization", "zscore")
        if standardization:
            self.logger.info(f"执行特征标准化: {standardization}")
            features_df = self._standardize_features(features_df, standardization)
        
        # 5. 存储特征
        if self.config.config.get("store_features", True):
            self._store_features(features_df, context)
        
        self._features_df = features_df
        
        # 计算特征统计
        self._feature_stats = self._calculate_feature_stats(features_df)
        
        self.logger.info(f"特征工程完成，输出特征: {features_df.shape}")
        
        return {
            "features": features_df,
            "features_shape": features_df.shape,
            "selected_features": self._selected_features,
            "feature_stats": self._feature_stats
        }
    
    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        features = df.copy()
        
        # 确保有OHLCV数据
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in features.columns:
                self.logger.warning(f"缺少 {col} 列，跳过相关特征计算")
        
        # 价格特征
        if "close" in features.columns:
            # 收益率
            features["returns"] = features["close"].pct_change()
            features["log_returns"] = np.log(features["close"] / features["close"].shift(1))
            
            # 价格变化
            features["price_change"] = features["close"].diff()
            features["price_change_pct"] = features["close"].pct_change() * 100
        
        # 移动平均线
        if "close" in features.columns:
            for window in [5, 10, 20, 50]:
                features[f"sma_{window}"] = features["close"].rolling(window=window).mean()
                features[f"ema_{window}"] = features["close"].ewm(span=window, adjust=False).mean()
                
                # 价格与均线距离
                features[f"dist_sma_{window}"] = (
                    (features["close"] - features[f"sma_{window}"]) / features[f"sma_{window}"] * 100
                )
        
        # RSI
        if "close" in features.columns:
            features["rsi_14"] = self._calculate_rsi(features["close"], 14)
            features["rsi_7"] = self._calculate_rsi(features["close"], 7)
        
        # MACD
        if "close" in features.columns:
            ema_12 = features["close"].ewm(span=12, adjust=False).mean()
            ema_26 = features["close"].ewm(span=26, adjust=False).mean()
            features["macd"] = ema_12 - ema_26
            features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
            features["macd_hist"] = features["macd"] - features["macd_signal"]
        
        # 布林带
        if "close" in features.columns:
            sma_20 = features["close"].rolling(window=20).mean()
            std_20 = features["close"].rolling(window=20).std()
            features["bb_upper"] = sma_20 + 2 * std_20
            features["bb_lower"] = sma_20 - 2 * std_20
            features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / sma_20
            features["bb_position"] = (features["close"] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"])
        
        # ATR (Average True Range)
        if all(col in features.columns for col in ["high", "low", "close"]):
            features["tr"] = np.maximum(
                features["high"] - features["low"],
                np.maximum(
                    abs(features["high"] - features["close"].shift(1)),
                    abs(features["low"] - features["close"].shift(1))
                )
            )
            features["atr_14"] = features["tr"].rolling(window=14).mean()
        
        # 成交量特征
        if "volume" in features.columns:
            features["volume_sma_20"] = features["volume"].rolling(window=20).mean()
            features["volume_ratio"] = features["volume"] / features["volume_sma_20"]
            features["obv"] = (np.sign(features["close"].diff()) * features["volume"]).cumsum()
        
        # 波动率
        if "returns" in features.columns:
            features["volatility_20"] = features["returns"].rolling(window=20).std() * np.sqrt(252)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算统计特征"""
        features = df.copy()
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        # 为每个数值列计算滚动统计
        for col in numeric_cols:
            if col in ["timestamp", "open", "high", "low", "close", "volume"]:
                continue
            
            for window in [5, 10, 20]:
                # 滚动均值和标准差
                features[f"{col}_roll_mean_{window}"] = features[col].rolling(window=window).mean()
                features[f"{col}_roll_std_{window}"] = features[col].rolling(window=window).std()
                
                # 滚动最大最小值
                features[f"{col}_roll_max_{window}"] = features[col].rolling(window=window).max()
                features[f"{col}_roll_min_{window}"] = features[col].rolling(window=window).min()
        
        # 时间特征
        if "timestamp" in features.columns:
            features["hour"] = pd.to_datetime(features["timestamp"]).dt.hour
            features["day_of_week"] = pd.to_datetime(features["timestamp"]).dt.dayofweek
            features["month"] = pd.to_datetime(features["timestamp"]).dt.month
        
        return features
    
    def _select_features(
        self,
        df: pd.DataFrame,
        method: str
    ) -> tuple[pd.DataFrame, List[str]]:
        """
        特征选择
        
        Args:
            df: 特征数据框
            method: 选择方法
            
        Returns:
            (选择后的数据框, 选择的特征列表)
        """
        # 排除非特征列
        exclude_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            return df, []
        
        feature_df = df[feature_cols].copy()
        
        # 处理无穷值和NaN
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna(axis=1, how="all")
        
        selected_features = list(feature_df.columns)
        
        if method == "variance":
            # 方差阈值选择
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            try:
                selector.fit(feature_df.fillna(0))
                selected_features = [feature_df.columns[i] for i in selector.get_support(indices=True)]
            except Exception as e:
                self.logger.warning(f"方差选择失败: {e}")
        
        elif method == "correlation":
            # 相关性选择：移除高度相关的特征
            corr_matrix = feature_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            selected_features = [f for f in selected_features if f not in to_drop]
        
        elif method == "all":
            # 保留所有特征
            pass
        
        # 保留原始列和选择的特征
        final_cols = exclude_cols + selected_features
        final_cols = [col for col in final_cols if col in df.columns]
        
        return df[final_cols].copy(), selected_features
    
    def _standardize_features(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        特征标准化
        
        Args:
            df: 特征数据框
            method: 标准化方法
            
        Returns:
            标准化后的数据框
        """
        features = df.copy()
        
        # 排除非数值列
        exclude_cols = ["timestamp"]
        numeric_cols = [col for col in features.columns 
                       if col not in exclude_cols and features[col].dtype in [np.float64, np.float32, np.int64]]
        
        if not numeric_cols:
            return features
        
        for col in numeric_cols:
            col_data = features[col].replace([np.inf, -np.inf], np.nan)
            
            if method == "zscore":
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    features[col] = (col_data - mean) / std
            
            elif method == "minmax":
                min_val = col_data.min()
                max_val = col_data.max()
                if max_val > min_val:
                    features[col] = (col_data - min_val) / (max_val - min_val)
            
            elif method == "robust":
                median = col_data.median()
                q75 = col_data.quantile(0.75)
                q25 = col_data.quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    features[col] = (col_data - median) / iqr
        
        return features
    
    def _store_features(self, df: pd.DataFrame, context: Dict[str, Any]) -> None:
        """存储特征到特征存储"""
        # 实际应存储到Feature Store
        self.logger.info(f"特征已准备，共 {len(df.columns)} 列")
    
    def _calculate_feature_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算特征统计信息"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats = {
            "total_features": len(df.columns),
            "numeric_features": len(numeric_cols),
            "missing_values": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0
        }
        
        return stats
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取阶段指标"""
        return {
            "total_features": len(self._features_df.columns) if self._features_df is not None else 0,
            "selected_features": len(self._selected_features),
            "feature_stats": self._feature_stats
        }
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """回滚特征工程阶段"""
        self.logger.info("回滚特征工程阶段")
        self._features_df = None
        self._selected_features = []
        self._feature_stats = {}
        return True
