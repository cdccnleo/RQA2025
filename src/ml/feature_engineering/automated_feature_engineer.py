#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化特征工程模块

功能：
- 时序特征自动提取
- 交叉特征生成
- 特征重要性分析
- 特征选择

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """特征集"""
    name: str
    features: pd.DataFrame
    feature_names: List[str]
    importance_scores: Dict[str, float]
    created_at: datetime


class AutomatedFeatureEngineer:
    """
    自动化特征工程
    
    自动提取和生成特征：
    - 时序特征（滞后、滚动统计）
    - 技术指标特征
    - 交叉特征
    - 特征选择
    """
    
    def __init__(self, max_features: int = 100):
        """
        初始化特征工程器
        
        Args:
            max_features: 最大特征数量
        """
        self.max_features = max_features
        self.feature_history: List[FeatureSet] = []
        
        logger.info(f"自动化特征工程器初始化完成，最大特征数: {max_features}")
    
    async def engineer_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        include_technical: bool = True,
        include_time: bool = True
    ) -> FeatureSet:
        """
        工程化特征
        
        Args:
            df: 原始数据
            target_col: 目标列
            include_technical: 是否包含技术指标
            include_time: 是否包含时间特征
            
        Returns:
            特征集
        """
        features_df = df.copy()
        feature_names = []
        
        # 1. 时序特征
        ts_features = self._extract_time_series_features(features_df)
        features_df = pd.concat([features_df, ts_features], axis=1)
        feature_names.extend(ts_features.columns.tolist())
        
        # 2. 技术指标特征
        if include_technical and 'close' in features_df.columns:
            tech_features = self._extract_technical_features(features_df)
            features_df = pd.concat([features_df, tech_features], axis=1)
            feature_names.extend(tech_features.columns.tolist())
        
        # 3. 时间特征
        if include_time and isinstance(features_df.index, pd.DatetimeIndex):
            time_features = self._extract_time_features(features_df)
            features_df = pd.concat([features_df, time_features], axis=1)
            feature_names.extend(time_features.columns.tolist())
        
        # 4. 交叉特征
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            cross_features = self._generate_cross_features(features_df[numeric_cols])
            features_df = pd.concat([features_df, cross_features], axis=1)
            feature_names.extend(cross_features.columns.tolist())
        
        # 5. 特征选择（如果特征过多）
        if len(feature_names) > self.max_features:
            feature_names = self._select_features(
                features_df, feature_names, target_col
            )
            features_df = features_df[feature_names]
        
        # 计算特征重要性
        importance_scores = self._calculate_importance(
            features_df, feature_names, target_col
        )
        
        feature_set = FeatureSet(
            name=f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            features=features_df,
            feature_names=feature_names,
            importance_scores=importance_scores,
            created_at=datetime.now()
        )
        
        self.feature_history.append(feature_set)
        
        logger.info(f"特征工程完成: 生成 {len(feature_names)} 个特征")
        
        return feature_set
    
    def _extract_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取时序特征"""
        features = pd.DataFrame(index=df.index)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 滞后特征
            for lag in [1, 3, 5]:
                features[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # 滚动统计
            for window in [5, 10, 20]:
                features[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
                features[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
                features[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()
                features[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
            
            # 变化率
            features[f'{col}_pct_change'] = df[col].pct_change()
            features[f'{col}_diff'] = df[col].diff()
        
        return features
    
    def _extract_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取技术指标特征"""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # 布林带
        features['bb_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        features['bb_upper'] = features['bb_middle'] + 2 * bb_std
        features['bb_lower'] = features['bb_middle'] - 2 * bb_std
        
        # 移动平均线
        for window in [5, 10, 20, 60]:
            features[f'ma_{window}'] = close.rolling(window).mean()
            features[f'ma_ratio_{window}'] = close / features[f'ma_{window}']
        
        return features
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取时间特征"""
        features = pd.DataFrame(index=df.index)
        
        dt_index = df.index
        
        features['hour'] = dt_index.hour
        features['day_of_week'] = dt_index.dayofweek
        features['day_of_month'] = dt_index.day
        features['month'] = dt_index.month
        features['quarter'] = dt_index.quarter
        features['is_month_start'] = dt_index.is_month_start.astype(int)
        features['is_month_end'] = dt_index.is_month_end.astype(int)
        
        return features
    
    def _generate_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交叉特征"""
        features = pd.DataFrame(index=df.index)
        
        numeric_cols = df.columns[:10]  # 限制数量
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # 乘法
                features[f'{col1}_mul_{col2}'] = df[col1] * df[col2]
                
                # 除法（避免除零）
                features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
                # 加法和减法
                features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
        
        return features
    
    def _select_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_col: Optional[str]
    ) -> List[str]:
        """特征选择"""
        if target_col and target_col in df.columns:
            # 基于相关性选择
            correlations = df[feature_names].corrwith(df[target_col]).abs()
            selected = correlations.nlargest(self.max_features).index.tolist()
        else:
            # 基于方差选择
            variances = df[feature_names].var()
            selected = variances.nlargest(self.max_features).index.tolist()
        
        return selected
    
    def _calculate_importance(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_col: Optional[str]
    ) -> Dict[str, float]:
        """计算特征重要性"""
        importance = {}
        
        if target_col and target_col in df.columns:
            # 使用相关性作为重要性
            for feature in feature_names:
                if feature in df.columns:
                    corr = df[feature].corr(df[target_col])
                    importance[feature] = abs(corr) if not pd.isna(corr) else 0.0
        else:
            # 默认等权重
            for feature in feature_names:
                importance[feature] = 1.0 / len(feature_names)
        
        return importance


# 全局实例
_engineer_instance: Optional[AutomatedFeatureEngineer] = None


def get_feature_engineer(max_features: int = 100) -> AutomatedFeatureEngineer:
    """获取特征工程器实例"""
    global _engineer_instance
    
    if _engineer_instance is None:
        _engineer_instance = AutomatedFeatureEngineer(max_features)
    
    return _engineer_instance
