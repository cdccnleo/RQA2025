#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强机器学习集成模块
整合现有ML功能并添加高级功能
"""

import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.infrastructure.integration import get_models_adapter

# 获取统一基础设施集成层的模型层适配器
try:
    models_adapter = get_models_adapter()
    logger = models_adapter.get_models_logger()
except Exception as e:
    # 降级处理
    logger = logging.getLogger(__name__)


@dataclass
class MLModelConfig:

    """机器学习模型配置"""
    training_period: int = 252
    prediction_window: int = 5
    confidence_threshold: float = 0.6


@dataclass
class PredictionResult:

    """预测结果"""
    prediction: int  # 1: 上涨, 0: 下跌
    confidence: float
    probability: float
    features_importance: Dict[str, float]


class EnhancedMLIntegration:

    """增强机器学习集成"""

    def __init__(self, config: Optional[MLModelConfig] = None):

        self.config = config or MLModelConfig()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.training_history = []
        self.prediction_history = []

    def prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        features = pd.DataFrame()

        # 价格特征
        features['returns'] = market_data['close'].pct_change()
        features['momentum'] = market_data['close'] / market_data['close'].shift(5) - 1
        features['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(5).mean()

        # 技术指标
        features['rsi'] = self._calculate_rsi(market_data['close'])
        features['volatility'] = features['returns'].rolling(20).std()

        return features.dropna()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def prepare_labels(self, market_data: pd.DataFrame) -> pd.Series:
        """准备标签数据"""
        future_returns = market_data['close'].shift(
            -self.config.prediction_window) / market_data['close'] - 1
        labels = (future_returns > 0.02).astype(int)  # 2 % 阈值
        return labels

    def train_model(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """训练模型"""
        logger.info("开始训练机器学习模型...")

        features = self.prepare_features(market_data)
        labels = self.prepare_labels(market_data)

        # 对齐数据
        common_index = features.index.intersection(labels.index)
        features = features.loc[common_index]
        labels = labels.loc[common_index]

        if len(features) < self.config.training_period:
            logger.warning(f"训练数据不足: {len(features)} < {self.config.training_period}")
            return {}

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练模型
        self.model.fit(X_train_scaled, y_train)

        # 评估
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # 保存特征重要性
        self.feature_importance = dict(zip(features.columns, self.model.feature_importances_))

        # 记录训练历史
        self.training_history.append({
            'timestamp': datetime.now(),
            'data_size': len(features),
            'accuracy': accuracy
        })

        logger.info(f"模型训练完成 - 准确率: {accuracy:.3f}")
        return {'accuracy': accuracy}

    def predict(self, market_data: pd.DataFrame) -> PredictionResult:
        """进行预测"""
        features = self.prepare_features(market_data)

        if len(features) == 0:
            return PredictionResult(
                prediction=0,
                confidence=0.0,
                probability=0.5,
                features_importance={}
            )

        latest_features = features.iloc[-1:].values
        features_scaled = self.scaler.transform(latest_features)

        try:
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            confidence = max(probability)

            # 记录预测历史
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'confidence': confidence
            })

            return PredictionResult(
                prediction=int(prediction),
                confidence=float(confidence),
                probability=float(max(probability)),
                features_importance=self.feature_importance
            )

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return PredictionResult(
                prediction=0,
                confidence=0.0,
                probability=0.5,
                features_importance={}
            )

    def get_model_performance(self) -> Dict[str, Any]:
        """获取模型性能统计"""
        if not self.training_history:
            return {}

        latest_training = self.training_history[-1]
        return {
            'latest_training': latest_training,
            'training_count': len(self.training_history),
            'prediction_count': len(self.prediction_history),
            'accuracy': latest_training.get('accuracy', 0)
        }

        if __name__ == "__main__":
            ml_integration = EnhancedMLIntegration()
            print("增强机器学习集成模块初始化完成")
