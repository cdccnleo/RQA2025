"""
model_predictor.py

模型预测服务模块

提供模型预测功能，支持：
- 加载训练好的模型
- 批量预测
- 预测结果缓存
- 预测结果转换为交易信号
- 预测结果持久化

适用于策略回测场景，支持模型驱动的回测。

作者: RQA2025 Team
日期: 2026-02-13
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """预测结果数据类"""
    model_id: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    signals: List[str] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPredictorConfig:
    """模型预测器配置"""
    enable_cache: bool = True
    cache_ttl_minutes: int = 30
    default_threshold: float = 0.5
    batch_size: int = 1000


class ModelPredictor:
    """
    模型预测器
    
    提供模型预测和信号生成功能，支持策略回测。
    
    Attributes:
        config: 预测器配置
        _model_cache: 模型缓存
        _prediction_cache: 预测结果缓存
        
    Example:
        >>> predictor = ModelPredictor()
        >>> 
        >>> # 加载模型并进行预测
        >>> result = predictor.predict(model_id, data)
        >>> 
        >>> # 获取交易信号
        >>> signals = result.signals  # ['buy', 'sell', 'hold', ...]
    """
    
    def __init__(self, config: Optional[ModelPredictorConfig] = None):
        """
        初始化模型预测器
        
        Args:
            config: 预测器配置
        """
        self.config = config or ModelPredictorConfig()
        self._model_cache: Dict[str, Any] = {}
        self._prediction_cache: Dict[str, PredictionResult] = {}
        
        logger.info(f"ModelPredictor 初始化完成: cache={self.config.enable_cache}")
    
    def _get_model_path(self, model_id: str) -> str:
        """获取模型文件路径

        模型文件存储在 models/{model_id}/model.pkl
        """
        return f"models/{model_id}/model.pkl"
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """
        加载训练好的模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            加载的模型，如果失败则返回None
        """
        # 检查缓存
        if model_id in self._model_cache:
            logger.debug(f"从缓存加载模型: {model_id}")
            return self._model_cache[model_id]
        
        try:
            model_path = self._get_model_path(model_id)
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 缓存模型
            self._model_cache[model_id] = model
            
            logger.info(f"模型加载成功: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败 {model_id}: {e}")
            return None
    
    def predict(
        self,
        model_id: str,
        data: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> Optional[PredictionResult]:
        """
        使用模型进行预测
        
        Args:
            model_id: 模型ID
            data: 输入数据
            threshold: 信号阈值（默认0.5）
            
        Returns:
            预测结果
        """
        threshold = threshold or self.config.default_threshold
        
        # 检查预测缓存
        cache_key = f"{model_id}_{hash(data.values.tobytes())}_{threshold}"
        if self.config.enable_cache and cache_key in self._prediction_cache:
            cached = self._prediction_cache[cache_key]
            if datetime.now() - cached.timestamp < timedelta(minutes=self.config.cache_ttl_minutes):
                logger.debug(f"使用缓存的预测结果: {model_id}")
                return cached
        
        # 加载模型
        model = self.load_model(model_id)
        if model is None:
            return None
        
        try:
            # 进行预测
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(data)
                predictions = model.predict(data)
            else:
                predictions = model.predict(data)
                probabilities = None
            
            # 生成交易信号
            signals, confidence = self._generate_signals(predictions, probabilities, threshold)
            
            # 构建结果
            result = PredictionResult(
                model_id=model_id,
                predictions=predictions,
                probabilities=probabilities,
                signals=signals,
                confidence=confidence,
                metadata={
                    "threshold": threshold,
                    "data_shape": data.shape if hasattr(data, 'shape') else None,
                    "model_type": type(model).__name__
                }
            )
            
            # 缓存结果
            if self.config.enable_cache:
                self._prediction_cache[cache_key] = result
            
            logger.info(f"预测完成: {model_id}, 生成了 {len(signals)} 个信号")
            return result
            
        except Exception as e:
            logger.error(f"预测失败 {model_id}: {e}")
            return None
    
    def _generate_signals(
        self,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray],
        threshold: float
    ) -> Tuple[List[str], List[float]]:
        """
        生成交易信号
        
        Args:
            predictions: 预测结果
            probabilities: 预测概率
            threshold: 信号阈值
            
        Returns:
            (信号列表, 置信度列表)
        """
        signals = []
        confidence = []
        
        for i, pred in enumerate(predictions):
            # 获取置信度
            if probabilities is not None and len(probabilities[i]) > 1:
                prob = probabilities[i][1] if pred == 1 else probabilities[i][0]
            else:
                prob = 0.5
            
            confidence.append(prob)
            
            # 生成信号
            if prob > threshold:
                signals.append('buy')
            elif prob < (1 - threshold):
                signals.append('sell')
            else:
                signals.append('hold')
        
        return signals, confidence
    
    def batch_predict(
        self,
        model_id: str,
        data_batches: List[Union[pd.DataFrame, np.ndarray]],
        threshold: Optional[float] = None
    ) -> List[Optional[PredictionResult]]:
        """
        批量预测
        
        Args:
            model_id: 模型ID
            data_batches: 数据批次列表
            threshold: 信号阈值
            
        Returns:
            预测结果列表
        """
        results = []
        for batch in data_batches:
            result = self.predict(model_id, batch, threshold)
            results.append(result)
        return results
    
    def get_signal_statistics(self, signals: List[str]) -> Dict[str, Any]:
        """
        获取信号统计信息
        
        Args:
            signals: 信号列表
            
        Returns:
            统计信息
        """
        total = len(signals)
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        hold_count = signals.count('hold')
        
        return {
            "total": total,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "buy_ratio": buy_count / total if total > 0 else 0,
            "sell_ratio": sell_count / total if total > 0 else 0,
            "hold_ratio": hold_count / total if total > 0 else 0
        }
    
    def clear_cache(self):
        """清除缓存"""
        self._model_cache.clear()
        self._prediction_cache.clear()
        logger.info("模型预测器缓存已清除")


# 全局预测器实例（单例模式）
_global_predictor: Optional[ModelPredictor] = None


def get_model_predictor(config: Optional[ModelPredictorConfig] = None) -> ModelPredictor:
    """
    获取全局模型预测器实例
    
    Args:
        config: 预测器配置
        
    Returns:
        模型预测器实例
    """
    global _global_predictor
    
    if _global_predictor is None:
        _global_predictor = ModelPredictor(config)
    
    return _global_predictor


def close_model_predictor():
    """关闭全局模型预测器实例"""
    global _global_predictor
    
    if _global_predictor:
        _global_predictor.clear_cache()
        _global_predictor = None
