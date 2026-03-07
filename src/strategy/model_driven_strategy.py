"""
model_driven_strategy.py

模型驱动的策略模块

提供基于模型预测的策略实现，支持：
- 加载训练好的模型
- 实时模型预测
- 预测结果转换为交易信号
- 模型置信度评估
- 动态阈值调整

适用于策略生命周期管理和策略执行，实现模型到策略的完整数据流。

作者: RQA2025 Team
日期: 2026-02-13
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ModelStrategyConfig:
    """模型策略配置"""
    model_id: str
    prediction_threshold: float = 0.5
    confidence_threshold: float = 0.7
    position_sizing: str = "equal"  # equal, confidence_based, kelly
    max_position_size: float = 0.2  # 最大仓位比例
    stop_loss: Optional[float] = None  # 止损比例
    take_profit: Optional[float] = None  # 止盈比例
    rebalancing_frequency: str = "daily"  # daily, weekly, monthly


@dataclass
class Signal:
    """交易信号数据类"""
    timestamp: datetime
    symbol: str
    action: str  # buy, sell, hold
    confidence: float
    predicted_return: float
    position_size: float
    reason: str


class ModelDrivenStrategy:
    """
    模型驱动的策略
    
    基于训练好的模型预测生成交易信号，实现模型到策略的完整数据流。
    
    Attributes:
        config: 模型策略配置
        model: 训练好的模型
        predictor: 模型预测器
        
    Example:
        >>> config = ModelStrategyConfig(model_id="model_123", prediction_threshold=0.6)
        >>> strategy = ModelDrivenStrategy(config)
        >>> 
        >>> # 生成交易信号
        >>> signals = strategy.generate_signals(market_data)
        >>> for signal in signals:
        ...     print(f"{signal.action} {signal.symbol} @ {signal.confidence:.2%} confidence")
    """
    
    def __init__(self, config: ModelStrategyConfig):
        """
        初始化模型驱动策略
        
        Args:
            config: 模型策略配置
        """
        self.config = config
        self.model = None
        self.predictor = None
        self._last_prediction_time = None
        self._prediction_cache = {}
        
        # 加载模型
        self._load_model()
        
        logger.info(f"模型驱动策略初始化完成: model_id={config.model_id}")
    
    def _load_model(self):
        """加载训练好的模型"""
        try:
            from src.ml.inference.model_predictor import get_model_predictor
            
            self.predictor = get_model_predictor()
            self.model = self.predictor.load_model(self.config.model_id)
            
            if self.model is None:
                raise ValueError(f"无法加载模型: {self.config.model_id}")
            
            logger.info(f"模型加载成功: {self.config.model_id}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_positions: Optional[Dict[str, float]] = None
    ) -> List[Signal]:
        """
        生成交易信号
        
        Args:
            data: 市场数据
            current_positions: 当前持仓
            
        Returns:
            交易信号列表
        """
        if self.model is None:
            logger.error("模型未加载")
            return []
        
        try:
            # 进行预测
            prediction_result = self.predictor.predict(
                self.config.model_id,
                data,
                threshold=self.config.prediction_threshold
            )
            
            if prediction_result is None:
                logger.warning("预测失败")
                return []
            
            signals = []
            timestamp = datetime.now()
            
            # 获取股票代码
            symbols = data.get('symbol', ['UNKNOWN'] * len(data))
            if isinstance(symbols, pd.Series):
                symbols = symbols.values
            
            # 生成信号
            for i, (symbol, action, confidence) in enumerate(zip(
                symbols,
                prediction_result.signals,
                prediction_result.confidence
            )):
                # 检查置信度阈值
                if confidence < self.config.confidence_threshold:
                    action = 'hold'
                
                # 计算仓位大小
                position_size = self._calculate_position_size(
                    action, confidence, current_positions
                )
                
                # 获取预测收益
                predicted_return = 0.0
                if prediction_result.predictions is not None and i < len(prediction_result.predictions):
                    predicted_return = float(prediction_result.predictions[i])
                
                signal = Signal(
                    timestamp=timestamp,
                    symbol=str(symbol),
                    action=action,
                    confidence=float(confidence),
                    predicted_return=predicted_return,
                    position_size=position_size,
                    reason=f"模型预测: {action} (置信度: {confidence:.2%})"
                )
                
                signals.append(signal)
            
            self._last_prediction_time = timestamp
            
            logger.info(f"生成 {len(signals)} 个交易信号")
            return signals
            
        except Exception as e:
            logger.error(f"生成信号失败: {e}")
            return []
    
    def _calculate_position_size(
        self,
        action: str,
        confidence: float,
        current_positions: Optional[Dict[str, float]]
    ) -> float:
        """
        计算仓位大小
        
        Args:
            action: 交易动作
            confidence: 置信度
            current_positions: 当前持仓
            
        Returns:
            仓位大小（0-1之间的比例）
        """
        if action == 'hold':
            return 0.0
        
        if self.config.position_sizing == "equal":
            # 等权重
            return self.config.max_position_size
        
        elif self.config.position_sizing == "confidence_based":
            # 基于置信度的仓位
            return self.config.max_position_size * confidence
        
        elif self.config.position_sizing == "kelly":
            # Kelly公式（简化版）
            # f* = (p*b - q) / b
            # 其中 p 是胜率，b 是盈亏比
            # 这里简化为使用置信度作为胜率
            p = confidence
            b = 2.0  # 假设盈亏比为2
            q = 1 - p
            
            kelly_fraction = (p * b - q) / b if b > 0 else 0
            kelly_fraction = max(0, min(kelly_fraction, 1))  # 限制在0-1之间
            
            return self.config.max_position_size * kelly_fraction
        
        else:
            return self.config.max_position_size
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            策略信息字典
        """
        return {
            "type": "model_driven",
            "model_id": self.config.model_id,
            "prediction_threshold": self.config.prediction_threshold,
            "confidence_threshold": self.config.confidence_threshold,
            "position_sizing": self.config.position_sizing,
            "max_position_size": self.config.max_position_size,
            "last_prediction_time": self._last_prediction_time.isoformat() if self._last_prediction_time else None,
            "model_loaded": self.model is not None
        }
    
    def update_config(self, **kwargs):
        """
        更新策略配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"策略配置更新: {key} = {value}")
    
    def get_performance_metrics(self, backtest_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        获取策略性能指标
        
        Args:
            backtest_result: 回测结果
            
        Returns:
            性能指标
        """
        metrics = {
            "strategy_type": "model_driven",
            "model_id": self.config.model_id,
            "config": {
                "prediction_threshold": self.config.prediction_threshold,
                "confidence_threshold": self.config.confidence_threshold,
                "position_sizing": self.config.position_sizing
            }
        }
        
        if backtest_result:
            metrics.update({
                "total_return": backtest_result.get("total_return", 0),
                "sharpe_ratio": backtest_result.get("sharpe_ratio", 0),
                "max_drawdown": backtest_result.get("max_drawdown", 0),
                "win_rate": backtest_result.get("win_rate", 0),
                "total_trades": backtest_result.get("total_trades", 0)
            })
        
        return metrics


# 模型策略工厂
class ModelStrategyFactory:
    """模型策略工厂"""
    
    _strategies: Dict[str, ModelDrivenStrategy] = {}
    
    @classmethod
    def create_strategy(cls, config: ModelStrategyConfig) -> ModelDrivenStrategy:
        """
        创建模型策略
        
        Args:
            config: 模型策略配置
            
        Returns:
            模型驱动策略实例
        """
        strategy = ModelDrivenStrategy(config)
        cls._strategies[config.model_id] = strategy
        return strategy
    
    @classmethod
    def get_strategy(cls, model_id: str) -> Optional[ModelDrivenStrategy]:
        """
        获取模型策略
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型驱动策略实例
        """
        return cls._strategies.get(model_id)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        列出所有模型策略
        
        Returns:
            模型ID列表
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def remove_strategy(cls, model_id: str):
        """
        移除模型策略
        
        Args:
            model_id: 模型ID
        """
        if model_id in cls._strategies:
            del cls._strategies[model_id]
            logger.info(f"模型策略已移除: {model_id}")


def create_model_strategy(
    model_id: str,
    prediction_threshold: float = 0.5,
    confidence_threshold: float = 0.7,
    position_sizing: str = "equal",
    **kwargs
) -> ModelDrivenStrategy:
    """
    创建模型驱动策略的便捷函数
    
    Args:
        model_id: 模型ID
        prediction_threshold: 预测阈值
        confidence_threshold: 置信度阈值
        position_sizing: 仓位管理方式
        **kwargs: 其他配置参数
        
    Returns:
        模型驱动策略实例
    """
    config = ModelStrategyConfig(
        model_id=model_id,
        prediction_threshold=prediction_threshold,
        confidence_threshold=confidence_threshold,
        position_sizing=position_sizing,
        **kwargs
    )
    
    return ModelStrategyFactory.create_strategy(config)
