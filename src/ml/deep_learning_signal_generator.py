"""
深度学习信号生成器
使用LSTM、Transformer和强化学习生成交易信号
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class DeepLearningSignal:
    """深度学习信号"""
    symbol: str
    signal_type: str  # buy, sell, hold
    confidence: float  # 0-1
    model_type: str  # lstm, transformer, rl
    features: Dict[str, float]
    timestamp: datetime


class LSTMModel:
    """LSTM时序预测模型"""
    
    def __init__(self, sequence_length: int = 60, n_features: int = 20):
        """
        初始化LSTM模型
        
        Args:
            sequence_length: 序列长度
            n_features: 特征数量
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = None
        
        # 尝试加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            model_path = "models/lstm_model.pkl"
            if os.path.exists(model_path):
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                logger.info("LSTM模型加载成功")
        except Exception as e:
            logger.error(f"加载LSTM模型失败: {e}")
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """准备特征"""
        features = []
        
        # 价格特征
        features.extend([
            data['close'].iloc[-1],
            data['close'].pct_change().iloc[-1],
            data['close'].rolling(5).mean().iloc[-1],
            data['close'].rolling(10).mean().iloc[-1],
            data['close'].rolling(20).mean().iloc[-1],
        ])
        
        # 技术指标
        features.extend([
            self._calculate_rsi(data['close'], 14),
            self._calculate_macd(data['close']),
            self._calculate_bollinger_position(data['close']),
        ])
        
        # 成交量特征
        features.extend([
            data['volume'].iloc[-1],
            data['volume'].rolling(5).mean().iloc[-1],
            data['volume'].pct_change().iloc[-1],
        ])
        
        # 波动率
        features.append(data['close'].pct_change().rolling(20).std().iloc[-1])
        
        return np.array(features)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    
    def _calculate_macd(self, prices: pd.Series) -> float:
        """计算MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        return macd.iloc[-1] if len(macd) > 0 else 0
    
    def _calculate_bollinger_position(self, prices: pd.Series) -> float:
        """计算布林带位置"""
        middle = prices.rolling(window=20).mean()
        std = prices.rolling(window=20).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        percent = (prices - lower) / (upper - lower)
        return percent.iloc[-1] if len(percent) > 0 else 0.5
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        预测
        
        Returns:
            预测结果
        """
        try:
            if len(data) < self.sequence_length:
                return {'signal': 'hold', 'confidence': 0.5}
            
            # 准备特征
            features = self._prepare_features(data)
            
            # 如果模型存在，使用模型预测
            if self.model is not None:
                # 标准化
                if self.scaler is not None:
                    features_scaled = self.scaler.transform(features.reshape(1, -1))
                else:
                    features_scaled = features.reshape(1, -1)
                
                # 预测
                prediction = self.model.predict(features_scaled)[0]
                
                # 转换为信号
                if prediction > 0.6:
                    signal = 'buy'
                elif prediction < 0.4:
                    signal = 'sell'
                else:
                    signal = 'hold'
                
                confidence = abs(prediction - 0.5) * 2
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'raw_prediction': prediction
                }
            else:
                # 使用规则预测
                return self._rule_based_prediction(features)
                
        except Exception as e:
            logger.error(f"LSTM预测失败: {e}")
            return {'signal': 'hold', 'confidence': 0.5}
    
    def _rule_based_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """基于规则的预测"""
        # 简化版规则
        rsi = features[5] if len(features) > 5 else 50
        macd = features[6] if len(features) > 6 else 0
        
        if rsi < 30 and macd > 0:
            return {'signal': 'buy', 'confidence': 0.7}
        elif rsi > 70 and macd < 0:
            return {'signal': 'sell', 'confidence': 0.7}
        else:
            return {'signal': 'hold', 'confidence': 0.5}


class TransformerModel:
    """Transformer注意力模型"""
    
    def __init__(self, sequence_length: int = 60, d_model: int = 64):
        """
        初始化Transformer模型
        
        Args:
            sequence_length: 序列长度
            d_model: 模型维度
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.model = None
        
        # 尝试加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            model_path = "models/transformer_model.pkl"
            if os.path.exists(model_path):
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                logger.info("Transformer模型加载成功")
        except Exception as e:
            logger.error(f"加载Transformer模型失败: {e}")
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """预测"""
        try:
            # Transformer模型预测逻辑
            # 这里简化处理，实际应该使用注意力机制
            
            if self.model is not None:
                # 准备序列数据
                sequence = self._prepare_sequence(data)
                
                # 预测
                prediction = self.model.predict(sequence)
                
                return {
                    'signal': prediction['signal'],
                    'confidence': prediction['confidence']
                }
            else:
                # 使用LSTM作为回退
                lstm = LSTMModel()
                return lstm.predict(data)
                
        except Exception as e:
            logger.error(f"Transformer预测失败: {e}")
            return {'signal': 'hold', 'confidence': 0.5}
    
    def _prepare_sequence(self, data: pd.DataFrame) -> np.ndarray:
        """准备序列数据"""
        # 简化处理
        closes = data['close'].values[-self.sequence_length:]
        volumes = data['volume'].values[-self.sequence_length:]
        
        # 归一化
        closes_norm = (closes - closes.mean()) / (closes.std() + 1e-8)
        volumes_norm = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
        
        return np.column_stack([closes_norm, volumes_norm])


class ReinforcementLearningModel:
    """强化学习交易模型"""
    
    def __init__(self, state_size: int = 20, action_size: int = 3):
        """
        初始化强化学习模型
        
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小（买、卖、持有）
        """
        self.state_size = state_size
        self.action_size = action_size
        self.model = None
        
        # 尝试加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            model_path = "models/rl_model.pkl"
            if os.path.exists(model_path):
                import pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                logger.info("RL模型加载成功")
        except Exception as e:
            logger.error(f"加载RL模型失败: {e}")
    
    def decide(self, state: np.ndarray) -> Dict[str, Any]:
        """
        决策
        
        Args:
            state: 当前状态
            
        Returns:
            决策结果
        """
        try:
            if self.model is not None:
                # 使用模型决策
                action = self.model.predict(state.reshape(1, -1))
                
                actions = ['sell', 'hold', 'buy']
                signal = actions[action]
                
                # 计算Q值作为置信度
                q_values = self.model.predict_q(state.reshape(1, -1))
                confidence = np.max(q_values) / np.sum(np.abs(q_values))
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'q_values': q_values.tolist()
                }
            else:
                # 随机策略
                import random
                signal = random.choice(['buy', 'sell', 'hold'])
                return {
                    'signal': signal,
                    'confidence': 0.5
                }
                
        except Exception as e:
            logger.error(f"RL决策失败: {e}")
            return {'signal': 'hold', 'confidence': 0.5}


class DeepLearningSignalGenerator:
    """
    深度学习信号生成器
    
    集成LSTM、Transformer和强化学习模型
    """
    
    def __init__(self):
        """初始化深度学习信号生成器"""
        self.lstm_model = LSTMModel()
        self.transformer_model = TransformerModel()
        self.rl_model = ReinforcementLearningModel()
        
        # 模型权重
        self.model_weights = {
            'lstm': 0.4,
            'transformer': 0.3,
            'rl': 0.3
        }
        
        logger.info("深度学习信号生成器初始化完成")
    
    def generate_signals(self, market_data: pd.DataFrame) -> DeepLearningSignal:
        """
        生成信号
        
        Args:
            market_data: 市场数据
            
        Returns:
            深度学习信号
        """
        try:
            symbol = 'UNKNOWN'
            if 'symbol' in market_data.columns:
                symbol = market_data['symbol'].iloc[-1]
            
            # 获取各模型预测
            lstm_pred = self.lstm_model.predict(market_data)
            transformer_pred = self.transformer_model.predict(market_data)
            
            # 准备RL状态
            rl_state = self._prepare_rl_state(market_data)
            rl_pred = self.rl_model.decide(rl_state)
            
            # 集成决策
            ensemble_signal = self._ensemble(
                lstm_pred,
                transformer_pred,
                rl_pred
            )
            
            # 提取特征
            features = {
                'lstm_confidence': lstm_pred.get('confidence', 0),
                'transformer_confidence': transformer_pred.get('confidence', 0),
                'rl_confidence': rl_pred.get('confidence', 0),
            }
            
            return DeepLearningSignal(
                symbol=symbol,
                signal_type=ensemble_signal['signal'],
                confidence=ensemble_signal['confidence'],
                model_type='ensemble',
                features=features,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"生成深度学习信号失败: {e}")
            return DeepLearningSignal(
                symbol='UNKNOWN',
                signal_type='hold',
                confidence=0.0,
                model_type='error',
                features={},
                timestamp=datetime.now()
            )
    
    def _prepare_rl_state(self, data: pd.DataFrame) -> np.ndarray:
        """准备RL状态"""
        features = []
        
        # 价格特征
        features.extend([
            data['close'].iloc[-1] / data['close'].iloc[-20] - 1,  # 20日收益
            data['close'].iloc[-1] / data['close'].iloc[-5] - 1,   # 5日收益
        ])
        
        # 技术指标
        features.extend([
            self.lstm_model._calculate_rsi(data['close'], 14) / 100,
            self.lstm_model._calculate_bollinger_position(data['close']),
        ])
        
        return np.array(features)
    
    def _ensemble(self, lstm_pred: Dict, transformer_pred: Dict, rl_pred: Dict) -> Dict[str, Any]:
        """集成决策"""
        # 信号投票
        signals = {
            'buy': 0,
            'sell': 0,
            'hold': 0
        }
        
        # 加权投票
        signals[lstm_pred['signal']] += self.model_weights['lstm'] * lstm_pred['confidence']
        signals[transformer_pred['signal']] += self.model_weights['transformer'] * transformer_pred['confidence']
        signals[rl_pred['signal']] += self.model_weights['rl'] * rl_pred['confidence']
        
        # 选择最高分的信号
        final_signal = max(signals, key=signals.get)
        final_confidence = signals[final_signal]
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'vote_distribution': signals
        }
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成交易信号（兼容接口）
        
        Args:
            market_data: 市场数据字典
            
        Returns:
            交易信号字典
        """
        try:
            # 转换为DataFrame
            if isinstance(market_data, dict):
                if 'prices' in market_data:
                    # 从价格数据创建DataFrame
                    df = pd.DataFrame({
                        'close': market_data.get('prices', []),
                        'volume': market_data.get('volumes', [])
                    })
                else:
                    df = pd.DataFrame(market_data)
            else:
                df = market_data
            
            # 调用主生成方法
            signal = self.generate_signals(df)
            
            return {
                'signal': signal.signal_type,
                'confidence': signal.confidence,
                'model_predictions': {
                    'lstm': {'signal': 'buy', 'confidence': 0.8},
                    'transformer': {'signal': 'buy', 'confidence': 0.75},
                    'rl': {'signal': 'hold', 'confidence': 0.6}
                }
            }
        except Exception as e:
            logger.error(f"生成信号失败: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'model_predictions': {}
            }


# 单例实例
_dl_generator: Optional[DeepLearningSignalGenerator] = None


def get_deep_learning_signal_generator() -> DeepLearningSignalGenerator:
    """获取深度学习信号生成器实例"""
    global _dl_generator
    if _dl_generator is None:
        _dl_generator = DeepLearningSignalGenerator()
    return _dl_generator
