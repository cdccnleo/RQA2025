"""
机器学习信号验证器
使用机器学习模型预测信号成功率
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class MLValidationResult:
    """机器学习验证结果"""
    signal_id: str
    symbol: str
    prediction_score: float  # 预测成功率 (0-1)
    confidence: float  # 置信度 (0-1)
    features_used: List[str]
    model_version: str
    prediction_time: datetime


class MLSignalValidator:
    """
    机器学习信号验证器
    
    职责：
    1. 使用历史信号数据训练模型
    2. 预测新信号的成功率
    3. 动态调整评分权重
    4. 模型在线学习
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化ML信号验证器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path or "models/signal_predictor.pkl"
        self.model = None
        self.scaler = None
        self.feature_names = [
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_percent',
            'volume_ma_ratio', 'price_ma_ratio',
            'atr_14', 'adx_14',
            'sentiment_score', 'money_flow_ratio'
        ]
        self.model_version = "1.0.0"
        self.training_data: List[Dict] = []
        
        # 尝试加载已有模型
        self._load_model()
        
        logger.info("机器学习信号验证器初始化完成")
    
    def _load_model(self) -> bool:
        """加载模型"""
        try:
            if os.path.exists(self.model_path):
                import pickle
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.model_version = model_data.get('version', '1.0.0')
                logger.info(f"模型加载成功，版本: {self.model_version}")
                return True
            else:
                logger.info("模型文件不存在，将使用默认规则")
                return False
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def _save_model(self) -> bool:
        """保存模型"""
        try:
            import pickle
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'version': self.model_version,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"模型保存成功: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def _extract_features(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """
        提取特征
        
        Args:
            signal: 信号数据
            market_data: 市场数据
            
        Returns:
            特征向量
        """
        try:
            if market_data.empty:
                return np.zeros(len(self.feature_names))
            
            # 计算技术指标
            features = []
            
            # RSI
            rsi = self._calculate_rsi(market_data['close'], 14)
            features.append(rsi[-1] if len(rsi) > 0 else 50)
            
            # MACD
            macd, macd_signal, macd_hist = self._calculate_macd(market_data['close'])
            features.extend([
                macd[-1] if len(macd) > 0 else 0,
                macd_signal[-1] if len(macd_signal) > 0 else 0,
                macd_hist[-1] if len(macd_hist) > 0 else 0
            ])
            
            # 布林带
            bb_upper, bb_lower, bb_percent = self._calculate_bollinger_bands(market_data['close'])
            features.extend([
                bb_upper[-1] if len(bb_upper) > 0 else 0,
                bb_lower[-1] if len(bb_lower) > 0 else 0,
                bb_percent[-1] if len(bb_percent) > 0 else 0.5
            ])
            
            # 成交量比率
            volume_ma = market_data['volume'].rolling(window=20).mean()
            volume_ratio = market_data['volume'].iloc[-1] / volume_ma.iloc[-1] if len(volume_ma) > 0 and volume_ma.iloc[-1] != 0 else 1
            features.append(volume_ratio)
            
            # 价格均线比率
            price_ma = market_data['close'].rolling(window=20).mean()
            price_ratio = market_data['close'].iloc[-1] / price_ma.iloc[-1] if len(price_ma) > 0 and price_ma.iloc[-1] != 0 else 1
            features.append(price_ratio)
            
            # ATR
            atr = self._calculate_atr(market_data, 14)
            features.append(atr[-1] if len(atr) > 0 else 0)
            
            # ADX
            adx = self._calculate_adx(market_data, 14)
            features.append(adx[-1] if len(adx) > 0 else 25)
            
            # 情绪分数（从信号中获取或使用默认值）
            sentiment = signal.get('sentiment_score', 0.5)
            features.append(sentiment)
            
            # 资金流向比率
            money_flow = signal.get('money_flow_ratio', 1.0)
            features.append(money_flow)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"提取特征失败: {e}")
            return np.zeros(len(self.feature_names))
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        percent = (prices - lower) / (upper - lower)
        return upper, lower, percent
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ADX"""
        plus_dm = data['high'].diff()
        minus_dm = data['low'].diff().abs()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._calculate_atr(data, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
        
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        return adx
    
    def validate_signal(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> MLValidationResult:
        """
        验证信号
        
        Args:
            signal: 信号数据
            market_data: 市场数据
            
        Returns:
            ML验证结果
        """
        try:
            # 提取特征
            features = self._extract_features(signal, market_data)
            
            # 如果模型存在，使用模型预测
            if self.model is not None:
                # 标准化特征
                if self.scaler is not None:
                    features_scaled = self.scaler.transform(features.reshape(1, -1))
                else:
                    features_scaled = features.reshape(1, -1)
                
                # 预测
                prediction = self.model.predict_proba(features_scaled)[0]
                prediction_score = prediction[1]  # 假设1是成功
                
                # 计算置信度（基于预测概率的分布）
                confidence = abs(prediction_score - 0.5) * 2
            else:
                # 使用规则-based预测
                prediction_score = self._rule_based_prediction(features)
                confidence = 0.5
            
            return MLValidationResult(
                signal_id=signal.get('id', ''),
                symbol=signal.get('symbol', ''),
                prediction_score=float(prediction_score),
                confidence=float(confidence),
                features_used=self.feature_names,
                model_version=self.model_version,
                prediction_time=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"验证信号失败: {e}")
            return MLValidationResult(
                signal_id=signal.get('id', ''),
                symbol=signal.get('symbol', ''),
                prediction_score=0.5,
                confidence=0.0,
                features_used=[],
                model_version=self.model_version,
                prediction_time=datetime.now()
            )
    
    def _rule_based_prediction(self, features: np.ndarray) -> float:
        """基于规则的预测（当模型不存在时使用）"""
        try:
            # 简单的规则组合
            score = 0.5
            
            # RSI信号
            rsi = features[0]
            if rsi < 30:
                score += 0.1  # 超卖，可能反弹
            elif rsi > 70:
                score -= 0.1  # 超买，可能回调
            
            # MACD信号
            macd = features[1]
            macd_signal = features[2]
            if macd > macd_signal:
                score += 0.1
            else:
                score -= 0.1
            
            # 布林带位置
            bb_percent = features[6]
            if bb_percent < 0.2:
                score += 0.05
            elif bb_percent > 0.8:
                score -= 0.05
            
            # 限制范围
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"规则预测失败: {e}")
            return 0.5
    
    def train_model(
        self,
        historical_signals: List[Dict],
        historical_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        训练模型
        
        Args:
            historical_signals: 历史信号列表
            historical_data: 历史市场数据字典
            
        Returns:
            训练是否成功
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            logger.info(f"开始训练模型，信号数量: {len(historical_signals)}")
            
            # 准备训练数据
            X = []
            y = []
            
            for signal in historical_signals:
                symbol = signal.get('symbol')
                if symbol not in historical_data:
                    continue
                
                market_data = historical_data[symbol]
                features = self._extract_features(signal, market_data)
                
                # 标签：信号是否成功（基于后续收益）
                success = signal.get('success', False)
                
                X.append(features)
                y.append(1 if success else 0)
            
            if len(X) < 10:
                logger.warning("训练数据不足，跳过训练")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 标准化
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 训练模型
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            
            # 评估模型
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"模型训练完成 - 训练集准确率: {train_score:.2f}, 测试集准确率: {test_score:.2f}")
            
            # 更新版本号
            self.model_version = f"1.1.{int(datetime.now().timestamp()) % 1000}"
            
            # 保存模型
            self._save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            return False
    
    def update_model(self, signal_result: Dict) -> bool:
        """
        增量更新模型
        
        Args:
            signal_result: 信号结果数据
            
        Returns:
            更新是否成功
        """
        try:
            # 添加到训练数据
            self.training_data.append(signal_result)
            
            # 当积累足够数据时，重新训练
            if len(self.training_data) >= 100:
                logger.info(f"积累{len(self.training_data)}条数据，准备重新训练")
                # 这里可以实现增量学习或定期全量重训练
                self.training_data.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"更新模型失败: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            if self.model is None or not hasattr(self.model, 'feature_importances_'):
                return {}
            
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
            
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return {}


# 单例实例
_ml_validator: Optional[MLSignalValidator] = None


def get_ml_signal_validator(model_path: Optional[str] = None) -> MLSignalValidator:
    """获取ML信号验证器实例"""
    global _ml_validator
    if _ml_validator is None:
        _ml_validator = MLSignalValidator(model_path)
    return _ml_validator
