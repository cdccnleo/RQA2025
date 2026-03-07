"""
真实模型训练器
基于scikit-learn、XGBoost、LightGBM实现真实的模型训练
支持量化交易特征工程和模型训练
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pickle
import json

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程类 - 计算量化交易技术指标"""
    
    @staticmethod
    def calculate_ma(prices: pd.Series, window: int) -> pd.Series:
        """计算移动平均线"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """计算指数移动平均线"""
        return prices.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, ma, lower
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """计算ATR（平均真实波幅）"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算OBV（能量潮）"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @classmethod
    def create_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建量化交易特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含特征的DataFrame
        """
        features = df.copy()
        
        # 清理数据：移除包含NaN的行，并确保数值列为float64类型
        features = features.dropna(subset=['close'])
        
        # 转换所有数值列为float64类型，避免类型混合问题
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce').astype('float64')
        
        # 价格特征
        features['returns'] = features['close'].pct_change()
        close_prices = features['close'].replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        log_vals = close_prices / close_prices.shift(1)
        log_vals = log_vals.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        features['log_returns'] = np.log(log_vals.clip(lower=1e-10))
        
        # 移动平均线
        for window in [5, 10, 20, 60]:
            features[f'ma_{window}'] = cls.calculate_ma(features['close'], window)
            features[f'ema_{window}'] = cls.calculate_ema(features['close'], window)
            features[f'ma_ratio_{window}'] = features['close'] / features[f'ma_{window}']
        
        # MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = \
            cls.calculate_macd(features['close'])
        
        # RSI
        features['rsi'] = cls.calculate_rsi(features['close'])
        features['rsi_6'] = cls.calculate_rsi(features['close'], 6)
        
        # 布林带
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = \
            cls.calculate_bollinger_bands(features['close'])
        features['bb_position'] = (features['close'] - features['bb_lower']) / \
                                  (features['bb_upper'] - features['bb_lower'])
        
        # ATR
        features['atr'] = cls.calculate_atr(features['high'], features['low'], features['close'])
        features['atr_ratio'] = features['atr'] / features['close']
        
        # OBV
        features['obv'] = cls.calculate_obv(features['close'], features['volume'])
        
        # 波动率
        features['volatility_20'] = features['returns'].rolling(window=20).std()
        
        # 价格动量
        for window in [5, 10, 20]:
            features[f'momentum_{window}'] = features['close'] / features['close'].shift(window) - 1
        
        # 成交量特征
        features['volume_ma_20'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma_20']
        
        # 删除NaN值
        features = features.dropna()
        
        return features
    
    @classmethod
    def create_labels(cls, df: pd.DataFrame, forward_periods: int = 5, threshold: float = 0.01) -> pd.Series:
        """
        创建标签（未来收益率方向）
        
        Args:
            df: 包含价格数据的DataFrame
            forward_periods: 前瞻期数
            threshold: 阈值
            
        Returns:
            标签Series (1: 上涨, 0: 下跌/持平)
        """
        future_returns = df['close'].shift(-forward_periods) / df['close'] - 1
        labels = (future_returns > threshold).astype(int)
        return labels


class RealModelTrainer:
    """真实模型训练器"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        
    def prepare_data(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        准备训练数据
        
        Args:
            data: 原始OHLCV数据
            feature_columns: 指定的特征列，None则使用所有特征
            
        Returns:
            (特征DataFrame, 标签Series, 特征列名列表)
        """
        # 创建特征
        features = self.feature_engineer.create_features(data)
        
        # 创建标签
        labels = self.feature_engineer.create_labels(data.loc[features.index])
        
        # 对齐数据 - 移除最后5行（没有标签）
        features = features.iloc[:-5]
        labels = labels.iloc[:-5]
        
        # 重置索引以确保一致性
        features = features.reset_index(drop=True)
        labels = labels.reset_index(drop=True)
        
        # 选择特征列
        if feature_columns is None:
            # 自动选择数值型特征列
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            feature_columns = [col for col in features.columns 
                             if col not in exclude_cols]
        
        X = features[feature_columns].copy()
        y = labels.copy()
        
        # 清理数据：移除包含NaN或Inf的行
        X = X.replace([np.inf, -np.inf], np.nan)
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)
        
        return X, y, feature_columns
    
    def train_model(self, model_type: str, data: pd.DataFrame, 
                   config: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            model_type: 模型类型 (RandomForest, XGBoost, LightGBM, LSTM, CNN, Transformer)
            data: 训练数据
            config: 训练配置
            progress_callback: 进度回调函数
            
        Returns:
            训练结果字典
        """
        try:
            logger.info(f"开始训练 {model_type} 模型")
            
            # 准备数据
            X, y, feature_columns = self.prepare_data(data, config.get('feature_columns'))
            
            if len(X) < 100:
                raise ValueError(f"训练数据不足: {len(X)} 条，需要至少100条")
            
            # 分割数据集
            train_size = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
            
            # 检查是否为深度学习模型
            deep_learning_models = ['LSTM', 'CNN', 'Transformer']
            
            if model_type in deep_learning_models:
                # 使用PyTorch深度学习模型
                return self._train_deep_learning_model(
                    model_type, X_train, y_train, X_test, y_test, 
                    feature_columns, config, progress_callback
                )
            else:
                # 使用传统机器学习模型
                return self._train_traditional_model(
                    model_type, X_train, y_train, X_test, y_test,
                    feature_columns, config, progress_callback
                )
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'model_type': model_type
            }
    
    def _train_traditional_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series, feature_columns: list,
                                config: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """训练传统机器学习模型"""
        
        # 创建模型
        model = self._create_model(model_type, config)
        
        if model is None:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 训练模型
        if progress_callback:
            progress_callback(10, "模型初始化完成")
        
        model.fit(X_train, y_train)
        
        if progress_callback:
            progress_callback(50, "模型训练完成")
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        if progress_callback:
            progress_callback(70, "预测完成")
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                pass
        
        if progress_callback:
            progress_callback(90, "指标计算完成")
        
        # 保存模型
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = {
            'model': model,
            'model_type': model_type,
            'feature_columns': feature_columns,
            'config': config,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
        }
        
        if progress_callback:
            progress_callback(100, "训练完成")
        
        result = {
            'model_id': model_id,
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'feature_columns': feature_columns,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'status': 'completed'
        }
        
        logger.info(f"传统模型训练完成: {model_id}, 准确率: {accuracy:.4f}")
        return result
    
    def _train_deep_learning_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series, feature_columns: list,
                                  config: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """训练深度学习模型"""
        
        from src.ml.pytorch_models import train_lstm_model, train_cnn_model, train_transformer_model
        
        if progress_callback:
            progress_callback(10, "深度学习模型初始化")
        
        # 根据模型类型选择训练函数
        if model_type == 'LSTM':
            model, training_info = train_lstm_model(
                X_train, y_train, X_test, y_test, config, progress_callback
            )
        elif model_type == 'CNN':
            model, training_info = train_cnn_model(
                X_train, y_train, X_test, y_test, config, progress_callback
            )
        elif model_type == 'Transformer':
            model, training_info = train_transformer_model(
                X_train, y_train, X_test, y_test, config, progress_callback
            )
        else:
            raise ValueError(f"不支持的深度学习模型类型: {model_type}")
        
        if progress_callback:
            progress_callback(90, "指标计算完成")
        
        # 保存模型
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = {
            'model': model,
            'model_type': model_type,
            'feature_columns': feature_columns,
            'config': config,
            'metrics': {
                'accuracy': training_info['accuracy'],
                'precision': training_info['precision'],
                'recall': training_info['recall'],
                'f1': training_info['f1'],
                'roc_auc': training_info['roc_auc']
            },
            'training_info': training_info
        }
        
        if progress_callback:
            progress_callback(100, "训练完成")
        
        result = {
            'model_id': model_id,
            'model_type': model_type,
            'accuracy': training_info['accuracy'],
            'precision': training_info['precision'],
            'recall': training_info['recall'],
            'f1': training_info['f1'],
            'roc_auc': training_info['roc_auc'],
            'feature_columns': feature_columns,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'status': 'completed',
            'device': training_info.get('device', 'cpu'),
            'epochs_trained': training_info.get('epochs_trained', 0)
        }
        
        logger.info(f"深度学习模型训练完成: {model_id}, 准确率: {training_info['accuracy']:.4f}")
        return result
    
    def _create_model(self, model_type: str, config: Dict[str, Any]):
        """创建模型实例"""
        
        if model_type == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 10),
                min_samples_split=config.get('min_samples_split', 2),
                random_state=42,
                n_jobs=-1
            )
        
        elif model_type == 'XGBoost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth', 6),
                    learning_rate=config.get('learning_rate', 0.1),
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            except ImportError:
                logger.warning("XGBoost未安装，使用RandomForest替代")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        elif model_type == 'LightGBM':
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth', -1),
                    learning_rate=config.get('learning_rate', 0.1),
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            except ImportError:
                logger.warning("LightGBM未安装，使用RandomForest替代")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        elif model_type == 'LSTM':
            # LSTM需要PyTorch，暂不支持，使用RandomForest替代
            logger.warning("LSTM需要PyTorch，暂不支持，使用RandomForest替代")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        elif model_type == 'CNN' or model_type == 'Transformer':
            # CNN和Transformer需要PyTorch，暂不支持
            logger.warning(f"{model_type}需要PyTorch，暂不支持，使用RandomForest替代")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        else:
            logger.warning(f"未知的模型类型: {model_type}，使用RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    def predict(self, model_id: str, data: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            model_id: 模型ID
            data: 预测数据
            
        Returns:
            预测结果
        """
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        model_info = self.models[model_id]
        model = model_info['model']
        feature_columns = model_info['feature_columns']
        
        # 准备特征
        features = self.feature_engineer.create_features(data)
        X = features[feature_columns]
        
        # 预测
        predictions = model.predict(X)
        return predictions
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """获取模型信息"""
        if model_id not in self.models:
            return None
        
        model_info = self.models[model_id]
        return {
            'model_id': model_id,
            'model_type': model_info['model_type'],
            'feature_columns': model_info['feature_columns'],
            'metrics': model_info['metrics'],
            'config': model_info['config']
        }
    
    def save_model(self, model_id: str, filepath: str):
        """保存模型到文件"""
        if model_id not in self.models:
            raise ValueError(f"模型不存在: {model_id}")
        
        model_info = self.models[model_id]
        with open(filepath, 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"模型已保存: {filepath}")
    
    def load_model(self, filepath: str) -> str:
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            model_info = pickle.load(f)
        
        model_id = f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = model_info
        
        logger.info(f"模型已加载: {model_id}")
        return model_id


# 全局训练器实例
_real_model_trainer = None

def get_real_model_trainer() -> RealModelTrainer:
    """获取全局真实模型训练器实例"""
    global _real_model_trainer
    if _real_model_trainer is None:
        _real_model_trainer = RealModelTrainer()
    return _real_model_trainer
