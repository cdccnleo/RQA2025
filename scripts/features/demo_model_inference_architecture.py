#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理架构演示脚本

展示重构后的模型推理架构，包括：
- 正确的层间依赖关系
- 特征层到模型层的数据流
- 模型推理功能
"""

from src.models.inference import ModelInferenceManager
from src.features.processors.gpu.gpu_technical_processor import GPUTechnicalProcessor
from src.utils.logger import get_logger
import sys
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


logger = get_logger(__name__)


class FeatureEngineer:
    """特征工程器 - 模拟特征层功能"""

    def __init__(self):
        self.logger = logger
        self.gpu_processor = GPUTechnicalProcessor()

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成特征 - 模拟特征层的主要功能"""
        try:
            self.logger.info(f"开始生成特征，数据形状: {data.shape}")

            # 模拟特征工程过程
            features = pd.DataFrame()

            # 1. 技术指标特征
            if 'close' in data.columns:
                # 使用GPU处理器计算技术指标
                technical_features = self._calculate_technical_features(data)
                features = pd.concat([features, technical_features], axis=1)

            # 2. 统计特征
            if 'volume' in data.columns:
                volume_features = self._calculate_volume_features(data)
                features = pd.concat([features, volume_features], axis=1)

            # 3. 价格特征
            price_features = self._calculate_price_features(data)
            features = pd.concat([features, price_features], axis=1)

            self.logger.info(f"特征生成完成，特征形状: {features.shape}")
            return features

        except Exception as e:
            self.logger.error(f"特征生成失败: {e}")
            return pd.DataFrame()

    def _calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        try:
            features = pd.DataFrame()

            if 'close' in data.columns:
                # 使用GPU处理器计算技术指标
                close_prices = data['close'].values

                # SMA
                sma_20 = self.gpu_processor.calculate_sma_gpu(close_prices, 20)
                features['sma_20'] = sma_20

                # EMA
                ema_20 = self.gpu_processor.calculate_ema_gpu(close_prices, 20)
                features['ema_20'] = ema_20

                # RSI
                rsi_14 = self.gpu_processor.calculate_rsi_gpu(close_prices, 14)
                features['rsi_14'] = rsi_14

                # MACD
                macd_result = self.gpu_processor.calculate_macd_gpu(close_prices)
                features['macd'] = macd_result['macd']
                features['macd_signal'] = macd_result['signal']
                features['macd_histogram'] = macd_result['histogram']

                # Bollinger Bands
                bb_result = self.gpu_processor.calculate_bollinger_bands_gpu(close_prices, 20)
                features['bb_upper'] = bb_result['upper']
                features['bb_middle'] = bb_result['middle']
                features['bb_lower'] = bb_result['lower']

                # ATR
                if 'high' in data.columns and 'low' in data.columns:
                    high_prices = data['high'].values
                    low_prices = data['low'].values
                    atr_14 = self.gpu_processor.calculate_atr_gpu(
                        high_prices, low_prices, close_prices, 14)
                    features['atr_14'] = atr_14

            return features

        except Exception as e:
            self.logger.error(f"计算技术指标特征失败: {e}")
            return pd.DataFrame()

    def _calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算成交量特征"""
        try:
            features = pd.DataFrame()

            if 'volume' in data.columns:
                volume = data['volume'].values

                # 成交量移动平均
                features['volume_sma_20'] = np.convolve(volume, np.ones(20)/20, mode='valid')
                features['volume_sma_20'] = np.concatenate(
                    [np.full(19, np.nan), features['volume_sma_20']])

                # 成交量比率
                features['volume_ratio'] = volume / features['volume_sma_20']

                # 成交量变化率
                features['volume_change'] = np.diff(volume, prepend=volume[0]) / volume

            return features

        except Exception as e:
            self.logger.error(f"计算成交量特征失败: {e}")
            return pd.DataFrame()

    def _calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算价格特征"""
        try:
            features = pd.DataFrame()

            if 'close' in data.columns:
                close = data['close'].values

                # 价格变化率
                features['price_change'] = np.diff(close, prepend=close[0]) / close

                # 价格波动率
                features['price_volatility'] = np.abs(features['price_change'])

                # 价格位置（相对于最高最低价）
                if 'high' in data.columns and 'low' in data.columns:
                    high = data['high'].values
                    low = data['low'].values
                    features['price_position'] = (close - low) / (high - low)

            return features

        except Exception as e:
            self.logger.error(f"计算价格特征失败: {e}")
            return pd.DataFrame()

    def get_feature_metadata(self) -> Dict[str, Any]:
        """获取特征元数据"""
        return {
            'feature_count': 15,  # 示例特征数量
            'feature_types': ['technical', 'volume', 'price'],
            'gpu_accelerated': True,
            'last_updated': time.time()
        }


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)

    # 生成时间序列数据
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

    # 生成价格数据
    base_price = 100
    price_changes = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]

    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    volume = np.random.randint(1000000, 10000000, n_samples)

    data = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })

    return data


def create_mock_model():
    """创建模拟模型"""
    try:
        import torch.nn as nn

        class MockModel(nn.Module):
            def __init__(self, input_size=15, output_size=1):
                super(MockModel, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, output_size)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        model = MockModel()
        model.eval()
        return model

    except ImportError:
        logger.warning("PyTorch不可用，使用模拟模型")
        return None


def save_mock_model(model, model_path: str):
    """保存模拟模型"""
    try:
        if model is not None:
            import torch
            torch.save(model, model_path)
            logger.info(f"模型已保存到: {model_path}")
        else:
            # 创建模拟模型文件
            with open(model_path, 'wb') as f:
                import pickle
                pickle.dump({'mock': True, 'timestamp': time.time()}, f)
            logger.info(f"模拟模型文件已创建: {model_path}")
    except Exception as e:
        logger.error(f"保存模型失败: {e}")


def main():
    """主函数"""
    logger.info("开始模型推理架构演示...")

    # 1. 创建示例数据
    logger.info("1. 创建示例数据...")
    sample_data = create_sample_data(1000)
    logger.info(f"示例数据创建完成，形状: {sample_data.shape}")

    # 2. 创建特征工程器（特征层）
    logger.info("2. 创建特征工程器...")
    feature_engineer = FeatureEngineer()

    # 3. 生成特征
    logger.info("3. 生成特征...")
    features = feature_engineer.generate_features(sample_data)
    logger.info(f"特征生成完成，特征形状: {features.shape}")
    logger.info(f"特征列: {list(features.columns)}")

    # 4. 创建模型推理管理器（模型层）
    logger.info("4. 创建模型推理管理器...")
    inference_manager = ModelInferenceManager({
        'use_gpu': True,
        'batch_size': 100,
        'enable_monitoring': True
    })

    # 5. 创建模拟模型
    logger.info("5. 创建模拟模型...")
    mock_model = create_mock_model()
    model_path = "models/mock_model.pth"
    save_mock_model(mock_model, model_path)

    # 6. 加载模型
    logger.info("6. 加载模型...")
    success = inference_manager.load_model("mock_model", model_path, "pytorch")
    if success:
        logger.info("模型加载成功")
    else:
        logger.warning("模型加载失败，使用模拟推理")

    # 7. 执行推理（使用特征层数据）
    logger.info("7. 执行推理...")
    if not features.empty:
        # 使用特征数据进行推理
        predictions = inference_manager.inference("mock_model", features.values)
        logger.info(f"推理完成，预测结果形状: {predictions.shape}")

        # 8. 展示完整的特征层到模型层流程
        logger.info("8. 展示完整流程...")
        raw_data = sample_data[['open', 'high', 'low', 'close', 'volume']]
        complete_predictions = inference_manager.predict_with_features(
            "mock_model", raw_data, feature_engineer)
        logger.info(f"完整流程预测结果形状: {complete_predictions.shape}")

        # 9. 获取统计信息
        logger.info("9. 获取统计信息...")
        inference_stats = inference_manager.get_inference_stats()
        logger.info(f"推理统计: {inference_stats}")

        # 10. 展示架构优势
        logger.info("10. 架构优势总结...")
        logger.info("✅ 职责分离清晰：特征层专注于特征工程，模型层专注于推理")
        logger.info("✅ 依赖关系正确：模型层依赖特征层，单向依赖")
        logger.info("✅ 数据流清晰：原始数据 → 特征层 → 特征数据 → 模型层 → 预测结果")
        logger.info("✅ 模块化设计：各模块独立，便于测试和维护")
        logger.info("✅ 扩展性好：可以轻松添加新的特征处理器和推理引擎")

    else:
        logger.error("特征生成失败，无法执行推理")

    logger.info("模型推理架构演示完成")


if __name__ == "__main__":
    main()
