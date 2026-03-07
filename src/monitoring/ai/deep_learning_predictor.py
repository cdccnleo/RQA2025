"""
RQA2025 深度学习预测器模块

此文件作为主入口，导入并导出各个模块的组件。

重构说明(2025-11-01):
- dl_models.py: LSTM、Autoencoder等模型定义
- dl_optimizer.py: GPU管理、模型优化、批量优化
- dl_predictor_core.py: DeepLearningPredictor主类

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging

# 导入所有组件
from .dl_models import (
    TimeSeriesDataset,
    LSTMPredictor,
    Autoencoder
)
from .dl_optimizer import (
    GPUResourceManager,
    AIModelOptimizer,
    DynamicBatchOptimizer
)
from .dl_predictor_core import (
    ModelCacheManager,
    DeepLearningPredictor
)

logger = logging.getLogger(__name__)


# 导出所有组件
__all__ = [
    # 数据集
    'TimeSeriesDataset',
    # 模型
    'LSTMPredictor',
    'Autoencoder',
    # 优化器
    'GPUResourceManager',
    'AIModelOptimizer',
    'DynamicBatchOptimizer',
    # 预测器
    'ModelCacheManager',
    'DeepLearningPredictor'
]


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)

    # 创建预测器
    predictor = DeepLearningPredictor()

    # 生成示例数据
    import numpy as np
    data = np.sin(np.linspace(0, 100, 1000))

    # 训练模型
    result = predictor.train_lstm(data)
    logger.info(f"训练结果: {result}")

    # 预测
    predictions = predictor.predict(data[-100:], steps=10)
    logger.info(f"预测值: {predictions}")
