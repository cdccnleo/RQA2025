#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单调试测试
"""

import numpy as np
import pandas as pd
from src.ml.core.ml_service import MLService


def test_simple_training():
    """简单的训练测试"""
    service = MLService()
    service.start()

    # 创建简单的数据
    X = np.random.randn(20, 2)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(20) * 0.1

    data = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1],
        'target': y
    })

    print("Data shape:", data.shape)
    print("Data columns:", data.columns.tolist())
    print("Data head:")
    print(data.head())

    # 训练模型
    config = {"algorithm": "linear_regression", "params": {}}
    result = service.train_model("simple_model", data, config)

    print("Training result:", result)

    if result:
        # 测试预测
        test_data = pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})
        prediction = service.predict(test_data)
        print("Prediction:", prediction)

        # 获取性能
        performance = service.get_model_performance("simple_model")
        print("Performance:", performance)

    service.stop()


if __name__ == "__main__":
    test_simple_training()
