"""数据标准化和反标准化工具

提供数据标准化和反标准化功能，用于机器学习模型训练和预测。

函数:
- normalize_data: 数据标准化
- denormalize_data: 数据反标准化
"""

def normalize_data(data, mean=None, std=None):
    """标准化数据

    参数:
        data: 要标准化的数据(numpy数组或pandas DataFrame)
        mean: 均值(可选)，如果未提供则计算数据的均值
        std: 标准差(可选)，如果未提供则计算数据的标准差

    返回:
        标准化后的数据和使用的均值、标准差
    """
    import numpy as np

    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
        std[std == 0] = 1.0  # 避免除以0

    normalized = (data - mean) / std
    return normalized, mean, std

def denormalize_data(normalized_data, mean, std):
    """反标准化数据

    参数:
        normalized_data: 标准化后的数据
        mean: 标准化时使用的均值
        std: 标准化时使用的标准差

    返回:
        原始数据
    """
    return normalized_data * std + mean

__all__ = ['normalize_data', 'denormalize_data']
