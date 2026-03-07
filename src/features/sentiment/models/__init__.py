"""
情感分析模型模块
"""

from .sentiment_model import SentimentModel


def load_pretrained_model(model_path: str = None) -> SentimentModel:
    """
    加载预训练模型

    Args:
        model_path: 模型路径

    Returns:
        情感分析模型实例
    """
    config = {
        'model_type': 'sentiment',
        'model_path': model_path
    }
    return SentimentModel(config)


__all__ = ['SentimentModel', 'load_pretrained_model']
