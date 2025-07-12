"""RQA2025模型模块 - 包含预测模型和评估工具

主要类:
- BaseModel: 模型基类
- ModelManager: 模型管理核心类
- ModelLSTM: LSTM模型实现
- ModelMonitor: 模型监控工具

使用示例:
    from src.models import ModelManager, BaseModel

    # 初始化模型管理器
    manager = ModelManager()

版本历史:
- v1.0 (2024-03-01): 初始版本
- v1.1 (2024-04-15): 重构模型结构
"""

from .base_model import BaseModel
from .model_manager import ModelManager
from .model_lstm import LSTMModelWrapper as ModelLSTM
from .model_monitor import ModelMonitor

__all__ = [
    'BaseModel',
    'ModelManager',
    'ModelLSTM', 
    'ModelMonitor'
]
