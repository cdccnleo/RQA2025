"""
层接口模块
提供层间通信接口
"""

from typing import Dict, Any, Protocol


class LayerInterface:
    """层接口基类"""

    def __init__(self, layer_name: str):
        self.layer_name = layer_name

    def communicate_up(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """向上一层通信"""
        return {"status": "success", "layer": self.layer_name}

    def communicate_down(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """向下一层通信"""
        return {"status": "success", "layer": self.layer_name}

    def get_layer_status(self) -> Dict[str, Any]:
        """获取层状态"""
        return {
            "layer_name": self.layer_name,
            "status": "active",
            "health": "good"
        }


class DataLayerInterface(LayerInterface):
    """数据层接口"""
    pass


class FeatureLayerInterface(LayerInterface):
    """特征层接口"""
    pass


class ModelLayerInterface(LayerInterface):
    """模型层接口"""
    pass


class StrategyLayerInterface(LayerInterface):
    """策略层接口"""
    pass


class RiskLayerInterface(LayerInterface):
    """风险层接口"""
    pass


class ExecutionLayerInterface(LayerInterface):
    """执行层接口"""
    pass


class MonitoringLayerInterface(LayerInterface):
    """监控层接口"""
    pass


__all__ = [
    'LayerInterface',
    'DataLayerInterface',
    'FeatureLayerInterface',
    'ModelLayerInterface',
    'StrategyLayerInterface',
    'RiskLayerInterface',
    'ExecutionLayerInterface',
    'MonitoringLayerInterface'
]
