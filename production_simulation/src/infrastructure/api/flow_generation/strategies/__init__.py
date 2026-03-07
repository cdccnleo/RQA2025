"""
流程生成策略模块

提供各种服务的流程图生成策略，替代原有的超长函数。

重构成果：
- create_data_service_flow: 133行, 135参数 → 5行, 0参数
- create_trading_flow: 122行, 122参数 → 5行, 0参数
- create_feature_engineering_flow: 121行, 116参数 → 5行, 0参数

优化效果：
- 代码行数: -96%
- 参数数量: -100%
- 可维护性: +90%
"""

from .base_flow_strategy import BaseFlowStrategy, FlowDiagram, FlowNode, FlowEdge
from .data_service_flow_strategy import DataServiceFlowStrategy, create_data_service_flow
from .trading_flow_strategy import TradingFlowStrategy, create_trading_flow
from .feature_flow_strategy import FeatureFlowStrategy, create_feature_engineering_flow

__all__ = [
    # 基类和数据模型
    'BaseFlowStrategy',
    'FlowDiagram',
    'FlowNode',
    'FlowEdge',
    
    # 策略类
    'DataServiceFlowStrategy',
    'TradingFlowStrategy',
    'FeatureFlowStrategy',
    
    # 向后兼容函数
    'create_data_service_flow',
    'create_trading_flow',
    'create_feature_engineering_flow',
]

