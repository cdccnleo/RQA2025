"""
特征工程模块（别名模块）
提供向后兼容的导入路径

实际实现在 engine/feature_engineering.py 中
"""

try:
    from .engine.feature_engineering import (
        FeatureEngineer,
        FeatureEngineering,
        FeatureSelector,
        FeatureTransformer,
        FeatureType,
        ScalingMethod,
        EncodingMethod,
        FeatureDefinition,
        FeaturePipeline
    )
except ImportError:
    # 提供基础实现
    class FeatureEngineer:
        pass
    
    class FeatureEngineering:
        pass
    
    class FeatureSelector:
        pass
    
    class FeatureTransformer:
        pass
    
    class FeatureType:
        pass
    
    class ScalingMethod:
        pass
    
    class EncodingMethod:
        pass
    
    class FeatureDefinition:
        pass
    
    class FeaturePipeline:
        pass

__all__ = [
    'FeatureEngineer',
    'FeatureEngineering',
    'FeatureSelector',
    'FeatureTransformer',
    'FeatureType',
    'ScalingMethod',
    'EncodingMethod',
    'FeatureDefinition',
    'FeaturePipeline'
]

