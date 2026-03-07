
# ============================================================================
# API服务层
# from .micro_service import MicroService  # 暂未实现
# 业务服务层
# 交易服务
# 交易服务层
# 微服务层
# 数据验证服务层
# 模型服务层

"""
服务层统一导出模块

提供完整的业务服务接口，包括交易服务、数据验证服务、模型服务等功能。

分层架构：
- 交易服务：TradingService
- 数据验证服务：DataValidationService
- 模型服务：ModelServing
- 业务服务：BusinessService
- API服务：APIService
- 微服务：MicroService

典型用法：
# 合理跨层级导入：infrastructure / services模块初始化文件导入自己的子模块
# 合理跨层级导入：这是模块内部的正常导入，不涉及跨层问题
trading_service = TradingService()
result = trading_service.execute_trade(order)

# 数据验证服务
validation_service = DataValidationService()
validation_result = validation_service.validate_data(data)

# 模型服务
model_service = ModelServing()
prediction = model_service.predict(features)
"""

# ============================================================================
# 服务基类
# ============================================================================
__version__ = "1.0.0"
__author__ = "RQA Team"
__description__ = "RQA服务层模块，提供完整的业务服务接口"

__all__ = []
# 服务基类
'BaseService',
'ServiceStatus',

# 交易服务
'TradingService',

# 数据验证服务
'DataValidationService',

# 模型服务
'ModelService',
'ModelServing',
'ABTestManager',

# 业务服务
'BusinessService',

# API服务
'APIService',
'APIVersion',
'RateLimitStrategy',
'APIEndpoint',

# 微服务
'MicroService',
'ServiceType',
'ServiceInfo',
'ServiceDiscovery',
'MicroServiceStatus',
'CacheService',
'CacheStrategy',
'CacheEntry',
