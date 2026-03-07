"""
业务流程相关枚举定义
包含业务流程状态和事件类型枚举
"""

from enum import Enum


class BusinessProcessState(Enum):

    """业务流程状态 - 增强版"""
    PENDING = "pending"                     # 待处理状态
    IDLE = "idle"                           # 空闲状态
    DATA_COLLECTING = "data_collecting"     # 数据采集中
    DATA_QUALITY_CHECKING = "data_quality_checking"  # 数据质量检查中
    FEATURE_EXTRACTING = "feature_extracting"  # 特征提取中
    GPU_ACCELERATING = "gpu_accelerating"   # GPU加速中
    MODEL_PREDICTING = "model_predicting"   # 模型预测中
    MODEL_ENSEMBLING = "model_ensembling"   # 模型集成中
    SIGNAL_GENERATING = "signal_generating"  # 信号生成中
    STRATEGY_DECIDING = "strategy_deciding"  # 策略决策中
    RISK_CHECKING = "risk_checking"         # 风险检查中
    ORDER_GENERATING = "order_generating"   # 订单生成中
    ORDER_ROUTING = "order_routing"         # 订单路由中
    EXECUTING = "executing"                 # 执行中
    MONITORING = "monitoring"               # 监控中
    COMPLETED = "completed"                 # 完成
    ERROR = "error"                         # 错误
    CANCELLED = "cancelled"                 # 取消


class EventType(Enum):

    """事件类型枚举 - 业务流程专用"""
    # 流程控制事件
    PROCESS_STARTED = "process_started"
    PROCESS_COMPLETED = "process_completed"
    PROCESS_ERROR = "process_error"
    PROCESS_CANCELLED = "process_cancelled"

    # 数据处理事件
    DATA_COLLECTION_STARTED = "data_collection_started"
    DATA_COLLECTION_COMPLETED = "data_collection_completed"
    DATA_QUALITY_CHECK_PASSED = "data_quality_check_passed"
    DATA_QUALITY_CHECK_FAILED = "data_quality_check_failed"

    # 特征处理事件
    FEATURE_EXTRACTION_STARTED = "feature_extraction_started"
    FEATURE_EXTRACTION_COMPLETED = "feature_extraction_completed"
    GPU_ACCELERATION_STARTED = "gpu_acceleration_started"
    GPU_ACCELERATION_COMPLETED = "gpu_acceleration_completed"

    # 模型处理事件
    MODEL_PREDICTION_STARTED = "model_prediction_started"
    MODEL_PREDICTION_COMPLETED = "model_prediction_completed"
    MODEL_ENSEMBLE_STARTED = "model_ensemble_started"
    MODEL_ENSEMBLE_COMPLETED = "model_ensemble_completed"

    # 策略决策事件
    SIGNAL_GENERATION_STARTED = "signal_generation_started"
    SIGNAL_GENERATION_COMPLETED = "signal_generation_completed"
    STRATEGY_DECISION_STARTED = "strategy_decision_started"
    STRATEGY_DECISION_COMPLETED = "strategy_decision_completed"

    # 风控事件
    RISK_CHECK_STARTED = "risk_check_started"
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"

    # 交易执行事件
    ORDER_GENERATION_STARTED = "order_generation_started"
    ORDER_GENERATION_COMPLETED = "order_generation_completed"
    ORDER_ROUTING_STARTED = "order_routing_started"
    ORDER_ROUTING_COMPLETED = "order_routing_completed"
    TRADE_EXECUTION_STARTED = "trade_execution_started"
    TRADE_EXECUTION_COMPLETED = "trade_execution_completed"

    # 监控事件
    PERFORMANCE_MONITORING_STARTED = "performance_monitoring_started"
    BUSINESS_MONITORING_STARTED = "business_monitoring_started"
    MONITORING_ALERT = "monitoring_alert"

    # 系统事件
    SYSTEM_HEALTH_CHECK = "system_health_check"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
