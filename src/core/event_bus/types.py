"""
事件总线类型定义
包含事件类型枚举和优先级枚举
"""

from enum import Enum


class EventType(Enum):

    """事件类型枚举 - 完善版"""
    # 数据层事件
    DATA_READY = "data_ready"
    DATA_COLLECTION_STARTED = "data_collection_started"
    DATA_COLLECTION_PROGRESS = "data_collection_progress"  # 数据采集进度更新
    DATA_COLLECTED = "data_collected"
    DATA_COLLECTION_COMPLETED = "data_collection_completed"  # 数据采集完成（用于触发后续业务流程）
    DATA_COLLECTION_FAILED = "data_collection_failed"  # 数据采集失败
    DATA_QUALITY_CHECKED = "data_quality_checked"
    DATA_QUALITY_ALERT = "data_quality_alert"
    DATA_QUALITY_UPDATED = "data_quality_updated"
    DATA_PERFORMANCE_UPDATED = "data_performance_updated"
    DATA_PERFORMANCE_ALERT = "data_performance_alert"
    DATA_STORED = "data_stored"
    DATA_VALIDATED = "data_validated"

    # 特征层事件
    FEATURE_EXTRACTED = "feature_extracted"
    FEATURE_EXTRACTION_STARTED = "feature_extraction_started"
    FEATURES_EXTRACTED = "features_extracted"
    GPU_ACCELERATION_STARTED = "gpu_acceleration_started"
    GPU_ACCELERATION_COMPLETED = "gpu_acceleration_completed"
    FEATURE_PROCESSING_COMPLETED = "feature_processing_completed"

    # 模型层事件
    MODEL_PREDICTION = "model_prediction"
    MODEL_PREDICTED = "model_predicted"
    MODEL_TRAINING_STARTED = "model_training_started"
    MODEL_TRAINING_COMPLETED = "model_training_completed"
    TRAINING_JOB_CREATED = "training_job_created"
    TRAINING_JOB_UPDATED = "training_job_updated"
    TRAINING_JOB_STOPPED = "training_job_stopped"
    TRAINING_JOB_DELETED = "training_job_deleted"
    MODEL_PREDICTION_STARTED = "model_prediction_started"
    MODEL_PREDICTION_READY = "model_prediction_ready"
    MODEL_ENSEMBLE_STARTED = "model_ensemble_started"
    MODEL_ENSEMBLE_READY = "model_ensemble_ready"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_EVALUATED = "model_evaluated"

    # 策略层事件
    SIGNAL_GENERATED = "signal_generated"
    STRATEGY_DECISION_STARTED = "strategy_decision_started"
    STRATEGY_DECISION_READY = "strategy_decision_ready"
    SIGNAL_GENERATION_STARTED = "signal_generation_started"
    SIGNALS_GENERATED = "signals_generated"
    PARAMETER_OPTIMIZATION_STARTED = "parameter_optimization_started"
    PARAMETER_OPTIMIZATION_COMPLETED = "parameter_optimization_completed"

    # 风控层事件
    RISK_CHECKED = "risk_checked"
    RISK_CHECK_STARTED = "risk_check_started"
    RISK_CHECK_COMPLETED = "risk_check_completed"
    RISK_REJECTED = "risk_rejected"
    RISK_ASSESSMENT_COMPLETED = "risk_assessment_completed"
    RISK_INTERCEPTED = "risk_intercepted"
    COMPLIANCE_VERIFICATION_STARTED = "compliance_verification_started"
    COMPLIANCE_VERIFIED = "compliance_verified"
    COMPLIANCE_CHECK_COMPLETED = "compliance_check_completed"
    COMPLIANCE_REJECTED = "compliance_rejected"
    RISK_REPORT_GENERATED = "risk_report_generated"
    ALERT_TRIGGERED = "alert_triggered"
    ALERT_RESOLVED = "alert_resolved"
    REAL_TIME_MONITORING_ALERT = "real_time_monitoring_alert"

    # 交易层事件
    ORDER_CREATED = "order_created"
    ORDER_GENERATION_STARTED = "order_generation_started"
    ORDERS_GENERATED = "orders_generated"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_MODIFIED = "order_modified"
    POSITION_UPDATED = "position_updated"
    TRADE_CONFIRMED = "trade_confirmed"

    # 监控层事件
    PERFORMANCE_ALERT = "performance_alert"
    BUSINESS_ALERT = "business_alert"
    TRADING_CYCLE_COMPLETED = "trading_cycle_completed"
    SYSTEM_HEALTH_CHECK = "system_health_check"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"

    # 基础设施层事件
    CONFIG_UPDATED = "config_updated"
    CACHE_UPDATED = "cache_updated"
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    SERVICE_HEALTH_CHECK = "service_health_check"

    # 核心服务层事件
    EVENT_BUS_STARTED = "event_bus_started"
    EVENT_BUS_STOPPED = "event_bus_stopped"
    SERVICE_REGISTERED = "service_registered"
    SERVICE_DISCOVERED = "service_discovered"
    APPLICATION_STARTUP_COMPLETE = "application_startup_complete"  # 应用启动完成事件

    # 工作流事件
    VALIDATION_COMPLETED = "validation_completed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_ERROR = "workflow_error"

    # API事件
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    API_ERROR = "api_error"

    # 服务通信事件
    SERVICE_COMMUNICATION = "service_communication"

    # 缓存事件
    CACHE_GET = "cache_get"
    CACHE_SET = "cache_set"
    CACHE_DELETE = "cache_delete"
    CACHE_CLEAR = "cache_clear"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"

    # 安全审计事件
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"


class EventPriority(Enum):

    """事件优先级枚举"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0
