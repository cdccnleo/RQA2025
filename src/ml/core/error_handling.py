#!/usr/bin/env python3
"""
RQA2025 ML错误处理系统

提供统一的ML错误处理、异常管理和恢复机制，
支持错误分类、自动恢复、日志记录和性能监控。
"""

import logging
import traceback
import threading
from functools import wraps
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json

try:  # pragma: no cover
    from src.infrastructure.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    import logging

    class _FallbackModelsAdapter:
        def get_models_logger(self):
            return logging.getLogger(__name__)

    def _get_models_adapter():
        return _FallbackModelsAdapter()

get_models_adapter = _get_models_adapter

# 全局错误处理器实例
_ml_error_handler = None

# 获取统一基础设施集成层的模型层适配器
try:
    models_adapter = _get_models_adapter()
    logger = models_adapter.get_models_logger()
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


class MLErrorCategory(Enum):

    """ML错误类别"""
    DATA_ERROR = "data_error"              # 数据相关错误
    MODEL_ERROR = "model_error"            # 模型相关错误
    TRAINING_ERROR = "training_error"      # 训练相关错误
    INFERENCE_ERROR = "inference_error"    # 推理相关错误
    CONFIGURATION_ERROR = "config_error"   # 配置相关错误
    RESOURCE_ERROR = "resource_error"      # 资源相关错误
    VALIDATION_ERROR = "validation_error"  # 验证相关错误
    DEPLOYMENT_ERROR = "deployment_error"  # 部署相关错误
    MONITORING_ERROR = "monitoring_error"  # 监控相关错误
    SYSTEM_ERROR = "system_error"          # 系统级错误


class MLErrorSeverity(Enum):

    """ML错误严重程度"""
    LOW = "low"          # 低严重性，不影响主要功能
    MEDIUM = "medium"    # 中等严重性，可能影响部分功能
    HIGH = "high"        # 高严重性，影响关键功能
    CRITICAL = "critical"  # 严重错误，可能导致系统瘫痪


@dataclass
class MLError:

    """ML错误信息"""
    error_id: str
    category: MLErrorCategory
    severity: MLErrorSeverity
    message: str
    exception: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    process_id: Optional[str] = None
    step_id: Optional[str] = None
    model_id: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ErrorRecoveryStrategy:

    """错误恢复策略"""
    strategy_id: str
    error_category: MLErrorCategory
    condition: Callable[[MLError], bool]
    recovery_action: Callable[[MLError], Any]
    max_attempts: int = 3
    cooldown_seconds: int = 60
    priority: int = 1


class MLException(Exception):

    """ML基础异常类"""

    def __init__(self, message: str, category: MLErrorCategory = MLErrorCategory.SYSTEM_ERROR,


                 severity: MLErrorSeverity = MLErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()

    def to_error(self) -> MLError:
        """转换为MLError对象"""
        return MLError(
            error_id=f"{self.category.value}_{int(self.timestamp.timestamp() * 1000)}",
            category=self.category,
            severity=self.severity,
            message=str(self),
            exception=self,
            context=self.context.copy(),
            stack_trace=traceback.format_exc(),
            tags=["exception"]
        )


class DataValidationError(MLException):

    """数据验证错误"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):

        super().__init__(message, MLErrorCategory.DATA_ERROR, MLErrorSeverity.HIGH, context)


class ModelLoadError(MLException):

    """模型加载错误"""

    def __init__(self, message: str, model_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):

        context = context or {}
        if model_id:
            context['model_id'] = model_id
            super().__init__(message, MLErrorCategory.MODEL_ERROR, MLErrorSeverity.HIGH, context)


class TrainingError(MLException):

    """训练错误"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):

        super().__init__(message, MLErrorCategory.TRAINING_ERROR, MLErrorSeverity.MEDIUM, context)


class InferenceError(MLException):

    """推理错误"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):

        super().__init__(message, MLErrorCategory.INFERENCE_ERROR, MLErrorSeverity.HIGH, context)


class ResourceExhaustionError(MLException):

    """资源耗尽错误"""

    def __init__(self, message: str, resource_type: str, context: Optional[Dict[str, Any]] = None):

        context = context or {}
        context['resource_type'] = resource_type
        super().__init__(message, MLErrorCategory.RESOURCE_ERROR, MLErrorSeverity.CRITICAL, context)


class MLErrorHandler:

    """ML错误处理器"""

    def __init__(self):

        self.errors: Dict[str, MLError] = {}
        self.error_history: List[MLError] = []
        self.recovery_strategies: List[ErrorRecoveryStrategy] = []
        self.error_callbacks: Dict[MLErrorCategory, List[Callable]] = defaultdict(list)
        self.error_counters: Counter = Counter()
        self.lock = threading.Lock()

        # 注册默认恢复策略
        self._register_default_recovery_strategies()

        logger.info("ML错误处理器已初始化")

    def _register_default_recovery_strategies(self):
        """注册默认错误恢复策略"""

        # 数据加载失败的恢复策略

        def data_load_recovery(error: MLError) -> Any:

            logger.info(f"尝试恢复数据加载错误: {error.message}")
            # 这里可以实现数据源切换、重试等逻辑
            return {"status": "recovered", "action": "retry_with_backup_source"}

        self.register_recovery_strategy(ErrorRecoveryStrategy(
            strategy_id="data_load_retry",
            error_category=MLErrorCategory.DATA_ERROR,
            condition=lambda e: "load" in e.message.lower(),
            recovery_action=data_load_recovery,
            max_attempts=2,
            cooldown_seconds=30
        ))

            # 模型推理失败的恢复策略

        def inference_recovery(error: MLError) -> Any:

            logger.info(f"尝试恢复推理错误: {error.message}")
            # 可以实现模型降级、批处理调整等
            return {"status": "recovered", "action": "switch_to_cpu_inference"}

        self.register_recovery_strategy(ErrorRecoveryStrategy(
            strategy_id="inference_fallback",
            error_category=MLErrorCategory.INFERENCE_ERROR,
            condition=lambda e: e.recovery_attempts < 2,
            recovery_action=inference_recovery,
            max_attempts=2,
            cooldown_seconds=10
        ))

            # 资源耗尽的恢复策略

        def resource_recovery(error: MLError) -> Any:

            logger.info(f"尝试恢复资源错误: {error.message}")
            # 可以实现资源清理、垃圾回收等
            import gc
            gc.collect()
            return {"status": "recovered", "action": "garbage_collection"}

        self.register_recovery_strategy(ErrorRecoveryStrategy(
            strategy_id="resource_cleanup",
            error_category=MLErrorCategory.RESOURCE_ERROR,
            condition=lambda e: e.severity != MLErrorSeverity.CRITICAL,
            recovery_action=resource_recovery,
            max_attempts=1,
            cooldown_seconds=300
        ))

    def register_recovery_strategy(self, strategy: ErrorRecoveryStrategy):
        """注册错误恢复策略"""
        self.recovery_strategies.append(strategy)
        self.recovery_strategies.sort(key=lambda s: s.priority, reverse=True)
        logger.info(f"已注册错误恢复策略: {strategy.strategy_id}")

    def register_error_callback(self, category: MLErrorCategory, callback: Callable):
        """注册错误回调"""
        self.error_callbacks[category].append(callback)
        logger.debug(f"已注册错误回调: {category.value}")

    def handle_error(self, error: Union[Exception, MLError], context: Optional[Dict[str, Any]] = None) -> MLError:
        """处理错误"""
        with self.lock:
            if isinstance(error, Exception):
                # 转换为MLError
                if hasattr(error, 'to_error'):
                    ml_error = error.to_error()
                else:
                    ml_error = MLError(
                        error_id=f"exception_{int(datetime.now().timestamp() * 1000)}",
                        category=MLErrorCategory.SYSTEM_ERROR,
                        severity=MLErrorSeverity.MEDIUM,
                        message=str(error),
                        exception=error,
                        context=context or {},
                        stack_trace=traceback.format_exc()
                    )
            else:
                # 已经是MLError，直接使用
                ml_error = error

        # 增强上下文信息
        if context:
            ml_error.context.update(context)

        # 记录错误
        self.errors[ml_error.error_id] = ml_error
        self.error_history.append(ml_error)
        self.error_counters[ml_error.category] += 1

        # 限制历史记录大小
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

        # 触发回调
        self._trigger_error_callbacks(ml_error)

        # 记录日志
        self._log_error(ml_error)

        # 尝试自动恢复
        recovery_result = self._attempt_recovery(ml_error)

        if recovery_result:
            ml_error.resolved = True
            ml_error.resolution_time = datetime.now()
            logger.info(f"错误已自动恢复: {ml_error.error_id}")

        return ml_error

    def _trigger_error_callbacks(self, error: MLError):
        """触发错误回调"""
        for callback in self.error_callbacks[error.category]:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"错误回调执行失败: {e}")

        # 全局错误回调
        for callback in self.error_callbacks[MLErrorCategory.SYSTEM_ERROR]:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"全局错误回调执行失败: {e}")

    def _log_error(self, error: MLError):
        """记录错误日志"""
        log_level = {
            MLErrorSeverity.LOW: logging.DEBUG,
            MLErrorSeverity.MEDIUM: logging.WARNING,
            MLErrorSeverity.HIGH: logging.ERROR,
            MLErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error.severity, logging.ERROR)

        logger.log(log_level, f"ML错误 [{error.category.value}]: {error.message}",
                   extra={
                       'error_id': error.error_id,
                       'severity': error.severity.value,
                       'context': error.context,
                       'process_id': error.process_id,
                       'model_id': error.model_id
                   })

    def _attempt_recovery(self, error: MLError) -> Optional[Any]:
        """尝试错误恢复"""
        if error.recovery_attempts >= error.max_recovery_attempts:
            return None

        # 查找适用的恢复策略
        for strategy in self.recovery_strategies:
            if strategy.error_category == error.category and strategy.condition(error):
                try:
                    error.recovery_attempts += 1
                    logger.info(
                        f"应用恢复策略: {strategy.strategy_id} "
                        f"(尝试 {error.recovery_attempts}/{error.max_recovery_attempts})"
                    )
                    result = strategy.recovery_action(error)
                    return result
                except Exception as e:
                    logger.error(f"恢复策略执行失败: {strategy.strategy_id}, 错误: {e}")
                    continue

        return None

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        with self.lock:
            stats = {
                'total_errors': len(self.error_history),
                'active_errors': len([e for e in self.errors.values() if not e.resolved]),
                'resolved_errors': len([e for e in self.errors.values() if e.resolved]),
                'error_distribution': {
                    category.value: count for category, count in self.error_counters.items()
                },
                'recent_errors': len([e for e in self.error_history
                                      if (datetime.now() - e.timestamp) < timedelta(hours=1)]),
                'severity_distribution': {
                    severity.value: len([e for e in self.error_history if e.severity == severity])
                    for severity in MLErrorSeverity
                }
            }

            # 恢复成功率
            recovery_attempts = sum(e.recovery_attempts for e in self.error_history)
            successful_recoveries = len([e for e in self.error_history if e.resolved])
            stats['recovery_success_rate'] = (
                successful_recoveries / recovery_attempts if recovery_attempts > 0 else 0
            )

            return stats

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的错误"""
        with self.lock:
            recent_errors = sorted(self.error_history[-limit:],
                                   key=lambda e: e.timestamp, reverse=True)

            return [{
                'error_id': e.error_id,
                'category': e.category.value,
                'severity': e.severity.value,
                'message': e.message,
                'timestamp': e.timestamp.isoformat(),
                'resolved': e.resolved,
                'recovery_attempts': e.recovery_attempts
            } for e in recent_errors]

    def resolve_error(self, error_id: str) -> bool:
        """手动解决错误"""
        with self.lock:
            if error_id in self.errors:
                self.errors[error_id].resolved = True
                self.errors[error_id].resolution_time = datetime.now()
                logger.info(f"错误已手动解决: {error_id}")
                return True
            return False

    def export_error_report(self, format: str = 'json') -> str:
        """导出错误报告"""
        stats = self.get_error_statistics()
        recent_errors = self.get_recent_errors(50)

        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'recent_errors': recent_errors,
            'active_errors': [
                {
                    'error_id': e.error_id,
                    'category': e.category.value,
                    'severity': e.severity.value,
                    'message': e.message,
                    'timestamp': e.timestamp.isoformat(),
                    'recovery_attempts': e.recovery_attempts
                }
                for e in self.errors.values() if not e.resolved
            ]
        }

        if format == 'json':
            return json.dumps(report, indent=2, default=str)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

            # 全局错误处理器实例已在模块级别定义


def get_ml_error_handler() -> MLErrorHandler:
    """获取ML错误处理器实例"""
    global _ml_error_handler
    if _ml_error_handler is None:
        _ml_error_handler = MLErrorHandler()
    return _ml_error_handler


def handle_ml_error(error: Union[Exception, MLError], context: Optional[Dict[str, Any]] = None) -> MLError:
    """处理ML错误"""
    handler = get_ml_error_handler()
    return handler.handle_error(error, context)


def register_error_recovery_strategy(strategy: ErrorRecoveryStrategy):
    """注册错误恢复策略"""
    handler = get_ml_error_handler()
    handler.register_recovery_strategy(strategy)


def register_error_callback(category: MLErrorCategory, callback: Callable):
    """注册错误回调"""
    handler = get_ml_error_handler()
    handler.register_error_callback(category, callback)


def get_error_statistics() -> Dict[str, Any]:
    """获取错误统计"""
    handler = get_ml_error_handler()
    return handler.get_error_statistics()

# 便捷装饰器


def ml_error_handler(category: MLErrorCategory = MLErrorCategory.SYSTEM_ERROR,
                     severity: MLErrorSeverity = MLErrorSeverity.MEDIUM):
    """ML错误处理装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ml_exception = MLException(
                    str(e),
                    category=category,
                    severity=severity,
                    context={
                        'function': func.__name__,
                        'args': str(args),
                        'kwargs': str(kwargs),
                    },
                )
                handle_ml_error(ml_exception)
                raise
        return wrapper
    return decorator


__all__ = [
    # 异常类
    'MLException', 'DataValidationError', 'ModelLoadError',
    'TrainingError', 'InferenceError', 'ResourceExhaustionError',

    # 枚举
    'MLErrorCategory', 'MLErrorSeverity',

    # 数据类
    'MLError', 'ErrorRecoveryStrategy',

    # 处理器
    'MLErrorHandler',

    # 全局函数
    'get_ml_error_handler', 'handle_ml_error',
    'register_error_recovery_strategy', 'register_error_callback',
    'get_error_statistics',

    # 装饰器
    'ml_error_handler'
]
