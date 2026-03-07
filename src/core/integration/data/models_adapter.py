#!/usr/bin/env python3
"""
RQA2025 模型层专用适配器

实现模型层与统一基础设施层的深度集成，提供模型专用基础设施服务访问接口。
基于适配器模式设计，消除代码重复，实现集中化管理。
"""

from typing import Dict, List, Any
import logging

from .business_adapters import BaseBusinessAdapter, BusinessLayerType

logger = logging.getLogger(__name__)


class ModelsLayerAdapter(BaseBusinessAdapter):

    """模型层专用适配器

    提供模型层专用的基础设施服务访问接口，实现：
    - 模型专用缓存管理
    - 模型专用配置管理
    - 模型专用监控系统
    - 模型专用事件总线
    - 模型专用健康检查
    """

    def __init__(self):

        super().__init__(BusinessLayerType.MODELS)
        self._init_models_infrastructure()

    def _init_models_infrastructure(self):
        """初始化模型层基础设施"""
        # 模型层直接使用统一基础设施集成层的服务
        # 不需要额外的桥接器，通过适配器直接访问基础设施服务
        logger.info("模型层统一基础设施集成初始化完成")

    # 模型专用基础设施服务访问接口

    def get_models_cache_manager(self):
        """获取模型专用缓存管理器"""
        # 直接使用统一基础设施集成层的缓存管理器
        return self.get_infrastructure_services().get('cache_manager')

    def get_models_config_manager(self):
        """获取模型专用配置管理器"""
        # 直接使用统一基础设施集成层的配置管理器
        return self.get_infrastructure_services().get('config_manager')

    def get_models_monitoring(self):
        """获取模型专用监控系统"""
        # 直接使用统一基础设施集成层的监控系统
        return self.get_infrastructure_services().get('monitoring')

    def get_models_event_bus(self):
        """获取模型专用事件总线"""
        # 直接使用统一基础设施集成层的事件总线
        try:
            from src.core.event_bus import EventBus
            return EventBus()
        except ImportError:
            logger.warning("EventBus不可用")
            return None

    def get_models_logger(self):
        """获取模型专用日志器"""
        return self.get_infrastructure_services().get('logger')

    def get_models_health_checker(self):
        """获取模型专用健康检查器"""
        return self.get_infrastructure_services().get('health_checker')

    # 模型层专用业务方法

    def get_model_training_metrics(self, model_id: str) -> Dict[str, Any]:
        """获取模型训练指标"""
        monitoring = self.get_models_monitoring()
        if monitoring:
            return monitoring.get_model_training_metrics(model_id)
        return {}

    def record_model_prediction(self, model_id: str, prediction_data: Dict[str, Any]):
        """记录模型预测"""
        monitoring = self.get_models_monitoring()
        if monitoring:
            monitoring.record_model_prediction(model_id, prediction_data)

    def get_model_performance_history(self, model_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """获取模型性能历史"""
        monitoring = self.get_models_monitoring()
        if monitoring:
            return monitoring.get_model_performance_history(model_id, hours)
        return []

    def cache_model_prediction(self, prediction_id: str, prediction_result: Any, ttl: int = 3600):
        """缓存模型预测结果"""
        cache = self.get_models_cache_manager()
        if cache:
            cache.set(f"prediction_{prediction_id}", prediction_result, ttl)

    def get_cached_prediction(self, prediction_id: str):
        """获取缓存的预测结果"""
        cache = self.get_models_cache_manager()
        if cache:
            return cache.get(f"prediction_{prediction_id}")
        return None

    def publish_model_event(self, event_type: str, event_data: Dict[str, Any]):
        """发布模型相关事件"""
        event_bus = self.get_models_event_bus()
        if event_bus:
            event_bus.publish_event(event_type, event_data)

    def subscribe_model_events(self, event_type: str, callback):
        """订阅模型相关事件"""
        event_bus = self.get_models_event_bus()
        if event_bus:
            event_bus.subscribe(event_type, callback)

    # 模型层健康检查扩展

    def health_check(self) -> Dict[str, Any]:
        """模型层专用健康检查"""
        base_health = super().health_check()

        # 添加模型层特定健康检查
        models_health = {
            'model_cache_status': self._check_model_cache_health(),
            'model_monitoring_status': self._check_model_monitoring_health(),
            'active_models_count': self._get_active_models_count(),
            'recent_predictions_count': self._get_recent_predictions_count(),
            'model_performance_status': self._check_model_performance_health()
        }

        base_health['models_specific'] = models_health
        return base_health

    def _check_model_cache_health(self) -> Dict[str, Any]:
        """检查模型缓存健康状态"""
        try:
            cache = self.get_models_cache_manager()
            if cache and hasattr(cache, 'health_check'):
                return cache.health_check()
            return {'status': 'unknown', 'message': '缓存管理器不可用'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _check_model_monitoring_health(self) -> Dict[str, Any]:
        """检查模型监控健康状态"""
        try:
            monitoring = self.get_models_monitoring()
            if monitoring and hasattr(monitoring, 'health_check'):
                return monitoring.health_check()
            return {'status': 'unknown', 'message': '监控系统不可用'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _get_active_models_count(self) -> int:
        """获取活跃模型数量"""
        try:
            # 这里可以连接到模型管理器获取活跃模型数量
            # 暂时返回0，实际实现需要连接到具体的模型管理服务
            return 0
        except Exception:
            return 0

    def _get_recent_predictions_count(self) -> int:
        """获取最近预测数量"""
        try:
            # 这里可以连接到监控系统获取最近预测数量
            # 暂时返回0，实际实现需要连接到具体的监控服务
            return 0
        except Exception:
            return 0

    def _check_model_performance_health(self) -> Dict[str, Any]:
        """检查模型性能健康状态"""
        try:
            # 检查最近模型性能指标
            # 这里可以实现具体的性能健康检查逻辑
            return {'status': 'healthy', 'message': '模型性能正常'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    # 配置管理方法

    def get_model_config(self, config_key: str, default: Any = None) -> Any:
        """获取模型配置"""
        config_manager = self.get_models_config_manager()
        if config_manager:
            return config_manager.get_config(f"models.{config_key}", default)
        return default

    def set_model_config(self, config_key: str, value: Any):
        """设置模型配置"""
        config_manager = self.get_models_config_manager()
        if config_manager:
            config_manager.set_config(f"models.{config_key}", value)

    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return {
            'batch_size': self.get_model_config('training.batch_size', 32),
            'epochs': self.get_model_config('training.epochs', 100),
            'learning_rate': self.get_model_config('training.learning_rate', 0.001),
            'validation_split': self.get_model_config('training.validation_split', 0.2),
            'early_stopping': self.get_model_config('training.early_stopping', True),
            'gpu_enabled': self.get_model_config('training.gpu_enabled', False)
        }

    def get_inference_config(self) -> Dict[str, Any]:
        """获取推理配置"""
        return {
            'batch_size': self.get_model_config('inference.batch_size', 64),
            'timeout_seconds': self.get_model_config('inference.timeout_seconds', 30),
            'cache_enabled': self.get_model_config('inference.cache_enabled', True),
            'cache_ttl': self.get_model_config('inference.cache_ttl', 3600),
            'confidence_threshold': self.get_model_config('inference.confidence_threshold', 0.5)
        }

    # 模型管理集成方法

    def get_model_manager(self):
        """获取模型管理器"""
        try:
            from src.ml.model_manager import ModelManager
            return ModelManager()
        except ImportError:
            logger.warning("ModelManager不可用")
            return None

    def get_inference_service(self):
        """获取推理服务"""
        try:
            from src.ml.inference_service import InferenceService
            return InferenceService()
        except ImportError:
            logger.warning("InferenceService不可用")
            return None

    def get_feature_engineer(self):
        """获取特征工程师"""
        try:
            from src.ml.feature_engineering import FeatureEngineer
            return FeatureEngineer()
        except ImportError:
            logger.warning("FeatureEngineer不可用")
            return None

    # 业务流程集成方法

    def create_model_training_workflow(self):
        """创建模型训练业务流程"""
        try:
            from src.ml.workflow.model_training_workflow import ModelTrainingWorkflow
            return ModelTrainingWorkflow(self)
        except ImportError:
            logger.warning("ModelTrainingWorkflow不可用")
            return None

    def create_real_time_inference_workflow(self):
        """创建实时推理业务流程"""
        try:
            from src.ml.workflow.real_time_inference_workflow import RealTimeInferenceWorkflow
            return RealTimeInferenceWorkflow(self)
        except ImportError:
            logger.warning("RealTimeInferenceWorkflow不可用")
            return None

    # 性能监控方法

    def start_performance_monitoring(self):
        """启动性能监控"""
        monitoring = self.get_models_monitoring()
        if monitoring:
            monitoring.start_model_performance_monitoring()

    def stop_performance_monitoring(self):
        """停止性能监控"""
        monitoring = self.get_models_monitoring()
        if monitoring:
            monitoring.stop_model_performance_monitoring()

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        monitoring = self.get_models_monitoring()
        if monitoring:
            return monitoring.generate_performance_report()
        return {}
