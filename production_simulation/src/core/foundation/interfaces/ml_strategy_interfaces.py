#!/usr/bin/env python3
"""
ML层和策略层标准化接口契约
Standardized Interface Contracts between ML Layer and Strategy Layer

定义ML层和策略层之间的标准接口，确保职责边界清晰，避免功能重叠。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd


# =============================================================================
# 数据格式标准
# =============================================================================

@dataclass
class MLFeatures:

    """
    ML特征数据标准格式
    """
    timestamp: datetime
    symbol: str
    features: Dict[str, Union[float, int, bool]]
    labels: Optional[Dict[str, Union[float, int, bool]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array(list(self.features.values()))

    def to_dataframe(self) -> pd.DataFrame:
        """转换为pandas DataFrame"""
        data = self.features.copy()
        if self.labels:
            data.update(self.labels)
        return pd.DataFrame([data])


@dataclass
class MLInferenceRequest:

    """
    ML推理请求标准格式
    """
    request_id: str
    model_id: str
    features: MLFeatures
    inference_type: str = "sync"  # sync, async, batch, streaming
    priority: int = 1  # 1 - 10, 10最高
    timeout_seconds: int = 30
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MLInferenceResponse:

    """
    ML推理响应标准格式
    """
    request_id: str
    success: bool
    prediction: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StrategyExecutionRequest:

    """
    策略执行请求标准格式
    """
    strategy_id: str
    market_data: MLFeatures
    execution_context: Optional[Dict[str, Any]] = None
    priority: int = 1
    timeout_seconds: int = 30


@dataclass
class StrategyExecutionResponse:

    """
    策略执行响应标准格式
    """
    strategy_id: str
    success: bool
    signals: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    execution_context: Optional[Dict[str, Any]] = None


# =============================================================================
# ML层接口定义
# =============================================================================


class IMLService(ABC):

    """
    ML层服务标准接口
    ML Layer Service Standard Interface

    定义ML层向策略层提供的标准化服务接口。
    ML层负责：算法实现、模型训练、推理服务、性能监控。
    """

    @abstractmethod
    def load_model(self, model_id: str, model_config: Dict[str, Any]) -> bool:
        """
        加载模型

        Args:
            model_id: 模型唯一标识
            model_config: 模型配置参数

        Returns:
            bool: 加载是否成功
        """

    @abstractmethod
    def unload_model(self, model_id: str) -> bool:
        """
        卸载模型

        Args:
            model_id: 模型唯一标识

        Returns:
            bool: 卸载是否成功
        """

    @abstractmethod
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出可用模型

        Args:
            model_type: 模型类型过滤器（可选）

        Returns:
            List[Dict[str, Any]]: 模型信息列表
        """

    @abstractmethod
    async def predict(self, request: MLInferenceRequest) -> MLInferenceResponse:
        """
        执行模型推理

        Args:
            request: 推理请求

        Returns:
            MLInferenceResponse: 推理响应
        """

    @abstractmethod
    async def predict_batch(self, requests: List[MLInferenceRequest]) -> List[MLInferenceResponse]:
        """
        批量执行模型推理

        Args:
            requests: 推理请求列表

        Returns:
            List[MLInferenceResponse]: 推理响应列表
        """

    @abstractmethod
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        获取模型性能指标

        Args:
            model_id: 模型唯一标识

        Returns:
            Dict[str, Any]: 性能指标数据
        """

    @abstractmethod
    def train_model(self, model_id: str, training_data: pd.DataFrame,


                    training_config: Dict[str, Any]) -> bool:
        """
        训练模型

        Args:
            model_id: 模型唯一标识
            training_data: 训练数据
            training_config: 训练配置

        Returns:
            bool: 训练是否成功
        """

    @abstractmethod
    def optimize_hyperparameters(self, model_id: str, param_space: Dict[str, Any],


                                 evaluation_data: pd.DataFrame) -> Dict[str, Any]:
        """
        超参数优化

        Args:
            model_id: 模型唯一标识
            param_space: 参数搜索空间
            evaluation_data: 评估数据

        Returns:
            Dict[str, Any]: 最优参数配置
        """

    @abstractmethod
    def get_service_status(self) -> Dict[str, Any]:
        """
        获取服务状态

        Returns:
            Dict[str, Any]: 服务状态信息
        """


class IMLFeatureEngineering(ABC):

    """
    ML特征工程标准接口
    ML Feature Engineering Standard Interface

    定义ML层特征工程服务的标准化接口。
    """

    @abstractmethod
    def extract_features(self, raw_data: pd.DataFrame,


                         feature_config: Dict[str, Any]) -> MLFeatures:
        """
        特征提取

        Args:
            raw_data: 原始数据
            feature_config: 特征配置

        Returns:
            MLFeatures: 提取的特征
        """

    @abstractmethod
    def preprocess_features(self, features: MLFeatures,


                            preprocessing_config: Dict[str, Any]) -> MLFeatures:
        """
        特征预处理

        Args:
            features: 原始特征
            preprocessing_config: 预处理配置

        Returns:
            MLFeatures: 预处理后的特征
        """

    @abstractmethod
    def select_features(self, features: MLFeatures, target: np.ndarray,


                        selection_config: Dict[str, Any]) -> MLFeatures:
        """
        特征选择

        Args:
            features: 输入特征
            target: 目标变量
            selection_config: 选择配置

        Returns:
            MLFeatures: 选择的特征
        """


# =============================================================================
# 策略层接口定义
# =============================================================================


class IStrategyService(ABC):

    """
    策略层服务标准接口
    Strategy Layer Service Standard Interface

    定义策略层向ML层提供的标准化服务接口。
    策略层负责：策略逻辑、参数调优、风险控制、市场适配。
    """

    @abstractmethod
    def create_strategy(self, strategy_config: Dict[str, Any]) -> bool:
        """
        创建策略

        Args:
            strategy_config: 策略配置

        Returns:
            bool: 创建是否成功
        """

    @abstractmethod
    def execute_strategy(self, request: StrategyExecutionRequest) -> StrategyExecutionResponse:
        """
        执行策略

        Args:
            request: 策略执行请求

        Returns:
            StrategyExecutionResponse: 策略执行响应
        """

    @abstractmethod
    def optimize_strategy_parameters(self, strategy_id: str,


                                     optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        策略参数优化

        Args:
            strategy_id: 策略ID
            optimization_config: 优化配置

        Returns:
            Dict[str, Any]: 最优参数配置
        """

    @abstractmethod
    def evaluate_strategy_performance(self, strategy_id: str,


                                      evaluation_data: pd.DataFrame) -> Dict[str, Any]:
        """
        策略性能评估

        Args:
            strategy_id: 策略ID
            evaluation_data: 评估数据

        Returns:
            Dict[str, Any]: 性能评估结果
        """

    @abstractmethod
    def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """
        获取策略状态

        Args:
            strategy_id: 策略ID

        Returns:
            Dict[str, Any]: 策略状态信息
        """


class IStrategyDataPreparation(ABC):

    """
    策略层数据准备标准接口
    Strategy Layer Data Preparation Standard Interface

    定义策略层数据准备和预处理的标准化接口。
    """

    @abstractmethod
    def prepare_market_data(self, raw_market_data: pd.DataFrame,


                            preparation_config: Dict[str, Any]) -> pd.DataFrame:
        """
        准备市场数据

        Args:
            raw_market_data: 原始市场数据
            preparation_config: 数据准备配置

        Returns:
            pd.DataFrame: 准备后的市场数据
        """

    @abstractmethod
    def validate_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证市场数据质量

        Args:
            market_data: 市场数据

        Returns:
            Dict[str, Any]: 验证结果
        """

    @abstractmethod
    def handle_data_anomalies(self, market_data: pd.DataFrame,


                              anomaly_config: Dict[str, Any]) -> pd.DataFrame:
        """
        处理数据异常

        Args:
            market_data: 市场数据
            anomaly_config: 异常处理配置

        Returns:
            pd.DataFrame: 处理后的市场数据
        """


# =============================================================================
# 协作协议定义
# =============================================================================


class MLStrategyCollaborationProtocol:

    """
    ML层和策略层协作协议
    ML and Strategy Layer Collaboration Protocol

    定义跨层协作的标准流程和规范。
    """

    def __init__(self, ml_service: IMLService, strategy_service: IStrategyService):

        self.ml_service = ml_service
        self.strategy_service = strategy_service
        self.collaboration_history = []

    async def execute_ml_enhanced_strategy(self, strategy_id: str,
                                           market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行ML增强策略的标准流程

        Args:
            strategy_id: 策略ID
            market_data: 市场数据

        Returns:
            Dict[str, Any]: 执行结果
        """
        collaboration_id = f"COLLAB_{datetime.now().strftime('%Y % m % d % H % M % S % f')}"

        try:
            # 1. 策略层：准备和验证数据
            prepared_data = await self._prepare_strategy_data(strategy_id, market_data)
            self._record_collaboration_step(collaboration_id, "data_preparation", "success")

            # 2. 策略层：提取策略级特征
            strategy_features = await self._extract_strategy_features(strategy_id, prepared_data)
            self._record_collaboration_step(collaboration_id, "feature_extraction", "success")

            # 3. 策略层：构建ML推理请求
            ml_request = await self._build_ml_inference_request(strategy_id, strategy_features)
            self._record_collaboration_step(collaboration_id, "ml_request_building", "success")

            # 4. ML层：执行推理
            ml_response = await self.ml_service.predict(ml_request)
            self._record_collaboration_step(collaboration_id, "ml_inference", "success")

            # 5. 策略层：处理ML推理结果
            strategy_result = await self._process_ml_response(strategy_id, ml_response)
            self._record_collaboration_step(collaboration_id, "result_processing", "success")

            # 6. 策略层：生成最终信号
            final_signals = await self._generate_final_signals(strategy_id, strategy_result)
            self._record_collaboration_step(collaboration_id, "signal_generation", "success")

            return {
                "collaboration_id": collaboration_id,
                "success": True,
                "signals": final_signals,
                "processing_time_ms": self._calculate_processing_time(collaboration_id)
            }

        except Exception as e:
            self._record_collaboration_step(collaboration_id, "error", str(e))
            return {
                "collaboration_id": collaboration_id,
                "success": False,
                "error": str(e)
            }

    async def _prepare_strategy_data(self, strategy_id: str,
                                     market_data: pd.DataFrame) -> pd.DataFrame:
        """策略层：数据准备"""
        # 调用策略层的数据准备服务
        return await self.strategy_service.prepare_market_data(strategy_id, market_data)

    async def _extract_strategy_features(self, strategy_id: str,
                                         prepared_data: pd.DataFrame) -> MLFeatures:
        """策略层：特征提取"""
        # 策略层负责提取策略相关的特征
        # 这里应该调用策略层的特征提取方法

    async def _build_ml_inference_request(self, strategy_id: str,
                                          features: MLFeatures) -> MLInferenceRequest:
        """策略层：构建ML推理请求"""
        return MLInferenceRequest(
            request_id=f"REQ_{datetime.now().strftime('%Y % m % d % H % M % S % f')}",
            model_id=f"strategy_{strategy_id}_model",
            features=features,
            inference_type="sync",
            priority=1
        )

    async def _process_ml_response(self, strategy_id: str,
                                   ml_response: MLInferenceResponse) -> Dict[str, Any]:
        """策略层：处理ML响应"""
        if not ml_response.success:
            raise Exception(f"ML inference failed: {ml_response.error_message}")

        # 策略层处理ML推理结果
        return {
            "prediction": ml_response.prediction,
            "confidence": ml_response.confidence,
            "processing_time_ms": ml_response.processing_time_ms
        }

    async def _generate_final_signals(self, strategy_id: str,
                                      strategy_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """策略层：生成最终信号"""
        # 基于ML结果和策略逻辑生成最终交易信号

    def _record_collaboration_step(self, collaboration_id: str, step: str, result: str):
        """记录协作步骤"""
        self.collaboration_history.append({
            "collaboration_id": collaboration_id,
            "timestamp": datetime.now(),
            "step": step,
            "result": result
        })

    def _calculate_processing_time(self, collaboration_id: str) -> float:
        """计算处理时间"""
        steps = [h for h in self.collaboration_history if h["collaboration_id"] == collaboration_id]
        if not steps:
            return 0.0

        start_time = min(s["timestamp"] for s in steps)
        end_time = max(s["timestamp"] for s in steps)

        return (end_time - start_time).total_seconds() * 1000


# =============================================================================
# 工厂函数和辅助工具
# =============================================================================


def create_ml_service(service_type: str = "default") -> IMLService:
    """
    创建ML服务实例

    Args:
        service_type: 服务类型

    Returns:
        IMLService: ML服务实例
    """
    if service_type == "inference":
        from src.ml.inference_service import InferenceService
        return InferenceService()
    elif service_type == "model_manager":
        from src.ml.model_manager import ModelManager
        return ModelManager()
    else:
        # 默认实现
        from src.ml.inference_service import InferenceService
        return InferenceService()


def create_strategy_service(service_type: str = "default") -> IStrategyService:
    """
    创建策略服务实例

    Args:
        service_type: 服务类型

    Returns:
        IStrategyService: 策略服务实例
    """
    if service_type == "unified":
        from src.strategy.core.strategy_service import UnifiedStrategyService
        return UnifiedStrategyService()
    else:
        # 默认实现
        from src.strategy.core.strategy_service import UnifiedStrategyService
        return UnifiedStrategyService()


def create_collaboration_protocol(ml_service: Optional[IMLService] = None,


                                  strategy_service: Optional[IStrategyService] = None) -> MLStrategyCollaborationProtocol:
    """
    创建协作协议实例

    Args:
        ml_service: ML服务实例（可选）
        strategy_service: 策略服务实例（可选）

    Returns:
        MLStrategyCollaborationProtocol: 协作协议实例
    """
    if ml_service is None:
        ml_service = create_ml_service()

    if strategy_service is None:
        strategy_service = create_strategy_service()

    return MLStrategyCollaborationProtocol(ml_service, strategy_service)

# =============================================================================
# 协议验证和测试工具
# =============================================================================


class InterfaceComplianceValidator:

    """
    接口合规性验证器
    """

    def __init__(self):

        self.validation_results = []

    def validate_ml_service_interface(self, service: IMLService) -> Dict[str, Any]:
        """验证ML服务接口合规性"""
        required_methods = [
            'load_model', 'unload_model', 'list_models', 'predict',
            'predict_batch', 'get_model_performance', 'train_model',
            'optimize_hyperparameters', 'get_service_status'
        ]

        compliance_results = {}
        for method_name in required_methods:
            has_method = hasattr(service, method_name)
            is_callable = callable(getattr(service, method_name, None))
            compliance_results[method_name] = has_method and is_callable

        overall_compliance = all(compliance_results.values())

        return {
            "service_type": "ML",
            "overall_compliance": overall_compliance,
            "method_compliance": compliance_results
        }

    def validate_strategy_service_interface(self, service: IStrategyService) -> Dict[str, Any]:
        """验证策略服务接口合规性"""
        required_methods = [
            'create_strategy', 'execute_strategy', 'optimize_strategy_parameters',
            'evaluate_strategy_performance', 'get_strategy_status'
        ]

        compliance_results = {}
        for method_name in required_methods:
            has_method = hasattr(service, method_name)
            is_callable = callable(getattr(service, method_name, None))
            compliance_results[method_name] = has_method and is_callable

        overall_compliance = all(compliance_results.values())

        return {
            "service_type": "Strategy",
            "overall_compliance": overall_compliance,
            "method_compliance": compliance_results
        }

    def validate_data_format_compatibility(self, ml_features: MLFeatures,


                                           strategy_data: pd.DataFrame) -> Dict[str, Any]:
        """验证数据格式兼容性"""
        # 验证特征数据格式
        # 验证时间戳格式
        # 验证数据类型一致性


# =============================================================================
# 使用示例和最佳实践
# =============================================================================

"""
使用示例：

# 1. 创建服务实例
ml_service = create_ml_service("inference")
    strategy_service = create_strategy_service("unified")

# 2. 创建协作协议
collaboration = create_collaboration_protocol(ml_service, strategy_service)

# 3. 执行ML增强策略
result = await collaboration.execute_ml_enhanced_strategy(
    strategy_id="ml_momentum_strategy",
    market_data=market_data_df
        )

# 4. 处理执行结果
if result["success"]:
    signals = result["signals"]
    # 处理交易信号
else:
    error_msg = result["error"]
    # 处理错误
"""

"""
最佳实践：

1. 接口合规性：
   - 所有实现必须严格遵守接口契约
   - 定期使用InterfaceComplianceValidator进行验证

2. 数据格式标准化：
   - 使用标准化的MLFeatures和StrategyExecutionRequest格式
   - 避免自定义数据格式导致的兼容性问题

3. 错误处理：
   - 在接口层面统一错误处理机制
   - 提供清晰的错误信息和错误码

4. 性能监控：
   - 对跨层调用进行性能监控
   - 定期review协作协议的性能表现

5. 版本管理：
   - 接口变更需要版本控制
   - 提供向后兼容性保证
"""
