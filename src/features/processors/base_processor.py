"""
特征处理器标准化接口实现
"""

from abc import abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass
import logging

from src.infrastructure.interfaces.standard_interfaces import FeatureProcessor, FeatureRequest
from src.features.core.validators import DataValidator, ValidationPipeline, ValidationResult
from src.features.core.exceptions import FeatureDataValidationError, FeatureProcessingError

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:

    """处理器配置"""
    processor_type: str
    feature_params: Dict[str, Any]
    validation_rules: Optional[Dict[str, Any]] = None
    enable_validation: bool = True  # 是否启用验证

    def __post_init__(self):

        if self.validation_rules is None:
            self.validation_rules = {}


class BaseFeatureProcessor(FeatureProcessor):

    """特征处理器基类"""

    def __init__(self, config: ProcessorConfig):

        self.config = config
        self.processor_type = config.processor_type
        self._features_cache = {}
        self._feature_info = {}
        self._validation_pipeline: Optional[ValidationPipeline] = None

        # 初始化验证管道
        if config.enable_validation:
            self._setup_validation_pipeline()

    def _setup_validation_pipeline(self):
        """设置验证管道"""
        self._validation_pipeline = ValidationPipeline(f"{self.processor_type}_validation")

        # 添加数据验证器
        data_validator = DataValidator(
            required_columns=self._get_required_columns(),
            min_rows=1,
            min_cols=1
        )
        self._validation_pipeline.add_validator(data_validator, "data")

    def _get_required_columns(self) -> List[str]:
        """获取必需列（子类可覆盖）"""
        return []

    def _validate_input(self, data: pd.DataFrame) -> ValidationResult:
        """验证输入数据"""
        if not self.config.enable_validation or self._validation_pipeline is None:
            return ValidationResult(is_valid=True)

        try:
            is_valid, results = self._validation_pipeline.validate_all({"data": data})

            # 获取数据验证结果
            data_result = results.get("DataValidator")
            if data_result:
                return data_result

            return ValidationResult(is_valid=is_valid)

        except Exception as e:
            logger.error(f"验证过程出错: {e}")
            return ValidationResult(
                is_valid=False,
                message=f"验证过程出错: {str(e)}"
            )

    def process(self, request: FeatureRequest) -> pd.DataFrame:
        """处理特征"""
        # 验证输入数据（使用解耦的验证模块）
        if self.config.enable_validation:
            validation_result = self._validate_input(request.data)
            if not validation_result.is_valid:
                missing_cols = validation_result.details.get("missing_columns", [])
                raise FeatureDataValidationError(
                    message=validation_result.message,
                    missing_columns=missing_cols,
                    data_shape=request.data.shape if not request.data.empty else None
                )

        # 检查请求的特征是否可用
        available_features = self.list_features()
        requested_features = request.features

        if not requested_features:
            # 如果没有指定特征，处理所有可用特征
            requested_features = available_features

        # 验证特征是否存在
        invalid_features = [f for f in requested_features if f not in available_features]
        if invalid_features:
            raise FeatureProcessingError(
                message=f"不支持的特征: {invalid_features}",
                processor_name=self.processor_type,
                feature_name=invalid_features[0] if invalid_features else None
            )

        # 处理特征
        result_data = request.data.copy()
        for feature_name in requested_features:
            try:
                feature_values = self._compute_feature(result_data, feature_name, request.params)
                result_data[f'feature_{feature_name}'] = feature_values
            except Exception as e:
                raise FeatureProcessingError(
                    message=f"计算特征 {feature_name} 失败: {str(e)}",
                    processor_name=self.processor_type,
                    feature_name=feature_name,
                    original_error=e
                )

        return result_data

    @abstractmethod
    def _compute_feature(self, data: pd.DataFrame, feature_name: str,


                         params: Dict[str, Any]) -> pd.Series:
        """计算单个特征"""

    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """获取特征信息"""
        if feature_name not in self._feature_info:
            self._feature_info[feature_name] = self._get_feature_metadata(feature_name)

        return self._feature_info[feature_name].copy()

    @abstractmethod
    def _get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """获取特征元数据"""

    def list_features(self) -> List[str]:
        """列出可用特征"""
        if not self._features_cache:
            self._features_cache = self._get_available_features()

        return self._features_cache.copy()

    @abstractmethod
    def _get_available_features(self) -> List[str]:
        """获取可用特征列表"""

    def validate_config(self) -> bool:
        """验证配置"""
        required_fields = ['processor_type', 'feature_params']
        for field in required_fields:
            if not hasattr(self.config, field) or getattr(self.config, field) is None:
                return False
        return True

    def get_processor_type(self) -> str:
        """获取处理器类型"""
        return self.processor_type

    def get_config(self) -> ProcessorConfig:
        """获取配置"""
        return self.config

    def clear_cache(self) -> None:
        """清空缓存"""
        self._features_cache.clear()
        self._feature_info.clear()

    def get_feature_params(self, feature_name: str) -> Dict[str, Any]:
        """获取特征参数"""
        return self.config.feature_params.get(feature_name, {})

    def set_feature_params(self, feature_name: str, params: Dict[str, Any]) -> None:
        """设置特征参数"""
        if feature_name not in self.config.feature_params:
            self.config.feature_params[feature_name] = {}
        self.config.feature_params[feature_name].update(params)
