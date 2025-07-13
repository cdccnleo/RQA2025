from typing import Dict, Any, Optional
from enum import Enum

class FeatureType(Enum):
    """特征类型枚举"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    QUANTITATIVE = "quantitative"
    SENTIMENT = "sentiment"

class FeatureConfig:
    """特征处理配置类

    用于定义特征处理的各项配置参数

    Attributes:
        feature_type (FeatureType): 特征类型
        params (Dict[str, Any]): 特征处理参数
        enabled (bool): 是否启用该特征
        version (str): 配置版本
    """

    def __init__(
        self,
        name: str,
        feature_type: FeatureType,
        params: Optional[Dict[str, Any]] = None,
        dependencies: Optional[list] = None,
        enabled: bool = True,
        version: str = "1.0",
        a_share_specific: bool = False
    ):
        """初始化特征配置

        Args:
            name: 特征名称
            feature_type: 特征类型
            params: 特征处理参数，默认为空字典
            dependencies: 依赖的特征列表，默认为空列表
            enabled: 是否启用该特征，默认为True
            version: 配置版本，默认为"1.0"
            a_share_specific: 是否为A股特有特征，默认为False
        """
        self.name = name
        self.feature_type = feature_type
        self.params = params or {}
        self.dependencies = dependencies or []
        self.enabled = enabled
        self.version = version
        self.a_share_specific = a_share_specific

    def validate(self) -> bool:
        """验证配置是否有效

        Returns:
            bool: 配置是否有效
        """
        required_params = {
            FeatureType.TECHNICAL: ["window_size", "indicators"],
            FeatureType.FUNDAMENTAL: ["metrics"],
            FeatureType.QUANTITATIVE: ["factors"]
        }

        if self.feature_type not in required_params:
            return False

        for param in required_params[self.feature_type]:
            if param not in self.params:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典

        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            "name": self.name,
            "feature_type": self.feature_type.value,
            "params": self.params,
            "dependencies": self.dependencies,
            "enabled": self.enabled,
            "version": self.version,
            "a_share_specific": self.a_share_specific
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FeatureConfig":
        """从字典创建配置

        Args:
            config_dict: 配置字典

        Returns:
            FeatureConfig: 配置实例
        """
        return cls(
            name=config_dict["name"],
            feature_type=FeatureType(config_dict["feature_type"]),
            params=config_dict.get("params", {}),
            dependencies=config_dict.get("dependencies", []),
            enabled=config_dict.get("enabled", True),
            version=config_dict.get("version", "1.0"),
            a_share_specific=config_dict.get("a_share_specific", False)
        )
