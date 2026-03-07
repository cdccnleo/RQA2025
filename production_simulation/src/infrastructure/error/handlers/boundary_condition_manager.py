"""
boundary_condition_manager 模块

专门管理边界条件检查和处理的组件。
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BoundaryConditionType(Enum):
    """边界条件类型"""
    VALUE_OUT_OF_RANGE = "value_out_of_range"
    NEGATIVE_VALUE = "negative_value"
    EMPTY_COLLECTION = "empty_collection"
    NULL_REFERENCE = "null_reference"
    INVALID_STATE = "invalid_state"


@dataclass
class BoundaryCondition:
    """边界条件"""
    condition_type: BoundaryConditionType
    severity: str
    description: str
    suggested_action: str
    context: Dict[str, Any]


@dataclass
class BoundaryConditionConfig:
    """边界条件配置参数对象"""
    condition_type: BoundaryConditionType
    severity: str
    description: str
    suggested_action: str
    context: Dict[str, Any]


@dataclass
class BoundaryCheckResult:
    """边界检查结果"""
    condition: BoundaryCondition
    triggered: bool
    actual_value: Any
    threshold_value: Any


class BoundaryConditionManager:
    """边界条件管理器 - 专门管理边界条件检查和处理"""

    def __init__(self):
        self._boundary_conditions: List[BoundaryCondition] = []
        self._setup_default_conditions()

    def _setup_default_conditions(self):
        """设置默认边界条件 - 使用参数对象"""
        # 值范围检查
        range_config = BoundaryConditionConfig(
            condition_type=BoundaryConditionType.VALUE_OUT_OF_RANGE,
            severity="warning",
            description="数值超出有效范围",
            suggested_action="检查输入值是否在允许范围内",
            context={}
        )
        self.add_boundary_condition_from_config(range_config)

        # 空值检查
        null_config = BoundaryConditionConfig(
            condition_type=BoundaryConditionType.NULL_REFERENCE,
            severity="error",
            description="发现空引用",
            suggested_action="验证对象是否已正确初始化",
            context={}
        )
        self.add_boundary_condition_from_config(null_config)

    def add_boundary_condition(self, condition_type: BoundaryConditionType,
                               severity: str, description: str, suggested_action: str,
                               context: Dict[str, Any]) -> None:
        """添加边界条件"""
        condition = BoundaryCondition(
            condition_type=condition_type,
            severity=severity,
            description=description,
            suggested_action=suggested_action,
            context=context
        )
        self._boundary_conditions.append(condition)
        logger.info(f"添加边界条件: {description}")

    def add_boundary_condition_from_config(self, config: BoundaryConditionConfig) -> None:
        """使用配置对象添加边界条件"""
        self.add_boundary_condition(
            config.condition_type,
            config.severity,
            config.description,
            config.suggested_action,
            config.context
        )

    def check_boundary_conditions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查边界条件"""
        results = []

        for condition in self._boundary_conditions:
            # 这里可以实现具体的边界条件检查逻辑
            # 为了简化，这里只是返回条件信息
            results.append({
                'condition_type': condition.condition_type.value,
                'severity': condition.severity,
                'description': condition.description,
                'suggested_action': condition.suggested_action,
                'triggered': False  # 简化为不触发
            })

        return results

    def get_boundary_conditions_count(self) -> int:
        """获取边界条件数量"""
        return len(self._boundary_conditions)
