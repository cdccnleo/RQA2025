from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ComponentFactory:

    """组件工厂"""

    def __init__(self):

        self._components = {}

    def create_component(self, component_type: str, config: Dict[str, Any]):
        """创建组件"""
        try:
            component = self._create_component_instance(component_type, config)
            if component and component.initialize(config):
                return component
            return None
        except Exception as e:
            logger.error(f"创建组件失败: {e}")
            return None

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        """创建组件实例"""
        return None


#!/usr/bin/env python3
"""
统一Health组件工厂

合并所有health_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:32:27
"""


class IHealthComponent(ABC):

    """Health组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def get_health_id(self) -> int:
        """获取health ID"""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """执行健康检查"""

    @abstractmethod
    def check_health_status(self) -> bool:
        """检查健康状态"""

    @abstractmethod
    def validate_component(self) -> Dict[str, Any]:
        """验证组件完整性"""


class HealthComponent(IHealthComponent):

    """统一Health组件实现"""

    def __init__(self, health_id: int, component_type: str = "Health"):
        """初始化组件"""
        self.health_id = health_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{health_id}"
        self.creation_time = datetime.now()

    def get_health_id(self) -> int:
        """获取health ID"""
        return self.health_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "health_id": self.health_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_engine_monitoring_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "health_id": self.health_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_health_processing"
            }
            return result
        except Exception as e:
            return {
                "health_id": self.health_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "health_id": self.health_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }

    def health_check(self) -> Dict[str, Any]:
        """执行健康检查"""
        try:
            # 基础健康检查
            basic_check = self._perform_basic_health_check()
            # 组件特定检查
            component_check = self._perform_component_health_check()
            # 综合评估
            overall_status = self._evaluate_overall_health(basic_check, component_check)

            return {
                "health_id": self.health_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "timestamp": datetime.now().isoformat(),
                "overall_status": overall_status,
                "basic_check": basic_check,
                "component_check": component_check,
                "check_duration_ms": 0,  # 可以后续添加计时
                "recommendations": self._generate_health_recommendations(overall_status)
            }
        except Exception as e:
            logger.error(f"Health check failed for component {self.component_name}: {str(e)}")
            return {
                "health_id": self.health_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    def check_health_status(self) -> bool:
        """检查健康状态"""
        try:
            health_result = self.health_check()
            return health_result.get("overall_status") in ["healthy", "good"]
        except Exception as e:
            logger.error(
                f"Health status check failed for component {self.component_name}: {str(e)}")
            return False

    def validate_component(self) -> Dict[str, Any]:
        """验证组件完整性"""
        validation_results = {
            "health_id": self.health_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "timestamp": datetime.now().isoformat(),
            "validations": []
        }

        # 验证必需属性
        required_attrs = ["health_id", "component_type", "component_name", "creation_time"]
        for attr in required_attrs:
            is_valid = hasattr(self, attr) and getattr(self, attr) is not None
            validation_results["validations"].append({
                "check_type": "attribute_validation",
                "attribute": attr,
                "is_valid": is_valid,
                "expected": f"{attr} should exist and not be None",
                "actual": getattr(self, attr, "MISSING")
            })

        # 验证组件状态
        try:
            status = self.get_status()
            status_valid = isinstance(status, dict) and "status" in status
            validation_results["validations"].append({
                "check_type": "status_validation",
                "is_valid": status_valid,
                "expected": "get_status() should return dict with 'status' key",
                "actual": type(status).__name__ if not status_valid else "VALID"
            })
        except Exception as e:
            validation_results["validations"].append({
                "check_type": "status_validation",
                "is_valid": False,
                "expected": "get_status() should not raise exception",
                "actual": f"Exception: {str(e)}"
            })

        # 计算总体验证结果
        all_valid = all(v["is_valid"] for v in validation_results["validations"])
        validation_results["overall_valid"] = all_valid
        validation_results["total_checks"] = len(validation_results["validations"])
        validation_results["passed_checks"] = sum(
            1 for v in validation_results["validations"] if v["is_valid"])
        validation_results["failed_checks"] = validation_results["total_checks"] - \
            validation_results["passed_checks"]

        return validation_results

    def _perform_basic_health_check(self) -> Dict[str, Any]:
        """执行基础健康检查"""
        return {
            "component_exists": True,
            "attributes_complete": all([
                hasattr(self, 'health_id'),
                hasattr(self, 'component_type'),
                hasattr(self, 'component_name'),
                hasattr(self, 'creation_time')
            ]),
            "creation_time_valid": isinstance(self.creation_time, datetime),
            "time_since_creation_hours": (datetime.now() - self.creation_time).total_seconds() / 3600
        }

    def _perform_component_health_check(self) -> Dict[str, Any]:
        """执行组件特定健康检查"""
        try:
            # 测试基本功能
            info = self.get_info()
            status = self.get_status()

            return {
                "info_accessible": isinstance(info, dict),
                "status_accessible": isinstance(status, dict),
                "info_has_required_fields": isinstance(info, dict) and "health_id" in info,
                "status_has_required_fields": isinstance(status, dict) and "status" in status,
                "component_functional": True
            }
        except Exception as e:
            return {
                "info_accessible": False,
                "status_accessible": False,
                "info_has_required_fields": False,
                "status_has_required_fields": False,
                "component_functional": False,
                "error": str(e)
            }

    def _evaluate_overall_health(self, basic_check: Dict[str, Any], component_check: Dict[str, Any]) -> str:
        """评估总体健康状态"""
        # 计算健康评分
        scores = []

        # 基础检查评分
        if basic_check.get("component_exists", False):
            scores.append(1.0)
        if basic_check.get("attributes_complete", False):
            scores.append(1.0)
        if basic_check.get("creation_time_valid", False):
            scores.append(1.0)

        # 组件检查评分
        if component_check.get("component_functional", False):
            scores.append(1.0)
        if component_check.get("info_accessible", False):
            scores.append(0.5)
        if component_check.get("status_accessible", False):
            scores.append(0.5)

        # 计算平均分
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score >= 0.9:
                return "excellent"
            elif avg_score >= 0.7:
                return "healthy"
            elif avg_score >= 0.5:
                return "warning"
            else:
                return "critical"
        else:
            return "unknown"

    def _generate_health_recommendations(self, overall_status: str) -> List[str]:
        """生成健康建议"""
        recommendations = []

        if overall_status in ["warning", "critical"]:
            recommendations.append("检查组件配置和依赖关系")
            recommendations.append("验证组件初始化过程")
            recommendations.append("检查系统资源使用情况")

        if overall_status == "critical":
            recommendations.append("立即检查组件错误日志")
            recommendations.append("考虑重启或重建组件")
            recommendations.append("联系技术支持团队")

        if overall_status in ["healthy", "excellent"]:
            recommendations.append("继续监控组件状态")
            recommendations.append("定期执行健康检查")

        return recommendations


class HealthComponentFactory:

    """Health组件工厂"""

    # 支持的health ID列表
    SUPPORTED_HEALTH_IDS = [4, 9]

    @staticmethod
    def create_component(health_id: int) -> HealthComponent:
        """创建指定ID的health组件"""
        if health_id not in HealthComponentFactory.SUPPORTED_HEALTH_IDS:
            raise ValueError(
                f"不支持的health ID: {health_id}。支持的ID: {HealthComponentFactory.SUPPORTED_HEALTH_IDS}")

        return HealthComponent(health_id, "Health")

    @staticmethod
    def get_available_healths() -> List[int]:
        """获取所有可用的health ID"""
        return sorted(list(HealthComponentFactory.SUPPORTED_HEALTH_IDS))

    @staticmethod
    def create_all_healths() -> Dict[int, HealthComponent]:
        """创建所有可用health"""
        return {
            health_id: HealthComponent(health_id, "Health")
            for health_id in HealthComponentFactory.SUPPORTED_HEALTH_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "HealthComponentFactory",
            "version": "2.0.0",
            "total_healths": len(HealthComponentFactory.SUPPORTED_HEALTH_IDS),
            "supported_ids": sorted(list(HealthComponentFactory.SUPPORTED_HEALTH_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_health_health_component_4(): return HealthComponentFactory.create_component(4)


def create_health_health_component_9(): return HealthComponentFactory.create_component(9)


__all__ = [
    "IHealthComponent",
    "HealthComponent",
    "HealthComponentFactory",
    "create_health_health_component_4",
    "create_health_health_component_9",
]
