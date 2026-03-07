from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import time

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


"""
统一Compliance组件工厂

合并所有compliance_*.py模板文件为统一的管理架构
生成时间: 2025-08-24 10:13:48
"""


class IComplianceComponent(ABC):

    """Compliance组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        pass

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        pass

    @abstractmethod
    def get_compliance_id(self) -> int:
        """获取compliance ID"""
        pass


class ComplianceComponent(IComplianceComponent):
    """合规组件实现类"""

    def __init__(self, compliance_id: int, component_type: str = "Compliance"):
        """初始化组件"""
        self.compliance_id = compliance_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{compliance_id}"
        self.creation_time = datetime.now()


    def get_compliance_id(self) -> int:
        """获取compliance ID"""
        return self.compliance_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "compliance_id": self.compliance_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat()
        }

    def check_compliance(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """检查合规性"""
        try:
            # 基本合规检查逻辑
            symbol = trade_data.get('symbol', '')
            quantity = trade_data.get('quantity', 0)
            price = trade_data.get('price', 0.0)
            trade_type = trade_data.get('trade_type', 'buy')
            market_volatility = trade_data.get('market_volatility', 0.0)
            liquidity = trade_data.get('liquidity', 1.0)

            # 简单的合规规则
            compliance_score = 1.0
            violations = []

            # 检查市场波动性
            if market_volatility > 0.1:
                compliance_score -= 0.2
                violations.append("高市场波动性")

            # 检查流动性
            if liquidity < 0.3:
                compliance_score -= 0.3
                violations.append("流动性不足")

            # 检查交易量
            if quantity > 10000:
                compliance_score -= 0.1
                violations.append("大额交易")

            return {
                "compliant": compliance_score >= 0.7,
                "compliance_score": compliance_score,
                "violations": violations,
                "component_id": self.compliance_id,
                "checked_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"合规检查失败: {e}")
            return {
                "compliant": False,
                "compliance_score": 0.0,
                "violations": [f"检查失败: {str(e)}"],
                "component_id": self.compliance_id,
                "checked_at": datetime.now().isoformat()
            }


    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "compliance_id": self.compliance_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_compliance_processing"
            }
            return result
        except Exception as e:
            return {
                "compliance_id": self.compliance_id,
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
            "compliance_id": self.compliance_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class ComplianceComponentFactory:

    """Compliance组件工厂"""

    # 支持的compliance ID列表
    SUPPORTED_COMPLIANCE_IDS = [1, 6]

    @staticmethod
    def create_component(compliance_id: int) -> ComplianceComponent:
        """创建指定ID的compliance组件"""
        if compliance_id not in ComplianceComponentFactory.SUPPORTED_COMPLIANCE_IDS:
            raise ValueError(
                f"不支持的compliance ID: {compliance_id}。支持的ID: {ComplianceComponentFactory.SUPPORTED_COMPLIANCE_IDS}")

        return ComplianceComponent(compliance_id, "Compliance")

    @staticmethod
    def get_available_compliances() -> List[int]:
        """获取所有可用的compliance ID"""
        return sorted(list(ComplianceComponentFactory.SUPPORTED_COMPLIANCE_IDS))

    @staticmethod
    def create_all_compliances() -> Dict[int, ComplianceComponent]:
        """创建所有可用compliance"""
        return {
            compliance_id: ComplianceComponent(compliance_id, "Compliance")
            for compliance_id in ComplianceComponentFactory.SUPPORTED_COMPLIANCE_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "ComplianceComponentFactory",
            "version": "2.0.0",
            "total_compliances": len(ComplianceComponentFactory.SUPPORTED_COMPLIANCE_IDS),
            "supported_ids": sorted(list(ComplianceComponentFactory.SUPPORTED_COMPLIANCE_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": f"统一Compliance组件工厂，替代原有的{len(ComplianceComponentFactory.SUPPORTED_COMPLIANCE_IDS)}个模板化文件"
        }

# 向后兼容：创建旧的组件实例
def create_compliance_compliance_component_1():
    return ComplianceComponentFactory.create_component(1)


def create_compliance_compliance_component_6():
    return ComplianceComponentFactory.create_component(6)


__all__ = [
    "IComplianceComponent",
    "ComplianceComponent",
    "ComplianceComponentFactory",
    "create_compliance_compliance_component_1",
    "create_compliance_compliance_component_6",
]
