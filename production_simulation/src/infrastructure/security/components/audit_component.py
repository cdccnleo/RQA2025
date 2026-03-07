from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

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
统一Audit组件工厂

合并所有audit_*.py模板文件为统一的管理架构
    生成时间: 2025 - 08 - 24 09:55:55
"""


class IAuditComponent(ABC):

    """Audit组件接口"""

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
    def get_audit_id(self) -> int:
        """获取audit ID"""


class AuditComponent(IAuditComponent):

    """统一Audit组件实现"""

    def __init__(self, audit_id: int, component_type: str = "Audit", *, managed: bool = False):
        """初始化组件"""
        self.audit_id = audit_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{audit_id}"
        self.creation_time = datetime.now()
        self.metadata: Dict[str, Any] = {
            "tags": [],
            "owner": "security_team",
        }
        self.process_count: int = 0
        self.error_count: int = 0
        self.last_processed_at: Optional[datetime] = None
        self._recent_events: deque = deque(maxlen=50)
        self._status: str = "active"
        self._managed: bool = managed

    def get_audit_id(self) -> int:
        """获取audit ID"""
        return self.audit_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        status = self._resolve_status()
        return {
            "audit_id": self.audit_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "status": status,
            "process_count": self.process_count,
            "error_count": self.error_count,
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
            "recent_events": list(self._recent_events),
            "metadata": dict(self.metadata),
            "description": f"统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_security_management_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            if isinstance(data, dict):
                normalized_data = data
            else:
                normalized_data = {"value": data}
            self.process_count += 1
            self.last_processed_at = datetime.now()
            event_record = {
                "received_at": self.last_processed_at.isoformat(),
                "data_preview": {k: normalized_data.get(k) for k in list(normalized_data.keys())[:5]},
            }
            self._recent_events.append(event_record)
            result = {
                "audit_id": self.audit_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": normalized_data,
                "processed_at": self.last_processed_at.isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_audit_processing"
            }

            if self._managed:
                self._status = "processed"
                self.metadata["process_count"] = self.process_count
                self.metadata["last_processed_at"] = self.last_processed_at.isoformat()
            return result
        except Exception as e:
            self.error_count += 1
            self.last_processed_at = datetime.now()
            self._status = "error"
            return {
                "audit_id": self.audit_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": self.last_processed_at.isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        status = self._resolve_status()
        return {
            "audit_id": self.audit_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": status,
            "creation_time": self.creation_time.isoformat(),
            "process_count": self.process_count,
            "error_count": self.error_count,
            "health": "good",
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
            "metadata": dict(self.metadata),
        }

    def _resolve_status(self) -> str:
        if self._status == "error":
            return "error"
        if self._managed and self.process_count > 0:
            return "processed"
        return "active"


class AuditComponentFactory:

    """Audit组件工厂"""

    # 支持的audit ID列表
    SUPPORTED_AUDIT_IDS = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58]

    @staticmethod
    def create_component(audit_id: int) -> AuditComponent:
        """创建指定ID的audit组件"""
        if audit_id not in AuditComponentFactory.SUPPORTED_AUDIT_IDS:
            raise ValueError(
                f"不支持的audit ID: {audit_id}。支持的ID: {AuditComponentFactory.SUPPORTED_AUDIT_IDS}")

        return AuditComponent(audit_id, "Audit", managed=True)

    @staticmethod
    def get_available_audits() -> List[int]:
        """获取所有可用的audit ID"""
        return sorted(list(AuditComponentFactory.SUPPORTED_AUDIT_IDS))

    @staticmethod
    def create_all_audits() -> Dict[int, AuditComponent]:
        """创建所有可用audit"""
        audits = {}
        for audit_id in AuditComponentFactory.SUPPORTED_AUDIT_IDS:
            audits[audit_id] = AuditComponentFactory.create_component(audit_id)
        return audits

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "AuditComponentFactory",
            "version": "2.0.0",
            "total_audits": len(AuditComponentFactory.SUPPORTED_AUDIT_IDS),
            "supported_ids": sorted(list(AuditComponentFactory.SUPPORTED_AUDIT_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一审计组件工厂"
        }


__all__ = [
    "IAuditComponent",
    "AuditComponent",
    "AuditComponentFactory"
]
