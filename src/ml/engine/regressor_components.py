"""轻量化回归组件，提供统一占位实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class RegressorComponent:
    regressor_id: int
    component_type: str = "Regressor"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_regressor_id(self) -> int:
        return self.regressor_id

    def get_info(self) -> Dict[str, Any]:
        return {
            "regressor_id": self.regressor_id,
            "component_type": self.component_type,
            "component_name": f"{self.component_type}_Component_{self.regressor_id}",
            "created_at": self.created_at.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "regressor_id": self.regressor_id,
            "status": "ready",
            "component_type": self.component_type,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "regressor_id": self.regressor_id,
            "component_type": self.component_type,
            "processed_at": datetime.utcnow().isoformat(),
            "input": data,
        }


class RegressorComponentFactory:
    def __init__(self, supported_ids: Optional[Iterable[int]] = None) -> None:
        self.supported_ids = list(supported_ids or [])
        self.components: Dict[int, RegressorComponent] = {}

    def register(self, component: RegressorComponent) -> None:
        if self.supported_ids and component.regressor_id not in self.supported_ids:
            raise ValueError("不支持的 regressor_id")
        self.components[component.regressor_id] = component

    def create_component(self, regressor_id: int, **kwargs: Any) -> RegressorComponent:
        if self.supported_ids and regressor_id not in self.supported_ids:
            raise ValueError("不支持的 regressor_id")
        component = RegressorComponent(regressor_id=regressor_id, **kwargs)
        self.components[regressor_id] = component
        return component

    def get_component(self, regressor_id: int) -> Optional[RegressorComponent]:
        return self.components.get(regressor_id)

    def list_components(self) -> List[int]:
        return sorted(self.components.keys())


__all__ = ["RegressorComponent", "RegressorComponentFactory"]

