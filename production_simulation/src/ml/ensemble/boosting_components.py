"""轻量化 Boosting 组件占位实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BoostingComponent:
    boosting_id: int
    component_type: str = "Boosting"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_boosting_id(self) -> int:
        return self.boosting_id

    def get_info(self) -> Dict[str, Any]:
        return {
            "boosting_id": self.boosting_id,
            "component_type": self.component_type,
            "component_name": f"{self.component_type}_Component_{self.boosting_id}",
            "created_at": self.created_at.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "boosting_id": self.boosting_id,
            "status": "ready",
            "component_type": self.component_type,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "boosting_id": self.boosting_id,
            "component_type": self.component_type,
            "processed_at": datetime.utcnow().isoformat(),
            "input": data,
        }


class BoostingComponentFactory:
    def __init__(self, supported_ids: Optional[Iterable[int]] = None) -> None:
        self.supported_ids = list(supported_ids or [])
        self.components: Dict[int, BoostingComponent] = {}

    def register(self, component: BoostingComponent) -> None:
        if self.supported_ids and component.boosting_id not in self.supported_ids:
            raise ValueError("不支持的 boosting_id")
        self.components[component.boosting_id] = component

    def create_component(self, boosting_id: int, **kwargs: Any) -> BoostingComponent:
        if self.supported_ids and boosting_id not in self.supported_ids:
            raise ValueError("不支持的 boosting_id")
        component = BoostingComponent(boosting_id=boosting_id, **kwargs)
        self.components[boosting_id] = component
        return component

    def get_component(self, boosting_id: int) -> Optional[BoostingComponent]:
        return self.components.get(boosting_id)

    def list_components(self) -> List[int]:
        return sorted(self.components.keys())


__all__ = ["BoostingComponent", "BoostingComponentFactory"]
