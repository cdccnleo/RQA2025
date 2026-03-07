"""轻量化 Stacking 组件占位实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class StackingComponent:
    stacking_id: int
    component_type: str = "Stacking"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_stacking_id(self) -> int:
        return self.stacking_id

    def get_info(self) -> Dict[str, Any]:
        return {
            "stacking_id": self.stacking_id,
            "component_type": self.component_type,
            "component_name": f"{self.component_type}_Component_{self.stacking_id}",
            "created_at": self.created_at.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "stacking_id": self.stacking_id,
            "status": "ready",
            "component_type": self.component_type,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "stacking_id": self.stacking_id,
            "component_type": self.component_type,
            "processed_at": datetime.utcnow().isoformat(),
            "input": data,
        }


class StackingComponentFactory:
    def __init__(self, supported_ids: Optional[Iterable[int]] = None) -> None:
        self.supported_ids = list(supported_ids or [])
        self.components: Dict[int, StackingComponent] = {}

    def register(self, component: StackingComponent) -> None:
        if self.supported_ids and component.stacking_id not in self.supported_ids:
            raise ValueError("不支持的 stacking_id")
        self.components[component.stacking_id] = component

    def create_component(self, stacking_id: int, **kwargs: Any) -> StackingComponent:
        if self.supported_ids and stacking_id not in self.supported_ids:
            raise ValueError("不支持的 stacking_id")
        component = StackingComponent(stacking_id=stacking_id, **kwargs)
        self.components[stacking_id] = component
        return component

    def get_component(self, stacking_id: int) -> Optional[StackingComponent]:
        return self.components.get(stacking_id)

    def list_components(self) -> List[int]:
        return sorted(self.components.keys())


__all__ = ["StackingComponent", "StackingComponentFactory"]
