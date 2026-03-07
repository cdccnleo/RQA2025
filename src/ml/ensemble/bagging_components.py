"""轻量化 Bagging 组件占位实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BaggingComponent:
    bagging_id: int
    component_type: str = "Bagging"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_bagging_id(self) -> int:
        return self.bagging_id

    def get_info(self) -> Dict[str, Any]:
        return {
            "bagging_id": self.bagging_id,
            "component_type": self.component_type,
            "component_name": f"{self.component_type}_Component_{self.bagging_id}",
            "created_at": self.created_at.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "bagging_id": self.bagging_id,
            "status": "ready",
            "component_type": self.component_type,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "bagging_id": self.bagging_id,
            "component_type": self.component_type,
            "processed_at": datetime.utcnow().isoformat(),
            "input": data,
        }


class BaggingComponentFactory:
    def __init__(self, supported_ids: Optional[Iterable[int]] = None) -> None:
        self.supported_ids = list(supported_ids or [])
        self.components: Dict[int, BaggingComponent] = {}

    def register(self, component: BaggingComponent) -> None:
        if self.supported_ids and component.bagging_id not in self.supported_ids:
            raise ValueError("不支持的 bagging_id")
        self.components[component.bagging_id] = component

    def create_component(self, bagging_id: int, **kwargs: Any) -> BaggingComponent:
        if self.supported_ids and bagging_id not in self.supported_ids:
            raise ValueError("不支持的 bagging_id")
        component = BaggingComponent(bagging_id=bagging_id, **kwargs)
        self.components[bagging_id] = component
        return component

    def get_component(self, bagging_id: int) -> Optional[BaggingComponent]:
        return self.components.get(bagging_id)

    def list_components(self) -> List[int]:
        return sorted(self.components.keys())


__all__ = ["BaggingComponent", "BaggingComponentFactory"]
