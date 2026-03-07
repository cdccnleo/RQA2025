"""轻量化集成组件占位模块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EnsembleComponent:
    ensemble_id: int
    component_type: str = "Ensemble"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_ensemble_id(self) -> int:
        return self.ensemble_id

    def get_info(self) -> Dict[str, Any]:
        return {
            "ensemble_id": self.ensemble_id,
            "component_type": self.component_type,
            "component_name": f"{self.component_type}_Component_{self.ensemble_id}",
            "created_at": self.created_at.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "ensemble_id": self.ensemble_id,
            "status": "ready",
            "component_type": self.component_type,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ensemble_id": self.ensemble_id,
            "component_type": self.component_type,
            "processed_at": datetime.utcnow().isoformat(),
            "input": data,
        }


class EnsembleComponentFactory:
    def __init__(self, supported_ids: Optional[Iterable[int]] = None) -> None:
        self.supported_ids = list(supported_ids or [])
        self.components: Dict[int, EnsembleComponent] = {}

    def register(self, component: EnsembleComponent) -> None:
        if self.supported_ids and component.ensemble_id not in self.supported_ids:
            raise ValueError("不支持的 ensemble_id")
        self.components[component.ensemble_id] = component

    def create_component(self, ensemble_id: int, **kwargs: Any) -> EnsembleComponent:
        if self.supported_ids and ensemble_id not in self.supported_ids:
            raise ValueError("不支持的 ensemble_id")
        component = EnsembleComponent(ensemble_id=ensemble_id, **kwargs)
        self.components[ensemble_id] = component
        return component

    def get_component(self, ensemble_id: int) -> Optional[EnsembleComponent]:
        return self.components.get(ensemble_id)

    def list_components(self) -> List[int]:
        return sorted(self.components.keys())


__all__ = ["EnsembleComponent", "EnsembleComponentFactory"]
