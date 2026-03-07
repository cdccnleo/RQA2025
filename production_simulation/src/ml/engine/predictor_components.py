"""轻量化预测组件，提供统一占位实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class PredictorComponent:
    predictor_id: int
    component_type: str = "Predictor"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_predictor_id(self) -> int:
        return self.predictor_id

    def get_info(self) -> Dict[str, Any]:
        return {
            "predictor_id": self.predictor_id,
            "component_type": self.component_type,
            "component_name": f"{self.component_type}_Component_{self.predictor_id}",
            "created_at": self.created_at.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "predictor_id": self.predictor_id,
            "status": "ready",
            "component_type": self.component_type,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "predictor_id": self.predictor_id,
            "component_type": self.component_type,
            "processed_at": datetime.utcnow().isoformat(),
            "input": data,
        }


class PredictorComponentFactory:
    def __init__(self, supported_ids: Optional[Iterable[int]] = None) -> None:
        self.supported_ids = list(supported_ids or [])
        self.components: Dict[int, PredictorComponent] = {}

    def register(self, component: PredictorComponent) -> None:
        if self.supported_ids and component.predictor_id not in self.supported_ids:
            raise ValueError("不支持的 predictor_id")
        self.components[component.predictor_id] = component

    def create_component(self, predictor_id: int, **kwargs: Any) -> PredictorComponent:
        if self.supported_ids and predictor_id not in self.supported_ids:
            raise ValueError("不支持的 predictor_id")
        component = PredictorComponent(predictor_id=predictor_id, **kwargs)
        self.components[predictor_id] = component
        return component

    def get_component(self, predictor_id: int) -> Optional[PredictorComponent]:
        return self.components.get(predictor_id)

    def list_components(self) -> List[int]:
        return sorted(self.components.keys())


__all__ = ["PredictorComponent", "PredictorComponentFactory"]

