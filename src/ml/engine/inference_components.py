"""轻量化推理组件，用于在 engine 层进行单元测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class InferenceComponent:
    inference_id: int
    component_type: str = "Inference"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_inference_id(self) -> int:
        return self.inference_id

    def get_info(self) -> Dict[str, Any]:
        return {
            "inference_id": self.inference_id,
            "component_type": self.component_type,
            "component_name": f"{self.component_type}_Component_{self.inference_id}",
            "created_at": self.created_at.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "inference_id": self.inference_id,
            "status": "idle",
            "component_type": self.component_type,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "inference_id": self.inference_id,
            "component_type": self.component_type,
            "processed_at": datetime.utcnow().isoformat(),
            "input": data,
        }


class InferenceComponentFactory:
    def __init__(self, supported_ids: Optional[Iterable[int]] = None) -> None:
        self.supported_ids = list(supported_ids or [])
        self.components: Dict[int, InferenceComponent] = {}

    def register(self, component: InferenceComponent) -> None:
        if self.supported_ids and component.inference_id not in self.supported_ids:
            raise ValueError("不支持的 inference_id")
        self.components[component.inference_id] = component

    def create_component(self, inference_id: int, **kwargs: Any) -> InferenceComponent:
        if self.supported_ids and inference_id not in self.supported_ids:
            raise ValueError("不支持的 inference_id")
        component = InferenceComponent(inference_id=inference_id, **kwargs)
        self.components[inference_id] = component
        return component

    def get_component(self, inference_id: int) -> Optional[InferenceComponent]:
        return self.components.get(inference_id)

    def list_components(self) -> List[int]:
        return sorted(self.components.keys())


__all__ = ["InferenceComponent", "InferenceComponentFactory"]

