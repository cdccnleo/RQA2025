"""轻量化分类组件占位实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ClassifierComponent:
    classifier_id: int
    component_type: str = "Classifier"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_classifier_id(self) -> int:
        return self.classifier_id

    def get_info(self) -> Dict[str, Any]:
        return {
            "classifier_id": self.classifier_id,
            "component_type": self.component_type,
            "component_name": f"{self.component_type}_Component_{self.classifier_id}",
            "created_at": self.created_at.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "classifier_id": self.classifier_id,
            "status": "ready",
            "component_type": self.component_type,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "classifier_id": self.classifier_id,
            "component_type": self.component_type,
            "processed_at": datetime.utcnow().isoformat(),
            "input": data,
        }


class ClassifierComponentFactory:
    def __init__(self, supported_ids: Optional[Iterable[int]] = None) -> None:
        self.supported_ids = list(supported_ids or [])
        self.components: Dict[int, ClassifierComponent] = {}

    def register(self, component: ClassifierComponent) -> None:
        if self.supported_ids and component.classifier_id not in self.supported_ids:
            raise ValueError("不支持的 classifier_id")
        self.components[component.classifier_id] = component

    def create_component(self, classifier_id: int, **kwargs: Any) -> ClassifierComponent:
        if self.supported_ids and classifier_id not in self.supported_ids:
            raise ValueError("不支持的 classifier_id")
        component = ClassifierComponent(classifier_id=classifier_id, **kwargs)
        self.components[classifier_id] = component
        return component

    def get_component(self, classifier_id: int) -> Optional[ClassifierComponent]:
        return self.components.get(classifier_id)

    def list_components(self) -> List[int]:
        return sorted(self.components.keys())


__all__ = ["ClassifierComponent", "ClassifierComponentFactory"]
