"""轻量化 Voting 组件占位实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class VotingComponent:
    voting_id: int
    component_type: str = "Voting"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_voting_id(self) -> int:
        return self.voting_id

    def get_info(self) -> Dict[str, Any]:
        return {
            "voting_id": self.voting_id,
            "component_type": self.component_type,
            "component_name": f"{self.component_type}_Component_{self.voting_id}",
            "created_at": self.created_at.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "voting_id": self.voting_id,
            "status": "ready",
            "component_type": self.component_type,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "voting_id": self.voting_id,
            "component_type": self.component_type,
            "processed_at": datetime.utcnow().isoformat(),
            "input": data,
        }


class VotingComponentFactory:
    def __init__(self, supported_ids: Optional[Iterable[int]] = None) -> None:
        self.supported_ids = list(supported_ids or [])
        self.components: Dict[int, VotingComponent] = {}

    def register(self, component: VotingComponent) -> None:
        if self.supported_ids and component.voting_id not in self.supported_ids:
            raise ValueError("不支持的 voting_id")
        self.components[component.voting_id] = component

    def create_component(self, voting_id: int, **kwargs: Any) -> VotingComponent:
        if self.supported_ids and voting_id not in self.supported_ids:
            raise ValueError("不支持的 voting_id")
        component = VotingComponent(voting_id=voting_id, **kwargs)
        self.components[voting_id] = component
        return component

    def get_component(self, voting_id: int) -> Optional[VotingComponent]:
        return self.components.get(voting_id)

    def list_components(self) -> List[int]:
        return sorted(self.components.keys())


__all__ = ["VotingComponent", "VotingComponentFactory"]
