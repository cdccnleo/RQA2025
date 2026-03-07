from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class ModelService:
    """简化版模型服务，提供保存与加载的内存实现。"""

    def __init__(self):
        self._store: Dict[Tuple[str, str], Tuple[Any, Optional[Dict[str, Any]]]] = {}

    def save_model(self, model_id: str, version: str, model: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        key = (model_id, version)
        if key in self._store:
            raise ValueError(f"模型 {model_id}@{version} 已存在")
        if metadata is None:
            metadata = {}
        metadata = metadata.copy()
        metadata.setdefault("status", "saved")
        self._store[key] = (model, metadata)

    def load_model(self, model_id: str, version: str) -> Any:
        if (model_id, version) not in self._store:
            raise KeyError(f"模型 {model_id}@{version} 不存在")
        return self._store[(model_id, version)][0]

    def list_models(self) -> Dict[str, Any]:
        return {(mid, ver): meta for (mid, ver), (_, meta) in self._store.items()}


__all__ = ["ModelService"]

