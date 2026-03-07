from datetime import datetime
from typing import Any, Dict, Optional, List

import pandas as pd


DEFAULT_FREQUENCY = "1d"
_UNSET = object()


class DataModel:
    """基础数据模型，提供数据访问、元数据管理与简单校验能力。"""

    def __init__(
        self,
        data: Any = _UNSET,
        name: str = "default",
        data_type: str = "unknown",
        frequency: Optional[str] = "daily",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.data_type = data_type
        self.frequency = frequency
        self._data = {} if data is _UNSET else data
        self._metadata = dict(metadata) if metadata else {}
        # 兼容历史属性名称
        self._user_metadata = self._metadata
        self.created_at = datetime.now()
        self.version = 1

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._data = value
        self.version += 1

    # 数据操作 -----------------------------------------------------------------
    def get_data(self, key: Optional[str] = None):
        """获取数据。若提供 key，则返回对应值，否则返回完整数据结构。"""
        if key is None:
            return self._data
        if isinstance(self._data, dict):
            return self._data.get(key)
        if hasattr(self._data, "get"):
            return self._data.get(key)
        return None

    def set_data(self, key: Optional[str], value: Any = None) -> None:
        """设置数据。支持 set_data(data) 或 set_data(key, value) 两种形式。"""
        if value is None:
            self._data = key
        else:
            if not isinstance(self._data, dict):
                self._data = {}
            self._data[key] = value
        self.version += 1

    def update_data(self, new_data: Any) -> bool:
        """更新数据并增加版本号。"""
        self._data = new_data
        self.version += 1
        return True

    def has_data(self, key: Optional[str] = None) -> bool:
        """检查是否存在数据或指定键的数据。"""
        if key is None:
            if isinstance(self._data, pd.DataFrame):
                return not self._data.empty
            return self._data is not None
        if isinstance(self._data, dict):
            return key in self._data
        if hasattr(self._data, "get"):
            return self._data.get(key) is not None
        return False

    def clear_data(self) -> None:
        """清空数据。"""
        if isinstance(self._data, pd.DataFrame):
            self._data = pd.DataFrame()
        elif isinstance(self._data, dict):
            self._data.clear()
        else:
            self._data = None

    def get_data_keys(self) -> List[str]:
        """获取数据键集合。"""
        if isinstance(self._data, dict):
            return list(self._data.keys())
        if isinstance(self._data, pd.DataFrame):
            return list(self._data.columns)
        return []

    # 元数据操作 ---------------------------------------------------------------
    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: Optional[str] = None, user_only: bool = False):
        base_metadata = {
            "name": self.name,
            "data_type": self.data_type,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }
        if self.frequency:
            base_metadata["frequency"] = self.frequency

        if user_only:
            metadata = dict(self._metadata)
        elif not self._metadata:
            # 默认没有用户元数据时返回空字典，保持向后兼容
            metadata = {}
        else:
            metadata = {**base_metadata, **self._metadata}

        if key is None:
            return metadata
        return metadata.get(key)

    def merge_metadata(self, new_metadata: Dict[str, Any]) -> bool:
        self._metadata.update(new_metadata or {})
        return True

    def get_metadata_keys(self) -> List[str]:
        return list(self._metadata.keys())

    # 信息获取 -----------------------------------------------------------------
    def get_frequency(self) -> str:
        if self.frequency:
            return self.frequency
        if isinstance(self._data, pd.DataFrame):
            if isinstance(self._data.index, pd.DatetimeIndex):
                freq = self._data.index.freq or self._data.index.inferred_freq
                if freq:
                    return str(freq)
            return "unknown"
        if not self._data:
            return DEFAULT_FREQUENCY
        return "unknown"

    def get_data_type(self) -> str:
        return self.data_type

    def get_summary(self) -> Dict[str, Any]:
        if isinstance(self._data, pd.DataFrame):
            shape = self._data.shape
            columns = list(self._data.columns)
        elif isinstance(self._data, dict):
            shape = (len(self._data),)
            columns = list(self._data.keys())
        else:
            shape = (0,)
            columns = []

        return {
            "name": self.name,
            "data_type": self.data_type,
            "shape": shape,
            "columns": columns,
            "frequency": self.get_frequency(),
            "version": self.version,
        }

    # 校验与序列化 -------------------------------------------------------------
    def validate(self) -> bool:
        if not self.name or not self.data_type:
            return False
        if self._data is None:
            return False
        if isinstance(self._data, pd.DataFrame) and self._data.empty:
            return False
        return True

    def to_dict(self) -> dict:
        data_copy = (
            self._data.copy()
            if isinstance(self._data, pd.DataFrame)
            else self._data.copy()
            if isinstance(self._data, dict)
            else self._data
        )
        return {
            "name": self.name,
            "data_type": self.data_type,
            "frequency": self.frequency,
            "data": data_copy,
            "metadata": dict(self._metadata),
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: dict):
        data_value = payload["data"] if "data" in payload else _UNSET
        frequency_value = payload["frequency"] if "frequency" in payload else None
        instance = cls(
            data=data_value,
            name=payload.get("name", "default"),
            data_type=payload.get("data_type", "unknown"),
            frequency=frequency_value,
            metadata=payload.get("metadata", {}),
        )
        created_at = payload.get("created_at")
        if created_at:
            try:
                instance.created_at = datetime.fromisoformat(created_at)
            except ValueError:
                pass
        instance.version = payload.get("version", 1)
        return instance

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            "name": self.name,
            "data_type": self.data_type,
            "has_data": self.has_data(),
            "version": self.version,
        }

        if isinstance(self._data, pd.DataFrame):
            stats.update({
                "row_count": len(self._data),
                "column_count": len(self._data.columns),
                "columns": list(self._data.columns),
                "dtypes": self._data.dtypes.astype(str).to_dict(),
            })
        elif isinstance(self._data, dict):
            stats.update({
                "item_count": len(self._data),
                "keys": list(self._data.keys()),
            })
        else:
            stats["data_type_info"] = str(type(self._data))

        return stats
