from abc import ABC, abstractmethod
from typing import Any


class IDataModel(ABC):

    @abstractmethod
    def get_data(self) -> Any:
        """
        获取数据，返回DataFrame或其他结构
        """
