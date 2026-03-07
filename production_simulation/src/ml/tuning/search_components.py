import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ComponentFactory:

    """占位组件工厂（向后兼容）"""

    def __init__(self):
        self._components: Dict[str, Any] = {}

    def create_component(self, component_type: str, config: Dict[str, Any]):
        return None


class ISearchComponent(ABC):

    """Search组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        pass

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        pass

    @abstractmethod
    def get_search_id(self) -> int:
        """获取search ID"""
        pass


class SearchComponent(ISearchComponent):

    """统一Search组件实现"""

    def __init__(self, search_id: int, component_type: str = "Search"):
        """初始化组件"""
        self.search_id = search_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{search_id}"
        self.creation_time = datetime.now()

    def get_search_id(self) -> int:
        """获取search ID"""
        return self.search_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
        "search_id": self.search_id,
        "component_name": self.component_name,
        "component_type": self.component_type,
        "creation_time": self.creation_time.isoformat(),
        "description": "统一{self.component_type}组件实现",
        "version": "2.0.0",
        "type": "unified_ml_tuning_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "search_id": self.search_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_search_processing"
            }
            return result
        except Exception as e:
            return {
            "search_id": self.search_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "input_data": data,
            "processed_at": datetime.now().isoformat(),
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
        "search_id": self.search_id,
        "component_name": self.component_name,
        "component_type": self.component_type,
        "status": "active",
        "creation_time": self.creation_time.isoformat(),
        "health": "good"
        }


class SearchComponentFactory:

    """Search组件工厂"""

    SUPPORTED_SEARCH_IDS = [4, 9, 14, 19, 24]

    @staticmethod
    def create_component(search_id: int) -> SearchComponent:
        """创建指定ID的search组件"""
        if search_id not in SearchComponentFactory.SUPPORTED_SEARCH_IDS:
            raise ValueError(
                f"不支持的search ID: {search_id}。支持的ID: {SearchComponentFactory.SUPPORTED_SEARCH_IDS}")

        return SearchComponent(search_id, "Search")

    @staticmethod
    def get_available_searches() -> List[int]:
        """获取所有可用的search ID"""
        return sorted(list(SearchComponentFactory.SUPPORTED_SEARCH_IDS))

    @staticmethod
    def create_all_searches() -> Dict[int, SearchComponent]:
        """创建所有可用search"""
        return {
            search_id: SearchComponent(search_id, "Search")
            for search_id in SearchComponentFactory.SUPPORTED_SEARCH_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "SearchComponentFactory",
            "version": "2.0.0",
            "total_searches": len(SearchComponentFactory.SUPPORTED_SEARCH_IDS),
            "supported_ids": sorted(list(SearchComponentFactory.SUPPORTED_SEARCH_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一Search组件工厂，替代原模板文件"
        }

        # 向后兼容：创建旧的组件实例

def create_search_search_component_4():
    return SearchComponentFactory.create_component(4)


def create_search_search_component_9():
    return SearchComponentFactory.create_component(9)


def create_search_search_component_14():
    return SearchComponentFactory.create_component(14)


def create_search_search_component_19():
    return SearchComponentFactory.create_component(19)


def create_search_search_component_24():
    return SearchComponentFactory.create_component(24)

__all__ = [
    "ISearchComponent",
    "SearchComponent",
    "SearchComponentFactory",
    "create_search_search_component_4",
    "create_search_search_component_9",
    "create_search_search_component_14",
    "create_search_search_component_19",
    "create_search_search_component_24",
        ]
