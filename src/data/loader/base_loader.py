from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generic, TypeVar
from dataclasses import dataclass
from datetime import datetime

T = TypeVar('T')  # 加载的数据类型

@dataclass
class LoaderConfig:
    """数据加载器配置"""
    batch_size: int = 100
    max_retries: int = 3
    timeout: int = 30  # 超时时间(秒)

class BaseDataLoader(ABC, Generic[T]):
    """数据加载器抽象基类，定义所有数据加载器的公共接口
    
    子类需要实现:
    - load(): 实际的数据加载逻辑
    - get_metadata(): 获取加载器元数据
    """

    def __init__(self, config: LoaderConfig = None):
        self.config = config or LoaderConfig()
        self._load_count = 0
        self._last_load_time = None

    @abstractmethod
    def load(self, *args, **kwargs) -> T:
        """加载数据的主方法

        参数:
            *args: 可变位置参数
            **kwargs: 可变关键字参数

        返回:
            加载的数据，具体类型由子类决定
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据

        返回:
            包含加载器元数据的字典，通常包括:
            - loader_type: 加载器类型
            - version: 版本号
            - description: 描述信息
        """
        pass

    def validate(self, data: T) -> bool:
        """验证加载的数据是否符合预期

        参数:
            data: 要验证的数据

        返回:
            bool: 数据是否有效
        """
        return data is not None

    def batch_load(self, params_list: List[Dict]) -> List[T]:
        """批量加载数据
        
        参数:
            params_list: 参数列表，每个元素是传递给load()的参数字典
            
        返回:
            加载的数据列表
        """
        results = []
        for params in params_list:
            try:
                data = self.load(**params)
                if self.validate(data):
                    results.append(data)
            except Exception as e:
                continue
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取加载器统计信息
        
        返回:
            包含统计信息的字典:
            - load_count: 总加载次数
            - last_load_time: 最后加载时间
        """
        return {
            "load_count": self._load_count,
            "last_load_time": self._last_load_time
        }

    def _update_stats(self):
        """更新加载统计信息"""
        self._load_count += 1
        self._last_load_time = datetime.now()
