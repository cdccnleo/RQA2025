
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
"""
基础设施层工具系统 - 存储适配器基类

提供统一的存储接口和基础实现。
"""


class StorageAdapter(ABC):
    """存储适配器基类"""

    def __init__(self, base_path: str = ""):
        """
        初始化存储适配器

        Args:
            base_path: 基础存储路径
        """
        self.base_path = Path(base_path) if base_path else Path(".")
        self._ensure_base_path()

    def _ensure_base_path(self) -> None:
        """确保基础路径存在"""
        try:
            if self.base_path != Path("."):
                self.base_path.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"创建存储目录失败，没有足够权限: {self.base_path}") from e
        except OSError as e:
            raise OSError(f"创建存储目录失败: {self.base_path}, 错误: {e}") from e
        except Exception as e:
            raise RuntimeError(f"存储初始化失败: {e}") from e

    @abstractmethod
    def save(self, key: str, data: Any, **kwargs) -> bool:
        """
        保存数据

        Args:
            key: 数据键
            data: 要保存的数据
            **kwargs: 额外参数

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def load(self, key: str, **kwargs) -> Optional[Any]:
        """
        加载数据

        Args:
            key: 数据键
            **kwargs: 额外参数

        Returns:
            Optional[Any]: 加载的数据，如果不存在返回None
        """

    @abstractmethod
    def delete(self, key: str, **kwargs) -> bool:
        """
        删除数据

        Args:
            key: 数据键
            **kwargs: 额外参数

        Returns:
            bool: 删除是否成功
        """

    @abstractmethod
    def exists(self, key: str, **kwargs) -> bool:
        """
        检查数据是否存在

        Args:
            key: 数据键
            **kwargs: 额外参数

        Returns:
            bool: 数据是否存在
        """

    @abstractmethod
    def list_keys(self, prefix: str = "", **kwargs) -> List[str]:
        """
        列出所有键

        Args:
            prefix: 键前缀过滤
            **kwargs: 额外参数

        Returns:
            List[str]: 键列表
        """

    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "adapter_type": self.__class__.__name__,
            "base_path": str(self.base_path),
        }
