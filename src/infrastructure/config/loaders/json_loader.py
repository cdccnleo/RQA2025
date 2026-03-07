
# from src.infrastructure.monitoring.decorators import monitor_resource

from __future__ import annotations

from ..config_exceptions import ConfigLoadError
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ..interfaces.unified_interface import ConfigLoaderStrategy, ConfigFormat, LoaderResult
"""
基础设施层 - 配置管理组件

json_loader 模块

配置管理相关的文件
提供配置管理相关的功能实现。
"""


class JSONLoader(ConfigLoaderStrategy):

    """JSON配置加载策略，支持单文件和批量加载"""

    def __init__(self):
        super().__init__("JSONLoader")
        self.format = ConfigFormat.JSON
        self._last_metadata: Dict[str, Any] = {}

    # @monitor_resource('json_loader')

    def load(self, source: str) -> LoaderResult:
        """
        加载单个JSON配置文件

        Args:
            source: JSON文件路径

        Returns:
            配置数据

        Raises:
            ConfigLoadError: 当文件不存在或IO错误时抛出
        """
        path = Path(source).absolute()
        if not path.exists():
            raise ConfigLoadError(
                f"JSON文件不存在: {source} (文件: {str(path)}, 错误类型: file_not_found)"
            )

        start_time = time.time()

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metadata = {
                'format': ConfigFormat.JSON.value,
                'source': str(path),
                'timestamp': time.time(),
                'load_time': max(time.time() - start_time, 0.0001),
                'size': path.stat().st_size if path.exists() else 0,
            }
            self._last_metadata = metadata

            if isinstance(data, dict):
                return LoaderResult(data, metadata)
            return data

        except json.JSONDecodeError as e:
            raise ConfigLoadError(
                f"JSON解析失败: {str(e)} (文件: {str(path)}, 位置: {e.pos}, 行: {e.lineno}, 列: {e.colno})"
            )

        except Exception as e:
            raise ConfigLoadError(
                f"加载JSON文件失败: {str(e)} (文件: {str(path)})"
            )

    def batch_load(self, sources: List[str]) -> Dict[str, LoaderResult]:
        """
        批量加载多个JSON配置文件

        Args:
            sources: JSON文件路径列表

        Returns:
            按文件路径索引的(配置数据, 元数据)字典

        Raises:
            ConfigLoadError: 当任一文件加载失败时抛出
        """
        results: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        for src in sources:
            try:
                result = self.load(src)
                if isinstance(result, LoaderResult):
                    results[src] = (dict(result), {**result.metadata})
                else:
                    results[src] = (result, {**self._last_metadata})
            except ConfigLoadError as e:
                raise ConfigLoadError(
                    f"批量加载失败: {src}",
                    details={'source': src, 'nested_error': str(e)}
                )

        return results

    def can_load(self, source: str) -> bool:
        """
        检查是否支持加载该配置源

        Args:
            source: 配置源标识(文件路径或URL)

        Returns:
            bool: 当源以.json结尾时返回True
        """
        return isinstance(source, str) and source.lower().endswith('.json')

    def can_handle_source(self, source: str) -> bool:
        """
        检查是否可以处理指定的配置源

        Args:
            source: 配置源标识

        Returns:
            bool: 是否可以处理
        """
        return self.can_load(source)

    def get_supported_formats(self) -> List[ConfigFormat]:
        """
        获取支持的配置格式

        Returns:
            List[ConfigFormat]: 支持的格式列表
        """
        return [ConfigFormat.JSON]

    def get_supported_extensions(self) -> List[str]:
        """
        获取支持的文件扩展名

        Returns:
            支持的扩展名列表
        """
        return ['.json']

    def get_last_metadata(self) -> Dict[str, Any]:
        return self._last_metadata.copy()




