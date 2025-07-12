import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
from ..strategy import ConfigLoaderStrategy
from ..exceptions import ConfigLoadError, ConfigValidationError
from src.infrastructure.monitoring.decorators import monitor_resource

class JSONLoader(ConfigLoaderStrategy):
    """JSON配置加载策略，支持单文件和批量加载"""

    @monitor_resource('json_loader')
    def load(self, source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        加载单个JSON配置文件
        
        Args:
            source: JSON文件路径
            
        Returns:
            Tuple[配置数据, 元数据]，元数据包含:
            - format: 文件格式(json)
            - source: 文件路径
            - load_time: 加载耗时(秒)
            - size: 文件大小(字节)
            - timestamp: 加载时间戳
            
        Raises:
            ConfigLoadError: 当文件不存在或IO错误时抛出
            ConfigValidationError: 当JSON解析失败时抛出
        """
        path = Path(source).absolute()
        if not path.exists():
            raise ConfigLoadError(
                f"JSON文件不存在: {source} (文件: {str(path)}, 错误类型: file_not_found)"
            )

        start_time = time.perf_counter()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data, {
                    'format': 'json',
                    'source': str(path),
                    'load_time': max(time.perf_counter() - start_time, 0.0001),
                    'size': path.stat().st_size,
                    'timestamp': time.time()
                }
        except json.JSONDecodeError as e:
            raise ConfigValidationError(
                f"JSON解析失败: {str(e)} (文件: {str(path)}, 位置: {e.pos}, 行: {e.lineno}, 列: {e.colno})"
            ) from e
        except Exception as e:
            raise ConfigLoadError(
                f"加载JSON文件失败: {str(e)} (文件: {str(path)})"
            ) from e

    def batch_load(self, sources: List[str]) -> Dict[str, Tuple[Dict, Dict]]:
        """
        批量加载多个JSON配置文件
        
        Args:
            sources: JSON文件路径列表
            
        Returns:
            按文件路径索引的(配置数据, 元数据)字典
            
        Raises:
            ConfigLoadError: 当任一文件加载失败时抛出
            ConfigValidationError: 当JSON解析失败时抛出
        """
        results = {}
        for src in sources:
            try:
                data, meta = self.load(src)
                results[src] = (data, meta)
            except (ConfigLoadError, ConfigValidationError) as e:
                raise ConfigLoadError(
                    f"批量加载失败: {src}",
                    context={'source': src, 'nested_error': e.context}
                ) from e
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
