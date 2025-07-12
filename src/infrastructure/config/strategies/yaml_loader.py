import yaml
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from ..strategy import ConfigLoaderStrategy
from ..exceptions import ConfigLoadError, ConfigValidationError
from src.infrastructure.monitoring.decorators import monitor_resource

class YAMLLoader(ConfigLoaderStrategy):
    """YAML配置加载策略"""

    @monitor_resource('yaml_loader')
    def load(self, source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        加载YAML配置文件并返回配置数据和元数据
        
        Args:
            source: YAML文件路径
            
        Returns:
            Tuple[配置数据, 元数据]
            
        Raises:
            ConfigLoadError: 当文件不存在或IO错误时抛出
            ConfigValidationError: 当YAML解析失败时抛出
        """
        path = Path(source).absolute()
        if not path.exists():
            raise ConfigLoadError(
                f"YAML文件不存在: {source}",
                context={'file': str(path), 'error_type': 'file_not_found'}
            )

        start_time = time.time()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data, {
                    'format': 'yaml',
                    'source': str(path),
                    'load_time': time.time() - start_time,
                    'size': path.stat().st_size,
                    'safe_load': True
                }
        except yaml.YAMLError as e:
            raise ConfigValidationError(
                f"YAML解析失败: {str(e)}",
                context={
                    'file': str(path),
                    'error_type': 'parse_error',
                    'mark': str(e.problem_mark) if hasattr(e, 'problem_mark') else None
                }
            ) from e
        except Exception as e:
            raise ConfigLoadError(
                f"加载YAML文件失败: {str(e)}",
                context={
                    'file': str(path),
                    'error_type': 'io_error',
                    'error': str(e)
                }
            ) from e

    def can_load(self, source: str) -> bool:
        """检查是否为YAML文件"""
        return isinstance(source, str) and source.lower().endswith(('.yaml', '.yml'))
