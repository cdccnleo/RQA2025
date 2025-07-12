import os
import time
from typing import Dict, Any, Tuple
from ..strategy import ConfigLoaderStrategy
from src.infrastructure.monitoring.system_monitor import ResourceMonitor

class EnvLoader(ConfigLoaderStrategy):
    """环境变量配置加载策略"""

    @ResourceMonitor('env_loader')
    def load(self, source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        加载环境变量配置并返回配置数据和元数据
        
        Args:
            source: 环境变量前缀
            
        Returns:
            Tuple[配置数据, 元数据]
            
        Raises:
            ConfigError: 当加载失败时抛出
        """
        if not self.can_load(source):
            raise ConfigError(
                f"无效的环境变量前缀: {source}",
                context={'prefix': source, 'error_type': 'invalid_prefix'}
            )

        start_time = time.time()
        prefix = f"{source}_"
        try:
            env_vars = {
                key[len(prefix):]: self._parse_value(os.getenv(key))
                for key in os.environ
                if key.startswith(prefix)
            }
            
            return env_vars, {
                'format': 'env',
                'prefix': prefix,
                'load_time': time.time() - start_time,
                'var_count': len(env_vars),
                'types': {
                    'string': sum(1 for v in env_vars.values() if isinstance(v, str)),
                    'number': sum(1 for v in env_vars.values() if isinstance(v, (int, float))),
                    'boolean': sum(1 for v in env_vars.values() if isinstance(v, bool)),
                    'null': sum(1 for v in env_vars.values() if v is None)
                }
            }
        except Exception as e:
            raise ConfigError(
                f"加载环境变量失败: {str(e)}",
                context={
                    'prefix': prefix,
                    'error_type': 'load_error',
                    'error': str(e)
                }
            ) from e

    def can_load(self, source: str) -> bool:
        """检查是否为有效的环境变量前缀"""
        return isinstance(source, str) and source.isidentifier()

    def _parse_value(self, value: str) -> Any:
        """解析环境变量值"""
        if value is None:
            return None
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
