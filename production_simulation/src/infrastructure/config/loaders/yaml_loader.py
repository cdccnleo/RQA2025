
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..config_exceptions import ConfigLoadError
from ..interfaces.unified_interface import ConfigLoaderStrategy, ConfigFormat, LoaderResult

try:  # pragma: no cover - 可选依赖
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)


class YAMLLoader(ConfigLoaderStrategy):
    """YAML配置加载策略，提供文件加载、批量处理及转换辅助功能"""

    def __init__(self):
        super().__init__("YAMLLoader")
        self.format = ConfigFormat.YAML
        self._yaml_available = self._check_yaml_availability()
        self._last_metadata: Dict[str, Any] = {}

    def _check_yaml_availability(self) -> bool:
        return yaml is not None

    def _ensure_available(self) -> None:
        if not self._yaml_available or yaml is None:
            raise ConfigLoadError("PyYAML is not installed. Install with: pip install PyYAML")

    def load(self, source: str) -> LoaderResult:
        self._ensure_available()

        path = Path(source).absolute()
        if not path.exists():
            raise ConfigLoadError(f"YAML文件不存在 (YAML file not found): {source}")

        start_time = time.time()

        try:
            with open(path, 'r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle)

            if not isinstance(data, dict):
                data = {}

            metadata = {
                'format': ConfigFormat.YAML.value,
                'source': str(path),
                'timestamp': time.time(),
                'load_time': max(time.time() - start_time, 0.0001),
                'size': path.stat().st_size,
                'yaml_version': getattr(yaml, '__version__', 'unknown') if yaml else 'unknown',
            }

            self._last_metadata = metadata
            return LoaderResult(data, metadata)

        except Exception as exc:
            if 'YAMLError' in type(exc).__name__:
                raise ConfigLoadError(f"YAML解析失败: {exc} (文件: {str(path)})")
            raise ConfigLoadError(f"加载YAML文件失败: {exc} (文件: {str(path)})")

    def get_last_metadata(self) -> Dict[str, Any]:
        return self._last_metadata.copy()

    def batch_load(self, sources: List[str]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        for src in sources:
            try:
                result = self.load(src)
                if isinstance(result, LoaderResult):
                    results[src] = dict(result)
                else:
                    results[src] = result
            except ConfigLoadError as exc:
                errors.append(f"Failed to load {src}: {exc}")
                logger.warning("Batch load error for %s: %s", src, exc)

        if errors and not results:
            raise ConfigLoadError('; '.join(errors))
        if errors:
            logger.warning("Batch load completed with errors: %s", '; '.join(errors))

        return results

    def can_load(self, source: str) -> bool:
        if not isinstance(source, str):
            return False
        return Path(source).suffix.lower() in {'.yaml', '.yml'}

    def can_handle_source(self, source: str) -> bool:
        return self.can_load(source)

    def get_supported_formats(self) -> List[ConfigFormat]:
        return [ConfigFormat.YAML]

    def get_supported_extensions(self) -> List[str]:
        return ['.yaml', '.yml', '.YAML', '.YML']

    def save(self, data: Dict[str, Any], file_path: str) -> bool:
        if not self._yaml_available or yaml is None:
            logger.error("PyYAML is not installed. Cannot save YAML files.")
            return False
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as handle:
                yaml.dump(data, handle, default_flow_style=False, allow_unicode=True, sort_keys=False)
            return True
        except Exception as exc:  # pragma: no cover - 写入失败不常发生
            logger.error("Failed to save YAML file %s: %s", file_path, exc)
            return False

    def validate_yaml_file(self, file_path: str) -> Tuple[bool, List[str]]:
        if not self._yaml_available or yaml is None:
            return False, ["PyYAML is not installed"]
        try:
            with open(file_path, 'r', encoding='utf-8') as handle:
                yaml.safe_load(handle)
            return True, []
        except Exception as exc:
            return False, [str(exc)]

    def get_yaml_info(self) -> Dict[str, Any]:
        info = {
            'available': self._yaml_available,
            'library': 'PyYAML',
            'version': getattr(yaml, '__version__', 'unknown') if yaml else 'unknown',
            'supported_features': []
        }
        if self._yaml_available and yaml is not None:
            info['supported_features'] = ['safe_load', 'dump', 'unicode_support']
        return info

    def merge_yaml_files(self, file_paths: List[str], output_path: Optional[str] = None) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for path in file_paths:
            try:
                result = self.load(path)
                merged = self._deep_merge(merged, dict(result))
            except ConfigLoadError as exc:
                logger.error("Failed to merge YAML file %s: %s", path, exc)
        if output_path:
            self.save(merged, output_path)
        return merged

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def convert_to_yaml(self, data: Dict[str, Any]) -> str:
        self._ensure_available()
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def convert_from_yaml(self, yaml_string: str) -> Dict[str, Any]:
        self._ensure_available()
        data = yaml.safe_load(yaml_string)
        return data if isinstance(data, dict) else {}




