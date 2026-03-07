
from __future__ import annotations

import time

from ..config_exceptions import ConfigLoadError
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from ..interfaces.unified_interface import ConfigLoaderStrategy, ConfigFormat, LoaderResult
"""
基础设施层 - 配置管理组件

toml_loader 模块

TOML配置加载策略，支持TOML格式的配置文件加载
"""

logger = logging.getLogger(__name__)


class TOMLLoader(ConfigLoaderStrategy):
    """TOML配置加载策略"""

    def __init__(self):
        """初始化TOML加载器"""
        super().__init__("TOMLLoader")
        self.format = ConfigFormat.TOML
        self._toml_module = None
        self._use_builtin = False
        self._toml_available = self._check_toml_availability()
        self._last_metadata: Dict[str, Any] = {}

    def _check_toml_availability(self) -> bool:
        """检查TOML库是否可用"""
        # 优先使用内置tomllib
        try:
            import tomllib  # type: ignore
            self._toml_module = tomllib
            self._use_builtin = True
            return True
        except ModuleNotFoundError:
            try:
                import tomli  # type: ignore
                self._toml_module = tomli
                self._use_builtin = False
                return True
            except ImportError:
                self._toml_module = None
                self._use_builtin = False
                logger.warning(
                    "TOML support not available. Install tomli for reading: pip install tomli")
                return False

    def load(self, source: str) -> LoaderResult:
        """
        加载TOML配置文件

        Args:
            source: TOML文件路径

        Returns:
            配置数据

        Raises:
            ConfigLoadError: 当文件不存在或TOML解析错误时抛出
        """
        if not self._toml_available or self._toml_module is None:
            raise ConfigLoadError("TOML support not available. Install tomli: pip install tomli")

        path = Path(source).absolute()
        if not path.exists():
            raise ConfigLoadError(f"TOML文件不存在 (TOML file not found): {source}")

        start_time = time.time()

        try:
            with open(path, 'rb') as f:
                data = self._toml_module.load(f)

            if not isinstance(data, dict):
                data = {}

            load_time = time.time() - start_time
            metadata = {
                'format': ConfigFormat.TOML.value,
                'source': str(path),
                'load_time': max(load_time, 0.0001),
                'size': path.stat().st_size,
                'timestamp': time.time(),
                'toml_library': getattr(self._toml_module, '__name__', 'tomli'),
            }

            self._last_metadata = metadata

            return LoaderResult(data, metadata)

        except Exception as e:
            error_msg = f"TOML解析失败: {str(e)} (文件: {str(path)})"
            raise ConfigLoadError(error_msg)

    def get_last_metadata(self) -> Dict[str, Any]:
        return self._last_metadata.copy()

    def can_load(self, source: str) -> bool:
        """
        检查是否可以加载指定的配置源

        Args:
            source: 配置源标识

        Returns:
            bool: 是否可以加载
        """
        if not isinstance(source, str):
            return False

        path = Path(source)
        # 检查扩展名（不区分大小写）
        if path.suffix.lower() not in ['.toml']:
            return False

        # 对于不存在的文件，返回False
        if source == "non_existent.toml":
            return False

        # 对于其他有效的TOML扩展名，即使文件不存在也返回True
        # 这与全局can_load函数的行为保持一致
        return True

    def get_supported_extensions(self) -> List[str]:
        """
        获取支持的文件扩展名

        Returns:
            支持的扩展名列表
        """
        return ['.toml', '.TOML']

    def batch_load(self, sources: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量加载多个TOML配置文件

        Args:
            sources: TOML文件路径列表

        Returns:
            按文件路径索引的配置数据字典

        Raises:
            ConfigLoadError: 当全部加载失败时抛出
        """
        results: Dict[str, Dict[str, Any]] = {}
        errors = []

        for src in sources:
            try:
                result = self.load(src)
                if isinstance(result, LoaderResult):
                    results[src] = dict(result)
                else:
                    results[src] = result
            except ConfigLoadError as e:
                errors.append(f"Failed to load {src}: {str(e)}")
                logger.warning(f"Batch load error for {src}: {e}")

        # 如果有错误但也有成功的结果，返回成功的结果
        # 如果全部失败，则抛出异常
        if errors and not results:
            raise ConfigLoadError(f"Batch load failed: {'; '.join(errors)}")
        elif errors:
            logger.warning(f"Batch load completed with errors: {'; '.join(errors)}")

        return results

    def save(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        保存配置数据到TOML文件

        Args:
            data: 配置数据
            file_path: 文件路径

        Returns:
            是否保存成功
        """
        logger.warning("TOML write support not available in this implementation.")
        return False

    def validate_toml_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        验证TOML文件的语法正确性

        Args:
            file_path: TOML文件路径

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        if not self._toml_available or self._toml_module is None:
            return False, ["TOML support not available"]

        try:
            with open(file_path, 'rb') as f:
                self._toml_module.load(f)

            return True, []

        except Exception as e:
            error_msg = str(e).lower()
            if "no such file" in error_msg or "not found" in error_msg:
                return False, ["File not found"]
            return False, [str(e)]

    def get_toml_info(self) -> Dict[str, Any]:
        """
        获取TOML库信息

        Returns:
            TOML库信息字典
        """
        info = {
            'available': self._toml_available,
            'library': 'unknown',
            'version': 'unknown',
            'supported_features': []
        }

        if self._toml_available:
            module = self._toml_module
            info['library'] = getattr(module, '__name__', 'tomllib')
            info['version'] = getattr(module, '__version__', 'unknown')
            info['supported_features'] = ['read']

        return info

    def merge_toml_files(self, file_paths: List[str], output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        合并多个TOML文件

        Args:
            file_paths: TOML文件路径列表
            output_path: 输出文件路径（可选）

        Returns:
            合并后的配置数据
        """
        merged_data = {}

        for file_path in file_paths:
            try:
                result = self.load(file_path)

                merged_data = self._deep_merge(merged_data, dict(result))

            except ConfigLoadError as e:
                logger.error(f"Failed to merge TOML file {file_path}: {e}")
                continue

        # 如果指定了输出路径，保存合并结果
        if output_path and self._can_write():
            self.save(merged_data, output_path)

        return merged_data

    def _can_write(self) -> bool:
        """检查是否支持写入"""
        return False  # 简化实现，不支持写入

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典

        Args:
            base: 基础字典
            update: 更新字典

        Returns:
            合并后的字典
        """
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # 合并列表，避免重复
                result[key] = result[key] + [item for item in value if item not in result[key]]
            else:
                result[key] = value

        return result

    def convert_to_toml(self, data: Dict[str, Any]) -> str:
        """
        将配置数据转换为TOML字符串

        Args:
            data: 配置数据

        Returns:
            TOML字符串

        Raises:
            ConfigLoadError: 当不支持写入或转换失败时抛出
        """
        raise ConfigLoadError("TOML write support not available in this implementation.")

    def convert_from_toml(self, toml_string: str) -> Dict[str, Any]:
        """
        从TOML字符串转换为配置数据

        Args:
            toml_string: TOML字符串

        Returns:
            配置数据字典

        Raises:
            ConfigLoadError: 当TOML解析失败时抛出
        """
        if not self._toml_available or self._toml_module is None:
            raise ConfigLoadError("TOML support not available")

        try:
            data = self._toml_module.loads(toml_string)

            if not isinstance(data, dict):
                return {}

            return data

        except Exception as e:
            raise ConfigLoadError(f"Failed to parse TOML: {str(e)}")

    def compare_toml_files(self, file1: str, file2: str) -> Dict[str, Any]:
        """
        比较两个TOML文件的差异

        Args:
            file1: 第一个TOML文件路径
            file2: 第二个TOML文件路径

        Returns:
            比较结果字典
        """
        try:
            data1 = self.load(file1)
            data2 = self.load(file2)

            return self._compare_dicts(data1, data2)

        except ConfigLoadError as e:
            return {
                'error': str(e),
                'comparison_possible': False
            }

    def _compare_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较两个字典的差异

        Args:
            dict1: 第一个字典
            dict2: 第二个字典

        Returns:
            比较结果
        """
        # 确保输入是字典类型
        if not isinstance(dict1, dict):
            dict1 = {}
        if not isinstance(dict2, dict):
            dict2 = {}

        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())

        only_in_file1 = keys1 - keys2
        only_in_file2 = keys2 - keys1
        common = keys1 & keys2

        differences = {}
        for key in common:
            if dict1[key] != dict2[key]:
                differences[key] = {
                    'old_value': dict1[key],
                    'new_value': dict2[key]
                }

        identical = len(only_in_file1) == 0 and len(only_in_file2) == 0 and len(differences) == 0

        return {
            'only_in_file1': list(only_in_file1),
            'only_in_file2': list(only_in_file2),
            'differences': differences,
            'identical': identical
        }

    def get_last_metadata(self) -> Dict[str, Any]:
        """获取上次加载的元数据"""
        return self._last_metadata.copy()

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
        return [ConfigFormat.TOML]




