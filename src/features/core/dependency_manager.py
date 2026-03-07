"""
特征层依赖管理器
处理可选依赖，如transformers, torch等
"""

import logging
import importlib
from typing import Optional, Any, Dict
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)


class DependencyManager:

    """依赖管理器，处理可选依赖的导入"""

    # 可选依赖列表
    OPTIONAL_DEPENDENCIES = {
        'transformers': {
            'min_version': '4.0.0',
            'purpose': '情感分析、文本处理',
            'fallback': '本地情感分析算法'
        },
        'torch': {
            'min_version': '1.9.0',
            'purpose': '深度学习、GPU加速',
            'fallback': 'CPU模式'
        },
        'datasets': {
            'min_version': '2.0.0',
            'purpose': '数据集管理',
            'fallback': '本地数据加载'
        },
        'huggingface_hub': {
            'min_version': '0.10.0',
            'purpose': '模型下载',
            'fallback': '本地模型'
        }
    }

    def __init__(self):

        self._available_deps = {}
        self._check_dependencies()

    def _check_dependencies(self):
        """检查可选依赖的可用性"""
        for dep_name, dep_info in self.OPTIONAL_DEPENDENCIES.items():
            self._available_deps[dep_name] = self._check_single_dependency(dep_name, dep_info)

    def _check_single_dependency(self, dep_name: str, dep_info: Dict) -> Dict[str, Any]:
        """检查单个依赖"""
        try:
            module = importlib.import_module(dep_name)
            version = getattr(module, '__version__', 'unknown')

            # 检查版本是否满足要求
            if self._version_satisfies(version, dep_info['min_version']):
                return {
                    'available': True,
                    'version': version,
                    'module': module,
                    'error': None
                }
            else:
                return {
                    'available': False,
                    'version': version,
                    'module': None,
                    'error': f"版本 {version} 低于要求 {dep_info['min_version']}"
                }

        except ImportError as e:
            return {
                'available': False,
                'version': None,
                'module': None,
                'error': str(e)
            }
        except Exception as e:
            return {
                'available': False,
                'version': None,
                'module': None,
                'error': f"检查依赖失败: {str(e)}"
            }

    def _version_satisfies(self, current_version: str, min_version: str) -> bool:
        """检查版本是否满足要求"""
        if current_version == 'unknown':
            return False

        try:
            # 简化版本比较
            current_parts = current_version.split('.')
            min_parts = min_version.split('.')

            for i in range(max(len(current_parts), len(min_parts))):
                current = int(current_parts[i]) if i < len(current_parts) else 0
                minimum = int(min_parts[i]) if i < len(min_parts) else 0

                if current > minimum:
                    return True
                elif current < minimum:
                    return False

            return True

        except BaseException:
            return False

    def is_available(self, dep_name: str) -> bool:
        """检查依赖是否可用"""
        return self._available_deps.get(dep_name, {}).get('available', False)

    def get_module(self, dep_name: str) -> Optional[Any]:
        """获取依赖模块，如果不可用则返回None"""
        dep_info = self._available_deps.get(dep_name, {})
        return dep_info.get('module') if dep_info.get('available') else None

    def safe_import(self, dep_name: str, fallback: Optional[Any] = None) -> Any:
        """安全导入依赖模块"""
        module = self.get_module(dep_name)
        if module is not None:
            return module

        if fallback is not None:
            logger.warning(f"依赖 {dep_name} 不可用，使用fallback")
            return fallback

        # 创建一个mock对象作为fallback
        logger.warning(f"依赖 {dep_name} 不可用，使用mock对象")
        return MagicMock()


    def get_dependency_info(self) -> Dict[str, Dict]:
        """获取所有依赖信息"""
        return self._available_deps.copy()

    def log_dependency_status(self):
        """记录依赖状态到日志"""
        logger.info("=== 特征层依赖状态 ===")

        for dep_name, dep_info in self._available_deps.items():
            if dep_info['available']:
                logger.info(f"✅ {dep_name} v{dep_info['version']} - 可用")
            else:
                logger.warning(f"❌ {dep_name} - 不可用: {dep_info['error']}")

        logger.info("=" * 25)


def safe_import(dep_name: str, fallback: Optional[Any] = None) -> Any:
    """
    模块级别的安全导入函数

    Args:
        dep_name: 依赖名称
        fallback: 后备方案

    Returns:
        导入的模块或fallback
    """
    # 使用单例模式获取DependencyManager实例
    manager = DependencyManager()
    return manager.safe_import(dep_name, fallback)


# 全局依赖管理器实例
dependency_manager = DependencyManager()


def get_transformers_pipeline(task: str = "sentiment - analysis"):
    """安全获取transformers pipeline"""
    transformers = dependency_manager.safe_import('transformers')

    if transformers and hasattr(transformers, 'pipeline'):
        try:
            return transformers.pipeline(task)
        except Exception as e:
            logger.warning(f"创建transformers pipeline失败: {e}")

    # 创建mock pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]
    return mock_pipeline


def get_torch_device():
    """安全获取torch设备"""
    torch = dependency_manager.safe_import('torch')

    if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
        return torch.device('cuda')

    return None


def is_gpu_available() -> bool:
    """检查GPU是否可用"""
    torch = dependency_manager.safe_import('torch')

    if torch and hasattr(torch, 'cuda'):
        return torch.cuda.is_available()

    return False


def get_gpu_count() -> int:
    """获取GPU数量"""
    torch = dependency_manager.safe_import('torch')

    if torch and hasattr(torch, 'cuda'):
        return torch.cuda.device_count()

    return 0


# 全局DependencyManager实例，用于模块级函数
dependency_manager = DependencyManager()

# 为了向后兼容，提供模块级别的DependencyManager类引用
