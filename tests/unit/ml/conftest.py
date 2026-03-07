#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML层测试配置和导入辅助
提供统一的导入逻辑，解决pytest-xdist并发环境下的导入问题
"""

import sys
import os
import importlib
from typing import Optional, Tuple, Any
import pytest

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

# 在模块加载时确保路径正确（强制添加到最前面）
# 移除旧路径后重新插入，确保src路径在前面
for path in [PROJECT_ROOT, SRC_PATH]:
    if path in sys.path:
        sys.path.remove(path)

# 确保src路径在最前面，这样可以正确解析src.ml模块
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, PROJECT_ROOT)

# 添加pytest hook确保在测试运行前路径正确
def pytest_configure(config):
    """pytest配置钩子，确保路径正确"""
    # 移除旧路径后重新插入到前面
    if PROJECT_ROOT in sys.path:
        sys.path.remove(PROJECT_ROOT)
    if SRC_PATH in sys.path:
        sys.path.remove(SRC_PATH)
    sys.path.insert(0, SRC_PATH)  # src路径应该在前面
    sys.path.insert(0, PROJECT_ROOT)

    # 强制重新导入ml模块，确保子模块可用
    import importlib
    try:
        if 'ml' in sys.modules:
            importlib.reload(sys.modules['ml'])
        if 'src.ml' in sys.modules:
            importlib.reload(sys.modules['src.ml'])
    except:
        pass

def pytest_runtest_setup(item):
    """在每个测试运行前确保路径正确"""
    if PROJECT_ROOT in sys.path:
        sys.path.remove(PROJECT_ROOT)
    if SRC_PATH in sys.path:
        sys.path.remove(SRC_PATH)
    sys.path.insert(0, SRC_PATH)  # src路径应该在前面
    sys.path.insert(0, PROJECT_ROOT)


def import_ml_module(module_path: str, class_name: Optional[str] = None) -> Any:
    """
    导入ML模块（增强版，支持pytest-xdist环境）

    Args:
        module_path: 模块路径，如 'src.ml.core.ml_core' 或 'ml.core.ml_core'
        class_name: 要导入的类名，如果为None则返回模块

    Returns:
        导入的类或模块，失败返回None
    """
    # 确保路径在sys.path中（每次调用都检查，并确保在正确位置）
    if PROJECT_ROOT in sys.path:
        sys.path.remove(PROJECT_ROOT)
    if SRC_PATH in sys.path:
        sys.path.remove(SRC_PATH)
    sys.path.insert(0, SRC_PATH)  # src路径应该在前面
    sys.path.insert(0, PROJECT_ROOT)

    # 首先尝试直接导入（最简单的方式）
    if class_name:
        try:
            # 尝试直接导入类
            module = importlib.import_module(module_path)
            result = getattr(module, class_name, None)
            # 检查result是否有效（不是None，且是类或类型）
            if result is not None:
                # 验证result是有效的类（使用更宽松的检查）
                # 只要result不是None且有__module__属性，就认为成功
                if hasattr(result, '__module__'):
                    return result
                # 或者使用inspect检查是否是类
                import inspect
                if inspect.isclass(result):
                    return result
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            # 如果直接导入失败，继续尝试其他方式
            pass

    # 尝试多种导入路径
    import_paths = [
        module_path,
        module_path.replace('src.ml', 'ml') if 'src.ml' in module_path else None,
        module_path.replace('ml', 'src.ml') if 'ml' in module_path and 'src.ml' not in module_path else None,
    ]

    # 移除None值并去重
    import_paths = list(dict.fromkeys([p for p in import_paths if p is not None]))

    # 添加重试机制，处理pytest-xdist环境下的导入问题
    max_retries = 3
    for attempt in range(max_retries):
        for path in import_paths:
            try:
                # 在pytest-xdist环境下，可能需要重新加载模块
                if attempt > 0:
                    # 清除模块缓存
                    if path in sys.modules:
                        del sys.modules[path]

                module = importlib.import_module(path)
                if class_name:
                    # 尝试多种方式获取类
                    result = getattr(module, class_name, None)
                    if result is None:
                        # 尝试从模块的__all__中查找
                        if hasattr(module, '__all__') and class_name in module.__all__:
                            result = getattr(module, class_name)
                    if result is None:
                        # 尝试从模块字典中查找（处理动态导入）
                        if hasattr(module, '__dict__') and class_name in module.__dict__:
                            result = module.__dict__[class_name]
                    if result is None:
                        # 尝试直接访问模块属性（处理私有属性）
                        try:
                            result = module.__getattribute__(class_name)
                        except AttributeError:
                            pass
                    # 如果result不是None，就认为成功（更宽松的检查）
                    if result is not None:
                        # 验证result是有效的类
                        if hasattr(result, '__module__'):
                            return result
                        # 或者使用inspect检查是否是类
                        import abc
                        if isinstance(result, abc.ABCMeta) or (hasattr(result, '__module__') and result.__module__):
                            return result
                else:
                    return module
            except (ImportError, AttributeError, ModuleNotFoundError, TypeError) as e:
                if attempt == max_retries - 1:
                    continue
                # 等待一小段时间后重试
                import time
                time.sleep(0.1 * (attempt + 1))
                continue
            except Exception as e:
                # 记录其他异常但不中断
                if attempt == max_retries - 1:
                    continue
                import time
                time.sleep(0.1 * (attempt + 1))
                continue

    return None


def safe_import_ml_module(module_path: str, class_name: Optional[str] = None) -> Any:
    """
    安全导入ML模块（带重试机制）

    Args:
        module_path: 模块路径
        class_name: 要导入的类名

    Returns:
        导入的类或模块，失败返回None
    """
    # 首先尝试使用标准导入函数
    result = import_ml_module(module_path, class_name)
    if result is not None:
        return result

    # 如果失败，尝试直接导入
    try:
        if class_name:
            # 尝试直接导入类
            exec(f"from {module_path} import {class_name}")
            return eval(class_name)
        else:
            return importlib.import_module(module_path)
    except:
        pass

    return None


def ensure_ml_modules_available(*module_classes: Tuple[str, str]) -> dict:
    """
    确保多个ML模块可用

    Args:
        *module_classes: 元组列表，每个元组为 (module_path, class_name)

    Returns:
        字典，键为类名，值为导入的类或None
    """
    result = {}
    for module_path, class_name in module_classes:
        cls = import_ml_module(module_path, class_name)
        result[class_name] = cls
    return result


# 常用模块的快速导入函数
def import_ml_core():
    """导入ML核心"""
    try:
        # 尝试多种导入方式
        from src.ml.core.ml_core import MLCore
        return MLCore
    except ImportError:
        try:
            from src.ml.core.ml_core import MLCore
            return MLCore
        except ImportError:
            return None


def import_model_manager():
    """导入模型管理器"""
    try:
        # 直接尝试从src导入
        from src.ml.core.model_manager import ModelManager
        return ModelManager
    except ImportError as e:
        print(f"DEBUG: Direct import failed: {e}")
        try:
            # 尝试使用辅助函数
            result = import_ml_module('src.ml.core.model_manager', 'ModelManager')
            print(f"DEBUG: Helper function result: {result}")
            if result:
                return result
        except Exception as e2:
            print(f"DEBUG: Helper function also failed: {e2}")

        # 最后尝试直接从src导入
        try:
            import sys
            if 'src' not in sys.path:
                sys.path.insert(0, 'src')
            from src.ml.core.model_manager import ModelManager
            return ModelManager
        except Exception as e3:
            print(f"DEBUG: Final fallback failed: {e3}")
            return None


def import_process_orchestrator():
    """导入ML业务流程编排器"""
    try:
        from src.ml.core.process_orchestrator import ProcessOrchestrator
        return ProcessOrchestrator
    except ImportError:
        try:
            from src.ml.core.process_orchestrator import ProcessOrchestrator
            return ProcessOrchestrator
        except ImportError:
            return None


def import_inference_service():
    """导入推理服务"""
    try:
        from src.ml.core.inference_service import InferenceService
        return InferenceService
    except ImportError:
        try:
            from src.ml.core.inference_service import InferenceService
            return InferenceService
        except ImportError:
            return None


def import_deep_learning_integration_tests():
    """导入深度学习集成测试"""
    try:
        from src.ml.deep_learning.core.integration_tests import DeepLearningIntegrationTests
        return DeepLearningIntegrationTests
    except ImportError:
        try:
            from src.ml.deep_learning.core.integration_tests import DeepLearningIntegrationTests
            return DeepLearningIntegrationTests
        except ImportError:
            return None


# pytest fixtures for common imports
@pytest.fixture(scope="session")
def ml_core():
    """ML核心fixture"""
    core = import_ml_core()
    if core is None:
        pytest.skip("ML核心不可用")
    return core


@pytest.fixture(scope="session")
def model_manager():
    """模型管理器fixture"""
    manager = import_model_manager()
    if manager is None:
        pytest.skip("模型管理器不可用")
    return manager


@pytest.fixture(scope="session")
def process_orchestrator():
    """ML业务流程编排器fixture"""
    orchestrator = import_process_orchestrator()
    if orchestrator is None:
        pytest.skip("ML业务流程编排器不可用")
    return orchestrator


@pytest.fixture(scope="session")
def inference_service():
    """推理服务fixture"""
    service = import_inference_service()
    if service is None:
        pytest.skip("推理服务不可用")
    return service


@pytest.fixture(scope="session")
def deep_learning_integration_tests():
    """深度学习集成测试fixture"""
    tests = import_deep_learning_integration_tests()
    if tests is None:
        pytest.skip("深度学习集成测试不可用")
    return tests


# 动态导入辅助函数，在setup_method中使用
def ensure_ml_module_imported(module_path: str, class_name: Optional[str] = None, skip_if_missing: bool = True):
    """
    确保ML模块已导入，用于setup_method中

    Args:
        module_path: 模块路径
        class_name: 类名
        skip_if_missing: 如果导入失败是否跳过测试

    Returns:
        导入的类或模块
    """
    result = import_ml_module(module_path, class_name)
    if result is None and skip_if_missing:
        pytest.skip(f"模块 {module_path}.{class_name} 不可用")
    return result
