#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest导入辅助模块
提供统一的导入装饰器和辅助函数
"""

import sys
import os
import pytest
from typing import Optional, Any, Callable
from functools import wraps

# 确保路径正确
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# 导入conftest中的函数
try:
    from conftest import import_risk_module, ensure_module_imported
except ImportError:
    # 如果无法导入conftest，定义基本函数
    def import_risk_module(module_path: str, class_name: Optional[str] = None) -> Any:
        import importlib
        try:
            module = importlib.import_module(module_path)
            if class_name:
                return getattr(module, class_name, None)
            return module
        except (ImportError, AttributeError):
            return None

    def ensure_module_imported(module_path: str, class_name: Optional[str] = None, skip_if_missing: bool = True):
        result = import_risk_module(module_path, class_name)
        if result is None and skip_if_missing:
            pytest.skip(f"模块 {module_path}.{class_name} 不可用")
        return result


def requires_module(module_path: str, class_name: Optional[str] = None):
    """
    装饰器：要求模块可用才能运行测试

    Usage:
        @requires_module('src.risk.models.risk_calculation_engine', 'RiskCalculationEngine')
        def test_something(self):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            module = import_risk_module(module_path, class_name)
            if module is None:
                pytest.skip(f"模块 {module_path}.{class_name} 不可用")
            # 将模块注入到kwargs中
            if class_name:
                kwargs[f'_{class_name.lower()}'] = module
            else:
                kwargs[f'_{module_path.split(".")[-1]}'] = module
            return func(*args, **kwargs)
        return wrapper
    return decorator


def setup_module_imports(test_class, *module_classes):
    """
    在setup_method中设置模块导入

    Usage:
        def setup_method(self):
            setup_module_imports(
                self,
                ('src.risk.models.risk_calculation_engine', 'RiskCalculationEngine'),
                ('src.risk.models.risk_types', 'RiskCalculationConfig')
            )
    """
    for module_path, class_name in module_classes:
        cls = ensure_module_imported(module_path, class_name, skip_if_missing=True)
        setattr(test_class, class_name, cls)
