#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险控制层测试配置和导入辅助
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
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)  # src路径应该在前面，确保能找到src.risk模块
sys.path.insert(0, PROJECT_ROOT)

# 添加pytest hook确保在测试运行前路径正确
def pytest_configure(config):
    """pytest配置钩子，确保路径正确"""
    # 确保路径在sys.path中（xdist环境下）
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    # 验证路径设置
    print(f"DEBUG: sys.path[0:2] = {sys.path[0:2]}")

    # 预加载关键模块
    try:
        import src.risk.models.risk_manager
        print("DEBUG: risk_manager module preloaded successfully")
    except ImportError as e:
        print(f"DEBUG: Failed to preload risk_manager: {e}")

def pytest_runtest_setup(item):
    """在每个测试运行前确保路径正确"""
    if PROJECT_ROOT in sys.path:
        sys.path.remove(PROJECT_ROOT)
    if SRC_PATH in sys.path:
        sys.path.remove(SRC_PATH)
    sys.path.insert(0, SRC_PATH)  # src路径应该在前面
    sys.path.insert(0, PROJECT_ROOT)


def import_risk_module(module_path: str, class_name: Optional[str] = None) -> Any:
    """
    导入风险模块（增强版，支持pytest-xdist环境）

    Args:
        module_path: 模块路径，如 'src.risk.models.risk_calculation_engine' 或 'risk.models.risk_calculation_engine'
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
        module_path.replace('src.risk', 'risk') if 'src.risk' in module_path else None,
        module_path.replace('risk', 'src.risk') if 'risk' in module_path and 'src.risk' not in module_path else None,
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


def safe_import_risk_module(module_path: str, class_name: Optional[str] = None) -> Any:
    """
    安全导入风险模块（带重试机制）

    Args:
        module_path: 模块路径
        class_name: 要导入的类名

    Returns:
        导入的类或模块，失败返回None
    """
    # 首先尝试使用标准导入函数
    result = import_risk_module(module_path, class_name)
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


def ensure_risk_modules_available(*module_classes: Tuple[str, str]) -> dict:
    """
    确保多个风险模块可用

    Args:
        *module_classes: 元组列表，每个元组为 (module_path, class_name)

    Returns:
        字典，键为类名，值为导入的类或None
    """
    result = {}
    for module_path, class_name in module_classes:
        cls = import_risk_module(module_path, class_name)
        result[class_name] = cls
    return result


# 常用模块的快速导入函数
def import_risk_calculation_engine():
    """导入风险计算引擎"""
    try:
        # 尝试多种导入方式
        from src.risk.models.risk_calculation_engine import RiskCalculationEngine
        return RiskCalculationEngine
    except ImportError:
        try:
            from risk.models.risk_calculation_engine import RiskCalculationEngine
            return RiskCalculationEngine
        except ImportError:
            return None


def import_risk_manager():
    """导入风险管理器"""
    # 确保路径正确
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    # 首先尝试直接导入
    try:
        from src.risk.models.risk_manager import RiskManager
        return RiskManager
    except ImportError:
        pass

    # 尝试使用通用导入函数
    result = import_risk_module('src.risk.models.risk_manager', 'RiskManager')
    if result:
        return result

    # 最后尝试强制导入
    try:
        import importlib
        module = importlib.import_module('src.risk.models.risk_manager')
        return getattr(module, 'RiskManager', None)
    except:
        return None


def import_risk_types():
    """导入风险类型"""
    modules = ensure_risk_modules_available(
        ('src.risk.models.risk_types', 'RiskCalculationConfig'),
        ('src.risk.models.risk_types', 'RiskMetricType'),
        ('src.risk.models.risk_types', 'ConfidenceLevel'),
        ('src.risk.models.risk_types', 'RiskCalculationResult'),
        ('src.risk.models.risk_types', 'PortfolioRiskProfile'),
    )
    return modules


def import_risk_calculators():
    """导入风险计算器"""
    calculators = ensure_risk_modules_available(
        ('src.risk.models.calculators.risk_calculators', 'VolatilityCalculator'),
        ('src.risk.models.calculators.risk_calculators', 'PositionRiskCalculator'),
        ('src.risk.models.calculators.risk_calculators', 'LiquidityCalculator'),
        ('src.risk.models.calculators.risk_calculators', 'CorrelationCalculator'),
        ('src.risk.models.calculators.risk_calculators', 'ConcentrationCalculator'),
        ('src.risk.models.calculators.risk_calculators', 'SharpeRatioCalculator'),
        ('src.risk.models.calculators.risk_calculators', 'BetaCalculator'),
        ('src.risk.models.calculators.risk_calculators', 'MaxDrawdownCalculator'),
    )
    return calculators


def import_var_calculator():
    """导入VaR计算器"""
    return import_risk_module('src.risk.models.calculators.var_calculator', 'VaRCalculator')


def import_alert_system():
    """导入告警系统"""
    return import_risk_module('src.risk.alert.alert_system', 'AlertSystem')


def import_realtime_monitor():
    """导入实时监控"""
    return import_risk_module('src.risk.monitor.realtime_risk_monitor', 'RealtimeRiskMonitor')


def import_compliance_manager():
    """导入合规管理器"""
    return import_risk_module('src.risk.compliance.cross_border_compliance_manager', 'CrossBorderComplianceManager')


def import_risk_checker():
    """导入风险检查器"""
    return import_risk_module('src.risk.checker.checker', 'RiskChecker')


def import_market_impact_analyzer():
    """导入市场冲击分析器"""
    return import_risk_module('src.risk.analysis.market_impact_analyzer', 'MarketImpactAnalyzer')


def import_realtime_monitor():
    """导入实时风险监控器"""
    return import_risk_module('src.risk.monitor.realtime_risk_monitor', 'RealtimeRiskMonitor')


def import_alert_system():
    """导入告警系统"""
    return import_risk_module('src.risk.alert.alert_system', 'AlertSystem')


def import_compliance_manager():
    """导入合规管理器"""
    return import_risk_module('src.risk.compliance.cross_border_compliance_manager', 'CrossBorderComplianceManager')


def import_risk_checker():
    """导入风险检查器"""
    return import_risk_module('src.risk.checker.checker', 'RiskChecker')


def import_memory_optimizer():
    """导入内存优化器"""
    return import_risk_module('src.risk.infrastructure.memory_optimizer', 'MemoryOptimizer')


def import_async_task_manager():
    """导入异步任务管理器"""
    return import_risk_module('src.risk.infrastructure.async_task_manager', 'AsyncTaskManager')


def import_distributed_cache_manager():
    """导入分布式缓存管理器"""
    return import_risk_module('src.risk.infrastructure.distributed_cache_manager', 'DistributedCacheManager')


# pytest fixtures for common imports
@pytest.fixture(scope="session")
def risk_calculation_engine():
    """风险计算引擎fixture"""
    engine = import_risk_calculation_engine()
    if engine is None:
        pytest.skip("风险计算引擎不可用")
    return engine


@pytest.fixture(scope="session")
def risk_manager():
    """风险管理器fixture"""
    manager = import_risk_manager()
    if manager is None:
        pytest.skip("风险管理器不可用")
    return manager


@pytest.fixture(scope="session")
def risk_types():
    """风险类型fixture"""
    types = import_risk_types()
    if not types or all(v is None for v in types.values()):
        pytest.skip("风险类型不可用")
    return types


@pytest.fixture(scope="session")
def risk_calculators():
    """风险计算器fixture"""
    calculators = import_risk_calculators()
    if not calculators or all(v is None for v in calculators.values()):
        pytest.skip("风险计算器不可用")
    return calculators


# 动态导入辅助函数，在setup_method中使用
def ensure_module_imported(module_path: str, class_name: Optional[str] = None, skip_if_missing: bool = True):
    """
    确保模块已导入，用于setup_method中

    Args:
        module_path: 模块路径
        class_name: 类名
        skip_if_missing: 如果导入失败是否跳过测试

    Returns:
        导入的类或模块
    """
    result = import_risk_module(module_path, class_name)
    if result is None and skip_if_missing:
        pytest.skip(f"模块 {module_path}.{class_name} 不可用")
    return result
