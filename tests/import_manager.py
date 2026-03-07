"""
中央导入管理器 - RQA2025测试框架

统一管理所有测试模块的导入，避免路径配置冲突
提供标准化的导入接口，确保测试环境的一致性

作者: AI Assistant
创建时间: 2025年12月3日
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib
import logging

# 配置日志
logger = logging.getLogger(__name__)

class ImportManager:
    """
    中央导入管理器

    负责统一管理所有模块的导入，提供标准化的导入接口
    """

    def __init__(self):
        self._project_root = None
        self._src_path = None
        self._module_cache = {}
        self._import_errors = []
        self._setup_paths()

    def _setup_paths(self):
        """设置项目路径"""
        # 从当前文件位置向上查找项目根目录
        current_file = Path(__file__).resolve()
        self._project_root = current_file.parent.parent  # tests/ -> project_root/
        self._src_path = self._project_root / "src"

        # 确保路径在sys.path的最前面
        project_root_str = str(self._project_root)
        src_path_str = str(self._src_path)

        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        if src_path_str not in sys.path:
            sys.path.insert(0, src_path_str)

        logger.info(f"项目根目录: {project_root_str}")
        logger.info(f"源码目录: {src_path_str}")

    def import_module(self, module_path: str, fallback: Any = None) -> Optional[Any]:
        """
        安全导入模块

        Args:
            module_path: 模块路径，如 'src.infrastructure.constants'
            fallback: 导入失败时的默认值

        Returns:
            导入的模块或fallback值
        """
        if module_path in self._module_cache:
            return self._module_cache[module_path]

        try:
            module = importlib.import_module(module_path)
            self._module_cache[module_path] = module
            logger.debug(f"成功导入模块: {module_path}")
            return module
        except ImportError as e:
            error_msg = f"导入失败: {module_path} - {str(e)}"
            self._import_errors.append(error_msg)
            logger.warning(error_msg)
            self._module_cache[module_path] = fallback
            return fallback
        except Exception as e:
            error_msg = f"导入异常: {module_path} - {str(e)}"
            self._import_errors.append(error_msg)
            logger.error(error_msg)
            self._module_cache[module_path] = fallback
            return fallback

    def import_class(self, module_path: str, class_name: str, fallback: Any = None) -> Optional[Any]:
        """
        安全导入类

        Args:
            module_path: 模块路径
            class_name: 类名
            fallback: 导入失败时的默认值

        Returns:
            导入的类或fallback值
        """
        module = self.import_module(module_path)
        if module and hasattr(module, class_name):
            try:
                cls = getattr(module, class_name)
                logger.debug(f"成功导入类: {module_path}.{class_name}")
                return cls
            except Exception as e:
                error_msg = f"获取类失败: {module_path}.{class_name} - {str(e)}"
                self._import_errors.append(error_msg)
                logger.error(error_msg)

        logger.warning(f"类不存在: {module_path}.{class_name}")
        return fallback

    def import_function(self, module_path: str, function_name: str, fallback: Any = None) -> Optional[Any]:
        """
        安全导入函数

        Args:
            module_path: 模块路径
            function_name: 函数名
            fallback: 导入失败时的默认值

        Returns:
            导入的函数或fallback值
        """
        module = self.import_module(module_path)
        if module and hasattr(module, function_name):
            try:
                func = getattr(module, function_name)
                if callable(func):
                    logger.debug(f"成功导入函数: {module_path}.{function_name}")
                    return func
            except Exception as e:
                error_msg = f"获取函数失败: {module_path}.{function_name} - {str(e)}"
                self._import_errors.append(error_msg)
                logger.error(error_msg)

        logger.warning(f"函数不存在: {module_path}.{function_name}")
        return fallback

    def get_import_errors(self) -> List[str]:
        """获取导入错误列表"""
        return self._import_errors.copy()

    def clear_cache(self):
        """清空模块缓存"""
        self._module_cache.clear()
        self._import_errors.clear()

# 全局导入管理器实例
import_manager = ImportManager()

# ============================================================================
# 预定义的常用导入函数
# ============================================================================

def import_infrastructure_constants():
    """导入基础设施常量"""
    return import_manager.import_module('src.infrastructure.constants')

def import_core_services():
    """导入核心服务"""
    return import_manager.import_module('src.core')

def import_data_management():
    """导入数据管理层"""
    return import_manager.import_module('src.data')

def import_ml_layer():
    """导入机器学习层"""
    return import_manager.import_module('src.ml')

def import_trading_engine():
    """导入交易引擎"""
    return import_manager.import_class('src.trading.core.trading_engine', 'TradingEngine')

def import_execution_engine():
    """导入执行引擎"""
    return import_manager.import_class('src.trading.execution.execution_engine', 'ExecutionEngine')

def import_risk_manager():
    """导入风险管理器"""
    return import_manager.import_class('src.risk.risk_manager', 'RiskManager')

def import_monitoring_system():
    """导入监控系统"""
    return import_manager.import_class('src.monitoring.monitoring_system', 'MonitoringSystem')

def import_mobile_trading():
    """导入移动端交易"""
    return import_manager.import_class('src.mobile.core.mobile_trading', 'MobileTradingService')

# ============================================================================
# 创建模拟类用于测试
# ============================================================================

# 使用扩展的Mock构建器创建更完整的Mock对象
def create_enhanced_mock(mock_type: str, **kwargs):
    """
    使用扩展的Mock构建器创建增强的Mock对象

    Args:
        mock_type: Mock类型
        **kwargs: 传递给Mock构建器的参数

    Returns:
        配置完整的Mock对象
    """
    try:
        from tests.fixtures.infrastructure_mocks import create_standard_mock
        return create_standard_mock(mock_type, **kwargs)
    except ImportError:
        # 如果无法导入，使用简单的MagicMock
        from unittest.mock import MagicMock
        return MagicMock()

class MockTradingEngine:
    """模拟交易引擎 - 使用增强Mock构建器"""
    def __init__(self, *args, **kwargs):
        # 使用增强的Mock构建器
        self._mock = create_enhanced_mock('trading', **kwargs)
        self.name = "MockTradingEngine"
        self.orders = []
        self.positions = {}

    def place_order(self, symbol, quantity, price=None, order_type="market"):
        return self._mock.submit_order(symbol, quantity, price, order_type)[2]  # 返回order_id

    def get_positions(self):
        return self.positions.copy()

class MockExecutionEngine:
    """模拟执行引擎 - 使用增强Mock构建器"""
    def __init__(self, *args, **kwargs):
        self._mock = create_enhanced_mock('async_processor', **kwargs)
        self.name = "MockExecutionEngine"
        self.executions = []

    def execute_order(self, order):
        return self._mock.submit_task(order)

class MockRiskManager:
    """模拟风险管理器 - 使用增强Mock构建器"""
    def __init__(self, *args, **kwargs):
        self._mock = create_enhanced_mock('risk', **kwargs)
        self.name = "MockRiskManager"
        self.checks = []

    def check_risk(self, order):
        return self._mock.assess_risk(order)['level'] == 'low'

class MockMonitoringSystem:
    """模拟监控系统 - 使用增强Mock构建器"""
    def __init__(self, *args, **kwargs):
        self._mock = create_enhanced_mock('monitor', **kwargs)
        self.name = "MockMonitoringSystem"
        self.metrics = {}
        self.alerts = []

    def record_metric(self, name, value):
        self._mock.record_metric(name, value)
        self.metrics[name] = value

    def get_metrics(self):
        return self._mock.get_metrics()

    def add_alert(self, alert):
        self._mock.add_alert(alert)
        self.alerts.append(alert)

# ============================================================================
# 便捷导入函数，支持自动降级到Mock
# ============================================================================

def get_trading_engine():
    """获取交易引擎，优先使用真实实现，降级到Mock"""
    engine_class = import_trading_engine()
    if engine_class:
        try:
            return engine_class()
        except Exception as e:
            logger.warning(f"创建真实交易引擎失败，使用Mock: {e}")
    return MockTradingEngine()

def get_execution_engine():
    """获取执行引擎，优先使用真实实现，降级到Mock"""
    engine_class = import_execution_engine()
    if engine_class:
        try:
            return engine_class()
        except Exception as e:
            logger.warning(f"创建真实执行引擎失败，使用Mock: {e}")
    return MockExecutionEngine()

def get_risk_manager():
    """获取风险管理器，优先使用真实实现，降级到Mock"""
    manager_class = import_risk_manager()
    if manager_class:
        try:
            return manager_class()
        except Exception as e:
            logger.warning(f"创建真实风险管理器失败，使用Mock: {e}")
    return MockRiskManager()

def get_monitoring_system():
    """获取监控系统，优先使用真实实现，降级到Mock"""
    system_class = import_monitoring_system()
    if system_class:
        try:
            return system_class()
        except Exception as e:
            logger.warning(f"创建真实监控系统失败，使用Mock: {e}")
    return MockMonitoringSystem()

# ============================================================================
# 导出接口
# ============================================================================

__all__ = [
    'ImportManager',
    'import_manager',
    'import_infrastructure_constants',
    'import_core_services',
    'import_data_management',
    'import_ml_layer',
    'import_trading_engine',
    'import_execution_engine',
    'import_risk_manager',
    'import_monitoring_system',
    'import_mobile_trading',
    'get_trading_engine',
    'get_execution_engine',
    'get_risk_manager',
    'get_monitoring_system',
    'MockTradingEngine',
    'MockExecutionEngine',
    'MockRiskManager',
    'MockMonitoringSystem'
]
