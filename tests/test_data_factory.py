#!/usr/bin/env python3
"""
测试数据工厂和Mock机制

提供统一的测试数据生成、Mock对象管理和测试环境配置
支持各层架构的测试数据需求
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pytest
import numpy as np


@dataclass
class TestDataConfig:
    """测试数据配置"""
    data_type: str
    size: str = "small"  # small, medium, large
    complexity: str = "simple"  # simple, medium, complex
    randomization: bool = True
    seed: Optional[int] = None


@dataclass
class MockConfig:
    """Mock配置"""
    target_class: Type
    method_configs: Dict[str, Any] = field(default_factory=dict)
    property_configs: Dict[str, Any] = field(default_factory=dict)
    side_effects: Dict[str, Callable] = field(default_factory=dict)


class TestDataFactory:
    """
    测试数据工厂

    提供各层架构的测试数据生成和管理
    支持数据模板、随机化生成、数据验证等功能
    """

    def __init__(self):
        self.templates = {}
        self.generators = {}
        self.validators = {}
        self._setup_default_templates()
        self._setup_default_generators()

    def _setup_default_templates(self):
        """设置默认数据模板"""

        # Trading层数据模板
        self.templates["trading_order"] = {
            "order_id": "ORDER_{timestamp}_{seq}",
            "symbol": "000001.SZ",
            "quantity": 100,
            "price": 10.0,
            "order_type": "limit",
            "side": "buy",
            "timestamp": "{timestamp}",
            "status": "pending"
        }

        self.templates["trading_execution"] = {
            "execution_id": "EXEC_{timestamp}_{seq}",
            "order_id": "ORDER_{timestamp}_{seq}",
            "symbol": "000001.SZ",
            "quantity": 100,
            "price": 10.0,
            "executed_quantity": 100,
            "execution_price": 10.0,
            "timestamp": "{timestamp}",
            "status": "completed"
        }

        # Strategy层数据模板
        self.templates["strategy_signal"] = {
            "signal_id": "SIG_{timestamp}_{seq}",
            "symbol": "000001.SZ",
            "signal_type": "buy",
            "strength": 0.8,
            "timestamp": "{timestamp}",
            "valid_until": "{timestamp_plus_1h}"
        }

        self.templates["strategy_position"] = {
            "position_id": "POS_{timestamp}_{seq}",
            "symbol": "000001.SZ",
            "quantity": 1000,
            "avg_price": 10.0,
            "current_price": 10.5,
            "pnl": 500.0,
            "timestamp": "{timestamp}"
        }

        # Data层数据模板
        self.templates["market_data"] = {
            "symbol": "000001.SZ",
            "timestamp": "{timestamp}",
            "open": 10.0,
            "high": 10.5,
            "low": 9.8,
            "close": 10.2,
            "volume": 1000000,
            "amount": 10000000.0
        }

        # Risk层数据模板
        self.templates["risk_metrics"] = {
            "portfolio_id": "PORT_{timestamp}_{seq}",
            "timestamp": "{timestamp}",
            "total_value": 1000000.0,
            "cash": 500000.0,
            "positions_value": 500000.0,
            "var_95": -50000.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.1
        }

    def _setup_default_generators(self):
        """设置默认数据生成器"""

        # 股票代码生成器
        def generate_symbol():
            exchanges = ["SZ", "SH", "BJ"]
            codes = [f"{random.randint(0, 999999):06d}.{random.choice(exchanges)}" for _ in range(100)]
            return random.choice(codes)

        # 时间戳生成器
        def generate_timestamp():
            now = datetime.now()
            random_days = random.randint(-30, 0)
            random_time = timedelta(
                days=random_days,
                hours=random.randint(9, 15),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            return (now + random_time).isoformat()

        # 价格生成器
        def generate_price(base_price: float = 10.0, volatility: float = 0.1):
            return round(base_price * (1 + random.uniform(-volatility, volatility)), 2)

        # 数量生成器
        def generate_quantity(min_qty: int = 100, max_qty: int = 10000):
            return random.randint(min_qty, max_qty)

        self.generators.update({
            "symbol": generate_symbol,
            "timestamp": generate_timestamp,
            "price": generate_price,
            "quantity": generate_quantity
        })

    def generate_data(self, template_name: str, config: TestDataConfig = None, **kwargs) -> Dict[str, Any]:
        """
        生成测试数据

        Args:
            template_name: 模板名称
            config: 测试数据配置
            **kwargs: 覆盖参数

        Returns:
            Dict[str, Any]: 生成的测试数据
        """
        if template_name not in self.templates:
            raise ValueError(f"未知的模板: {template_name}")

        config = config or TestDataConfig(data_type=template_name)

        # 获取模板
        template = self.templates[template_name].copy()

        # 设置随机种子
        if config.seed is not None:
            random.seed(config.seed)

        # 生成数据
        data = {}
        seq = random.randint(1000, 9999)

        for key, value in template.items():
            if isinstance(value, str) and "{" in value:
                # 处理模板变量
                data[key] = self._resolve_template_variables(value, seq)
            elif key in kwargs:
                # 使用提供的参数
                data[key] = kwargs[key]
            elif config.randomization and key in self.generators:
                # 使用生成器
                generator = self.generators[key]
                if callable(generator):
                    data[key] = generator()
                else:
                    data[key] = generator
            else:
                data[key] = value

        return data

    def _resolve_template_variables(self, template_str: str, seq: int) -> str:
        """解析模板变量"""

        now = datetime.now()

        # 时间戳变量
        if "{timestamp}" in template_str:
            template_str = template_str.replace("{timestamp}", now.strftime("%Y%m%d_%H%M%S"))

        if "{timestamp_plus_1h}" in template_str:
            future_time = now + timedelta(hours=1)
            template_str = template_str.replace("{timestamp_plus_1h}", future_time.strftime("%Y%m%d_%H%M%S"))

        # 序列号变量
        if "{seq}" in template_str:
            template_str = template_str.replace("{seq}", str(seq))

        return template_str

    def generate_bulk_data(self, template_name: str, count: int, config: TestDataConfig = None) -> List[Dict[str, Any]]:
        """
        生成批量测试数据

        Args:
            template_name: 模板名称
            count: 生成数量
            config: 测试数据配置

        Returns:
            List[Dict[str, Any]]: 测试数据列表
        """
        data_list = []
        for i in range(count):
            # 为每个数据项设置不同的种子以确保多样性
            if config and config.seed is not None:
                item_config = TestDataConfig(
                    data_type=config.data_type,
                    size=config.size,
                    complexity=config.complexity,
                    randomization=config.randomization,
                    seed=config.seed + i
                )
            else:
                item_config = config

            data = self.generate_data(template_name, item_config)
            data_list.append(data)

        return data_list

    def add_template(self, name: str, template: Dict[str, Any]):
        """添加自定义模板"""
        self.templates[name] = template

    def add_generator(self, name: str, generator: Callable):
        """添加自定义生成器"""
        self.generators[name] = generator


class MockFactory:
    """
    Mock对象工厂

    提供统一的Mock对象创建和管理
    支持复杂对象的Mock配置和行为定义
    """

    def __init__(self):
        self.mock_configs = {}
        self.created_mocks = []
        self._setup_default_configs()

    def _setup_default_configs(self):
        """设置默认Mock配置"""

        # Trading相关Mock配置
        from src.trading.execution_engine import ExecutionEngine

        self.mock_configs["execution_engine"] = MockConfig(
            target_class=ExecutionEngine,
            method_configs={
                "execute_order": {
                    "return_value": {"status": "completed", "execution_id": "EXEC_001"},
                    "side_effect": None
                },
                "cancel_execution": {
                    "return_value": True,
                    "side_effect": None
                },
                "get_execution_status": {
                    "return_value": "completed",
                    "side_effect": None
                }
            },
            property_configs={
                "active_executions": [],
                "completed_executions": 10
            }
        )

        # Risk相关Mock配置
        self.mock_configs["risk_controller"] = MockConfig(
            target_class=object,  # 通用对象
            method_configs={
                "check_order": {
                    "return_value": {"allowed": True, "reason": "approved"},
                    "side_effect": None
                },
                "calculate_var": {
                    "return_value": -50000.0,
                    "side_effect": None
                }
            }
        )

        # Data相关Mock配置
        self.mock_configs["market_data_provider"] = MockConfig(
            target_class=object,
            method_configs={
                "get_latest_price": {
                    "return_value": 10.5,
                    "side_effect": None
                },
                "get_historical_data": {
                    "return_value": [{"timestamp": "2023-01-01", "price": 10.0}],
                    "side_effect": None
                }
            }
        )

    def create_mock(self, config_name: str, **overrides) -> Mock:
        """
        创建Mock对象

        Args:
            config_name: Mock配置名称
            **overrides: 覆盖配置

        Returns:
            Mock: 配置好的Mock对象
        """
        if config_name not in self.mock_configs:
            raise ValueError(f"未知的Mock配置: {config_name}")

        config = self.mock_configs[config_name]

        # 创建Mock对象
        mock_obj = Mock()

        # 配置方法
        for method_name, method_config in config.method_configs.items():
            if method_name in overrides:
                method_config = overrides[method_name]

            if "return_value" in method_config:
                getattr(mock_obj, method_name).return_value = method_config["return_value"]

            if "side_effect" in method_config:
                getattr(mock_obj, method_name).side_effect = method_config["side_effect"]

        # 配置属性
        for prop_name, prop_value in config.property_configs.items():
            if prop_name in overrides:
                prop_value = overrides[prop_name]
            setattr(mock_obj, prop_name, prop_value)

        # 记录创建的Mock
        self.created_mocks.append(mock_obj)

        return mock_obj

    def create_context_manager_mock(self, config_name: str = None, **config):
        """
        创建上下文管理器Mock

        Args:
            config_name: Mock配置名称
            **config: Mock配置

        Returns:
            Mock: 上下文管理器Mock
        """
        mock_obj = Mock()
        mock_obj.__enter__ = Mock(return_value=mock_obj)
        mock_obj.__exit__ = Mock(return_value=None)

        if config_name and config_name in self.mock_configs:
            base_config = self.mock_configs[config_name]
            # 应用基础配置
            for method_name, method_config in base_config.method_configs.items():
                if "return_value" in method_config:
                    getattr(mock_obj, method_name).return_value = method_config["return_value"]

        # 应用覆盖配置
        for key, value in config.items():
            if hasattr(mock_obj, key):
                getattr(mock_obj, key).return_value = value
            else:
                setattr(mock_obj, key, value)

        return mock_obj

    def reset_all_mocks(self):
        """重置所有Mock对象"""
        for mock_obj in self.created_mocks:
            mock_obj.reset_mock()
        self.created_mocks.clear()

    def add_mock_config(self, name: str, config: MockConfig):
        """添加自定义Mock配置"""
        self.mock_configs[name] = config


class TestEnvironmentManager:
    """
    测试环境管理器

    管理测试环境的设置、清理和资源管理
    """

    def __init__(self):
        self.environments = {}
        self.current_env = None

    def setup_environment(self, env_name: str, config: Dict[str, Any]):
        """
        设置测试环境

        Args:
            env_name: 环境名称
            config: 环境配置
        """
        self.environments[env_name] = config
        self.current_env = env_name

        # 设置环境变量
        if "env_vars" in config:
            for key, value in config["env_vars"].items():
                import os
                os.environ[key] = str(value)

        # 创建临时文件/目录
        if "temp_files" in config:
            import tempfile
            for temp_config in config["temp_files"]:
                if temp_config["type"] == "file":
                    with tempfile.NamedTemporaryFile(delete=False, **temp_config.get("kwargs", {})) as f:
                        if "content" in temp_config:
                            f.write(temp_config["content"].encode())
                        temp_config["_path"] = f.name

                elif temp_config["type"] == "directory":
                    temp_dir = tempfile.mkdtemp(**temp_config.get("kwargs", {}))
                    temp_config["_path"] = temp_dir

    def cleanup_environment(self, env_name: str = None):
        """
        清理测试环境

        Args:
            env_name: 环境名称，默认为当前环境
        """
        env_name = env_name or self.current_env
        if not env_name or env_name not in self.environments:
            return

        config = self.environments[env_name]

        # 清理临时文件
        if "temp_files" in config:
            import os
            import shutil
            for temp_config in config["temp_files"]:
                if "_path" in temp_config:
                    path = temp_config["_path"]
                    if os.path.isfile(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)

        # 清理环境变量
        if "env_vars" in config:
            import os
            for key in config["env_vars"].keys():
                if key in os.environ:
                    del os.environ[key]

        # 删除环境配置
        del self.environments[env_name]
        if self.current_env == env_name:
            self.current_env = None


# ==================== 全局实例 ====================

_test_data_factory = None
_mock_factory = None
_test_env_manager = None

def get_test_data_factory() -> TestDataFactory:
    """获取测试数据工厂全局实例"""
    global _test_data_factory
    if _test_data_factory is None:
        _test_data_factory = TestDataFactory()
    return _test_data_factory

def get_mock_factory() -> MockFactory:
    """获取Mock工厂全局实例"""
    global _mock_factory
    if _mock_factory is None:
        _mock_factory = MockFactory()
    return _mock_factory

def get_test_env_manager() -> TestEnvironmentManager:
    """获取测试环境管理器全局实例"""
    global _test_env_manager
    if _test_env_manager is None:
        _test_env_manager = TestEnvironmentManager()
    return _test_env_manager


# ==================== 便捷函数 ====================

def generate_test_data(template_name: str, **kwargs) -> Dict[str, Any]:
    """便捷的测试数据生成函数"""
    factory = get_test_data_factory()
    return factory.generate_data(template_name, **kwargs)

def generate_bulk_test_data(template_name: str, count: int, **kwargs) -> List[Dict[str, Any]]:
    """便捷的批量测试数据生成函数"""
    factory = get_test_data_factory()
    config = TestDataConfig(data_type=template_name, **kwargs)
    return factory.generate_bulk_data(template_name, count, config)

def create_mock(config_name: str, **overrides) -> Mock:
    """便捷的Mock创建函数"""
    factory = get_mock_factory()
    return factory.create_mock(config_name, **overrides)

def setup_test_environment(env_name: str, **config):
    """便捷的测试环境设置函数"""
    manager = get_test_env_manager()
    manager.setup_environment(env_name, config)

def cleanup_test_environment(env_name: str = None):
    """便捷的测试环境清理函数"""
    manager = get_test_env_manager()
    manager.cleanup_environment(env_name)


# ==================== Pytest Fixtures ====================

@pytest.fixture
def test_data_factory():
    """测试数据工厂fixture"""
    return get_test_data_factory()

@pytest.fixture
def mock_factory():
    """Mock工厂fixture"""
    return get_mock_factory()

@pytest.fixture
def test_env_manager():
    """测试环境管理器fixture"""
    return get_test_env_manager()

@pytest.fixture
def sample_trading_order():
    """示例交易订单fixture"""
    return generate_test_data("trading_order")

@pytest.fixture
def sample_market_data():
    """示例市场数据fixture"""
    return generate_test_data("market_data")

@pytest.fixture
def mock_execution_engine():
    """Mock执行引擎fixture"""
    return create_mock("execution_engine")

@pytest.fixture
def mock_risk_controller():
    """Mock风险控制器fixture"""
    return create_mock("risk_controller")


# ==================== 测试数据验证器 ====================

class TestDataValidator:
    """测试数据验证器"""

    @staticmethod
    def validate_trading_order(order: Dict[str, Any]) -> bool:
        """验证交易订单数据"""
        required_fields = ["order_id", "symbol", "quantity", "price", "order_type", "side"]
        return all(field in order for field in required_fields)

    @staticmethod
    def validate_market_data(data: Dict[str, Any]) -> bool:
        """验证市场数据"""
        required_fields = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        return all(field in data for field in required_fields)

    @staticmethod
    def validate_strategy_signal(signal: Dict[str, Any]) -> bool:
        """验证策略信号"""
        required_fields = ["signal_id", "symbol", "signal_type", "strength", "timestamp"]
        return all(field in signal for field in required_fields)


if __name__ == "__main__":
    # 示例用法
    print("🧪 测试数据工厂和Mock机制演示")
    print("=" * 50)

    # 创建测试数据
    factory = get_test_data_factory()

    print("📊 生成测试数据:")
    order = factory.generate_data("trading_order")
    print(f"交易订单: {order}")

    market_data = factory.generate_data("market_data")
    print(f"市场数据: {market_data}")

    signal = factory.generate_data("strategy_signal")
    print(f"策略信号: {signal}")

    print("\n🎭 创建Mock对象:")
    mock_factory = get_mock_factory()

    # 创建Mock执行引擎
    mock_engine = mock_factory.create_mock("execution_engine")
    result = mock_engine.execute_order({"order_id": "test"})
    print(f"Mock执行结果: {result}")

    print("\n✅ 测试数据工厂和Mock机制初始化完成！")
