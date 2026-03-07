#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一Mock管理器 - RQA2025量化交易系统

提供标准化的Mock对象创建和管理，支持各层级的测试需求
创建时间: 2025-12-04
"""

from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class UnifiedMockManager:
    """统一Mock管理器"""

    def __init__(self):
        self._mock_cache = {}
        self._patchers = []

    def create_standard_mock(self, **kwargs) -> Mock:
        """创建标准Mock对象"""
        mock = Mock(**kwargs)
        # 设置基本属性
        mock.__class__.__name__ = "MockObject"
        return mock

    def create_infrastructure_mock(self) -> Mock:
        """创建基础设施Mock"""
        mock = Mock()

        # 缓存相关方法
        mock.get = Mock(return_value="cached_value")
        mock.set = Mock(return_value=True)
        mock.delete = Mock(return_value=True)
        mock.exists = Mock(return_value=True)

        # 配置相关方法
        mock.load_config = Mock(return_value={"setting": "value"})
        mock.save_config = Mock(return_value=True)
        mock.get_config = Mock(return_value={"config": "data"})

        # 日志相关方法
        mock.info = Mock()
        mock.warning = Mock()
        mock.error = Mock()
        mock.debug = Mock()

        # 健康检查相关
        mock.check_health = Mock(return_value={"status": "healthy", "score": 95})

        return mock

    def create_data_mock(self) -> Mock:
        """创建数据层Mock"""
        mock = Mock()

        # 数据加载器方法
        mock.load = Mock(return_value=pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'price': np.random.normal(100, 5, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }))

        # 数据处理器方法
        mock.process = Mock(return_value={"processed": True, "records": 100})
        mock.validate = Mock(return_value=True)
        mock.clean = Mock(return_value={"cleaned_records": 95})

        # 缓存相关
        mock.get_cached = Mock(return_value=None)
        mock.set_cache = Mock(return_value=True)

        # 数据库相关
        mock.connect = Mock(return_value=True)
        mock.disconnect = Mock()
        mock.execute_query = Mock(return_value=[])

        return mock

    def create_strategy_mock(self) -> Mock:
        """创建策略层Mock"""
        mock = Mock()

        # 策略基本属性
        mock.strategy_id = "test_strategy"
        mock.name = "Test Strategy"
        mock.status = "active"

        # 策略核心方法
        mock.execute = Mock(return_value={
            "signal": "BUY",
            "confidence": 0.8,
            "quantity": 100,
            "price": 150.0
        })

        mock.initialize = Mock(return_value=True)
        mock.start = Mock()
        mock.stop = Mock()

        # 参数管理
        mock.set_parameters = Mock(return_value=True)
        mock.get_parameters = Mock(return_value={"param1": 100, "param2": 0.05})
        mock.validate_parameters = Mock(return_value=True)

        # 性能跟踪
        mock.get_performance_stats = Mock(return_value={
            "total_trades": 50,
            "win_rate": 0.65,
            "total_pnl": 2500.0,
            "sharpe_ratio": 1.2
        })

        return mock

    def create_risk_mock(self) -> Mock:
        """创建风险控制Mock"""
        mock = Mock()

        # 风险评估方法
        mock.assess_portfolio_risk = Mock(return_value={
            "var_95": 0.05,
            "cvar_95": 0.08,
            "max_drawdown": 0.12,
            "volatility": 0.18
        })

        mock.calculate_var = Mock(return_value=5000.0)
        mock.check_risk_limits = Mock(return_value=True)

        # 监控方法
        mock.monitor_positions = Mock(return_value=[])
        mock.get_risk_alerts = Mock(return_value=[])

        # 压力测试
        mock.run_stress_test = Mock(return_value={
            "scenario_results": {"crash": -0.15, "recovery": 0.08},
            "breach_probability": 0.02
        })

        return mock

    def create_trading_mock(self) -> Mock:
        """创建交易层Mock"""
        mock = Mock()

        # 订单管理
        mock.place_order = Mock(return_value={"order_id": "12345", "status": "filled"})
        mock.cancel_order = Mock(return_value=True)
        mock.get_order_status = Mock(return_value="filled")

        # 交易执行
        mock.execute_trade = Mock(return_value={
            "trade_id": "T123",
            "executed_quantity": 100,
            "executed_price": 150.0,
            "commission": 15.0
        })

        # 持仓管理
        mock.get_positions = Mock(return_value=[
            {"symbol": "AAPL", "quantity": 100, "avg_price": 145.0, "current_value": 15000}
        ])

        mock.update_position = Mock(return_value=True)

        return mock

    def create_ml_mock(self) -> Mock:
        """创建机器学习Mock"""
        mock = Mock()

        # 模型训练
        mock.train = Mock(return_value={"accuracy": 0.85, "loss": 0.15})
        mock.fit = Mock(return_value=mock)

        # 预测
        mock.predict = Mock(return_value=np.array([0.8, 0.6, 0.9]))
        mock.predict_proba = Mock(return_value=np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]]))

        # 模型评估
        mock.score = Mock(return_value=0.85)
        mock.evaluate = Mock(return_value={
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85
        })

        # 特征处理
        mock.transform = Mock(return_value=np.random.rand(100, 20))
        mock.fit_transform = Mock(return_value=np.random.rand(100, 20))

        return mock

    def create_monitoring_mock(self) -> Mock:
        """创建监控层Mock"""
        mock = Mock()

        # 指标收集
        mock.collect_metrics = Mock(return_value={
            "cpu_usage": 45.2,
            "memory_usage": 2.1,
            "response_time": 125.0,
            "error_rate": 0.02
        })

        # 告警管理
        mock.check_alerts = Mock(return_value=[])
        mock.create_alert = Mock(return_value={"alert_id": "A123", "severity": "warning"})
        mock.resolve_alert = Mock(return_value=True)

        # 健康检查
        mock.health_check = Mock(return_value={
            "status": "healthy",
            "components": {
                "database": "up",
                "cache": "up",
                "api": "up"
            }
        })

        return mock

    def create_business_objects_mock(self) -> Dict[str, Mock]:
        """创建业务对象Mock集合"""
        return {
            "portfolio": self._create_portfolio_mock(),
            "market_data": self._create_market_data_mock(),
            "order": self._create_order_mock(),
            "trade": self._create_trade_mock(),
            "position": self._create_position_mock()
        }

    def _create_portfolio_mock(self) -> Mock:
        """创建投资组合Mock"""
        mock = Mock()
        mock.total_value = 1000000.0
        mock.positions = [
            {"symbol": "AAPL", "quantity": 1000, "avg_price": 145.0},
            {"symbol": "GOOGL", "quantity": 50, "avg_price": 2750.0}
        ]
        mock.get_total_value = Mock(return_value=1000000.0)
        mock.get_positions = Mock(return_value=mock.positions)
        return mock

    def _create_market_data_mock(self) -> Mock:
        """创建市场数据Mock"""
        mock = Mock()
        mock.symbol = "AAPL"
        mock.price = 150.0
        mock.volume = 1000000
        mock.timestamp = datetime.now()
        mock.high = 152.0
        mock.low = 148.0
        mock.open = 149.0
        mock.close = 150.0
        return mock

    def _create_order_mock(self) -> Mock:
        """创建订单Mock"""
        mock = Mock()
        mock.order_id = "O12345"
        mock.symbol = "AAPL"
        mock.side = "BUY"
        mock.quantity = 100
        mock.price = 150.0
        mock.order_type = "LIMIT"
        mock.status = "filled"
        return mock

    def _create_trade_mock(self) -> Mock:
        """创建交易Mock"""
        mock = Mock()
        mock.trade_id = "T12345"
        mock.symbol = "AAPL"
        mock.side = "BUY"
        mock.quantity = 100
        mock.price = 150.0
        mock.timestamp = datetime.now()
        mock.commission = 15.0
        return mock

    def _create_position_mock(self) -> Mock:
        """创建持仓Mock"""
        mock = Mock()
        mock.symbol = "AAPL"
        mock.quantity = 1000
        mock.avg_price = 145.0
        mock.current_price = 150.0
        mock.unrealized_pnl = 5000.0
        mock.market_value = 150000.0
        return mock

    def patch_module(self, module_name: str, mock_obj: Mock) -> None:
        """打补丁到模块"""
        patcher = patch(module_name, mock_obj)
        patcher.start()
        self._patchers.append(patcher)

    def patch_function(self, target: str, mock_function: Mock) -> None:
        """打补丁到函数"""
        patcher = patch(target, mock_function)
        patcher.start()
        self._patchers.append(patcher)

    def patch_object(self, target: Any, attribute: str, mock_obj: Mock) -> None:
        """打补丁到对象属性"""
        patcher = patch.object(target, attribute, mock_obj)
        patcher.start()
        self._patchers.append(patcher)

    def cleanup(self) -> None:
        """清理所有补丁"""
        for patcher in self._patchers:
            patcher.stop()
        self._patchers.clear()
        self._mock_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ==================== 便捷函数 ====================

def create_mock_with_methods(methods: Dict[str, Any]) -> Mock:
    """创建带有指定方法的Mock对象"""
    mock = Mock()
    for method_name, return_value in methods.items():
        setattr(mock, method_name, Mock(return_value=return_value))
    return mock


def create_data_frame_mock(rows: int = 100, columns: List[str] = None) -> Mock:
    """创建DataFrame Mock"""
    if columns is None:
        columns = ['timestamp', 'price', 'volume', 'high', 'low']

    mock = Mock()
    mock.shape = (rows, len(columns))
    mock.columns = columns
    mock.index = range(rows)

    # 添加基本方法
    mock.head = Mock(return_value=f"DataFrame with {rows} rows and {len(columns)} columns")
    mock.tail = Mock(return_value=f"DataFrame with {rows} rows and {len(columns)} columns")
    mock.describe = Mock(return_value={})
    mock.info = Mock()

    return mock


def create_database_connection_mock() -> Mock:
    """创建数据库连接Mock"""
    mock = Mock()
    mock.connect = Mock(return_value=True)
    mock.disconnect = Mock()
    mock.execute = Mock(return_value=[])
    mock.fetchone = Mock(return_value=None)
    mock.fetchall = Mock(return_value=[])
    mock.commit = Mock()
    mock.rollback = Mock()
    return mock


def create_redis_client_mock() -> Mock:
    """创建Redis客户端Mock"""
    mock = Mock()
    mock.get = Mock(return_value=None)
    mock.set = Mock(return_value=True)
    mock.delete = Mock(return_value=1)
    mock.exists = Mock(return_value=1)
    mock.expire = Mock(return_value=True)
    return mock


def create_logger_mock() -> Mock:
    """创建日志器Mock"""
    mock = Mock()
    mock.debug = Mock()
    mock.info = Mock()
    mock.warning = Mock()
    mock.error = Mock()
    mock.critical = Mock()
    mock.log = Mock()
    return mock


def create_async_function_mock(return_value: Any = None) -> Mock:
    """创建异步函数Mock"""
    async def async_mock(*args, **kwargs):
        return return_value

    mock = Mock(side_effect=async_mock)
    return mock


# ==================== 上下文管理器 ====================

class MockContext:
    """Mock上下文管理器"""

    def __init__(self, mock_manager: UnifiedMockManager):
        self.mock_manager = mock_manager
        self.patches = []

    def patch(self, target: str, mock_obj: Mock) -> None:
        """添加补丁"""
        patcher = patch(target, mock_obj)
        patcher.start()
        self.patches.append(patcher)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for patcher in self.patches:
            patcher.stop()


# ==================== 全局实例 ====================

# 创建全局Mock管理器实例
global_mock_manager = UnifiedMockManager()


def get_mock_manager() -> UnifiedMockManager:
    """获取全局Mock管理器实例"""
    return global_mock_manager


# ==================== 快速Mock创建函数 ====================

def mock_infrastructure() -> Mock:
    """快速创建基础设施Mock"""
    return global_mock_manager.create_infrastructure_mock()


def mock_data() -> Mock:
    """快速创建数据Mock"""
    return global_mock_manager.create_data_mock()


def mock_strategy() -> Mock:
    """快速创建策略Mock"""
    return global_mock_manager.create_strategy_mock()


def mock_risk() -> Mock:
    """快速创建风险控制Mock"""
    return global_mock_manager.create_risk_mock()


def mock_trading() -> Mock:
    """快速创建交易Mock"""
    return global_mock_manager.create_trading_mock()


def mock_ml() -> Mock:
    """快速创建机器学习Mock"""
    return global_mock_manager.create_ml_mock()


def mock_monitoring() -> Mock:
    """快速创建监控Mock"""
    return global_mock_manager.create_monitoring_mock()


# ==================== 批量Mock创建 ====================

def create_layer_mocks() -> Dict[str, Mock]:
    """创建各层级Mock集合"""
    return {
        "infrastructure": mock_infrastructure(),
        "data": mock_data(),
        "strategy": mock_strategy(),
        "risk": mock_risk(),
        "trading": mock_trading(),
        "ml": mock_ml(),
        "monitoring": mock_monitoring()
    }


def create_business_mocks() -> Dict[str, Mock]:
    """创建业务对象Mock集合"""
    return global_mock_manager.create_business_objects_mock()
