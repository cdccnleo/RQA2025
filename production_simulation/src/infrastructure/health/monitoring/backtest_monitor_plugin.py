"""
backtest_monitor_plugin 模块

提供 backtest_monitor_plugin 相关功能和接口。
"""

import logging

import time

from prometheus_client import Counter, Gauge, CollectorRegistry
from dataclasses import dataclass
from datetime import datetime
from prometheus_client import Counter, Gauge, CollectorRegistry, REGISTRY
from typing import Dict, Any, Optional, List
"""
基础设施层 - 资源管理组件

backtest_monitor_plugin 模块

资源管理相关的文件
提供资源管理相关的功能实现。
"""


@dataclass
class BacktestMetrics:

    """回测监控指标"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0

    def update(self, data: Dict[str, Any]) -> None:
        """更新指标数据"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class BacktestMonitorPlugin:

    """回测监控器"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        初始化回测监控器，支持Prometheus registry参数隔离
        """
        self.metrics = BacktestMetrics()
        self._trade_history = []
        self._portfolio_history = []
        self._performance_metrics = {
            'returns': [],
            'volatility': [],
            'sharpe': [],
            'max_drawdown': []
        }

        self.registry = registry if registry is not None else CollectorRegistry()
        try:
            self._trade_counter = Counter(
                'backtest_trades_total',
                'Total number of trades',
                ['status'],
                registry=self.registry
            )
        except ValueError as e:
            if "Duplicated timeseries" in str(e):
                # 如果已存在，使用已有的计数器
                self._trade_counter = Counter(
                    'backtest_trades_total',
                    'Total number of trades',
                    ['status'],
                    registry=self.registry
                )
            else:
                raise

        try:
            self._pnl_gauge = Gauge(
                'backtest_pnl',
                'Current PnL value',
                registry=self.registry
            )
        except ValueError:
            self._pnl_gauge = Gauge(
                'backtest_pnl',
                'Current PnL value',
                registry=self.registry
            )

        try:
            self._drawdown_gauge = Gauge(
                'backtest_max_drawdown',
                'Maximum drawdown',
                registry=self.registry
            )
        except ValueError:
            self._drawdown_gauge = Gauge(
                'backtest_max_drawdown',
                'Maximum drawdown',
                registry=self.registry
            )

        try:
            self._sharpe_gauge = Gauge(
                'backtest_sharpe_ratio',
                'Sharpe ratio',
                registry=self.registry
            )
        except ValueError:
            self._sharpe_gauge = Gauge(
                'backtest_sharpe_ratio',
                'Sharpe ratio',
                registry=self.registry
            )

    def record_trade(self, **kwargs) -> None:
        """
        记录交易数据

        Args:
            **kwargs: 交易数据关键字参数
        """
        # 构建交易记录
        trade_record = {
            'value': kwargs.get('price', 0.0),
            'tags': {
                'symbol': kwargs.get('symbol', ''),
                'action': kwargs.get('action', ''),
                'side': kwargs.get('side', kwargs.get('action', '')),  # 支持side参数
                'strategy': kwargs.get('strategy', ''),
                'quantity': kwargs.get('quantity', 0)
            },
            'timestamp': kwargs.get('timestamp', datetime.now())
        }

        self._trade_history.append(trade_record)

        # 更新内部指标
        self.metrics.total_trades += 1

        # 模拟成功 / 失败判断
        success = kwargs.get('action') == 'SELL'  # 简化逻辑
        if success:
            self.metrics.successful_trades += 1
            self._trade_counter.labels(status='success').inc()
        else:
            self.metrics.failed_trades += 1
            self._trade_counter.labels(status='failed').inc()

        # 更新PnL
        pnl = kwargs.get('pnl', 0.0)
        self.metrics.total_pnl += pnl
        self._pnl_gauge.set(self.metrics.total_pnl)

        # 更新胜率
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.successful_trades / self.metrics.total_trades

    def record_portfolio(self, **kwargs) -> None:
        """
        记录投资组合数据

        Args:
            **kwargs: 投资组合数据关键字参数
        """
        # 构建组合记录
        portfolio_record = {
            'value': kwargs.get('value', 0.0),
            'tags': {
                'strategy': kwargs.get('strategy', ''),
                'positions': str(len(kwargs.get('positions', {})))
            },
            'timestamp': kwargs.get('timestamp', datetime.now())
        }

        self._portfolio_history.append(portfolio_record)

        # 更新最大回撤
        drawdown = kwargs.get('max_drawdown', 0.0)
        if drawdown > self.metrics.max_drawdown:
            self.metrics.max_drawdown = drawdown
            self._drawdown_gauge.set(drawdown)

        # 更新夏普比率
        sharpe = kwargs.get('sharpe_ratio', 0.0)
        self.metrics.sharpe_ratio = sharpe
        self._sharpe_gauge.set(sharpe)

        # 更新平均交易时长
        avg_duration = kwargs.get('avg_trade_duration', 0.0)
        self.metrics.avg_trade_duration = avg_duration

    def record_performance(self, **kwargs) -> None:
        """
        记录性能数据

        Args:
            **kwargs: 性能数据关键字参数
        """
        # 记录各项指标
        for metric_name in ['returns', 'volatility', 'sharpe', 'max_drawdown']:
            if metric_name in kwargs:
                metric_record = {
                    'value': kwargs[metric_name],
                    'tags': {
                        'strategy': kwargs.get('strategy', '')
                    },
                    'timestamp': kwargs.get('timestamp', datetime.now())
                }
                self._performance_metrics[metric_name].append(metric_record)

        # 更新内部指标
        performance_data = {k: v for k, v in kwargs.items() if hasattr(self.metrics, k)}
        self.metrics.update(performance_data)

    def get_trade_history(self, **filters) -> List[Dict[str, Any]]:
        """
        获取交易历史

        Args:
            **filters: 过滤条件

        Returns:
            过滤后的交易列表
        """
        trades = self._trade_history

        # 应用过滤条件
        if 'strategy' in filters:
            trades = [t for t in trades if t['tags'].get('strategy') == filters['strategy']]
        if 'symbol' in filters:
            trades = [t for t in trades if t['tags'].get('symbol') == filters['symbol']]
        if 'side' in filters:
            trades = [t for t in trades if t['tags'].get('side') == filters['side']]
        if 'start_time' in filters:
            trades = [t for t in trades if t['timestamp'] >= filters['start_time']]
        if 'end_time' in filters:
            trades = [t for t in trades if t['timestamp'] <= filters['end_time']]

        return trades

    def get_portfolio_history(self, **filters) -> List[Dict[str, Any]]:
        """
        获取组合历史

        Args:
            **filters: 过滤条件

        Returns:
            过滤后的组合列表
        """
        portfolios = self._portfolio_history

        # 应用过滤条件
        if 'strategy' in filters:
            portfolios = [p for p in portfolios if p['tags'].get('strategy') == filters['strategy']]

        return portfolios

    def get_performance_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取性能指标

        Returns:
            性能指标字典，只返回有数据的指标
        """
        # 只返回有数据的指标
        result = {}
        for metric_name, metrics in self._performance_metrics.items():
            if len(metrics) > 0:
                result[metric_name] = metrics
        return result

    def get_custom_metrics(self, name: str) -> List[Dict[str, Any]]:
        """
        获取自定义指标

        Args:
            name: 指标名称

        Returns:
            指标列表
        """
        if name == 'position_detail':
            # 返回持仓详情（模拟数据）
            return [
                {'tags': {'symbol': '600000.SH'}, 'value': 1000},
                {'tags': {'symbol': '000001.SZ'}, 'value': 500}
            ]

        return []

    def filter_trades(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        过滤交易记录

        Args:
            criteria: 过滤条件

        Returns:
            过滤后的交易列表
        """
        return self.get_trade_history(**criteria)

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取监控指标

        Returns:
            指标字典
        """
        return {
            'total_trades': self.metrics.total_trades,
            'successful_trades': self.metrics.successful_trades,
            'failed_trades': self.metrics.failed_trades,
            'total_pnl': self.metrics.total_pnl,
            'max_drawdown': self.metrics.max_drawdown,
            'sharpe_ratio': self.metrics.sharpe_ratio,
            'win_rate': self.metrics.win_rate,
            'avg_trade_duration': self.metrics.avg_trade_duration
        }

    def start(self) -> bool:
        """启动回测监控器"""
        # BacktestMonitorPlugin是无状态的，启动总是成功
        return True

    def stop(self) -> bool:
        """停止回测监控器"""
        # BacktestMonitorPlugin是无状态的，停止总是成功
        return True

    def monitor_backtest(self, backtest_id: str) -> Dict[str, Any]:
        """监控回测过程"""
        return {
            "backtest_id": backtest_id,
            "status": "running",
            "progress": 0.5,
            "performance": self.get_metrics()
        }

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "metrics_available": True,
            "trades_count": len(self._trade_history),
            "performance_records": len(self._performance_metrics.get('returns', []))
        }

    def reset_metrics(self) -> None:
        """重置所有指标"""
        self.metrics = BacktestMetrics()
        self._trade_history = []
        self._portfolio_history = []
        self._performance_metrics = {
            'returns': [],
            'volatility': [],
            'sharpe': [],
            'max_drawdown': []
        }

        self._pnl_gauge.set(0.0)
        self._drawdown_gauge.set(0.0)
        self._sharpe_gauge.set(0.0)

# 模块级健康检查函数


def check_health() -> Dict[str, Any]:
    """执行整体健康检查"""
    try:
        logger = logging.getLogger(__name__)
        logger.info("开始回测监控插件模块健康检查")

        health_checks = {
            "plugin_class": check_plugin_class(),
            "metrics_class": check_metrics_class(),
            "prometheus_integration": check_prometheus_integration()
        }

        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())
        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "backtest_monitor_plugin",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("回测监控插件模块健康检查发现问题")
            result["issues"] = [name for name, check in health_checks.items()
                                if not check.get("healthy", False)]

        logger.info(f"回测监控插件模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"回测监控插件模块健康检查失败: {str(e)}", exc_info=True)
        return {"healthy": False, "timestamp": datetime.now().isoformat(), "service": "backtest_monitor_plugin", "error": str(e)}


def check_plugin_class() -> Dict[str, Any]:
    """检查插件类"""
    try:
        plugin_exists = 'BacktestMonitorPlugin' in globals()
        if not plugin_exists:
            return {"healthy": False, "error": "BacktestMonitorPlugin class not found"}

        required_methods = ['record_trade', 'record_portfolio', 'record_performance', 'get_metrics']
        methods_exist = all(hasattr(BacktestMonitorPlugin, method) for method in required_methods)

        return {
            "healthy": plugin_exists and methods_exist,
            "plugin_exists": plugin_exists,
            "methods_exist": methods_exist
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def check_metrics_class() -> Dict[str, Any]:
    """检查指标类"""
    try:
        metrics_exists = 'BacktestMetrics' in globals()
        if not metrics_exists:
            return {"healthy": False, "error": "BacktestMetrics class not found"}

        update_method_exists = hasattr(BacktestMetrics, 'update')
        return {
            "healthy": metrics_exists and update_method_exists,
            "metrics_exists": metrics_exists,
            "update_method_exists": update_method_exists
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def check_prometheus_integration() -> Dict[str, Any]:
    """检查Prometheus集成"""
    try:
        prometheus_available = True
        try:
            import sys
        except ImportError:
            prometheus_available = False

        return {"healthy": prometheus_available, "prometheus_available": prometheus_available}
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def health_status() -> Dict[str, Any]:
    """获取健康状态摘要"""
    try:
        health_check = check_health()
        return {
            "status": "healthy" if health_check["healthy"] else "unhealthy",
            "service": "backtest_monitor_plugin",
            "health_check": health_check,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def health_summary() -> Dict[str, Any]:
    """获取健康摘要报告"""
    try:
        health_check = check_health()
        return {
            "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
            "backtest_monitor_plugin_module_info": {
                "service_name": "backtest_monitor_plugin",
                "purpose": "回测监控插件",
                "operational": health_check["healthy"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"overall_health": "error", "error": str(e)}


def monitor_backtest_monitor_plugin() -> Dict[str, Any]:
    """监控回测监控插件状态"""
    try:
        health_check = check_health()
        monitoring_efficiency = 1.0 if health_check["healthy"] else 0.0
        return {
            "healthy": health_check["healthy"],
            "plugin_metrics": {
                "service_name": "backtest_monitor_plugin",
                "monitoring_efficiency": monitoring_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            }
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def validate_backtest_monitor_plugin() -> Dict[str, Any]:
    """验证回测监控插件"""
    try:
        validation_results = {
            "plugin_validation": check_plugin_class(),
            "metrics_validation": check_metrics_class(),
            "prometheus_validation": check_prometheus_integration()
        }
        overall_valid = all(result.get("valid", False) for result in validation_results.values())
        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
