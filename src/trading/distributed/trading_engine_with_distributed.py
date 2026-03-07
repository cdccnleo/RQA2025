import logging
"""
集成分布式能力的交易引擎

from src.engine.logging.unified_logger import get_unified_logger
本模块展示了如何将分布式锁、配置中心、分布式监控集成到交易系统中，
实现高可靠性、可配置、可监控的交易执行。
"""

import time
import threading
from typing import Dict, Any
from dataclasses import dataclass

# 导入分布式能力组件
from src.infrastructure.logging.distributed_lock import DistributedLockManager
from src.infrastructure.config.config_center import ConfigCenterManager
from src.infrastructure.logging.distributed_monitoring import DistributedMonitoringManager

# 导入现有交易组件
from ...trading_engine import TradingEngine
from ...execution.execution_engine import ExecutionEngine
from src.trading import ChinaRiskController
from ...execution.order_manager import OrderManager
logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:

    """交易配置数据类"""
    max_order_size: int = 1000
    max_position_per_symbol: float = 0.1
    risk_threshold: float = 0.02
    execution_timeout: int = 30
    retry_count: int = 3


class DistributedTradingEngine:

    """
    集成分布式能力的交易引擎

    特性:
    - 使用分布式锁确保订单执行的互斥性
    - 通过配置中心实现动态配置管理
    - 集成分布式监控实现全链路监控
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化分布式交易引擎

        Args:
            config: 配置字典，包含分布式组件和交易组件的配置
        """
        self.config = config
        self._lock = threading.Lock()

        # 初始化分布式组件
        self._init_distributed_components()

        # 初始化交易组件
        self._init_trading_components()

        # 加载初始配置
        self._load_initial_config()

        # 设置配置变更监听
        self._setup_config_watcher()

        logger.info("分布式交易引擎初始化完成")

    def _init_distributed_components(self):
        """初始化分布式组件"""
        try:
            # 初始化分布式锁管理器
            lock_config = self.config.get('distributed_lock', {})
            self.lock_manager = DistributedLockManager(lock_config)

            # 初始化配置中心管理器
            config_center_config = self.config.get('config_center', {})
            self.config_manager = ConfigCenterManager(config_center_config)

            # 初始化分布式监控管理器
            monitoring_config = self.config.get('distributed_monitoring', {})
            self.monitoring_manager = DistributedMonitoringManager(monitoring_config)

            logger.info("分布式组件初始化成功")
        except Exception as e:
            logger.error(f"分布式组件初始化失败: {e}")
            raise

    def _init_trading_components(self):
        """初始化交易组件"""
        try:
            # 初始化交易引擎
            trading_config = self.config.get('trading', {})
            self.trading_engine = TradingEngine(trading_config)

            # 初始化订单管理器
            order_config = self.config.get('order_manager', {})
            self.order_manager = OrderManager(order_config)

            # 初始化执行引擎
            self.execution_engine = ExecutionEngine(self.order_manager)

            # 初始化风控控制器
            risk_config = self.config.get('risk', {})
            self.risk_controller = ChinaRiskController(risk_config)

            logger.info("交易组件初始化成功")
        except Exception as e:
            logger.error(f"交易组件初始化失败: {e}")
            raise

    def _load_initial_config(self):
        """加载初始配置"""
        try:
            # 从配置中心加载交易配置
            trading_config = self.config_manager.get_config("trading / parameters")
            if trading_config:
                self.trading_config = TradingConfig(**trading_config.value)
                logger.info("从配置中心加载交易配置成功")
            else:
                # 使用默认配置
                self.trading_config = TradingConfig()
                # 将默认配置保存到配置中心
                self.config_manager.set_config("trading / parameters", self.trading_config.__dict__)
                logger.info("使用默认交易配置")

            # 从配置中心加载风控配置
            risk_config = self.config_manager.get_config("risk / thresholds")
            if risk_config:
                self.risk_controller.update_thresholds(risk_config.value)
                logger.info("从配置中心加载风控配置成功")

        except Exception as e:
            logger.error(f"加载初始配置失败: {e}")
            # 使用默认配置继续运行
            self.trading_config = TradingConfig()

    def _setup_config_watcher(self):
        """设置配置变更监听"""
        try:
            # 监听交易配置变更
            self.config_manager.watch_config("trading / parameters", self._on_trading_config_change)

            # 监听风控配置变更
            self.config_manager.watch_config("risk / thresholds", self._on_risk_config_change)

            logger.info("配置变更监听设置成功")
        except Exception as e:
            logger.error(f"设置配置变更监听失败: {e}")

    def _on_trading_config_change(self, config_key: str, new_value: Dict[str, Any]):
        """交易配置变更回调"""
        try:
            with self._lock:
                self.trading_config = TradingConfig(**new_value)
                logger.info(f"交易配置已更新: {config_key}")
        except Exception as e:
            logger.error(f"更新交易配置失败: {e}")

    def _on_risk_config_change(self, config_key: str, new_value: Dict[str, Any]):
        """风控配置变更回调"""
        try:
            self.risk_controller.update_thresholds(new_value)
            logger.info(f"风控配置已更新: {config_key}")
        except Exception as e:
            logger.error(f"更新风控配置失败: {e}")

    def _setup_monitoring(self):
        """设置监控指标"""
        try:
            # 注册业务指标
            self.monitoring_manager.register_metric("trading.order_count", "counter")
            self.monitoring_manager.register_metric("trading.order_success_rate", "gauge")
            self.monitoring_manager.register_metric("trading.order_execution_time", "histogram")
            self.monitoring_manager.register_metric("trading.risk_rejection_rate", "gauge")
            self.monitoring_manager.register_metric("trading.position_value", "gauge")

            # 设置告警规则
            self.monitoring_manager.add_alert_rule({
                "name": "trading_error_rate_high",
                "condition": "trading.order_success_rate < 0.95",
                "severity": "critical",
                "message": "交易成功率过低"
            })

            self.monitoring_manager.add_alert_rule({
                "name": "trading_risk_rejection_high",
                "condition": "trading.risk_rejection_rate > 0.1",
                "severity": "warning",
                "message": "风控拒绝率过高"
            })

            logger.info("监控指标设置成功")
        except Exception as e:
            logger.error(f"设置监控指标失败: {e}")

    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行订单（集成分布式锁）

        Args:
            order: 订单信息

        Returns:
            执行结果
        """
        start_time = time.time()
        order_id = order.get('order_id', f"order_{int(time.time())}")
        symbol = order.get('symbol', '')

        # 记录订单开始指标
        self.monitoring_manager.record_metric("trading.order_count", 1)

        try:
            # 获取订单锁
            lock_key = f"order:{symbol}:{order_id}"
            with self.lock_manager.acquire_lock(lock_key, timeout=self.trading_config.execution_timeout):
                logger.info(f"获取订单锁成功: {lock_key}")

                # 风控检查
                risk_result = self.risk_controller.check(order)
                if not risk_result['passed']:
                    # 记录风控拒绝指标
                    self.monitoring_manager.record_metric("trading.risk_rejection_rate", 1.0)
                    logger.warning(f"订单被风控拒绝: {order_id}, 原因: {risk_result['reason']}")
                    return {
                        'success': False,
                        'order_id': order_id,
                        'reason': f"风控拒绝: {risk_result['reason']}",
                        'execution_time': time.time() - start_time
                    }

                # 执行订单
                execution_result = self.execution_engine.execute_order(order)

                # 记录成功指标
                self.monitoring_manager.record_metric("trading.order_success_rate", 1.0)

                # 记录执行时间
                execution_time = time.time() - start_time
                self.monitoring_manager.record_metric(
                    "trading.order_execution_time", execution_time)

                logger.info(f"订单执行成功: {order_id}, 耗时: {execution_time:.3f}秒")

                return {
                    'success': True,
                    'order_id': order_id,
                    'result': execution_result,
                    'execution_time': execution_time
                }

        except Exception as e:
            # 记录错误指标
            self.monitoring_manager.record_metric("trading.order_success_rate", 0.0)
            logger.error(f"订单执行失败: {order_id}, 错误: {e}")

            return {
                'success': False,
                'order_id': order_id,
                'reason': f"执行异常: {str(e)}",
                'execution_time': time.time() - start_time
            }

    def update_position(self, symbol: str, quantity: int) -> Dict[str, Any]:
        """
        更新持仓（集成分布式锁）

        Args:
            symbol: 股票代码
            quantity: 数量变化

        Returns:
            更新结果
        """
        start_time = time.time()

        try:
            # 获取持仓锁
            lock_key = f"position:{symbol}"
            with self.lock_manager.acquire_lock(lock_key, timeout=60):
                logger.info(f"获取持仓锁成功: {lock_key}")

                # 更新持仓
                self.order_manager.update_position(symbol, quantity)

                # 记录持仓价值指标
                if hasattr(self.order_manager, 'get_position_value'):
                    position_value = self.order_manager.get_position_value(symbol)
                    self.monitoring_manager.record_metric("trading.position_value", position_value)

                logger.info(f"持仓更新成功: {symbol}, 变化: {quantity}")

                return {
                    'success': True,
                    'symbol': symbol,
                    'quantity_change': quantity,
                    'execution_time': time.time() - start_time
                }

        except Exception as e:
            logger.error(f"持仓更新失败: {symbol}, 错误: {e}")

            return {
                'success': False,
                'symbol': symbol,
                'reason': f"更新异常: {str(e)}",
                'execution_time': time.time() - start_time
            }

    def get_trading_status(self) -> Dict[str, Any]:
        """
        获取交易状态

        Returns:
            交易状态信息
        """
        try:
            # 获取锁状态
            lock_status = self.lock_manager.get_lock_info()

            # 获取配置状态
            config_status = {
                'trading_config': self.trading_config.__dict__,
                'config_center_connected': self.config_manager.is_connected()
            }

            # 获取监控状态
            monitoring_status = {
                'metrics_count': len(self.monitoring_manager.get_metrics()),
                'alerts_count': len(self.monitoring_manager.get_alerts())
            }

            return {
                'lock_status': lock_status,
                'config_status': config_status,
                'monitoring_status': monitoring_status,
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"获取交易状态失败: {e}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }

    def shutdown(self):
        """关闭分布式交易引擎"""
        try:
            # 关闭分布式组件
            if hasattr(self, 'lock_manager'):
                self.lock_manager.shutdown()

            if hasattr(self, 'config_manager'):
                self.config_manager.shutdown()

            if hasattr(self, 'monitoring_manager'):
                self.monitoring_manager.shutdown()

            logger.info("分布式交易引擎已关闭")

        except Exception as e:
            logger.error(f"关闭分布式交易引擎失败: {e}")


# 使用示例

def create_distributed_trading_engine() -> DistributedTradingEngine:
    """
    创建分布式交易引擎实例

    Returns:
        分布式交易引擎实例
    """
    config = {
        'distributed_lock': {
            'redis_endpoints': ['localhost:6379'],
            'lock_timeout': 30,
            'retry_count': 3
        },
        'config_center': {
            'etcd_endpoints': ['localhost:2379'],
            'encryption_key': 'your - encryption - key'
        },
        'distributed_monitoring': {
            'prometheus_endpoints': ['localhost:9090'],
            'metrics_interval': 30
        },
        'trading': {
            'initial_capital': 1000000.0,
            'max_position_per_symbol': 0.1
        },
        'risk': {
            'market_type': 'A',
            'per_trade_risk': 0.01
        }
    }

    return DistributedTradingEngine(config)


if __name__ == "__main__":
    # 示例用法
    engine = create_distributed_trading_engine()

    # 执行订单
    order = {
        'symbol': '600519.SH',
        'order_type': 'buy',
        'quantity': 100,
        'price': 180.0
    }

    result = engine.execute_order(order)
    print(f"订单执行结果: {result}")

    # 获取状态
    status = engine.get_trading_status()
    print(f"交易状态: {status}")

    # 关闭引擎
    engine.shutdown()
