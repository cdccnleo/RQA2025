"""
交易层健康检查组件

负责检查交易层各服务的健康状态。
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TradingHealthChecker:
    """交易层健康检查器"""

    def __init__(self, adapter):
        """初始化健康检查器
        
        Args:
            adapter: TradingLayerAdapter实例，用于获取服务
        """
        self.adapter = adapter

    def check_trading_engine_health(self) -> Dict[str, Any]:
        """检查交易引擎健康状态"""
        engine = self.adapter.get_trading_engine()
        if engine and hasattr(engine, 'health_check'):
            try:
                return engine.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif engine:
            return {'status': 'healthy', 'engine_available': True}
        return {'status': 'unknown'}

    def check_order_manager_health(self) -> Dict[str, Any]:
        """检查订单管理器健康状态"""
        manager = self.adapter.get_order_manager()
        if manager and hasattr(manager, 'health_check'):
            try:
                return manager.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif manager:
            return {'status': 'healthy', 'manager_available': True}
        return {'status': 'unknown'}

    def check_execution_engine_health(self) -> Dict[str, Any]:
        """检查执行引擎健康状态"""
        engine = self.adapter.get_execution_engine()
        if engine and hasattr(engine, 'health_check'):
            try:
                return engine.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif engine:
            return {'status': 'healthy', 'engine_available': True}
        return {'status': 'unknown'}

    def check_portfolio_manager_health(self) -> Dict[str, Any]:
        """检查投资组合管理器健康状态"""
        manager = self.adapter.get_portfolio_manager()
        if manager and hasattr(manager, 'health_check'):
            try:
                return manager.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif manager:
            return {'status': 'healthy', 'manager_available': True}
        return {'status': 'unknown'}

    def check_all_services_health(self) -> Dict[str, Any]:
        """检查所有交易服务的健康状态"""
        return {
            'trading_engine': self.check_trading_engine_health(),
            'order_manager': self.check_order_manager_health(),
            'execution_engine': self.check_execution_engine_health(),
            'portfolio_manager': self.check_portfolio_manager_health()
        }

