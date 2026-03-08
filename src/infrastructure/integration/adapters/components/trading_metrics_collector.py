"""
交易层指标收集组件

负责收集交易层的性能指标。
"""

import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class TradingMetricsCollector:
    """交易层指标收集器"""

    def __init__(self, adapter):
        """初始化指标收集器
        
        Args:
            adapter: TradingLayerAdapter实例，用于获取服务和组件
        """
        self.adapter = adapter

    def collect_trading_metrics(self) -> Dict[str, Any]:
        """收集交易层性能指标"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'layer_type': 'trading',
            'infrastructure_metrics': {},
            'trading_metrics': {}
        }

        # 获取基础设施服务指标
        infrastructure_services = self.adapter.get_infrastructure_services()
        for service_name, service in infrastructure_services.items():
            if service and hasattr(service, 'get_metrics'):
                try:
                    metrics['infrastructure_metrics'][service_name] = service.get_metrics()
                except Exception as e:
                    metrics['infrastructure_metrics'][service_name] = {'error': str(e)}

        # 获取交易组件指标
        components = {
            'trading_engine': self.adapter.get_trading_engine(),
            'order_manager': self.adapter.get_order_manager(),
            'execution_engine': self.adapter.get_execution_engine(),
            'portfolio_manager': self.adapter.get_portfolio_manager()
        }

        for component_name, component in components.items():
            if component and hasattr(component, 'get_metrics'):
                try:
                    metrics['trading_metrics'][component_name] = component.get_metrics()
                except Exception as e:
                    metrics['trading_metrics'][component_name] = {'error': str(e)}

        return metrics

