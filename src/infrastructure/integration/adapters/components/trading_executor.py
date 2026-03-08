"""
交易执行组件

负责执行交易逻辑，包括缓存检查、交易执行、结果记录等。
"""

import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class TradingExecutor:
    """交易执行器"""

    def __init__(self, adapter):
        """初始化交易执行器
        
        Args:
            adapter: TradingLayerAdapter实例，用于获取服务
        """
        self.adapter = adapter

    def execute_trade(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """执行交易"""
        result = self._initialize_trade_result(trade_request)

        try:
            # 获取基础设施服务
            services = self._get_infrastructure_services()

            # 记录交易开始
            self._record_trade_start(services, result)

            # 检查缓存避免重复交易
            if self._check_trade_cache(trade_request, services, result):
                return result

            # 执行实际交易
            self._execute_actual_trade(trade_request, services, result)

            # 记录交易完成
            self._record_trade_completion(services)

        except Exception as e:
            self._handle_trade_error(e, result)

        return result

    def _initialize_trade_result(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """初始化交易结果对象"""
        return {
            'timestamp': datetime.now().isoformat(),
            'layer_type': 'trading',
            'trade_request': trade_request,
            'executed': False,
            'infrastructure_used': []
        }

    def _get_infrastructure_services(self) -> Dict[str, Any]:
        """获取所需的基础设施服务"""
        services = self.adapter.get_infrastructure_services()
        return {
            'cache_manager': services.get('cache_manager'),
            'monitoring': services.get('monitoring')
        }

    def _record_trade_start(self, services: Dict[str, Any], result: Dict[str, Any]) -> None:
        """记录交易开始"""
        if services.get('monitoring'):
            services['monitoring'].record_metric('trade_execution_start', 1, {'layer': 'trading'})
            result['infrastructure_used'].append('monitoring')

    def _check_trade_cache(self, trade_request: Dict[str, Any],
                          services: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """检查交易缓存，避免重复执行"""
        cache_manager = services.get('cache_manager')
        if not cache_manager:
            return False

        cache_key = f"trade_{hash(str(trade_request))}"
        cached_result = cache_manager.get(cache_key)

        if cached_result:
            result['cached_result'] = cached_result
            result['executed'] = True
            result['infrastructure_used'].append('cache')
            return True

        return False

    def _execute_actual_trade(self, trade_request: Dict[str, Any],
                             services: Dict[str, Any], result: Dict[str, Any]) -> None:
        """执行实际的交易逻辑"""
        trading_engine = self.adapter.get_trading_engine()
        if trading_engine:
            execution_result = trading_engine.execute_trade(trade_request)
            result['execution_result'] = execution_result
            result['executed'] = True

            # 缓存执行结果
            self._cache_execution_result(trade_request, execution_result, services, result)

    def _cache_execution_result(self, trade_request: Dict[str, Any],
                               execution_result: Any, services: Dict[str, Any],
                               result: Dict[str, Any]) -> None:
        """缓存交易执行结果"""
        cache_manager = services.get('cache_manager')
        if cache_manager:
            cache_key = f"trade_{hash(str(trade_request))}"
            cache_manager.set(cache_key, execution_result, 300)  # 缓存5分钟
            result['infrastructure_used'].append('cache')

    def _record_trade_completion(self, services: Dict[str, Any]) -> None:
        """记录交易完成"""
        monitoring = services.get('monitoring')
        if monitoring:
            monitoring.record_metric('trade_execution_complete', 1, {'layer': 'trading'})

    def _handle_trade_error(self, error: Exception, result: Dict[str, Any]) -> None:
        """处理交易执行错误"""
        result['error'] = str(error)
        logger.error(f"交易执行失败: {error}")

        # 记录错误指标
        services = self._get_infrastructure_services()
        monitoring = services.get('monitoring')
        if monitoring:
            monitoring.record_metric('trade_execution_error', 1, {'layer': 'trading', 'error': str(error)})

