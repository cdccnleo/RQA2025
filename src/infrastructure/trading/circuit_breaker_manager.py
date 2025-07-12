import json
from typing import Dict, Optional
from datetime import datetime
from .circuit_breaker import InstrumentedCircuitBreaker, CircuitBreakerManager
from prometheus_client import start_http_server

class EnhancedCircuitBreakerManager(CircuitBreakerManager):
    """增强型熔断器管理器（生产级实现）"""

    def __init__(self, prometheus_port: Optional[int] = 9090):
        super().__init__()
        self.prometheus_port = prometheus_port
        self._init_metrics_server()

    def _init_metrics_server(self):
        """启动Prometheus指标服务"""
        if self.prometheus_port:
            try:
                start_http_server(self.prometheus_port)
            except Exception as e:
                print(f"Failed to start metrics server: {e}")

    def load_config(self, config_path: str) -> Dict:
        """从配置文件加载熔断器配置"""
        try:
            with open(config_path) as f:
                configs = json.load(f)

            for name, params in configs.items():
                self.get_breaker(name, **params)

            return {"status": "success", "loaded": len(configs)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def export_config(self) -> Dict:
        """导出当前所有熔断器配置"""
        return {
            name: {
                "failure_threshold": breaker.failure_threshold,
                "recovery_timeout": breaker.recovery_timeout,
                "trading_hours": breaker.trading_hours
            }
            for name, breaker in self.breakers.items()
        }

    def manual_override(self,
                       breaker_name: str,
                       action: str,
                       operator: str = "admin") -> Dict:
        """手动操作熔断器状态"""
        if breaker_name not in self.breakers:
            return {"status": "error", "message": "Breaker not found"}

        breaker = self.breakers[breaker_name]
        if action == "reset":
            breaker._operator = operator
            breaker.manual_reset()
            return {"status": "success"}
        elif action == "trip":
            with breaker._lock:
                breaker._operator = operator
                breaker._trip()
            return {"status": "success"}
        else:
            return {"status": "error", "message": "Invalid action"}

class CircuitBreakerAdminAPI:
    """熔断器管理API（支持HTTP/CLI）"""

    def __init__(self, manager: EnhancedCircuitBreakerManager):
        self.manager = manager

    def handle_request(self,
                      action: str,
                      params: Dict) -> Dict:
        """处理API请求"""
        try:
            if action == "status":
                return {
                    "breakers": self.manager.get_all_status(),
                    "timestamp": datetime.now().isoformat()
                }
            elif action == "config":
                return {"configs": self.manager.export_config()}
            elif action == "override":
                return self.manager.manual_override(
                    params.get("name"),
                    params.get("action"),
                    params.get("operator", "api")
                )
            else:
                return {"status": "error", "message": "Unknown action"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# 示例配置
SAMPLE_CONFIG = {
    "order_api": {
        "failure_threshold": 5,
        "recovery_timeout": 30,
        "trading_hours": {
            "morning": {"start": "09:30", "end": "11:30"},
            "afternoon": {"start": "13:00", "end": "15:00"}
        }
    },
    "market_data": {
        "failure_threshold": 10,
        "recovery_timeout": 60
    }
}
