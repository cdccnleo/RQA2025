"""
CPU优化器

负责处理所有CPU相关的优化操作。
"""

from typing import Dict, Any, Optional
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .optimization_config import CpuOptimizationConfig


class CpuOptimizer:
    """CPU优化器"""
    
    def __init__(self, logger: Optional[ILogger] = None, error_handler: Optional[IErrorHandler] = None):
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()
    
    def optimize_cpu_from_config(self, config: CpuOptimizationConfig,
                                current_resources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """基于参数对象进行CPU优化"""
        return self.optimize_cpu(config.to_dict(), current_resources)
    
    def optimize_cpu(self, config: Dict[str, Any], current_resources: Dict[str, Any]) -> Dict[str, Any]:
        """CPU优化"""
        try:
            result = {
                "type": "cpu_optimization",
                "status": "applied",
                "actions": []
            }

            cpu_usage = current_resources.get("cpu_usage", 0)

            # CPU亲和性设置
            if config.get("cpu_affinity", {}).get("enabled", False):
                result["actions"].append("配置CPU亲和性")

            # 进程优先级调整
            if cpu_usage > config.get("priority_threshold", 90):
                result["actions"].append("调整进程优先级")

            # 节能模式
            if config.get("power_saving", False):
                result["actions"].append("启用CPU节能模式")

            return result

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "CPU优化失败"})
            return {
                "type": "cpu_optimization",
                "status": "failed",
                "error": str(e)
            }
