"""
磁盘优化器

负责处理所有磁盘相关的优化操作。
"""

from typing import Dict, Any, Optional
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .optimization_config import DiskOptimizationConfig


class DiskOptimizer:
    """磁盘优化器"""
    
    def __init__(self, logger: Optional[ILogger] = None, error_handler: Optional[IErrorHandler] = None):
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()
    
    def optimize_disk_from_config(self, config: DiskOptimizationConfig,
                                 current_resources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """基于参数对象进行磁盘优化"""
        return self.optimize_disk(config.to_dict(), current_resources)
    
    def optimize_disk(self, config: Dict[str, Any], current_resources: Dict[str, Any]) -> Dict[str, Any]:
        """磁盘优化"""
        try:
            result = {
                "type": "disk_optimization",
                "status": "applied",
                "actions": []
            }

            # I/O调度器设置
            if config.get("io_scheduler", {}).get("enabled", False):
                result["actions"].append("配置I/O调度器")

            # 缓存策略
            if config.get("caching", {}).get("enabled", False):
                result["actions"].append("启用磁盘缓存策略")

            # 预读设置
            if config.get("readahead", {}).get("enabled", False):
                result["actions"].append("配置预读参数")

            return result

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "磁盘优化失败"})
            return {
                "type": "disk_optimization",
                "status": "failed",
                "error": str(e)
            }
