"""
内存优化器

负责处理所有内存相关的优化操作。
"""

import gc
from typing import Dict, Any, Optional
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .optimization_config import MemoryOptimizationConfig


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, logger: Optional[ILogger] = None, error_handler: Optional[IErrorHandler] = None):
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()
    
    def optimize_memory_from_config(self, config: MemoryOptimizationConfig,
                                   current_resources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """基于参数对象进行内存优化"""
        return self.optimize_memory(config.to_dict(), current_resources)
    
    def optimize_memory(self, config: Dict[str, Any], current_resources: Dict[str, Any]) -> Dict[str, Any]:
        """内存优化"""
        try:
            result = {
                "type": "memory_optimization",
                "status": "applied",
                "actions": []
            }

            memory_usage = current_resources.get("memory_usage", 0)

            # 如果内存使用率过高，强制垃圾回收
            if memory_usage > config.get("gc_threshold", 80):
                collected = self._perform_garbage_collection()
                result["actions"].append(f"执行垃圾回收，清理 {collected} 个对象")

            # 内存池化建议
            if config.get("enable_pooling", False):
                result["actions"].append("启用对象池化以减少内存分配")

            # 大对象监控
            if config.get("monitor_large_objects", False):
                result["actions"].append("启用大对象监控")

            return result

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "内存优化失败"})
            return {
                "type": "memory_optimization",
                "status": "failed",
                "error": str(e)
            }
    
    def _perform_garbage_collection(self) -> int:
        """执行垃圾回收"""
        try:
            return gc.collect()
        except Exception as e:
            self.logger.log_warning(f"垃圾回收失败: {e}")
            return 0
