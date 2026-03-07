"""
优化配置管理器

负责处理并行化和检查点配置的优化操作。
"""

from typing import Dict, Any, Optional
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .optimization_config import ParallelizationConfig, CheckpointingConfig


class OptimizationConfigManager:
    """优化配置管理器"""
    
    def __init__(self, logger: Optional[ILogger] = None, error_handler: Optional[IErrorHandler] = None):
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()
    
    def configure_parallelization_from_config(self, config: ParallelizationConfig) -> Optional[Dict[str, Any]]:
        """基于参数对象配置并行化"""
        return self.configure_parallelization(config.to_dict())
    
    def configure_checkpointing_from_config(self, config: CheckpointingConfig) -> Optional[Dict[str, Any]]:
        """基于参数对象配置检查点"""
        return self.configure_checkpointing(config.to_dict())
    
    def configure_parallelization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """配置并行化"""
        try:
            result = {
                "type": "parallelization_config",
                "status": "applied",
                "actions": []
            }

            # 线程池大小
            pool_size = config.get("thread_pool_size", 4)
            result["actions"].append(f"设置线程池大小为 {pool_size}")

            # 进程池大小
            if config.get("process_pool_size"):
                proc_size = config.get("process_pool_size")
                result["actions"].append(f"设置进程池大小为 {proc_size}")

            # 异步配置
            if config.get("async_enabled", False):
                result["actions"].append("启用异步处理")

            return result

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "并行化配置失败"})
            return {
                "type": "parallelization_config",
                "status": "failed",
                "error": str(e)
            }

    def configure_checkpointing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """配置检查点"""
        try:
            result = {
                "type": "checkpointing_config",
                "status": "applied",
                "actions": []
            }

            # 检查点间隔
            interval = config.get("interval_seconds", 300)
            result["actions"].append(f"设置检查点间隔为 {interval} 秒")

            # 检查点存储路径
            if config.get("storage_path"):
                path = config.get("storage_path")
                result["actions"].append(f"设置检查点存储路径: {path}")

            # 压缩启用
            if config.get("compression", {}).get("enabled", False):
                result["actions"].append("启用检查点压缩")

            return result

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "检查点配置失败"})
            return {
                "type": "checkpointing_config",
                "status": "failed",
                "error": str(e)
            }
