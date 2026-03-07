
import gc

from .optimization_config import (
    ResourceOptimizationConfig,
    MemoryOptimizationConfig,
    CpuOptimizationConfig,
    DiskOptimizationConfig,
    ParallelizationConfig,
    CheckpointingConfig
)
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .system_resource_analyzer import SystemResourceAnalyzer
from .optimization_memory_optimizer import MemoryOptimizer
from .optimization_cpu_optimizer import CpuOptimizer
from .optimization_disk_optimizer import DiskOptimizer
from .optimization_config_manager import OptimizationConfigManager
from datetime import datetime
from typing import Dict, List, Optional, Any

"""
资源优化引擎

Phase 3: 质量提升 - 文件拆分优化

负责执行资源优化策略和配置调整。
"""


class ResourceOptimizationEngine:
    """资源优化引擎"""

    def __init__(self, system_analyzer: Optional[SystemResourceAnalyzer] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.system_analyzer = system_analyzer or SystemResourceAnalyzer()
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

        # 优化配置
        self._optimization_configs = {}
        
        # 初始化优化器组件
        self.memory_optimizer = MemoryOptimizer(logger, error_handler)
        self.cpu_optimizer = CpuOptimizer(logger, error_handler)
        self.disk_optimizer = DiskOptimizer(logger, error_handler)
        self.config_manager = OptimizationConfigManager(logger, error_handler)

    def optimize_resources(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行资源优化（支持字典和参数对象）"""
        # 转换为参数对象格式（如果传入的是字典）
        if isinstance(optimization_config, dict):
            config_obj = ResourceOptimizationConfig.from_dict(optimization_config)
        elif isinstance(optimization_config, ResourceOptimizationConfig):
            config_obj = optimization_config
        else:
            raise ValueError("optimization_config必须是字典或ResourceOptimizationConfig对象")

        return self.optimize_resources_with_config(config_obj)

    def optimize_resources_with_config(self, config: ResourceOptimizationConfig) -> Dict[str, Any]:
        """执行资源优化（使用参数对象）"""
        try:
            # 验证配置
            validation_issues = config.validate()
            if validation_issues:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "validation_errors": validation_issues
                }

            # 初始化优化结果
            optimization_result = self._initialize_optimization_result_from_config(config)

            # 收集当前资源状态
            current_resources = self._collect_current_resources()

            # 应用各项优化
            optimizations_applied = self._apply_resource_optimizations_from_config(
                config, current_resources)

            # 完成优化结果
            return self._finalize_optimization_result(optimization_result, optimizations_applied)

        except Exception as e:
            return self._handle_optimization_error(e)

    def _initialize_optimization_result_from_config(self, config: ResourceOptimizationConfig) -> Dict[str, Any]:
        """从参数对象初始化优化结果"""
        return {
            "timestamp": datetime.now().isoformat(),
            "config": config.to_dict(),
            "optimizations": [],
            "status": "success"
        }

    def _initialize_optimization_result(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """初始化优化结果（向后兼容）"""
        return {
            "timestamp": datetime.now().isoformat(),
            "config": optimization_config,
            "optimizations": [],
            "status": "success"
        }

    def _collect_current_resources(self) -> Dict[str, Any]:
        """收集当前资源状态"""
        try:
            return self.system_analyzer.get_resource_summary()
        except Exception as e:
            self.logger.log_warning(f"获取资源状态失败: {e}")
            return {}

    def _apply_resource_optimizations_from_config(self, config: ResourceOptimizationConfig,
                                                  current_resources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从参数对象应用各项资源优化"""
        optimizations_applied = []

        # 使用策略映射来简化复杂的条件逻辑
        optimization_strategies = self._get_optimization_strategies(config)

        # 按照优先级顺序应用优化
        for opt_type in config.optimization_priority:
            strategy = optimization_strategies.get(opt_type)
            if strategy:
                optimization_result = strategy(current_resources)
                if optimization_result:
                    optimizations_applied.append(optimization_result)

        return optimizations_applied

    def _get_optimization_strategies(self, config: ResourceOptimizationConfig) -> Dict[str, callable]:
        """获取优化策略映射"""
        return {
            "memory": lambda resources: self._apply_memory_optimization_strategy(config, resources),
            "cpu": lambda resources: self._apply_cpu_optimization_strategy(config, resources),
            "disk": lambda resources: self._apply_disk_optimization_strategy(config, resources),
            "parallelization": lambda resources: self._apply_parallelization_strategy(config),
            "checkpointing": lambda resources: self._apply_checkpointing_strategy(config),
        }

    def _apply_memory_optimization_strategy(self, config: ResourceOptimizationConfig, 
                                          current_resources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用内存优化策略"""
        if config.memory_optimization.enabled:
            return self.memory_optimizer.optimize_memory_from_config(
                config.memory_optimization, current_resources)
        return None

    def _apply_cpu_optimization_strategy(self, config: ResourceOptimizationConfig, 
                                       current_resources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用CPU优化策略"""
        if config.cpu_optimization.enabled:
            return self.cpu_optimizer.optimize_cpu_from_config(
                config.cpu_optimization, current_resources)
        return None

    def _apply_disk_optimization_strategy(self, config: ResourceOptimizationConfig, 
                                        current_resources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用磁盘优化策略"""
        if config.disk_optimization.enabled:
            return self.disk_optimizer.optimize_disk_from_config(
                config.disk_optimization, current_resources)
        return None

    def _apply_parallelization_strategy(self, config: ResourceOptimizationConfig) -> Optional[Dict[str, Any]]:
        """应用并行化策略"""
        if config.parallelization.enabled:
            return self.config_manager.configure_parallelization_from_config(config.parallelization)
        return None

    def _apply_checkpointing_strategy(self, config: ResourceOptimizationConfig) -> Optional[Dict[str, Any]]:
        """应用检查点策略"""
        if config.checkpointing.enabled:
            return self.config_manager.configure_checkpointing_from_config(config.checkpointing)
        return None

    def _apply_resource_optimizations(self, optimization_config: Dict[str, Any],
                                      current_resources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用各项资源优化"""
        optimizations_applied = []

        # 内存优化
        memory_opt = self._apply_memory_optimization(optimization_config, current_resources)
        if memory_opt:
            optimizations_applied.append(memory_opt)

        # CPU优化
        cpu_opt = self._apply_cpu_optimization(optimization_config, current_resources)
        if cpu_opt:
            optimizations_applied.append(cpu_opt)

        # 磁盘优化
        disk_opt = self._apply_disk_optimization(optimization_config, current_resources)
        if disk_opt:
            optimizations_applied.append(disk_opt)

        # 并行化配置
        parallel_opt = self._apply_parallelization_config(optimization_config)
        if parallel_opt:
            optimizations_applied.append(parallel_opt)

        # 检查点配置
        checkpoint_opt = self._apply_checkpointing_config(optimization_config)
        if checkpoint_opt:
            optimizations_applied.append(checkpoint_opt)

        return optimizations_applied

    def _apply_memory_optimization(self, optimization_config: Dict[str, Any],
                                   current_resources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用内存优化"""
        memory_config = optimization_config.get("memory_optimization", {})
        if not memory_config.get("enabled", False):
            return None

        return self.memory_optimizer.optimize_memory(memory_config, current_resources)

    def _apply_cpu_optimization(self, optimization_config: Dict[str, Any],
                                current_resources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用CPU优化"""
        cpu_config = optimization_config.get("cpu_optimization", {})
        if not cpu_config.get("enabled", False):
            return None

        return self.cpu_optimizer.optimize_cpu(cpu_config, current_resources)

    def _apply_disk_optimization(self, optimization_config: Dict[str, Any],
                                 current_resources: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用磁盘优化"""
        disk_config = optimization_config.get("disk_optimization", {})
        if not disk_config.get("enabled", False):
            return None

        return self.disk_optimizer.optimize_disk(disk_config, current_resources)

    def _apply_parallelization_config(self, optimization_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用并行化配置"""
        parallel_config = optimization_config.get("parallelization", {})
        if not parallel_config.get("enabled", False):
            return None

        return self.config_manager.configure_parallelization(parallel_config)

    def _apply_checkpointing_config(self, optimization_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用检查点配置"""
        checkpoint_config = optimization_config.get("checkpointing", {})
        if not checkpoint_config.get("enabled", False):
            return None

        return self.config_manager.configure_checkpointing(checkpoint_config)

    def _finalize_optimization_result(self, optimization_result: Dict[str, Any],
                                      optimizations_applied: List[Dict[str, Any]]) -> Dict[str, Any]:
        """完成优化结果"""
        optimization_result["optimizations"] = optimizations_applied
        self.logger.log_info(f"资源优化完成，应用了 {len(optimizations_applied)} 个优化")
        return optimization_result

    def _handle_optimization_error(self, error: Exception) -> Dict[str, Any]:
        """处理优化错误"""
        self.error_handler.handle_error(error, {"context": "资源优化失败"})
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(error)
        }

    # 这些方法现在由专门的优化器类处理，保持向后兼容性

    def get_optimization_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []

        try:
            current_resources = self.system_analyzer.get_resource_summary()

            # 基于当前资源状态生成建议
            cpu_usage = current_resources.get("cpu_usage", 0)
            memory_usage = current_resources.get("memory_usage", 0)
            thread_count = current_resources.get("thread_count", 0)

            if cpu_usage > 80:
                recommendations.append("CPU使用率较高，建议启用CPU优化")
            if memory_usage > 80:
                recommendations.append("内存使用率较高，建议启用内存优化")
            if thread_count > 50:
                recommendations.append("线程数量较多，建议配置并行化优化")

        except Exception:
            recommendations.append("建议定期检查资源使用情况并应用适当优化")

        return recommendations
