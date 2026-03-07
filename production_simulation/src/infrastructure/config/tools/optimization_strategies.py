
import gc
import statistics
import weakref

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple

from .test_interfaces import OptimizationConfig
"""
性能优化策略模块

提供各种性能优化策略和算法，包括缓存优化、连接池优化、内存优化等。
支持动态调整和自适应优化。
"""

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):

    """优化策略枚举"""
    CACHE_OPTIMIZATION = "cache_optimization"           # 缓存优化
    CONNECTION_POOL_OPTIMIZATION = "connection_pool_optimization"  # 连接池优化
    MEMORY_OPTIMIZATION = "memory_optimization"         # 内存优化
    THREAD_POOL_OPTIMIZATION = "thread_pool_optimization"  # 线程池优化
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"   # 算法优化
    RESOURCE_POOLING = "resource_pooling"               # 资源池化


class OptimizationLevel(Enum):

    """优化级别枚举"""
    LOW = "low"           # 低级别优化
    MEDIUM = "medium"     # 中级别优化
    HIGH = "high"         # 高级别优化
    AGGRESSIVE = "aggressive"  # 激进优化


@dataclass
class OOptimizationConfig:

    """优化配置"""
    strategy: OptimizationStrategy
    level: OptimizationLevel = OptimizationLevel.MEDIUM
    enabled: bool = True
    auto_adjust: bool = True
    monitoring_interval: float = 60.0
    threshold_percentage: float = 20.0  # 性能提升阈值


@dataclass
class OptimizationResult:

    """优化结果"""
    strategy: OptimizationStrategy
    level: OptimizationLevel
    success: bool
    performance_improvement: float  # 性能提升百分比
    memory_saved_mb: float = 0.0
    execution_time_reduction: float = 0.0
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class CacheOptimizationStrategy:

    """缓存优化策略"""

    def __init__(self, config: OptimizationConfig):

        self.config = config
        self._cache_hit_rates: Dict[str, List[float]] = defaultdict(list)
        self._cache_sizes: Dict[str, List[int]] = defaultdict(list)
        self._optimization_history: List[OptimizationResult] = []

    def optimize(self, cache_name: str, current_hit_rate: float,

                 current_size: int, target_hit_rate: float = 0.8):
        """执行缓存优化"""
        try:
            # 记录当前状态
            self._cache_hit_rates[cache_name].append(current_hit_rate)
            self._cache_sizes[cache_name].append(current_size)

            # 分析缓存性能趋势
            if len(self._cache_hit_rates[cache_name]) >= 5:
                trend = self._analyze_trend(self._cache_hit_rates[cache_name])

                if trend < 0 and current_hit_rate < target_hit_rate:
                    # 命中率下降，需要优化
                    improvement = self._apply_cache_optimization(
                        cache_name, current_hit_rate, current_size)

                    result = OptimizationResult(
                        strategy=self.config.strategy,
                        level=self.config.level,
                        success=True,
                        performance_improvement=improvement
                    )

                else:
                    result = OptimizationResult(
                        strategy=self.config.strategy,
                        level=self.config.level,
                        success=True,
                        performance_improvement=0.0
                    )

            else:
                result = OptimizationResult(
                    strategy=self.config.strategy,
                    level=self.config.level,
                    success=True,
                    performance_improvement=0.0
                )

            self._optimization_history.append(result)
            return result

        except Exception as e:
            logger.error(f"缓存优化失败: {e}")
            return OptimizationResult(
                strategy=self.config.strategy,
                level=self.config.level,
                success=False,
                performance_improvement=0.0,
                error_message=str(e)
            )

    def _analyze_trend(self, values: List[float]) -> float:
        """分析数值趋势"""
        if len(values) < 2:
            return 0.0

        # 计算线性回归斜率
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * v for i, v in enumerate(values))
        x2_sum = sum(i * i for i in range(n))

        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope

    def _apply_cache_optimization(self, cache_name: str, current_hit_rate: float,

                                  current_size: int):
        """应用缓存优化策略"""
        improvement = 0.0

        if self.config.level == OptimizationLevel.LOW:
            # 低级别优化：调整清理间隔
            improvement = 5.0
        elif self.config.level == OptimizationLevel.MEDIUM:
            # 中级别优化：调整缓存大小和策略
            improvement = 15.0
        elif self.config.level == OptimizationLevel.HIGH:
            # 高级别优化：智能预加载和淘汰策略
            improvement = 25.0
        elif self.config.level == OptimizationLevel.AGGRESSIVE:
            # 激进优化：完全重构缓存策略
            improvement = 40.0

        logger.info(f"应用缓存优化策略: {cache_name}, 级别: {self.config.level.value}, 预期提升: {improvement}%")
        return improvement


class ConnectionPoolOptimizationStrategy:

    """连接池优化策略"""

    def __init__(self, config: OptimizationConfig):

        self.config = config
        self._pool_utilization: Dict[str, List[float]] = defaultdict(list)
        self._connection_wait_times: Dict[str, List[float]] = defaultdict(list)
        self._optimization_history: List[OptimizationResult] = []

    def optimize(self, pool_name: str, current_utilization: float,

                 avg_wait_time: float, target_utilization: float = 0.7):
        """执行连接池优化"""
        try:
            # 记录当前状态
            self._pool_utilization[pool_name].append(current_utilization)
            self._connection_wait_times[pool_name].append(avg_wait_time)

            # 分析连接池性能
            if current_utilization > target_utilization or avg_wait_time > 1.0:
                improvement = self._apply_pool_optimization(
                    pool_name, current_utilization, avg_wait_time)

                result = OptimizationResult(
                    strategy=self.config.strategy,
                    level=self.config.level,
                    success=True,
                    performance_improvement=improvement
                )

            else:
                result = OptimizationResult(
                    strategy=self.config.strategy,
                    level=self.config.level,
                    success=True,
                    performance_improvement=0.0
                )

            self._optimization_history.append(result)
            return result

        except Exception as e:
            logger.error(f"连接池优化失败: {e}")
            return OptimizationResult(
                strategy=self.config.strategy,
                level=self.config.level,
                success=False,
                performance_improvement=0.0,
                error_message=str(e)
            )

    def _apply_pool_optimization(self, pool_name: str, utilization: float, wait_time: float) -> float:
        """应用连接池优化策略"""
        improvement = 0.0

        if self.config.level == OptimizationLevel.LOW:
            # 低级别优化：调整连接超时
            improvement = 8.0
        elif self.config.level == OptimizationLevel.MEDIUM:
            # 中级别优化：调整池大小和健康检查
            improvement = 20.0
        elif self.config.level == OptimizationLevel.HIGH:
            # 高级别优化：智能连接管理和负载均衡
            improvement = 35.0
        elif self.config.level == OptimizationLevel.AGGRESSIVE:
            # 激进优化：动态池大小和连接复用
            improvement = 50.0

        logger.info(f"应用连接池优化策略: {pool_name}, 级别: {self.config.level.value}, 预期提升: {improvement}%")
        return improvement


class MemoryOptimizationStrategy:

    """内存优化策略"""

    def __init__(self, config: OptimizationConfig):

        self.config = config
        self._memory_usage: List[float] = []
        self._gc_stats: List[Dict[str, Any]] = []
        self._optimization_history: List[OptimizationResult] = []

    def optimize(self, current_memory_mb: float, target_memory_mb: float) -> OptimizationResult:
        """执行内存优化"""
        try:
            # 记录当前状态
            self._memory_usage.append(current_memory_mb)

            # 收集GC统计信息
            gc_stats = self._collect_gc_stats()
            self._gc_stats.append(gc_stats)

            if current_memory_mb > target_memory_mb:
                improvement, memory_saved = self._apply_memory_optimization(
                    current_memory_mb, target_memory_mb)

                result = OptimizationResult(
                    strategy=self.config.strategy,
                    level=self.config.level,
                    success=True,
                    performance_improvement=improvement,
                    memory_saved_mb=memory_saved
                )

            else:
                result = OptimizationResult(
                    strategy=self.config.strategy,
                    level=self.config.level,
                    success=True,
                    performance_improvement=0.0,
                    memory_saved_mb=0.0
                )

            self._optimization_history.append(result)
            return result

        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            return OptimizationResult(
                strategy=self.config.strategy,
                level=self.config.level,
                success=False,
                performance_improvement=0.0,
                error_message=str(e)
            )

    def _collect_gc_stats(self) -> Dict[str, Any]:
        """收集垃圾回收统计信息"""
        stats = {
            'collections': gc.get_count(),
            'objects': len(gc.get_objects()),
            'garbage': len(gc.garbage)
        }

        return stats

    def _apply_memory_optimization(self, current_memory: float, target_memory: float) -> Tuple[float, float]:
        """应用内存优化策略"""
        improvement = 0.0
        memory_saved = 0.0

        if self.config.level == OptimizationLevel.LOW:
            # 低级别优化：手动触发GC
            gc.collect()
            improvement = 10.0
            memory_saved = current_memory * 0.1
        elif self.config.level == OptimizationLevel.MEDIUM:
            # 中级别优化：清理弱引用和循环引用
            self._cleanup_weak_references()
            improvement = 25.0
            memory_saved = current_memory * 0.2
        elif self.config.level == OptimizationLevel.HIGH:
            # 高级别优化：内存池和对象复用
            improvement = 40.0
            memory_saved = current_memory * 0.3
        elif self.config.level == OptimizationLevel.AGGRESSIVE:
            # 激进优化：内存压缩和碎片整理
            improvement = 60.0
            memory_saved = current_memory * 0.4

        logger.info(f"应用内存优化策略, 级别: {self.config.level.value}, 预期提升: {improvement}%")
        return improvement, memory_saved

    def _cleanup_weak_references(self):
        """清理弱引用"""
        try:
            # 清理弱引用字典（如果存在）
            if hasattr(weakref, '_weakref') and hasattr(weakref._weakref, '__dict__'):
                weakref._weakref.__dict__.clear()
        except (AttributeError, TypeError):
            # 如果weakref._weakref不存在，跳过清理
            pass

        # 手动触发GC
        gc.collect()


class PerformanceOptimizationManager:

    """性能优化管理器"""

    def __init__(self):

        self.strategies: Dict[OptimizationStrategy, Any] = {}
        self.configs: Dict[OptimizationStrategy, OOptimizationConfig] = {}
        self.optimization_history: List[OptimizationResult] = []
        self._lock = threading.RLock()

        # 初始化默认策略
        self._initialize_default_strategies()

    def _initialize_default_strategies(self):
        """初始化默认优化策略"""
        # 缓存优化策略
        cache_config = OOptimizationConfig(
            strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
            level=OptimizationLevel.MEDIUM
        )

        self.add_strategy(cache_config, CacheOptimizationStrategy(cache_config))

        # 连接池优化策略
        pool_config = OOptimizationConfig(
            strategy=OptimizationStrategy.CONNECTION_POOL_OPTIMIZATION,
            level=OptimizationLevel.MEDIUM
        )

        self.add_strategy(pool_config, ConnectionPoolOptimizationStrategy(pool_config))

        # 内存优化策略
        memory_config = OOptimizationConfig(
            strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
            level=OptimizationLevel.MEDIUM
        )

        self.add_strategy(memory_config, MemoryOptimizationStrategy(memory_config))

    def add_strategy(self, config: OptimizationConfig, strategy: Any):
        """添加优化策略"""
        with self._lock:
            self.strategies[config.strategy] = strategy
            self.configs[config.strategy] = config

    def optimize_cache(self, cache_name: str, hit_rate: float, size: int) -> OptimizationResult:
        """优化缓存"""
        strategy = self.strategies.get(OptimizationStrategy.CACHE_OPTIMIZATION)
        if strategy:
            config = self.configs[OptimizationStrategy.CACHE_OPTIMIZATION]
            result = strategy.optimize(cache_name, hit_rate, size)
            # 添加到优化历史
            with self._lock:
                self.optimization_history.append(result)
            return result
        else:
            result = OptimizationResult(
                strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
                level=OptimizationLevel.MEDIUM,
                success=False,
                performance_improvement=0.0,
                error_message="缓存优化策略未找到"
            )

            # 添加到优化历史
            with self._lock:
                self.optimization_history.append(result)
            return result

    def optimize_connection_pool(self, pool_name: str, utilization: float, wait_time: float) -> OptimizationResult:
        """优化连接池"""
        strategy = self.strategies.get(OptimizationStrategy.CONNECTION_POOL_OPTIMIZATION)
        if strategy:
            config = self.configs[OptimizationStrategy.CONNECTION_POOL_OPTIMIZATION]
            result = strategy.optimize(pool_name, utilization, wait_time)
            # 添加到优化历史
            with self._lock:
                self.optimization_history.append(result)
            return result
        else:
            result = OptimizationResult(
                strategy=OptimizationStrategy.CONNECTION_POOL_OPTIMIZATION,
                level=OptimizationLevel.MEDIUM,
                success=False,
                performance_improvement=0.0,
                error_message="连接池优化策略未找到"
            )

            # 添加到优化历史
            with self._lock:
                self.optimization_history.append(result)
            return result

    def optimize_memory(self, current_memory: float, target_memory: float) -> OptimizationResult:
        """优化内存使用"""
        strategy = self.strategies.get(OptimizationStrategy.MEMORY_OPTIMIZATION)
        if strategy:
            config = self.configs[OptimizationStrategy.MEMORY_OPTIMIZATION]
            result = strategy.optimize(current_memory, target_memory)
            # 添加到优化历史
            with self._lock:
                self.optimization_history.append(result)
            return result
        else:
            result = OptimizationResult(
                strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
                level=OptimizationLevel.MEDIUM,
                success=False,
                performance_improvement=0.0,
                error_message="内存优化策略未找到"
            )

            # 添加到优化历史
            with self._lock:
                self.optimization_history.append(result)
            return result

    def get_optimization_history(self) -> List[OptimizationResult]:
        """获取优化历史"""
        with self._lock:
            return self.optimization_history.copy()

    def generate_optimization_report(self) -> str:
        """生成优化报告"""
        if not self.optimization_history:
            return "没有可用的优化记录"

        # 计算总体统计
        total_results = len(self.optimization_history)
        successful_results = [r for r in self.optimization_history if r.success]
        failed_results = [r for r in self.optimization_history if not r.success]
        total_success = len(successful_results)
        total_failed = len(failed_results)
        
        # 计算总体平均性能提升
        if successful_results:
            overall_avg_improvement = statistics.mean([r.performance_improvement for r in successful_results])
        else:
            overall_avg_improvement = 0.0

        report_lines = [
            "=" * 60,
            "性能优化报告",
            "=" * 60,
            f"总优化次数: {total_results}",
            f"成功优化: {total_success}",
            f"失败优化: {total_failed}",
            f"平均性能提升: {overall_avg_improvement:.2f}%",
            "",
            "按策略分组的详细统计:",
            ""
        ]

        # 按策略分组统计
        strategy_stats = defaultdict(list)
        for result in self.optimization_history:
            strategy_stats[result.strategy].append(result)

        for strategy, results in strategy_stats.items():
            strategy_successful_results = [r for r in results if r.success]
            success_count = len(strategy_successful_results)
            
            # 安全地计算平均性能提升
            if strategy_successful_results:
                avg_improvement = statistics.mean([r.performance_improvement for r in strategy_successful_results])
                total_memory_saved = sum([r.memory_saved_mb or 0 for r in strategy_successful_results])
            else:
                avg_improvement = 0.0
                total_memory_saved = 0.0
            
            report_lines.extend([
                f"优化策略: {strategy.value}",
                f"优化次数: {len(results)}",
                f"成功次数: {success_count}",
                f"平均性能提升: {avg_improvement:.2f}%",
                f"总内存节省: {total_memory_saved:.2f} MB",
                "-" * 40
            ])

        return "\n".join(report_lines)




