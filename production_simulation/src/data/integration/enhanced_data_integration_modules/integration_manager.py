"""
集成管理器主类模块

包含 EnhancedDataIntegration 主类和所有数据加载方法。
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# 导入基础模块
from .config import IntegrationConfig
from .components import (
    TaskPriority,
    LoadTask,
    create_enhanced_loader,
    DynamicThreadPoolManager,
    ConnectionPoolManager,
    MemoryOptimizer,
    FinancialDataOptimizer,
)
from .cache_utils import (
    check_cache_for_symbols,
    check_cache_for_indices,
    check_cache_for_financial,
    cache_data,
    cache_index_data,
    cache_financial_data,
)
from .performance_utils import (
    check_data_quality,
    update_avg_response_time as update_avg_response_time_util,
    get_integration_stats,
)

# 先初始化 logger，避免导入错误时 logger 未定义
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    import logging
    def get_infrastructure_logger(name):
        _logger = logging.getLogger(name)
        _logger.warning("无法导入基础设施层日志，使用标准logging")
        return _logger

logger = get_infrastructure_logger("enhanced_data_integration")

# 导入外部依赖（logger 已定义，可以使用）
try:
    from ...cache.enhanced_cache_strategy import create_enhanced_cache_strategy
except ImportError:
    create_enhanced_cache_strategy = None
    logger.warning("无法导入 create_enhanced_cache_strategy")

try:
    from ...loader.financial_loader import FinancialDataLoader
    from ...loader.index_loader import IndexDataLoader
    from ...loader.stock_loader import StockDataLoader
except ImportError:
    FinancialDataLoader = None
    IndexDataLoader = None
    StockDataLoader = None
    logger.warning("无法导入数据加载器")

try:
    from ...quality.enhanced_quality_monitor import create_enhanced_quality_monitor
except ImportError:
    create_enhanced_quality_monitor = None
    logger.warning("无法导入 create_enhanced_quality_monitor")


class EnhancedDataIntegration:
    """
    增强版数据层集成管理器

    功能：
    - 集成增强版并行加载器
    - 集成增强版缓存策略
    - 集成增强版质量监控
    - 提供统一的API接口
    - 支持配置驱动的优化
    - 新增：性能优化和自适应缓存策略
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        """
        初始化增强版数据层集成管理器

        Args:
            config: 集成配置
        """
        self.config = config or IntegrationConfig()

        # 初始化组件
        self._init_components()

        # 性能监控
        self._performance_metrics = {
            "avg_response_time": 0.0,
            "total_requests": 0,
            "cache_hit_rate": 0.0,
            "quality_score": 0.0,
            "memory_usage": 0.0,
            "thread_utilization": 0.0,
        }

        # 自适应缓存策略
        self._adaptive_cache_config = {
            "hit_rate_threshold": 0.8,
            "memory_threshold": 0.85,
            "response_time_threshold": 1000,  # ms
            "quality_threshold": 0.95,
        }

        # 缓存预热状态
        self._cache_warming_status = {
            "is_warming": False,
            "warmed_items": 0,
            "total_items": 0,
            "warming_progress": 0.0,
        }

        # 启动性能监控
        self._start_performance_monitoring()

    def _init_components(self):
        """初始化各个组件"""
        # 并行加载管理器
        self.parallel_manager = create_enhanced_loader(
            config={
                "max_workers": self.config.parallel_loading["max_workers"],
                "enable_auto_scaling": self.config.parallel_loading["enable_auto_scaling"],
                "batch_size": self.config.parallel_loading["batch_size"],
                "max_queue_size": self.config.parallel_loading["max_queue_size"],
            }
        )

        # 缓存策略
        if create_enhanced_cache_strategy is None:
            raise ImportError("create_enhanced_cache_strategy 未导入，请检查依赖")
        self.cache_strategy = create_enhanced_cache_strategy(
            config={
                "approach": self.config.cache_strategy["approach"],
                "max_size": self.config.cache_strategy["max_size"],
                "max_items": self.config.cache_strategy["max_items"],
                "enable_preload": self.config.cache_strategy["enable_preload"],
                "enable_adaptive_ttl": self.config.cache_strategy["enable_adaptive_ttl"],
            }
        )

        # 质量监控器
        if create_enhanced_quality_monitor is None:
            raise ImportError("create_enhanced_quality_monitor 未导入，请检查依赖")
        self.quality_monitor = create_enhanced_quality_monitor(
            enable_alerting=self.config.quality_monitor["enable_alerting"],
            enable_trend_analysis=self.config.quality_monitor["enable_trend_analysis"],
        )

        # 数据管理器
        from ...data_manager import DataManagerSingleton
        self.data_manager = DataManagerSingleton.get_instance()

        # 数据加载器
        try:
            self.stock_loader = StockDataLoader(save_path="data / cache / stock")
        except Exception as e:
            logger.warning(f"StockDataLoader初始化失败: {e}")
            self.stock_loader = None

        try:
            self.index_loader = IndexDataLoader(save_path="data / cache / index")
        except Exception as e:
            logger.warning(f"IndexDataLoader初始化失败: {e}")
            self.index_loader = None

        try:
            self.financial_loader = FinancialDataLoader(save_path="data / cache / financial")
        except Exception as e:
            logger.warning(f"FinancialDataLoader初始化失败: {e}")
            self.financial_loader = None

        # 性能优化组件
        self._init_performance_optimization()

        # 初始化企业级特性
        self._init_enterprise_features()

        # 启动性能监控
        self._start_performance_monitoring()

    def _init_performance_optimization(self):
        """初始化性能优化组件"""
        # 动态线程池管理器
        self._thread_pool_manager = DynamicThreadPoolManager(
            initial_size=self.config.parallel_loading["max_workers"],
            max_size=self.config.parallel_loading["max_workers"] * 2,
            min_size=4,
        )

        # 连接池管理器
        if self.config.performance_optimization["enable_connection_pooling"]:
            self._connection_pool = ConnectionPoolManager(
                max_size=self.config.performance_optimization["max_connection_pool_size"],
                timeout=self.config.performance_optimization["connection_timeout"],
            )

        # 内存优化器
        self._memory_optimizer = MemoryOptimizer(
            enable_compression=self.config.performance_optimization["enable_data_compression"],
            compression_level=self.config.performance_optimization["compression_level"],
        )

        # 财务数据优化器
        if self.config.performance_optimization["enable_financial_optimization"]:
            self._financial_optimizer = FinancialDataOptimizer()

    def _init_enterprise_features(self):
        """初始化企业级特性"""
        logger.info("初始化企业级特性")

        # 分布式支持
        self.distributed_manager = self._init_distributed_manager()

        # 实时数据流
        self.realtime_stream = self._init_realtime_stream()

        # 监控可视化
        self.monitoring_dashboard = self._init_monitoring_dashboard()

        # 集群管理
        self.cluster_manager = self._init_cluster_manager()

        # 负载均衡
        self.load_balancer = self._init_load_balancer()

        logger.info("企业级特性初始化完成")

    def _init_distributed_manager(self):
        """初始化分布式管理器"""
        return {
            "enabled": True,
            "nodes": [],
            "node_id": f"node_{hash(self) % 10000}",
            "coordinator": None,
            "sync_interval": 30,
        }

    def _init_realtime_stream(self):
        """初始化实时数据流"""
        return {
            "enabled": True,
            "stream_processors": [],
            "quality_monitors": [],
            "alert_system": {
                "enabled": True,
                "thresholds": {
                    "response_time": 1000,  # ms
                    "error_rate": 0.05,     # 5%
                    "quality_score": 0.8    # 80%
                }
            }
        }

    def _init_monitoring_dashboard(self):
        """初始化监控面板"""
        return {
            "enabled": True,
            "metrics": {
                "performance": [],
                "quality": [],
                "errors": [],
                "throughput": []
            },
            "visualization": {
                "charts": [],
                "alerts": [],
                "reports": []
            }
        }

    def _init_cluster_manager(self):
        """初始化集群管理器"""
        return {
            "enabled": True,
            "cluster_nodes": [],
            "leader_election": False,
            "health_check_interval": 60
        }

    def _init_load_balancer(self):
        """初始化负载均衡器"""
        return {
            "enabled": True,
            "approach": "round_robin",
            "health_checks": True,
            "failover": True
        }

    def _start_performance_monitoring(self):
        """启动性能监控"""
        def monitor_performance():
            while True:
                try:
                    # 更新性能指标
                    self._update_performance_metrics()

                    # 自适应调整
                    self._adaptive_adjustment()

                    # 缓存预热检查
                    self._check_cache_warming()

                    time.sleep(30)  # 每30秒检查一次
                except Exception as e:
                    logger.error(f"性能监控错误: {e}")
                    time.sleep(60)

        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
        monitor_thread.start()

    def _update_performance_metrics(self):
        """更新性能指标"""
        # 计算平均响应时间
        if self._performance_metrics["total_requests"] > 0:
            self._performance_metrics["avg_response_time"] = (
                self._performance_metrics["avg_response_time"] * 0.9
                + self._get_current_response_time() * 0.1
            )

        # 更新缓存命中率
        self._performance_metrics["cache_hit_rate"] = self.cache_strategy.get_hit_rate()

        # 更新质量分数
        self._performance_metrics["quality_score"] = self.quality_monitor.get_overall_quality_score()

        # 更新内存使用率
        self._performance_metrics["memory_usage"] = self._get_memory_usage()

        # 更新线程利用率
        self._performance_metrics["thread_utilization"] = self._thread_pool_manager.get_utilization()

    def _adaptive_adjustment(self):
        """自适应调整策略"""
        # 根据性能指标调整缓存策略
        if self._performance_metrics["cache_hit_rate"] < self._adaptive_cache_config["hit_rate_threshold"]:
            self._optimize_cache_strategy()

        # 根据内存使用率调整
        if self._performance_metrics["memory_usage"] > self._adaptive_cache_config["memory_threshold"]:
            self._optimize_memory_usage()

        # 根据响应时间调整线程池
        if self._performance_metrics["avg_response_time"] > self._adaptive_cache_config["response_time_threshold"]:
            self._optimize_thread_pool()

    def _optimize_cache_strategy(self):
        """优化缓存策略"""
        logger.info("优化缓存策略")

        # 增加缓存大小
        current_max_size = self.cache_strategy.get_max_size()
        new_max_size = int(current_max_size * 1.2)
        self.cache_strategy.set_max_size(new_max_size)

        # 调整TTL策略
        self.cache_strategy.optimize_ttl_strategy()

        # 启动缓存预热
        self._start_cache_warming()

    def _optimize_memory_usage(self):
        """优化内存使用"""
        logger.info("优化内存使用")

        # 清理过期缓存
        self.cache_strategy.cleanup_expired()

        # 压缩缓存数据
        self._memory_optimizer.compress_cache_data(self.cache_strategy)

        # 调整缓存大小
        current_max_size = self.cache_strategy.get_max_size()
        new_max_size = int(current_max_size * 0.8)
        self.cache_strategy.set_max_size(new_max_size)

    def _optimize_thread_pool(self):
        """优化线程池"""
        logger.info("优化线程池")

        # 增加工作线程数
        current_workers = self._thread_pool_manager.get_current_size()
        new_workers = min(current_workers + 2, self._thread_pool_manager.get_max_size())
        self._thread_pool_manager.resize(new_workers)

        # 优化任务分配策略
        if hasattr(self.parallel_manager, 'optimize_task_distribution'):
            self.parallel_manager.optimize_task_distribution()

    def _start_cache_warming(self):
        """启动缓存预热"""
        if self._cache_warming_status["is_warming"]:
            return

        self._cache_warming_status["is_warming"] = True
        self._cache_warming_status["warmed_items"] = 0

        def warm_cache():
            try:
                # 预热常用股票数据
                common_symbols = ["600519.SH", "000001.SZ", "000002.SZ", "600036.SH"]
                for symbol in common_symbols:
                    self._preload_stock_data(symbol)
                    self._cache_warming_status["warmed_items"] += 1

                # 预热指数数据
                common_indices = ["000300.SH", "000905.SH", "399006.SZ"]
                for index in common_indices:
                    self._preload_index_data(index)
                    self._cache_warming_status["warmed_items"] += 1

                self._cache_warming_status["is_warming"] = False
                logger.info("缓存预热完成")

            except Exception as e:
                logger.error(f"缓存预热失败: {e}")
                self._cache_warming_status["is_warming"] = False

        # 启动预热线程
        warming_thread = threading.Thread(target=warm_cache, daemon=True)
        warming_thread.start()

    def _preload_stock_data(self, symbol: str):
        """预加载股票数据"""
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

            if self.stock_loader:
                data = self.stock_loader.load_data(symbol, start_date=start_date, end_date=end_date)

                if data is not None and not data.empty:
                    self.cache_strategy.set(
                        f"stock_{symbol}_{start_date}_{end_date}_1d",
                        data,
                        ttl=3600,  # 1小时TTL
                    )

        except Exception as e:
            logger.warning(f"预加载股票数据失败 {symbol}: {e}")

    def _preload_index_data(self, index: str):
        """预加载指数数据"""
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

            if self.index_loader:
                data = self.index_loader.load_data(index, start_date=start_date, end_date=end_date)

                if data is not None and not data.empty:
                    self.cache_strategy.set(
                        f"index_{index}_{start_date}_{end_date}_1d",
                        data,
                        ttl=3600,  # 1小时TTL
                    )

        except Exception as e:
            logger.warning(f"预加载指数数据失败 {index}: {e}")

    def _get_current_response_time(self) -> float:
        """获取当前响应时间"""
        # 这里可以实现更复杂的响应时间计算
        return 100.0  # 默认值

    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.5  # 默认值

    def _check_cache_warming(self):
        """检查缓存预热状态"""
        if self._cache_warming_status["total_items"] > 0:
            self._cache_warming_status["warming_progress"] = (
                self._cache_warming_status["warmed_items"]
                / self._cache_warming_status["total_items"]
            )

    # ========================================================================
    # 缓存方法（使用工具模块）
    # ========================================================================

    def _check_cache_for_symbols(self, symbols: List[str], start_date: str, end_date: str, frequency: str) -> Dict[str, pd.DataFrame]:
        """检查股票数据缓存（封装工具函数）"""
        return check_cache_for_symbols(self.cache_strategy, symbols, start_date, end_date, frequency)

    def _check_cache_for_indices(self, indices: List[str], start_date: str, end_date: str, frequency: str) -> Dict[str, pd.DataFrame]:
        """检查指数数据缓存（封装工具函数）"""
        return check_cache_for_indices(self.cache_strategy, indices, start_date, end_date, frequency)

    def _check_cache_for_financial(self, symbols: List[str], start_date: str, end_date: str, data_type: str) -> Dict[str, pd.DataFrame]:
        """检查财务数据缓存（封装工具函数）"""
        return check_cache_for_financial(symbols, start_date, end_date, data_type, self.cache_strategy)

    def _cache_data(self, symbol: str, data: pd.DataFrame, start_date: str, end_date: str, frequency: str):
        """缓存股票数据（封装工具函数）"""
        cache_data(self.cache_strategy, symbol, data, start_date, end_date, frequency)

    def _cache_index_data(self, index: str, data: pd.DataFrame, start_date: str, end_date: str, frequency: str):
        """缓存指数数据（封装工具函数）"""
        cache_index_data(self.cache_strategy, index, data, start_date, end_date, frequency)

    def _cache_financial_data(self, symbol: str, data: pd.DataFrame, start_date: str, end_date: str, data_type: str):
        """缓存财务数据（封装工具函数）"""
        cache_financial_data(self.cache_strategy, symbol, data, start_date, end_date, data_type)

    # ========================================================================
    # 质量和性能方法（使用工具模块）
    # ========================================================================

    def _check_data_quality(self, data: pd.DataFrame, identifier: str):
        """检查数据质量（封装工具函数）"""
        return check_data_quality(data, identifier, self.quality_monitor)

    def _update_avg_response_time(self, response_time: float):
        """更新平均响应时间（封装工具函数）"""
        update_avg_response_time_util(self._performance_metrics, response_time)

    def get_integration_stats(self) -> Dict[str, Any]:
        """获取集成统计信息（封装工具函数）"""
        return get_integration_stats(self._performance_metrics, self.cache_strategy, self.parallel_manager)

    # ========================================================================
    # 数据加载方法
    # ========================================================================

    def load_stock_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        frequency: str = "1d",
        priority: TaskPriority = TaskPriority.NORMAL,
        enable_cache: bool = True,
        enable_quality_check: bool = True,
    ) -> Dict[str, Any]:
        """
        加载股票数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            priority: 任务优先级
            enable_cache: 是否启用缓存
            enable_quality_check: 是否启用质量检查

        Returns:
            包含数据和统计信息的字典
        """
        start_time = time.time()

        try:
            # 检查缓存
            cached_data = {}
            if enable_cache:
                cached_data = self._check_cache_for_symbols(symbols, start_date, end_date, frequency)

            # 确定需要加载的股票
            symbols_to_load = [s for s in symbols if s not in cached_data]

            # 并行加载数据
            loaded_data = {}
            if symbols_to_load:
                loaded_data = self._load_data_parallel(symbols_to_load, start_date, end_date, frequency, priority)

            # 合并缓存和加载的数据
            all_data = {**cached_data, **loaded_data}

            # 质量检查
            quality_metrics = {}
            if enable_quality_check:
                for symbol, data in all_data.items():
                    if data is not None and not data.empty:
                        metrics = self._check_data_quality(data, symbol)
                        if metrics is not None:
                            # 将QualityMetrics对象转换为字典格式
                            quality_metrics[symbol] = {
                                "completeness": metrics.completeness,
                                "accuracy": metrics.accuracy,
                                "consistency": metrics.consistency,
                                "timeliness": metrics.timeliness,
                                "validity": metrics.validity,
                                "uniqueness": metrics.uniqueness,
                                "overall_quality": metrics.overall_score,
                                "timestamp": metrics.timestamp,
                                "data_type": metrics.data_type,
                                "details": metrics.details,
                            }
                        else:
                            # 如果质量检查失败，提供默认值
                            quality_metrics[symbol] = {
                                "completeness": 0.0,
                                "accuracy": 0.0,
                                "consistency": 0.0,
                                "timeliness": 0.0,
                                "validity": 0.0,
                                "uniqueness": 0.0,
                                "overall_quality": 0.0,
                                "timestamp": datetime.now().isoformat(),
                                "data_type": "stock",
                                "details": {},
                            }

            # 缓存新加载的数据
            if enable_cache:
                for symbol, data in loaded_data.items():
                    if data is not None and not data.empty:
                        self._cache_data(symbol, data, start_date, end_date, frequency)

            # 更新统计信息
            response_time = (time.time() - start_time) * 1000
            self._update_avg_response_time(response_time)

            # 计算缓存命中率
            cache_hit_rate = len(cached_data) / len(symbols) if symbols else 0.0

            # 检查是否有数据加载成功
            if not all_data:
                error_msg = "API Error" if symbols else "No symbols provided"
                return {
                    "success": False,
                    "data": {},
                    "quality_metrics": {},
                    "stats": {
                        "response_time": response_time,
                        "cache_hit_rate": 0.0,
                        "loaded_count": 0,
                        "cached_count": 0,
                        "memory_usage": self._get_memory_usage(),
                    },
                    "error": error_msg,
                }

            return {
                "success": True,
                "data": all_data,
                "stats": {
                    "response_time": response_time,
                    "cache_hit_rate": cache_hit_rate,
                    "loaded_count": len(loaded_data),
                    "cached_count": len(cached_data),
                    "memory_usage": self._get_memory_usage(),
                },
                "quality_metrics": quality_metrics,
            }

        except Exception as e:
            logger.error(f"加载股票数据失败: {e}")
            return {
                "success": False,
                "data": {},
                "quality_metrics": {},
                "stats": {
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "cache_hit_rate": 0.0,
                    "loaded_count": 0,
                    "cached_count": 0,
                },
                "performance": {
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "cache_hit_rate": 0.0,
                    "loaded_count": 0,
                    "cached_count": 0,
                },
            }

    def load_index_data(
        self,
        indices: List[str],
        start_date: str,
        end_date: str,
        frequency: str = "1d",
        enable_cache: bool = True,
        enable_quality_check: bool = True,
    ) -> Dict[str, Any]:
        """
        加载指数数据

        Args:
            indices: 指数代码列表
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
            enable_cache: 是否启用缓存
            enable_quality_check: 是否启用质量检查

        Returns:
            包含数据和统计信息的字典
        """
        start_time = time.time()

        try:
            # 检查缓存
            cached_data = {}
            if enable_cache:
                cached_data = self._check_cache_for_indices(indices, start_date, end_date, frequency)

            # 确定需要加载的指数
            indices_to_load = [i for i in indices if i not in cached_data]

            # 并行加载数据
            loaded_data = {}
            if indices_to_load:
                loaded_data = self._load_index_data_parallel(indices_to_load, start_date, end_date, frequency)

            # 合并缓存和加载的数据
            all_data = {**cached_data, **loaded_data}

            # 质量检查
            quality_metrics = {}
            if enable_quality_check:
                for index, data in all_data.items():
                    if data is not None and not data.empty:
                        metrics = self._check_data_quality(data, index)
                        quality_metrics[index] = metrics

            # 缓存新加载的数据
            if enable_cache:
                for index, data in loaded_data.items():
                    if data is not None and not data.empty:
                        self._cache_index_data(index, data, start_date, end_date, frequency)

            # 更新统计信息
            response_time = (time.time() - start_time) * 1000
            self._update_avg_response_time(response_time)

            # 计算缓存命中率
            cache_hit_rate = len(cached_data) / len(indices) if indices else 0.0

            return {
                "success": True,
                "data": all_data,
                "quality_metrics": quality_metrics,
                "stats": {
                    "response_time": response_time,
                    "cache_hit_rate": cache_hit_rate,
                    "loaded_count": len(loaded_data),
                    "cached_count": len(cached_data),
                },
                "performance": {
                    "response_time_ms": response_time,
                    "cache_hit_rate": cache_hit_rate,
                    "loaded_count": len(loaded_data),
                    "cached_count": len(cached_data),
                },
            }

        except Exception as e:
            logger.error(f"加载指数数据失败: {e}")
            return {
                "success": False,
                "data": {},
                "quality_metrics": {},
                "stats": {
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "cache_hit_rate": 0.0,
                    "loaded_count": 0,
                    "cached_count": 0,
                },
            }

    def load_financial_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_type: str = "financial",
        enable_cache: bool = True,
        enable_quality_check: bool = True,
    ) -> Dict[str, Any]:
        """
        加载财务数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型
            enable_cache: 是否启用缓存
            enable_quality_check: 是否启用质量检查

        Returns:
            包含数据和统计信息的字典
        """
        start_time = time.time()

        try:
            # 检查缓存
            cached_data = {}
            if enable_cache:
                cached_data = self._check_cache_for_financial(symbols, start_date, end_date, data_type)

            # 确定需要加载的股票
            symbols_to_load = [s for s in symbols if s not in cached_data]

            # 并行加载数据
            loaded_data = {}
            if symbols_to_load:
                loaded_data = self._load_financial_data_parallel(symbols_to_load, start_date, end_date, data_type)

            # 合并缓存和加载的数据
            all_data = {**cached_data, **loaded_data}

            # 质量检查
            quality_metrics = {}
            if enable_quality_check:
                for symbol, data in all_data.items():
                    if data is not None and not data.empty:
                        metrics = self._check_data_quality(data, symbol)
                        if metrics is not None:
                            quality_metrics[symbol] = {
                                "completeness": metrics.completeness,
                                "accuracy": metrics.accuracy,
                                "consistency": metrics.consistency,
                                "timeliness": metrics.timeliness,
                                "validity": metrics.validity,
                                "uniqueness": metrics.uniqueness,
                                "overall_quality": metrics.overall_score,
                                "timestamp": metrics.timestamp,
                                "data_type": metrics.data_type,
                                "details": metrics.details,
                            }
                        else:
                            quality_metrics[symbol] = {
                                "completeness": 0.0,
                                "accuracy": 0.0,
                                "consistency": 0.0,
                                "timeliness": 0.0,
                                "validity": 0.0,
                                "uniqueness": 0.0,
                                "overall_quality": 0.0,
                                "timestamp": datetime.now().isoformat(),
                                "data_type": "financial",
                                "details": {},
                            }

            # 缓存新加载的数据
            if enable_cache:
                for symbol, data in loaded_data.items():
                    if data is not None and not data.empty:
                        self._cache_financial_data(symbol, data, start_date, end_date, data_type)

            # 更新统计信息
            response_time = (time.time() - start_time) * 1000
            self._update_avg_response_time(response_time)

            # 计算缓存命中率
            cache_hit_rate = len(cached_data) / len(symbols) if symbols else 0.0

            return {
                "success": True,
                "data": all_data,
                "stats": {
                    "response_time": response_time,
                    "cache_hit_rate": cache_hit_rate,
                    "loaded_count": len(loaded_data),
                    "cached_count": len(cached_data),
                    "memory_usage": self._get_memory_usage(),
                },
                "quality_metrics": quality_metrics,
            }

        except Exception as e:
            logger.error(f"加载财务数据失败: {e}")
            return {
                "success": False,
                "data": {},
                "quality_metrics": {},
                "stats": {
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "cache_hit_rate": 0.0,
                    "loaded_count": 0,
                    "cached_count": 0,
                },
            }

    def _load_data_parallel(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        frequency: str,
        priority: TaskPriority,
    ) -> Dict[str, pd.DataFrame]:
        """并行加载股票数据"""
        results = {}

        if not self.stock_loader:
            logger.error("StockDataLoader未初始化")
            return results

        # 创建任务
        tasks = []
        for symbol in symbols:
            task = LoadTask(
                task_id=f"stock_{symbol}_{start_date}_{end_date}",
                loader=self.stock_loader,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                priority=priority,
                kwargs={"symbol": symbol},
            )
            tasks.append(task)

        # 提交任务
        task_ids = []
        for task in tasks:
            task_id = self.parallel_manager.submit_task(task)
            task_ids.append(task_id)

        # 执行任务并收集结果
        try:
            processing = self.parallel_manager.execute_tasks(timeout=30)

            # 处理结果
            for i, task_id in enumerate(task_ids):
                if task_id in processing:
                    result = processing[task_id]
                    if result is not None and not result.empty:
                        results[symbols[i]] = result
                else:
                    logger.error(f"任务 {task_id} 未返回结果")

        except Exception as e:
            logger.error(f"并行加载数据失败: {e}")

        return results

    def _load_index_data_parallel(
        self, indices: List[str], start_date: str, end_date: str, frequency: str
    ) -> Dict[str, pd.DataFrame]:
        """并行加载指数数据"""
        results = {}

        if not self.index_loader:
            logger.error("IndexDataLoader未初始化")
            return results

        # 创建任务
        tasks = []
        for index in indices:
            task = LoadTask(
                task_id=f"index_{index}_{start_date}_{end_date}",
                loader=self.index_loader,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                priority=TaskPriority.NORMAL,
                kwargs={"index": index},
            )
            tasks.append(task)

        # 提交任务
        task_ids = []
        for task in tasks:
            task_id = self.parallel_manager.submit_task(task)
            task_ids.append(task_id)

        # 执行任务并收集结果
        try:
            processing = self.parallel_manager.execute_tasks(timeout=30)

            # 处理结果
            for i, task_id in enumerate(task_ids):
                if task_id in processing:
                    result = processing[task_id]
                    if result is not None and not result.empty:
                        results[indices[i]] = result
                else:
                    logger.error(f"任务 {task_id} 未返回结果")

        except Exception as e:
            logger.error(f"并行加载指数数据失败: {e}")

        return results

    def _load_financial_data_parallel(
        self, symbols: List[str], start_date: str, end_date: str, data_type: str
    ) -> Dict[str, pd.DataFrame]:
        """并行加载财务数据"""
        results = {}

        if not self.financial_loader:
            logger.error("FinancialDataLoader未初始化")
            return results

        # 创建任务
        tasks = []
        for symbol in symbols:
            task = LoadTask(
                task_id=f"financial_{symbol}_{start_date}_{end_date}",
                loader=self.financial_loader,
                start_date=start_date,
                end_date=end_date,
                frequency="1d",
                priority=TaskPriority.NORMAL,
                kwargs={"symbol": symbol},
            )
            tasks.append(task)

        # 提交任务
        task_ids = []
        for task in tasks:
            task_id = self.parallel_manager.submit_task(task)
            task_ids.append(task_id)

        # 执行任务并收集结果
        try:
            processing = self.parallel_manager.execute_tasks(timeout=30)

            # 处理结果
            for i, task_id in enumerate(task_ids):
                if task_id in processing:
                    result = processing[task_id]
                    if result is not None and not result.empty:
                        results[symbols[i]] = result
                else:
                    logger.error(f"任务 {task_id} 未返回结果")

        except Exception as e:
            logger.error(f"并行加载财务数据失败: {e}")

        return results

    # ========================================================================
    # 企业级特性方法
    # ========================================================================

    def get_enterprise_features_status(self) -> Dict[str, Any]:
        """获取企业级特性状态"""
        return {
            "distributed_manager": {
                "enabled": self.distributed_manager["enabled"],
                "node_id": self.distributed_manager["node_id"],
                "nodes_count": len(self.distributed_manager["nodes"])
            },
            "realtime_stream": {
                "enabled": self.realtime_stream["enabled"],
                "processors_count": len(self.realtime_stream["stream_processors"]),
                "monitors_count": len(self.realtime_stream["quality_monitors"])
            },
            "monitoring_dashboard": {
                "enabled": self.monitoring_dashboard["enabled"],
                "metrics_count": sum(len(metrics) for metrics in self.monitoring_dashboard["metrics"].values())
            },
            "cluster_manager": {
                "enabled": self.cluster_manager["enabled"],
                "nodes_count": len(self.cluster_manager["cluster_nodes"]),
                "leader_election": self.cluster_manager["leader_election"]
            },
            "load_balancer": {
                "enabled": self.load_balancer["enabled"],
                "approach": self.load_balancer["approach"],
                "health_checks": self.load_balancer["health_checks"]
            }
        }

    def add_distributed_node(self, node_info: Dict[str, Any]) -> bool:
        """添加分布式节点"""
        try:
            node_id = node_info.get("node_id", f"node_{len(self.distributed_manager['nodes'])}")
            node_info["node_id"] = node_id
            node_info["status"] = "active"
            node_info["last_heartbeat"] = time.time()

            self.distributed_manager["nodes"].append(node_info)
            logger.info(f"添加分布式节点: {node_id}")
            return True
        except Exception as e:
            logger.error(f"添加分布式节点失败: {e}")
            return False

    def start_realtime_stream_processing(self, stream_config: Dict[str, Any]) -> bool:
        """启动实时数据流处理"""
        try:
            processor = {
                "id": f"processor_{len(self.realtime_stream['stream_processors'])}",
                "config": stream_config,
                "status": "active",
                "start_time": time.time()
            }

            self.realtime_stream["stream_processors"].append(processor)
            logger.info(f"启动实时数据流处理器: {processor['id']}")
            return True
        except Exception as e:
            logger.error(f"启动实时数据流处理失败: {e}")
            return False

    def add_monitoring_metric(self, metric_type: str, metric_data: Dict[str, Any]) -> bool:
        """添加监控指标"""
        try:
            if metric_type in self.monitoring_dashboard["metrics"]:
                metric_data["timestamp"] = time.time()
                self.monitoring_dashboard["metrics"][metric_type].append(metric_data)
                logger.debug(f"添加监控指标: {metric_type}")
                return True
            else:
                logger.warning(f"未知的监控指标类型: {metric_type}")
                return False
        except Exception as e:
            logger.error(f"添加监控指标失败: {e}")
            return False

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """获取监控面板数据"""
        return {
            "metrics": self.monitoring_dashboard["metrics"],
            "visualization": self.monitoring_dashboard["visualization"],
            "alerts": self._get_active_alerts(),
            "performance_summary": self._get_performance_summary()
        }

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        alerts = []
        current_time = time.time()

        # 检查响应时间告警
        if hasattr(self, "_avg_response_time") and self._avg_response_time > self.realtime_stream["alert_system"]["thresholds"]["response_time"]:
            alerts.append({
                "type": "performance",
                "message": f"响应时间过高: {self._avg_response_time:.2f}ms",
                "severity": "warning",
                "timestamp": current_time
            })

        # 检查错误率告警
        if hasattr(self, "_error_count") and hasattr(self, "_total_requests"):
            error_rate = self._error_count / max(self._total_requests, 1)
            if error_rate > self.realtime_stream["alert_system"]["thresholds"]["error_rate"]:
                alerts.append({
                    "type": "error",
                    "message": f"错误率过高: {error_rate:.2%}",
                    "severity": "critical",
                    "timestamp": current_time
                })

        return alerts

    def _get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "avg_response_time": getattr(self, "_avg_response_time", 0),
            "total_requests": getattr(self, "_total_requests", 0),
            "error_count": getattr(self, "_error_count", 0),
            "cache_hit_rate": getattr(self, "_cache_hit_rate", 0),
            "memory_usage": self._get_memory_usage(),
            "uptime": time.time() - getattr(self, "_start_time", time.time())
        }

    def shutdown(self):
        """关闭集成管理器"""
        from .performance_utils import shutdown as shutdown_util
        shutdown_util(self.parallel_manager, self.cache_strategy, self.quality_monitor)

