
from ..utils.memory_leak_detector import MemoryLeakDetector
from ..utils.optimization_report_generator import OptimizationReportGenerator
from .resource_optimization_engine import ResourceOptimizationEngine
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .system_resource_analyzer import SystemResourceAnalyzer
from ..utils.thread_analyzer import ThreadAnalyzer
from datetime import datetime
from typing import Dict, List, Optional, Any
"""
资源优化器 (重构后版本)

Phase 3: 质量提升 - 文件拆分优化

整合所有资源优化组件的简化协调器。
"""


class ResourceOptimizer:
    """资源优化器 (重构后版本)"""

    def __init__(self, logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        # 初始化基础组件
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()
        
        # 初始化优化历史记录
        self.optimization_history = []
        
        # 初始化兼容性属性
        self.monitoring_data = []
        self.optimization_suggestions = []

        # 初始化专用组件
        self.system_analyzer = SystemResourceAnalyzer(self.logger, self.error_handler)
        self.thread_analyzer = ThreadAnalyzer(self.logger, self.error_handler)
        self.memory_detector = MemoryLeakDetector(self.logger, self.error_handler)
        self.report_generator = OptimizationReportGenerator(
            self.system_analyzer, self.thread_analyzer, self.memory_detector,
            self.logger, self.error_handler
        )
        self.optimization_engine = ResourceOptimizationEngine(
            self.system_analyzer, self.logger, self.error_handler
        )

        self.logger.log_info("资源优化器已初始化 (重构后版本)")

    def get_system_resources(self, analysis_depth: str = "basic") -> Dict[str, Any]:
        """获取系统资源状态"""
        resources = self.system_analyzer.get_system_resources(analysis_depth)

        # 为向后兼容 Tests / 旧接口提供扁平化字段
        cpu_snapshot = resources.get("cpu", {})
        memory_snapshot = resources.get("memory", {})

        if "cpu_percent" not in resources:
            usage = cpu_snapshot.get("usage_percent")
            if usage is not None:
                resources["cpu_percent"] = usage
        if "memory_percent" not in resources:
            mem_usage = memory_snapshot.get("usage_percent") or memory_snapshot.get("percent")
            if mem_usage is not None:
                resources["memory_percent"] = mem_usage
        if "disk_percent" not in resources:
            disk_snapshot = resources.get("io", {}).get("disk", {})
            disk_percent = disk_snapshot.get("usage_percent") or disk_snapshot.get("percent")
            if disk_percent is not None:
                resources["disk_percent"] = disk_percent

        return resources

    def analyze_threads(self, include_stacks: bool = False) -> Dict[str, Any]:
        """分析线程状态"""
        return self.thread_analyzer.analyze_threads(include_stacks)

    def detect_memory_leaks(self) -> List[str]:
        """检测内存泄漏"""
        return self.memory_detector.detect_memory_leaks()

    def generate_optimization_report(self, report_type: str = "summary") -> Dict[str, Any]:
        """生成优化报告"""
        return self.report_generator.generate_optimization_report(report_type)

    def optimize_resources(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行资源优化"""
        if config is None:
            # 如果没有提供配置，使用默认配置
            config = {
                'cpu_threshold': 80.0,
                'memory_threshold': 80.0,
                'optimization_level': 'basic'
            }
        
        # 执行优化
        result = self.optimization_engine.optimize_resources(config)
        
        # 记录优化历史
        history_entry = {
            'timestamp': datetime.now(),
            'config': config,
            'result': result
        }
        self.optimization_history.append(history_entry)
        
        # 限制历史记录长度
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-50:]
        
        return result

    def get_recommendations(self) -> List[str]:
        """获取优化建议"""
        return self.optimization_engine.get_optimization_recommendations()

    def monitor_performance(self, operation_name: str):
        """性能监控装饰器"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = datetime.now()

                try:
                    self.logger.log_info(f"开始执行操作: {operation_name}")
                    result = func(*args, **kwargs)
                    end_time = datetime.now()

                    duration = (end_time - start_time).total_seconds()
                    self.logger.log_info(f"操作完成: {operation_name}, 耗时: {duration:.2f}秒")

                    return result

                except Exception as e:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    self.error_handler.handle_error(e, {
                        "context": f"操作执行失败: {operation_name}",
                        "duration_seconds": duration
                    })
                    raise

            return wrapper
        return decorator
