
from .memory_leak_detector import MemoryLeakDetector
from .optimization_report_generator import OptimizationReportGenerator
from .resource_optimization_engine import ResourceOptimizationEngine
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .system_resource_analyzer import SystemResourceAnalyzer
from .thread_analyzer import ThreadAnalyzer
from datetime import datetime
from typing import Dict, List, Optional, Any
"""
资源优化器 (重构后版本)

Phase 3: 质量提升 - 文件拆分优化

整合所有资源优化组件的简化协调器。
"""


class NewResourceOptimizer:
    """资源优化器 (重构后版本)"""

    def __init__(self, logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        # 初始化基础组件
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

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
        return self.system_analyzer.get_system_resources(analysis_depth)

    def analyze_threads(self, include_stacks: bool = False) -> Dict[str, Any]:
        """分析线程状态"""
        return self.thread_analyzer.analyze_threads(include_stacks)

    def detect_memory_leaks(self) -> List[str]:
        """检测内存泄漏"""
        return self.memory_detector.detect_memory_leaks()

    def generate_optimization_report(self, report_type: str = "summary") -> Dict[str, Any]:
        """生成优化报告"""
        return self.report_generator.generate_optimization_report(report_type)

    def optimize_resources(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """执行资源优化"""
        return self.optimization_engine.optimize_resources(config)

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
