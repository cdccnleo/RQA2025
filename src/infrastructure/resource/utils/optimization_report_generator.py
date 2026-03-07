"""
optimization_report_generator 模块

提供 optimization_report_generator 相关功能和接口。
"""

import json

import yaml

from .memory_leak_detector import MemoryLeakDetector
from ..core.shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from ..core.system_resource_analyzer import SystemResourceAnalyzer
from .thread_analyzer import ThreadAnalyzer
from datetime import datetime
from typing import Dict, List, Optional, Any
"""
优化报告生成器

Phase 3: 质量提升 - 文件拆分优化

负责生成资源优化的详细报告。
"""


class OptimizationReportGenerator:
    """优化报告生成器"""

    def __init__(self, system_analyzer: Optional[SystemResourceAnalyzer] = None,
                 thread_analyzer: Optional[ThreadAnalyzer] = None,
                 memory_detector: Optional[MemoryLeakDetector] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.system_analyzer = system_analyzer or SystemResourceAnalyzer()
        self.thread_analyzer = thread_analyzer or ThreadAnalyzer()
        self.memory_detector = memory_detector or MemoryLeakDetector()

        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

    def generate_optimization_report(self, report_type: str = "summary") -> Dict[str, Any]:
        """生成优化报告"""
        try:
            if report_type == "detailed":
                return self._generate_detailed_report()
            else:
                return self._generate_summary_report()

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "生成优化报告失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _generate_summary_report(self) -> Dict[str, Any]:
        """生成汇总报告"""
        try:
            # 收集所有数据
            system_resources = self.system_analyzer.get_resource_summary()
            thread_info = self.thread_analyzer.get_thread_summary()
            memory_issues = self.memory_detector.detect_memory_leaks()

            # 生成报告
            report = {
                "timestamp": datetime.now().isoformat(),
                "report_type": "summary",
                "system_resources": system_resources,
                "thread_info": thread_info,
                "memory_issues": memory_issues,
                "recommendations": self._generate_recommendations(system_resources, thread_info, memory_issues)
            }

            return report

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "生成汇总报告失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _generate_detailed_report(self) -> Dict[str, Any]:
        """生成详细报告"""
        try:
            # 收集详细数据
            system_resources = self.system_analyzer.get_system_resources("detailed")
            thread_analysis = self.thread_analyzer.analyze_threads(include_stacks=True)
            thread_issues = self.thread_analyzer.detect_thread_issues()
            memory_report = self.memory_detector.get_memory_report()

            # 生成详细报告
            report = {
                "timestamp": datetime.now().isoformat(),
                "report_type": "detailed",
                "sections": {
                    "system_resources": system_resources,
                    "thread_analysis": thread_analysis,
                    "thread_issues": thread_issues,
                    "memory_analysis": memory_report,
                    "performance_trends": self._analyze_performance_trends(),
                    "optimization_suggestions": self._generate_detailed_recommendations()
                }
            }

            return report

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "生成详细报告失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _generate_recommendations(self, system_resources: Dict, thread_info: Dict,
                                  memory_issues: List[str]) -> List[str]:
        """生成建议"""
        recommendations = []

        try:
            # CPU使用率建议
            cpu_usage = system_resources.get("cpu_usage", 0)
            if cpu_usage > 90:
                recommendations.append("CPU使用率过高，考虑优化CPU密集型操作或增加CPU资源")
            elif cpu_usage > 70:
                recommendations.append("CPU使用率较高，建议监控CPU使用趋势")

            # 内存使用建议
            memory_usage = system_resources.get("memory_usage", 0)
            if memory_usage > 90:
                recommendations.append("内存使用率过高，可能存在内存泄漏，建议进行内存分析")
            elif memory_usage > 80:
                recommendations.append("内存使用率较高，建议监控内存使用趋势")

            # 线程数量建议
            thread_count = thread_info.get("thread_count", 0)
            if thread_count > 100:
                recommendations.append("线程数量过多，考虑使用线程池或异步处理")
            elif thread_count > 50:
                recommendations.append("线程数量偏高，建议优化并发处理逻辑")

            # 内存问题建议
            if memory_issues:
                recommendations.append(f"发现 {len(memory_issues)} 个内存相关问题，建议进行详细检查")

            # 网络和磁盘I/O建议
            network_bytes = (system_resources.get("network_bytes_sent", 0) +
                             system_resources.get("network_bytes_recv", 0))
            disk_bytes = (system_resources.get("disk_read_bytes", 0) +
                          system_resources.get("disk_write_bytes", 0))

            if network_bytes > 100 * 1024 * 1024:  # 100MB
                recommendations.append("网络I/O负载较高，考虑优化网络通信")

            if disk_bytes > 500 * 1024 * 1024:  # 500MB
                recommendations.append("磁盘I/O负载较高，考虑优化存储操作")

        except Exception:
            recommendations.append("建议定期监控系统资源使用情况")

        return recommendations

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        return {
            "cpu_trend": "stable",
            "memory_trend": "stable",
            "thread_trend": "stable",
            "io_trend": "stable",
            "analysis_period": "last_24_hours",
            "data_points": 0
        }

    def _generate_detailed_recommendations(self) -> List[Dict[str, Any]]:
        """生成详细建议"""
        return [
            {
                "category": "performance",
                "priority": "high",
                "title": "CPU优化",
                "description": "分析CPU使用率并优化热点代码",
                "actions": ["性能分析", "代码优化", "负载均衡"]
            },
            {
                "category": "memory",
                "priority": "high",
                "title": "内存管理",
                "description": "监控内存使用并处理泄漏问题",
                "actions": ["内存分析", "垃圾回收优化", "对象池化"]
            },
            {
                "category": "concurrency",
                "priority": "medium",
                "title": "并发优化",
                "description": "优化线程使用和并发处理",
                "actions": ["线程池", "异步处理", "锁优化"]
            }
        ]

    def export_report(self, report: Dict[str, Any], format: str = "json") -> str:
        """导出报告"""
        try:
            if format.lower() == "json":
                return json.dumps(report, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                return yaml.dump(report, default_flow_style=False)
            else:
                raise ValueError(f"不支持的导出格式: {format}")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "导出报告失败"})
            return f"导出失败: {str(e)}"
