"""
thread_analyzer 模块

提供 thread_analyzer 相关功能和接口。
"""


import threading
import traceback

from ..core.shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from datetime import datetime
from typing import Dict, List, Optional, Any
"""
线程分析器

Phase 3: 质量提升 - 文件拆分优化

负责分析线程状态、堆栈跟踪和线程相关问题。
"""


class ThreadAnalyzer:
    """线程分析器"""

    def __init__(self, logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

    def analyze_threads(self, include_stacks: bool = False) -> Dict[str, Any]:
        """分析线程状态"""
        try:
            current_thread = threading.current_thread()
            all_threads = threading.enumerate()

            thread_info = {
                "timestamp": datetime.now().isoformat(),
                "current_thread": {
                    "name": current_thread.name,
                    "ident": current_thread.ident,
                    "daemon": current_thread.daemon,
                    "alive": current_thread.is_alive()
                },
                "thread_count": len(all_threads),
                "threads": []
            }

            # 分析所有线程
            for thread in all_threads:
                thread_data = {
                    "name": thread.name,
                    "ident": thread.ident,
                    "daemon": thread.daemon,
                    "alive": thread.is_alive()
                }

                # 获取线程堆栈（如果启用）
                if include_stacks:
                    try:
                        thread_data["stack"] = self._get_thread_stack(thread)
                    except Exception as e:
                        thread_data["stack_error"] = str(e)

                thread_info["threads"].append(thread_data)

            # 线程统计
            thread_info["statistics"] = {
                "daemon_threads": sum(1 for t in all_threads if t.daemon),
                "non_daemon_threads": sum(1 for t in all_threads if not t.daemon),
                "alive_threads": sum(1 for t in all_threads if t.is_alive()),
                "dead_threads": sum(1 for t in all_threads if not t.is_alive())
            }

            return thread_info

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "分析线程状态失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _get_thread_stack(self, thread: threading.Thread) -> List[str]:
        """获取线程堆栈跟踪"""
        try:
            # 注意：获取其他线程的堆栈在某些情况下可能不可靠
            # 这里使用一个简化的方法
            if thread == threading.current_thread():
                return traceback.format_stack()
            else:
                # 对于其他线程，我们返回线程的基本信息
                return [f"Thread {thread.name} (ID: {thread.ident})"]
        except Exception:
            return ["Unable to get stack trace"]

    def detect_thread_issues(self) -> Dict[str, Any]:
        """检测线程相关问题"""
        try:
            issues = {
                "timestamp": datetime.now().isoformat(),
                "problems": [],
                "warnings": []
            }

            all_threads = threading.enumerate()

            # 检查线程数量
            thread_count = len(all_threads)
            if thread_count > 100:
                issues["problems"].append({
                    "type": "high_thread_count",
                    "message": f"线程数量过多: {thread_count}",
                    "severity": "high"
                })
            elif thread_count > 50:
                issues["warnings"].append({
                    "type": "moderate_thread_count",
                    "message": f"线程数量偏高: {thread_count}",
                    "severity": "medium"
                })

            # 检查死线程
            dead_threads = [t for t in all_threads if not t.is_alive()]
            if dead_threads:
                issues["warnings"].append({
                    "type": "dead_threads",
                    "message": f"发现 {len(dead_threads)} 个死线程",
                    "severity": "low",
                    "threads": [t.name for t in dead_threads]
                })

            # 检查长时间运行的线程（简化检查）
            # 这里可以扩展为更复杂的线程监控逻辑

            return issues

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "检测线程问题失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def get_thread_summary(self) -> Dict[str, Any]:
        """获取线程汇总信息"""
        analysis = self.analyze_threads(include_stacks=False)

        if "error" in analysis:
            return analysis

        return {
            "thread_count": analysis["thread_count"],
            "alive_threads": analysis["statistics"]["alive_threads"],
            "daemon_threads": analysis["statistics"]["daemon_threads"],
            "current_thread": analysis["current_thread"]["name"],
            "timestamp": analysis["timestamp"]
        }

    def analyze_thread_stacks(self) -> Dict[str, Any]:
        """分析线程堆栈"""
        try:
            all_threads = threading.enumerate()
            stack_analysis = {
                "timestamp": datetime.now().isoformat(),
                "thread_count": len(all_threads),
                "thread_stacks": []
            }

            for thread in all_threads:
                thread_stack_info = {
                    "thread_name": thread.name,
                    "thread_id": thread.ident,
                    "daemon": thread.daemon,
                    "alive": thread.is_alive()
                }

                try:
                    # 获取线程堆栈
                    stack = self._get_thread_stack(thread)
                    thread_stack_info["stack"] = stack
                    thread_stack_info["stack_lines"] = len(stack)
                except Exception as e:
                    thread_stack_info["stack_error"] = str(e)
                    thread_stack_info["stack"] = []

                stack_analysis["thread_stacks"].append(thread_stack_info)

            return stack_analysis

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "分析线程堆栈失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def get_deadlock_risk(self) -> Dict[str, Any]:
        """获取死锁风险分析"""
        try:
            all_threads = threading.enumerate()
            deadlock_analysis = {
                "timestamp": datetime.now().isoformat(),
                "risk_level": "low",  # low, medium, high
                "risk_factors": [],
                "recommendations": []
            }

            # 分析线程数量
            thread_count = len(all_threads)
            if thread_count > 100:
                deadlock_analysis["risk_level"] = "high"
                deadlock_analysis["risk_factors"].append({
                    "factor": "high_thread_count",
                    "description": f"线程数量过多: {thread_count}",
                    "risk_score": 3
                })
            elif thread_count > 50:
                deadlock_analysis["risk_level"] = "medium"
                deadlock_analysis["risk_factors"].append({
                    "factor": "moderate_thread_count",
                    "description": f"线程数量偏高: {thread_count}",
                    "risk_score": 2
                })

            # 检查死线程（可能是死锁的迹象）
            dead_threads = [t for t in all_threads if not t.is_alive()]
            if dead_threads:
                deadlock_analysis["risk_factors"].append({
                    "factor": "dead_threads",
                    "description": f"发现 {len(dead_threads)} 个死线程",
                    "risk_score": 1,
                    "threads": [{"name": t.name, "id": t.ident} for t in dead_threads]
                })

            # 根据风险因素调整风险级别
            if deadlock_analysis["risk_level"] == "low" and deadlock_analysis["risk_factors"]:
                deadlock_analysis["risk_level"] = "medium"

            # 生成建议
            if deadlock_analysis["risk_level"] == "high":
                deadlock_analysis["recommendations"].append("考虑减少并发线程数量")
                deadlock_analysis["recommendations"].append("检查锁的获取顺序")
            elif deadlock_analysis["risk_level"] == "medium":
                deadlock_analysis["recommendations"].append("监控线程状态")

            deadlock_analysis["total_risk_score"] = sum(factor.get("risk_score", 0) for factor in deadlock_analysis["risk_factors"])

            return deadlock_analysis

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "分析死锁风险失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def get_thread_count(self) -> int:
        """获取当前线程数量"""
        try:
            return len(threading.enumerate())
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取线程数量失败"})
            return 0

    def get_thread_info(self) -> Dict[str, Any]:
        """获取线程信息"""
        try:
            all_threads = threading.enumerate()
            current_thread = threading.current_thread()
            
            thread_info = {
                "timestamp": datetime.now().isoformat(),
                "total_threads": len(all_threads),
                "current_thread": {
                    "name": current_thread.name,
                    "ident": current_thread.ident,
                    "daemon": current_thread.daemon,
                    "alive": current_thread.is_alive()
                },
                "thread_list": []
            }
            
            # 收集所有线程的基本信息
            for thread in all_threads:
                thread_data = {
                    "name": thread.name,
                    "ident": thread.ident,
                    "daemon": thread.daemon,
                    "alive": thread.is_alive(),
                    "native_id": getattr(thread, 'native_id', None)
                }
                thread_info["thread_list"].append(thread_data)
            
            return thread_info
            
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取线程信息失败"})
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "total_threads": 0,
                "thread_list": []
            }
