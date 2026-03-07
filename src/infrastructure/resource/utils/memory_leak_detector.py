"""
memory_leak_detector 模块

提供 memory_leak_detector 相关功能和接口。
"""

import sys

import gc
import threading
import psutil
import weakref

from ..core.shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from datetime import datetime
from typing import Dict, List, Optional, Any
"""
内存泄漏检测器

Phase 3: 质量提升 - 文件拆分优化

负责检测内存泄漏和内存使用异常。
"""


class MemoryLeakDetector:
    """内存泄漏检测器"""

    def __init__(self, logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

        # 内存监控历史
        self._memory_history: List[Dict[str, Any]] = []
        self._max_history_size = 100

        # 对象引用跟踪
        self._object_refs = weakref.WeakSet()

    def detect_memory_leaks(self) -> List[str]:
        """检测内存泄漏"""
        issues = []

        try:
            # 1. 检查内存使用趋势
            trend_issues = self._check_memory_trend()
            issues.extend(trend_issues)

            # 2. 检查对象引用
            ref_issues = self._check_object_references()
            issues.extend(ref_issues)

            # 3. 检查循环引用
            cycle_issues = self._check_circular_references()
            issues.extend(cycle_issues)

            # 4. 检查大对象
            large_obj_issues = self._check_large_objects()
            issues.extend(large_obj_issues)

            # 记录检测结果
            if issues:
                self.logger.log_warning(f"检测到 {len(issues)} 个内存相关问题")
            else:
                self.logger.log_info("内存检查完成，未发现明显问题")

            return issues

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "内存泄漏检测失败"})
            return [f"检测失败: {str(e)}"]

    def _check_memory_trend(self) -> List[str]:
        """检查内存使用趋势"""
        issues = []

        try:
            # 获取当前内存使用
            process = psutil.Process()
            current_memory = process.memory_info().rss

            # 记录到历史
            self._memory_history.append({
                "timestamp": datetime.now(),
                "memory_mb": current_memory / 1024 / 1024
            })

            # 保持历史大小
            if len(self._memory_history) > self._max_history_size:
                self._memory_history.pop(0)

            # 检查趋势（需要至少5个数据点）
            if len(self._memory_history) >= 5:
                recent = self._memory_history[-5:]

                # 计算增长率
                start_memory = recent[0]["memory_mb"]
                end_memory = recent[-1]["memory_mb"]
                growth_rate = (end_memory - start_memory) / start_memory if start_memory > 0 else 0

                if growth_rate > 0.5:  # 50%增长
                    issues.append(f"内存使用快速增长: {growth_rate:.1%} (过去5次测量)")
                elif growth_rate > 0.2:  # 20%增长
                    issues.append(f"内存使用稳步增长: {growth_rate:.1%} (过去5次测量)")
        except Exception:
            pass

        return issues

    def _check_object_references(self) -> List[str]:
        """检查对象引用"""
        issues = []

        try:
            # 获取当前对象数量
            gc.collect()  # 强制垃圾回收

            # 统计对象类型
            objects_by_type = {}
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                objects_by_type[obj_type] = objects_by_type.get(obj_type, 0) + 1

            # 检查可疑的对象数量
            suspicious_types = ['dict', 'list', 'tuple', 'str']
            for obj_type, count in objects_by_type.items():
                if obj_type in suspicious_types and count > 10000:
                    issues.append(f"对象类型 '{obj_type}' 数量异常: {count}")

        except Exception:
            pass

        return issues

    def _check_circular_references(self) -> List[str]:
        """检查循环引用"""
        issues = []

        try:
            # 强制垃圾回收并检查循环引用
            collected = gc.collect()

            if collected > 0:
                issues.append(f"检测到 {collected} 个循环引用对象，已清理")

        except Exception:
            pass

        return issues

    def _check_large_objects(self) -> List[str]:
        """检查大对象"""
        issues = []

        try:
            # 获取大对象（>1MB）
            large_objects = []
            for obj in gc.get_objects():
                try:
                    size = sys.getsizeof(obj)
                    if size > 1024 * 1024:  # 1MB
                        large_objects.append((type(obj).__name__, size))
                except Exception as e:
                    continue

            # 按大小排序，取前10个
            large_objects.sort(key=lambda x: x[1], reverse=True)
            large_objects = large_objects[:10]

            for obj_type, size in large_objects:
                size_mb = size / 1024 / 1024
                issues.append(f"大对象检测: {obj_type} ({size_mb:.1f}MB)")
        except Exception:
            pass

        return issues

    def start_memory_monitoring(self, interval_seconds: int = 60):
        """开始内存监控"""
        def monitor_loop():
            while True:
                try:
                    issues = self.detect_memory_leaks()
                    if issues:
                        for issue in issues:
                            self.logger.log_warning(f"内存监控: {issue}")
                    else:
                        self.logger.log_info("内存监控: 正常")

                    threading.Event().wait(interval_seconds)

                except Exception as e:
                    self.error_handler.handle_error(e, {"context": "内存监控循环异常"})
                    break

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

        self.logger.log_info(f"内存监控已启动，间隔: {interval_seconds}秒")

    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存报告"""
        try:
            process = psutil.Process()

            return {
                "timestamp": datetime.now().isoformat(),
                "process_memory": {
                    "rss_mb": process.memory_info().rss / 1024 / 1024,
                    "vms_mb": process.memory_info().vms / 1024 / 1024,
                    "percent": process.memory_percent()
                },
                "system_memory": {
                    "total_mb": psutil.virtual_memory().total / 1024 / 1024,
                    "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                    "used_mb": psutil.virtual_memory().used / 1024 / 1024,
                    "percent": psutil.virtual_memory().percent
                },
                "issues": self.detect_memory_leaks()
            }

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "生成内存报告失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
