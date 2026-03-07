"""
performance_baseline 模块

提供 performance_baseline 相关功能和接口。
"""

import logging


from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List
"""
基础设施层 - 性能基准模块

提供性能基准测试和比较的功能。
"""

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """性能基准数据结构"""

    test_name: str = "default_test"
    test_category: str = "default_category"
    baseline_execution_time: float = 1.0
    baseline_operations_per_second: float = 1.0
    baseline_memory_usage: float = 0.0
    baseline_cpu_usage: float = 0.0
    threshold_percentage: float = 10.0  # 允许的性能变化百分比
    created_at: float = None
    updated_at: float = None
    sample_count: int = 1
    min_execution_time: float = None
    max_execution_time: float = None
    std_deviation: float = 0.0

    def __post_init__(self):
        """初始化后的处理"""
        current_time = datetime.now().timestamp()
        if self.created_at is None:
            self.created_at = current_time
        if self.updated_at is None:
            self.updated_at = current_time
        if self.min_execution_time is None:
            self.min_execution_time = self.baseline_execution_time
        if self.max_execution_time is None:
            self.max_execution_time = self.baseline_execution_time

    def update_baseline(
        self, execution_time: float, operations_per_second: float, memory_usage: float, cpu_usage: float
    ) -> None:
        """更新基准值"""
        self.sample_count += 1
        self.updated_at = datetime.now().timestamp()

        # 更新执行时间统计
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)

        # 使用移动平均更新基准值
        alpha = 0.1  # 平滑因子
        self.baseline_execution_time = self.baseline_execution_time * \
            (1 - alpha) + execution_time * alpha
        self.baseline_operations_per_second = (
            self.baseline_operations_per_second * (1 - alpha) + operations_per_second * alpha
        )
        self.baseline_memory_usage = self.baseline_memory_usage * (1 - alpha) + memory_usage * alpha
        self.baseline_cpu_usage = self.baseline_cpu_usage * (1 - alpha) + cpu_usage * alpha

    def is_within_threshold(self, current_value: float, metric: str) -> bool:
        """检查当前值是否在基准阈值内"""
        if metric == "execution_time":
            baseline = self.baseline_execution_time
        elif metric == "operations_per_second":
            baseline = self.baseline_operations_per_second
        elif metric == "memory_usage":
            baseline = self.baseline_memory_usage
        elif metric == "cpu_usage":
            baseline = self.baseline_cpu_usage
        else:
            return True

        deviation = abs(current_value - baseline) / baseline * 100
        return deviation <= self.threshold_percentage

    def compare_performance(self, current_execution_time: float) -> Dict[str, Any]:
        """比较执行时间性能"""
        if current_execution_time <= 0:
            return {
                'within_threshold': True,
                'change_percentage': 0.0,
                'status': 'invalid'
            }

        if self.baseline_execution_time <= 0:
            return {
                'within_threshold': True,
                'change_percentage': 0.0,
                'status': 'invalid'
            }

        change_percentage = (
            (current_execution_time - self.baseline_execution_time) / self.baseline_execution_time) * 100
        within_threshold = abs(change_percentage) <= self.threshold_percentage

        if change_percentage > 0:
            status = 'degraded'
        elif change_percentage < 0:
            status = 'improved'
        else:
            status = 'stable'

        return {
            'within_threshold': within_threshold,
            'change_percentage': round(change_percentage, 2),
            'status': status
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "test_name": self.test_name,
            "test_category": self.test_category,
            "baseline_metrics": {
                "execution_time": self.baseline_execution_time,
                "operations_per_second": self.baseline_operations_per_second,
                "memory_usage": self.baseline_memory_usage,
                "cpu_usage": self.baseline_cpu_usage,
            },
            "statistics": {
                "sample_count": self.sample_count,
                "min_execution_time": self.min_execution_time,
                "max_execution_time": self.max_execution_time,
                "threshold_percentage": self.threshold_percentage,
            },
            "timestamps": {
                "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
                "updated_at": datetime.fromtimestamp(self.updated_at).isoformat(),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """从字典创建实例"""
        return cls(**data)


class PerformanceBaselineManager:
    """性能基准管理器"""

    def __init__(self):
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.baseline_file = "performance_baselines.json"

    def add_baseline(self, baseline: PerformanceBaseline) -> None:
        """添加基准"""
        self.baselines[baseline.test_name] = baseline

    def get_baseline(self, test_name: str) -> Optional[PerformanceBaseline]:
        """获取基准"""
        return self.baselines.get(test_name)

    def update_baseline_from_result(
        self, test_name: str, execution_time: float, operations_per_second: float, memory_usage: float, cpu_usage: float
    ) -> None:
        """从测试结果更新基准"""
        baseline = self.baselines.get(test_name)
        if baseline:
            baseline.update_baseline(execution_time, operations_per_second, memory_usage, cpu_usage)
        else:
            # 创建新基准
            baseline = PerformanceBaseline(
                test_name=test_name,
                test_category="auto_generated",
                baseline_execution_time=execution_time,
                baseline_operations_per_second=operations_per_second,
                baseline_memory_usage=memory_usage,
                baseline_cpu_usage=cpu_usage,
            )
            self.baselines[test_name] = baseline

    def compare_with_baseline(
        self,
        test_name: str,
        current_execution_time: float,
        current_ops_per_sec: float,
        current_memory: float,
        current_cpu: float,
    ) -> Dict[str, Any]:
        """与基准比较"""
        try:
            baseline = self.baselines.get(test_name)
            if not baseline:
                return {"error": f"基准 '{test_name}' 不存在"}

            return {
                "test_name": test_name,
                "execution_time": self._compare_metric(
                    current_execution_time, baseline.baseline_execution_time, baseline, "execution_time"
                ),
                "operations_per_second": self._compare_metric(
                    current_ops_per_sec, baseline.baseline_operations_per_second, baseline, "operations_per_second"
                ),
                "memory_usage": self._compare_metric(
                    current_memory, baseline.baseline_memory_usage, baseline, "memory_usage"
                ),
                "cpu_usage": self._compare_metric(current_cpu, baseline.baseline_cpu_usage, baseline, "cpu_usage"),
            }
        except Exception as e:
            logger.error(f"比较基准时发生错误: {e}")
            return {"error": f"比较基准失败: {str(e)}"}

    def _compare_metric(
        self, current_value: float, baseline_value: float, baseline: PerformanceBaseline, metric_name: str
    ) -> Dict[str, Any]:
        """比较单个指标"""
        try:
            # 避免除零错误
            if baseline_value == 0:
                deviation_percent = 0.0 if current_value == 0 else float("inf")
            else:
                deviation_percent = (current_value - baseline_value) / baseline_value * 100

            return {
                "current": current_value,
                "baseline": baseline_value,
                "deviation_percent": deviation_percent,
                "within_threshold": baseline.is_within_threshold(current_value, metric_name),
            }
        except Exception as e:
            logger.warning(f"比较指标 '{metric_name}' 时发生错误: {e}")
            return {"current": current_value, "baseline": baseline_value, "error": f"指标比较失败: {str(e)}"}

    def get_all_baselines(self) -> List[PerformanceBaseline]:
        """获取所有基准"""
        return list(self.baselines.values())

    def clear_baselines(self) -> None:
        """清除所有基准"""
        self.baselines.clear()

    def compare_performance_with_baseline(self, test_name: str, current_execution_time: float) -> Optional[Dict[str, Any]]:
        """简化版本的性能比较方法"""
        baseline = self.baselines.get(test_name)
        if not baseline:
            return None

        return baseline.compare_performance(current_execution_time)

    def list_baselines(self, category: Optional[str] = None) -> List[PerformanceBaseline]:
        """列出所有基准，可按类别过滤"""
        if category is None:
            return list(self.baselines.values())
        else:
            return [baseline for baseline in self.baselines.values() if baseline.test_category == category]

    def remove_baseline(self, test_name: str) -> bool:
        """删除基准"""
        if test_name in self.baselines:
            del self.baselines[test_name]
            return True
        return False

    def update_baseline_stats(self, test_name: str, execution_time: float, operations_per_second: float, memory_usage: float, cpu_usage: float) -> None:
        """更新基准统计信息"""
        baseline = self.baselines.get(test_name)
        if baseline:
            baseline.update_baseline(execution_time, operations_per_second, memory_usage, cpu_usage)

    def save_baselines(self) -> Dict[str, Dict[str, Any]]:
        """保存所有基准"""
        return {name: baseline.to_dict() for name, baseline in self.baselines.items()}

    def load_baselines(self, data: Dict[str, Dict[str, Any]]) -> None:
        """加载基准"""
        for name, baseline_data in data.items():
            self.baselines[name] = PerformanceBaseline.from_dict(baseline_data)
