import logging
import time
import threading
from typing import Dict, Any
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """内存快照"""

    timestamp: float
    allocated: float
    reserved: float
    total: float
    usage_percent: float
    fragmentation_ratio: float = 0.0


class GpuMemoryManager:
    """
    GPU内存管理器 - 增强版
    支持智能显存监控、预测性清理、多策略内存优化
    """

    def __init__(
        self,
        cleanup_threshold: float = 0.8,
        cleanup_interval: int = 300,
        prediction_window: int = 10,
        adaptive_threshold: bool = True,
    ):
        """
        初始化GPU内存管理器

        Args:
            cleanup_threshold: 显存使用率阈值，超过则自动回收（0 - 1）
            cleanup_interval: 自动回收最小间隔（秒）
            prediction_window: 预测窗口大小，用于预测内存使用趋势
            adaptive_threshold: 是否启用自适应阈值调整
        """
        self.cleanup_threshold = cleanup_threshold
        self.cleanup_interval = cleanup_interval
        self.prediction_window = prediction_window
        self.adaptive_threshold = adaptive_threshold
        self._last_cleanup_time = 0

        # 内存历史记录
        self.memory_history = deque(maxlen=100)
        self._lock = threading.RLock()

        # 性能监控
        self.cleanup_count = 0
        self.last_memory_pressure = 0.0
        self.memory_pressure_threshold = 0.7

        # 预测相关
        self.prediction_enabled = True
        self.memory_trend_slope = 0.0

        # 启动监控线程
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._memory_monitor_loop, daemon=True
        )
        self._monitor_thread.start()

        logger.info(
            f"GPU内存管理器初始化完成 - 阈值: {cleanup_threshold * 100:.1f}%, 间隔: {cleanup_interval}s"
        )

    def get_memory_info(self) -> Dict[str, float]:
        """获取当前GPU显存信息（单位MB）- 增强版"""
        with self._lock:
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024 / 1024
                    reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    total = (
                        torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    )

                    # 计算碎片化率
                    fragmentation_ratio = (
                        (reserved - allocated) / reserved if reserved > 0 else 0.0
                    )

                    info = {
                        "allocated": allocated,
                        "reserved": reserved,
                        "total": total,
                        "available": total - reserved,
                        "usage_percent": (
                            (allocated / total) * 100 if total > 0 else 0.0
                        ),
                        "fragmentation_ratio": fragmentation_ratio,
                        "timestamp": time.time(),
                    }

                    # 记录历史
                    snapshot = MemorySnapshot(
                        timestamp=time.time(),
                        allocated=allocated,
                        reserved=reserved,
                        total=total,
                        usage_percent=info["usage_percent"],
                        fragmentation_ratio=fragmentation_ratio,
                    )
                    self.memory_history.append(snapshot)

                    return info
            except Exception as e:
                logger.warning(f"获取GPU内存信息失败: {e}")

            return {
                "allocated": 0.0,
                "reserved": 0.0,
                "total": 8192.0,
                "available": 8192.0,
                "usage_percent": 0.0,
                "fragmentation_ratio": 0.0,
                "timestamp": time.time(),
            }

    def need_cleanup(self) -> bool:
        """判断是否需要自动回收显存 - 增强版"""
        with self._lock:
            info = self.get_memory_info()
            now = time.time()

            # 基本阈值检查
            usage_threshold_exceeded = (
                info["usage_percent"] > self.cleanup_threshold * 100
            )
            time_threshold_exceeded = (
                now - self._last_cleanup_time
            ) > self.cleanup_interval

            if not (usage_threshold_exceeded and time_threshold_exceeded):
                return False

            # 预测性检查 - 如果内存使用呈上升趋势，更早触发清理
            if (
                self.prediction_enabled
                and len(self.memory_history) >= self.prediction_window
            ):
                trend = self._analyze_memory_trend()
                if trend > 0.1:  # 使用率每分钟上升超过0.1%
                    logger.info(
                        f"检测到内存使用上升趋势({trend:.3f}%/min)，提前触发清理"
                    )
                    return True

            # 碎片化检查 - 如果碎片化严重，也触发清理
            if info.get("fragmentation_ratio", 0) > 0.3:  # 碎片化超过30%
                logger.info(
                    f"检测到严重内存碎片化({info['fragmentation_ratio']:.1f})，触发清理"
                )
                return True

            # 内存压力检查
            memory_pressure = self._calculate_memory_pressure()
            if memory_pressure > self.memory_pressure_threshold:
                logger.info(f"内存压力过高({memory_pressure:.2f})，触发清理")
                return True

            logger.info(f"GPU显存使用率{info['usage_percent']:.1f}%超阈值，准备回收")
            return True

    def cleanup(self, force: bool = False):
        """回收GPU显存，force=True时无视间隔强制回收"""
        try:
            import torch

            if torch.cuda.is_available():
                before_info = self.get_memory_info()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after_info = self.get_memory_info()

                self._last_cleanup_time = time.time()
                self.cleanup_count += 1

                freed_memory = before_info["allocated"] - after_info["allocated"]
                logger.info(
                    f"已执行GPU显存回收 - 释放内存: {freed_memory:.1f}MB，当前使用率: {after_info['usage_percent']:.1f}%"
                )
        except ImportError:
            logger.warning("未安装torch，无法回收GPU显存")
        except Exception as e:
            logger.error(f"GPU显存回收失败: {e}")

    def auto_cleanup(self):
        """如有需要自动回收显存"""
        if self.need_cleanup():
            self.cleanup()

    def set_threshold(self, threshold: float):

        if 0.0 < threshold < 1.0:
            self.cleanup_threshold = threshold
            logger.info(f"设置GPU显存回收阈值为{threshold * 100:.1f}%")
        else:
            logger.warning("阈值需在0 - 1之间")

    def set_interval(self, interval: int):

        if interval > 0:
            self.cleanup_interval = interval
            logger.info(f"设置GPU显存回收最小间隔为{interval}秒")
        else:
            logger.warning("间隔需大于0")

    def _analyze_memory_trend(self) -> float:
        """分析内存使用趋势"""
        if len(self.memory_history) < self.prediction_window:
            return 0.0

        # 取最近的预测窗口数据
        recent_data = list(self.memory_history)[-self.prediction_window:]

        if len(recent_data) < 2:
            return 0.0

        # 计算线性回归斜率（每分钟使用率变化）
        n = len(recent_data)
        x_values = [
            (snapshot.timestamp - recent_data[0].timestamp) / 60
            for snapshot in recent_data
        ]  # 分钟
        y_values = [snapshot.usage_percent for snapshot in recent_data]

        # 计算斜率
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        self.memory_trend_slope = slope

        return slope

    def _calculate_memory_pressure(self) -> float:
        """计算内存压力指标"""
        if not self.memory_history:
            return 0.0

        # 基于最近的内存使用情况计算压力
        recent_usage = [snapshot.usage_percent for snapshot in self.memory_history]
        avg_usage = sum(recent_usage) / len(recent_usage)

        # 计算方差作为压力指标
        variance = sum((usage - avg_usage) ** 2 for usage in recent_usage) / len(
            recent_usage
        )
        pressure = variance / 100.0  # 归一化到0 - 1范围

        self.last_memory_pressure = pressure
        return pressure

    def _memory_monitor_loop(self):
        """内存监控循环"""
        while self._monitoring:
            try:
                # 获取内存信息（会自动记录历史）
                info = self.get_memory_info()

                # 检查是否需要清理
                if self.need_cleanup():
                    self.cleanup()

                # 自适应阈值调整
                if self.adaptive_threshold:
                    self._adjust_threshold_dynamically()

                # 每30秒检查一次
                time.sleep(30)

            except Exception as e:
                logger.error(f"内存监控循环异常: {e}")
                time.sleep(60)  # 出错后等待更长时间

    def _adjust_threshold_dynamically(self):
        """动态调整清理阈值"""
        if len(self.memory_history) < 10:
            return

        # 基于历史数据调整阈值
        recent_usage = [snapshot.usage_percent for snapshot in self.memory_history][
            -10:
        ]
        avg_usage = sum(recent_usage) / len(recent_usage)
        max_usage = max(recent_usage)

        # 如果平均使用率较高，降低阈值
        if avg_usage > 70:
            new_threshold = min(self.cleanup_threshold, 0.75)
            if new_threshold != self.cleanup_threshold:
                self.cleanup_threshold = new_threshold
                logger.info(f"动态调整清理阈值为{new_threshold * 100:.1f}%")

        # 如果最大使用率较低，提高阈值
        elif max_usage < 60:
            new_threshold = min(self.cleanup_threshold + 0.05, 0.9)
            if new_threshold != self.cleanup_threshold:
                self.cleanup_threshold = new_threshold
                logger.info(f"动态调整清理阈值为{new_threshold * 100:.1f}%")

    def get_memory_statistics(self) -> Dict[str, float]:
        """获取内存统计信息"""
        with self._lock:
            if not self.memory_history:
                return {}

            usage_values = [snapshot.usage_percent for snapshot in self.memory_history]
            fragmentation_values = [
                snapshot.fragmentation_ratio for snapshot in self.memory_history
            ]

            return {
                "avg_usage": sum(usage_values) / len(usage_values),
                "max_usage": max(usage_values),
                "min_usage": min(usage_values),
                "usage_std": (
                    sum(
                        (x - sum(usage_values) / len(usage_values)) ** 2
                        for x in usage_values
                    )
                    / len(usage_values)
                )
                ** 0.5,
                "avg_fragmentation": sum(fragmentation_values)
                / len(fragmentation_values),
                "max_fragmentation": max(fragmentation_values),
                "cleanup_count": self.cleanup_count,
                "trend_slope": self.memory_trend_slope,
                "memory_pressure": self.last_memory_pressure,
            }

    def predict_memory_usage(self, minutes_ahead: int = 5) -> float:
        """预测未来内存使用率"""
        with self._lock:
            if len(self.memory_history) < self.prediction_window:
                return 0.0

            current_usage = self.memory_history[-1].usage_percent
            predicted_usage = current_usage + (self.memory_trend_slope * minutes_ahead)

            # 限制在合理范围内
            return max(0.0, min(100.0, predicted_usage))

    def optimize_memory_allocation(self) -> Dict[str, Any]:
        """优化内存分配策略"""
        with self._lock:
            stats = self.get_memory_statistics()
            recommendations = []

            # 基于统计数据生成优化建议
            if stats.get("avg_fragmentation", 0) > 0.2:
                recommendations.append(
                    {
                        "type": "fragmentation",
                        "action": "increase_cleanup_frequency",
                        "reason": f"内存碎片化率过高({stats['avg_fragmentation']:.1f})",
                    }
                )

            if stats.get("memory_pressure", 0) > 0.5:
                recommendations.append(
                    {
                        "type": "pressure",
                        "action": "reduce_batch_size",
                        "reason": f"内存压力过高({stats['memory_pressure']:.2f})",
                    }
                )

            if stats.get("trend_slope", 0) > 0.2:
                recommendations.append(
                    {
                        "type": "trend",
                        "action": "enable_predictive_cleanup",
                        "reason": f"内存使用呈上升趋势({stats['trend_slope']:.3f}%/min)",
                    }
                )

            return {
                "recommendations": recommendations,
                "statistics": stats,
                "timestamp": time.time(),
            }

    def shutdown(self):
        """关闭内存管理器"""
        self._monitoring = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logger.info("GPU内存管理器已关闭")
