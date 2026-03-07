"""
metrics_storage 模块

提供 metrics_storage 相关功能和接口。
"""

import json
import logging

import threading

from .constants import (
    DEFAULT_HISTORY_SIZE, MIN_HISTORY_SIZE, MAX_HISTORY_SIZE,
    DEFAULT_HISTORY_HOURS, HISTORY_HOURS_OPTIONS
)
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
"""
指标数据存储管理器

根据Phase 8.2.3指标收集体系优化，分离SystemMetricsCollector的存储职责。
提供标准化的指标数据存储和查询功能。
"""

logger = logging.getLogger(__name__)


class MetricsStorage:
    """
    指标数据存储管理器

    负责指标数据的存储、查询和管理，分离自SystemMetricsCollector。
    提供线程安全的指标历史数据管理。
    """

    def __init__(self, history_size: int = DEFAULT_HISTORY_SIZE):
        """
        初始化指标存储器

        Args:
            history_size: 历史数据存储大小限制
        """
        # 验证历史大小参数
        if history_size < MIN_HISTORY_SIZE:
            logger.warning(f"历史大小 {history_size} 小于最小值 {MIN_HISTORY_SIZE}，使用最小值")
            history_size = MIN_HISTORY_SIZE
        elif history_size > MAX_HISTORY_SIZE:
            logger.warning(f"历史大小 {history_size} 大于最大值 {MAX_HISTORY_SIZE}，使用最大值")
            history_size = MAX_HISTORY_SIZE

        self.history_size = history_size
        self._lock = threading.RLock()  # 线程安全锁

        # 历史数据存储 - 使用deque实现高效的固定大小队列
        self.metrics_history = deque(maxlen=history_size)

        # 统计信息
        self._total_stored = 0
        self._storage_start_time = datetime.now()

        logger.info(f"指标存储器初始化完成，历史大小: {history_size}")

    def store_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        存储指标数据

        Args:
            metrics: 要存储的指标数据字典

        Returns:
            bool: 存储是否成功
        """
        if not isinstance(metrics, dict):
            logger.error(f"指标数据必须是字典类型，收到: {type(metrics)}")
            return False

        try:
            with self._lock:
                # 确保时间戳存在
                if 'timestamp' not in metrics:
                    metrics['timestamp'] = datetime.now().isoformat()

                # 添加存储时间戳
                metrics['_stored_at'] = datetime.now().isoformat()

                # 存储到历史队列
                self.metrics_history.append(metrics.copy())
                self._total_stored += 1

                logger.debug(f"指标数据已存储，总数: {len(self.metrics_history)}")
                return True

        except Exception as e:
            logger.error(f"存储指标数据失败: {e}")
            return False

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的指标数据

        Returns:
            Optional[Dict[str, Any]]: 最新的指标数据，如果没有数据则返回None
        """
        try:
            with self._lock:
                if self.metrics_history:
                    return self.metrics_history[-1].copy()
                return None
        except Exception as e:
            logger.error(f"获取最新指标失败: {e}")
            return None

    def get_metrics_history(self, hours: int = DEFAULT_HISTORY_HOURS) -> List[Dict[str, Any]]:
        """
        获取指定时间范围内的历史指标数据

        Args:
            hours: 历史时间范围(小时)，默认24小时

        Returns:
            List[Dict[str, Any]]: 历史指标数据列表
        """
        # 验证时间参数
        if hours not in HISTORY_HOURS_OPTIONS:
            logger.warning(f"时间范围 {hours} 不在标准选项中，使用默认值 {DEFAULT_HISTORY_HOURS}")
            hours = DEFAULT_HISTORY_HOURS

        try:
            with self._lock:
                if not self.metrics_history:
                    return []

                # 计算时间阈值
                cutoff_time = datetime.now() - timedelta(hours=hours)

                # 过滤历史数据
                filtered_history = []
                for metrics in reversed(self.metrics_history):
                    try:
                        # 解析时间戳
                        timestamp_str = metrics.get('timestamp', metrics.get('_stored_at', ''))
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if timestamp >= cutoff_time:
                                filtered_history.append(metrics.copy())
                            else:
                                break  # 由于是逆序，遇到第一个不符合条件的就可以停止
                    except (ValueError, TypeError) as e:
                        logger.warning(f"解析时间戳失败: {timestamp_str}, 错误: {e}")
                        continue

                # 返回正序结果
                return list(reversed(filtered_history))

        except Exception as e:
            logger.error(f"获取历史指标失败: {e}")
            return []

    def get_metrics_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        获取指定时间范围内的指标数据

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[Dict[str, Any]]: 指定时间范围内的指标数据
        """
        try:
            with self._lock:
                if not self.metrics_history:
                    return []

                filtered_history = []
                for metrics in self.metrics_history:
                    try:
                        timestamp_str = metrics.get('timestamp', metrics.get('_stored_at', ''))
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if start_time <= timestamp <= end_time:
                                filtered_history.append(metrics.copy())
                    except (ValueError, TypeError) as e:
                        logger.warning(f"解析时间戳失败: {timestamp_str}, 错误: {e}")
                        continue

                return filtered_history

        except Exception as e:
            logger.error(f"按时间范围获取指标失败: {e}")
            return []

    def get_metrics_count(self) -> int:
        """
        获取当前存储的指标数量

        Returns:
            int: 当前指标数量
        """
        with self._lock:
            return len(self.metrics_history)

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储器统计信息

        Returns:
            Dict[str, Any]: 存储统计信息
        """
        try:
            with self._lock:
                latest_metrics = self.get_latest_metrics()

                return {
                    "history_size": self.history_size,
                    "current_count": len(self.metrics_history),
                    "total_stored": self._total_stored,
                    "storage_start_time": self._storage_start_time.isoformat(),
                    "uptime_seconds": (datetime.now() - self._storage_start_time).total_seconds(),
                    "utilization_percent": (len(self.metrics_history) / self.history_size) * 100,
                    "has_latest_data": latest_metrics is not None,
                    "latest_timestamp": latest_metrics.get('timestamp') if latest_metrics else None
                }

        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {"error": str(e)}

    def clear_history(self) -> bool:
        """
        清空历史数据

        Returns:
            bool: 清空是否成功
        """
        try:
            with self._lock:
                self.metrics_history.clear()
                logger.info("指标历史数据已清空")
                return True
        except Exception as e:
            logger.error(f"清空历史数据失败: {e}")
            return False

    def get_average_metrics(self, hours: int = 1) -> Optional[Dict[str, Any]]:
        """
        获取指定时间范围内的平均指标

        Args:
            hours: 时间范围(小时)

        Returns:
            Optional[Dict[str, Any]]: 平均指标数据
        """
        try:
            history = self.get_metrics_history(hours)
            if not history:
                return None

            # 提取数值型指标进行平均计算
            numeric_metrics = {}
            count = len(history)

            for metrics in history:
                for key, value in metrics.items():
                    # 只处理数值类型且不是时间戳的指标
                    if isinstance(value, (int, float)) and key not in ['timestamp', '_stored_at']:
                        if key not in numeric_metrics:
                            numeric_metrics[key] = []
                        numeric_metrics[key].append(value)

            # 计算平均值
            averages = {}
            for key, values in numeric_metrics.items():
                if values:
                    averages[key] = sum(values) / len(values)

            return {
                "time_range_hours": hours,
                "sample_count": count,
                "averages": averages,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"计算平均指标失败: {e}")
            return None

    def export_metrics(self, format_type: str = "json") -> Optional[str]:
        """
        导出指标数据

        Args:
            format_type: 导出格式 ("json", "csv")

        Returns:
            Optional[str]: 导出的数据字符串
        """
        try:
            with self._lock:
                if format_type.lower() == "json":
                    return json.dumps(list(self.metrics_history), indent=2, default=str)
                elif format_type.lower() == "csv":
                    if not self.metrics_history:
                        return ""

                    # 获取所有可能的列
                    all_keys = set()
                    for metrics in self.metrics_history:
                        all_keys.update(metrics.keys())

                    # 生成CSV
                    lines = [",".join(sorted(all_keys))]
                    for metrics in self.metrics_history:
                        row = []
                        for key in sorted(all_keys):
                            value = metrics.get(key, "")
                            row.append(str(value))
                        lines.append(",".join(row))

                    return "\n".join(lines)
                else:
                    logger.error(f"不支持的导出格式: {format_type}")
                    return None

        except Exception as e:
            logger.error(f"导出指标数据失败: {e}")
            return None
