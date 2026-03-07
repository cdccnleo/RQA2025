"""
cloud_auto_scaling 模块

提供 cloud_auto_scaling 相关功能和接口。
"""

from ..core.imports import (
    Dict, Any, List, Optional, logging, time, threading, datetime
)
from .cloud_native_configs import (
    AutoScalingConfig, ScalingPolicy
)

"""
云原生自动伸缩管理器

实现基于指标的自动伸缩功能，支持CPU、内存和自定义指标
"""

logger = logging.getLogger(__name__)


class AutoScalingManager:
    """自动伸缩管理器"""

    def __init__(self, config: AutoScalingConfig):
        self.config = config
        self._lock = threading.RLock()
        self._current_replicas = config.min_replicas
        self._scaling_history: List[Dict[str, Any]] = []
        self._last_scale_time = 0
        self._cooldown_active = False
        self._metrics_cache: Dict[str, List[float]] = {}
        self._scale_up_count = 0
        self._scale_down_count = 0

    def should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """检查是否应该扩容"""
        if not self.config.enabled:
            return False

        # 检查冷却期
        if self._is_in_cooldown():
            return False

        # 检查是否已达到最大副本数
        if self._current_replicas >= self.config.max_replicas:
            return False

        # 检查metrics是否有效
        if not metrics or not isinstance(metrics, dict):
            return False

        # 根据策略检查扩容条件
        if self.config.scaling_policy == ScalingPolicy.CPU_UTILIZATION:
            return self._check_cpu_scale_up(metrics)
        elif self.config.scaling_policy == ScalingPolicy.MEMORY_UTILIZATION:
            return self._check_memory_scale_up(metrics)
        elif self.config.scaling_policy == ScalingPolicy.CUSTOM_METRIC:
            return self._check_custom_scale_up(metrics)
        elif self.config.scaling_policy == ScalingPolicy.REQUEST_RATE:
            return self._check_request_rate_scale_up(metrics)

        return False

    def should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """检查是否应该缩容"""
        if not self.config.enabled:
            return False

        # 检查冷却期
        if self._is_in_cooldown():
            return False

        # 检查是否已达到最小副本数
        if self._current_replicas <= self.config.min_replicas:
            return False

        # 检查metrics是否有效
        if not metrics or not isinstance(metrics, dict):
            return False

        # 根据策略检查缩容条件
        if self.config.scaling_policy == ScalingPolicy.CPU_UTILIZATION:
            return self._check_cpu_scale_down(metrics)
        elif self.config.scaling_policy == ScalingPolicy.MEMORY_UTILIZATION:
            return self._check_memory_scale_down(metrics)
        elif self.config.scaling_policy == ScalingPolicy.CUSTOM_METRIC:
            return self._check_custom_scale_down(metrics)
        elif self.config.scaling_policy == ScalingPolicy.REQUEST_RATE:
            return self._check_request_rate_scale_down(metrics)

        return False

    def _check_cpu_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """检查CPU扩容条件"""
        cpu_usage = metrics.get('cpu_percent', 0)
        return cpu_usage > self.config.target_cpu_utilization

    def _check_cpu_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """检查CPU缩容条件"""
        cpu_usage = metrics.get('cpu_percent', 0)
        # 缩容条件应该比扩容条件更保守
        scale_down_threshold = self.config.target_cpu_utilization * 0.7
        return cpu_usage < scale_down_threshold

    def _check_memory_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """检查内存扩容条件"""
        memory_usage = metrics.get('memory_percent', 0)
        return memory_usage > self.config.target_memory_utilization

    def _check_memory_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """检查内存缩容条件"""
        memory_usage = metrics.get('memory_percent', 0)
        # 缩容条件应该比扩容条件更保守
        scale_down_threshold = self.config.target_memory_utilization * 0.7
        return memory_usage < scale_down_threshold

    def _check_custom_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """检查自定义指标扩容条件"""
        for metric_name in self.config.custom_metrics:
            value = metrics.get(metric_name, 0)
            # 简单的阈值检查，可以根据需要扩展为更复杂的逻辑
            if value > 80:  # 假设80为扩容阈值
                return True
        return False

    def _check_custom_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """检查自定义指标缩容条件"""
        for metric_name in self.config.custom_metrics:
            value = metrics.get(metric_name, 0)
            # 缩容条件更保守
            if value < 30:  # 假设30为缩容阈值
                return True
        return False

    def _check_request_rate_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """检查请求率扩容条件"""
        request_rate = metrics.get('requests_per_second', 0)
        # 假设1000 req/s 为扩容阈值
        return request_rate > 1000

    def _check_request_rate_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """检查请求率缩容条件"""
        request_rate = metrics.get('requests_per_second', 0)
        # 缩容条件更保守
        return request_rate < 300

    def _is_in_cooldown(self) -> bool:
        """检查是否在冷却期内"""
        if self._cooldown_active:
            time_since_last_scale = time.time() - self._last_scale_time
            if time_since_last_scale < self.config.cooldown_period_seconds:
                return True
            else:
                self._cooldown_active = False
        return False

    def scale_up(self, reason: str = "自动扩容", metrics: Optional[Dict[str, Any]] = None) -> bool:
        """执行扩容操作"""
        with self._lock:
            # 如果没有提供metrics，使用默认的高负载metrics
            if metrics is None:
                metrics = {"cpu_percent": 90.0, "memory_percent": 85.0}

            if not self.should_scale_up(metrics):
                return False

            new_replicas = min(self._current_replicas + 1, self.config.max_replicas)
            if new_replicas == self._current_replicas:
                return False

            try:
                # 这里应该实现实际的扩容逻辑
                # 例如调用Kubernetes API或其他云服务API
                success = self._perform_scale_operation(new_replicas)

                if success:
                    self._record_scaling_event(
                        "scale_up", self._current_replicas, new_replicas, reason)
                    self._current_replicas = new_replicas
                    self._scale_up_count += 1
                    self._enter_cooldown()
                    logger.info(
                        f"扩容成功: {self._current_replicas - 1} -> {self._current_replicas} ({reason})")
                    return True
                else:
                    logger.error("扩容操作失败")
                    return False

            except Exception as e:
                logger.error(f"扩容异常: {e}")
                return False

    def scale_down(self, reason: str = "自动缩容", metrics: Optional[Dict[str, Any]] = None) -> bool:
        """执行缩容操作"""
        with self._lock:
            # 如果没有提供metrics，使用默认的低负载metrics
            if metrics is None:
                metrics = {"cpu_percent": 20.0, "memory_percent": 25.0}

            if not self.should_scale_down(metrics):
                return False

            new_replicas = max(self._current_replicas - 1, self.config.min_replicas)
            if new_replicas == self._current_replicas:
                return False

            try:
                # 这里应该实现实际的缩容逻辑
                success = self._perform_scale_operation(new_replicas)

                if success:
                    self._record_scaling_event(
                        "scale_down", self._current_replicas, new_replicas, reason)
                    self._current_replicas = new_replicas
                    self._scale_down_count += 1
                    self._enter_cooldown()
                    logger.info(
                        f"缩容成功: {self._current_replicas + 1} -> {self._current_replicas} ({reason})")
                    return True
                else:
                    logger.error("缩容操作失败")
                    return False

            except Exception as e:
                logger.error(f"缩容异常: {e}")
                return False

    def _perform_scale_operation(self, target_replicas: int) -> bool:
        """执行伸缩操作"""
        try:
            # 这里应该实现具体的伸缩逻辑
            # 例如:
            # - Kubernetes: kubectl scale deployment
            # - AWS ECS: update service desired count
            # - Azure: update container app replica count
            logger.info(f"执行伸缩操作: {self._current_replicas} -> {target_replicas} (模拟)")
            return True
        except Exception as e:
            logger.error(f"伸缩操作失败: {e}")
            return False

    def _record_scaling_event(self, event_type: str, old_replicas: int,
                              new_replicas: int, reason: str):
        """记录伸缩事件"""
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "old_replicas": old_replicas,
            "new_replicas": new_replicas,
            "reason": reason,
            "cooldown_remaining": self.config.cooldown_period_seconds
        }
        self._scaling_history.append(event)

        # 限制历史记录大小
        if len(self._scaling_history) > 100:
            self._scaling_history.pop(0)

    def _enter_cooldown(self):
        """进入冷却期"""
        self._last_scale_time = time.time()
        self._cooldown_active = True

    def get_current_replicas(self) -> int:
        """获取当前副本数"""
        with self._lock:
            return self._current_replicas

    def get_scaling_status(self) -> Dict[str, Any]:
        """获取伸缩状态"""
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "current_replicas": self._current_replicas,
                "min_replicas": self.config.min_replicas,
                "max_replicas": self.config.max_replicas,
                "scaling_policy": self.config.scaling_policy.value,
                "in_cooldown": self._is_in_cooldown(),
                "cooldown_remaining": max(0, self.config.cooldown_period_seconds -
                                          (time.time() - self._last_scale_time)) if self._cooldown_active else 0,
                "scale_up_count": self._scale_up_count,
                "scale_down_count": self._scale_down_count,
                "last_scale_time": datetime.fromtimestamp(self._last_scale_time).isoformat() if self._last_scale_time > 0 else None
            }

    def get_scaling_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取伸缩历史"""
        with self._lock:
            history = self._scaling_history[-limit:] if limit > 0 else self._scaling_history
            return [
                {
                    "timestamp": event["timestamp"].isoformat(),
                    "event_type": event["event_type"],
                    "old_replicas": event["old_replicas"],
                    "new_replicas": event["new_replicas"],
                    "reason": event["reason"]
                }
                for event in history
            ]

    def manual_scale(self, target_replicas: int, reason: str = "手动伸缩") -> bool:
        """手动伸缩"""
        with self._lock:
            if not self.config.enabled:
                logger.warning("自动伸缩已禁用")
                return False

            if target_replicas < self.config.min_replicas or target_replicas > self.config.max_replicas:
                logger.error(
                    f"目标副本数 {target_replicas} 超出范围 [{self.config.min_replicas}, {self.config.max_replicas}]")
                return False

            if target_replicas == self._current_replicas:
                logger.info("目标副本数与当前副本数相同，无需伸缩")
                return True

            try:
                success = self._perform_scale_operation(target_replicas)
                if success:
                    direction = "up" if target_replicas > self._current_replicas else "down"
                    self._record_scaling_event(f"manual_scale_{direction}",
                                               self._current_replicas, target_replicas, reason)
                    self._current_replicas = target_replicas
                    if direction == "up":
                        self._scale_up_count += 1
                    else:
                        self._scale_down_count += 1
                    self._enter_cooldown()
                    logger.info(f"手动伸缩成功: {self._current_replicas} -> {target_replicas} ({reason})")
                    return True
                else:
                    logger.error("手动伸缩操作失败")
                    return False

            except Exception as e:
                logger.error(f"手动伸缩异常: {e}")
                return False

    def update_metrics(self, metrics: Dict[str, Any]):
        """更新指标缓存"""
        with self._lock:
            for key, value in metrics.items():
                if key not in self._metrics_cache:
                    self._metrics_cache[key] = []
                self._metrics_cache[key].append(value)

                # 保持最近10个值
                if len(self._metrics_cache[key]) > 10:
                    self._metrics_cache[key].pop(0)

    def get_average_metric(self, metric_name: str, window_size: int = 5) -> Optional[float]:
        """获取指标平均值"""
        with self._lock:
            if metric_name not in self._metrics_cache:
                return None

            values = self._metrics_cache[metric_name][-window_size:]
            return sum(values) / len(values) if values else None

    def reset_statistics(self):
        """重置统计信息"""
        with self._lock:
            self._scale_up_count = 0
            self._scale_down_count = 0
            self._scaling_history.clear()
            logger.info("伸缩统计信息已重置")

    def validate_scaling_config(self) -> Dict[str, Any]:
        """验证伸缩配置"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }

        # 检查基本配置
        if self.config.min_replicas < 0:
            validation_result["valid"] = False
            validation_result["errors"].append("最小副本数不能为负数")

        if self.config.max_replicas <= 0:
            validation_result["valid"] = False
            validation_result["errors"].append("最大副本数必须大于0")

        if self.config.min_replicas >= self.config.max_replicas:
            validation_result["valid"] = False
            validation_result["errors"].append("最小副本数不能大于或等于最大副本数")

        if self._current_replicas < self.config.min_replicas or \
           self._current_replicas > self.config.max_replicas:
            validation_result["valid"] = False
            validation_result["errors"].append("当前副本数超出允许范围")

        # 检查阈值设置
        if self.config.scale_down_threshold >= self.config.scale_up_threshold:
            validation_result["valid"] = False
            validation_result["errors"].append("缩容阈值应该低于扩容阈值")

        # 推荐配置
        if self.config.cooldown_period_seconds < 60:
            validation_result["recommendations"].append("冷却期建议至少60秒")

        if not self.config.custom_metrics and self.config.scaling_policy == ScalingPolicy.CUSTOM_METRIC:
            validation_result["warnings"].append("选择了自定义指标策略但未配置自定义指标")

        return validation_result




