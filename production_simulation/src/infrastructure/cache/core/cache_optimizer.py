"""
缓存优化器

提供缓存性能优化和策略调整功能
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
logger = logging.getLogger(__name__)

# 常量定义
CACHE_HIT_RATE_WARNING = 0.7
CACHE_HIT_RATE_CRITICAL = 0.5
CACHE_RESPONSE_TIME_WARNING = 100.0  # ms
CACHE_RESPONSE_TIME_CRITICAL = 500.0  # ms
CACHE_MEMORY_USAGE_WARNING = 0.8
CACHE_MEMORY_USAGE_CRITICAL = 0.9
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL = 300
MAX_CACHE_SIZE = 10000
MIN_CACHE_SIZE = 100

class CachePolicy(Enum):
    """缓存策略枚举"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"


class CacheOptimizer:
    """
    缓存优化器

    负责分析缓存性能并提供优化建议
    """

    def __init__(self, policy: CachePolicy = CachePolicy.LRU):
        self.policy = policy
        self.metrics_history: List[Dict[str, Any]] = []
        self._optimization_history: List[Dict[str, Any]] = []
        self._recommendation_cache: Dict[str, Any] = {}  # 缓存优化建议
        self._size_history: List[int] = []

    def analyze_performance(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析缓存性能"""
        self.metrics_history.append(current_metrics)

        analysis = {
            'hit_rate': current_metrics.get('hit_rate', 0.0),
            'avg_response_time': current_metrics.get('avg_response_time', 0.0),
            'recommendations': self._generate_recommendations(current_metrics)
        }

        return analysis

    def optimize_strategy(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """优化缓存策略"""
        optimized_config = current_config.copy()

        if self.policy == CachePolicy.LFU:
            optimized_config['eviction_policy'] = 'LFU'
        elif self.policy == CachePolicy.FIFO:
            optimized_config['eviction_policy'] = 'FIFO'
        elif self.policy == CachePolicy.RANDOM:
            optimized_config['eviction_policy'] = 'RANDOM'
        else:
            optimized_config['eviction_policy'] = 'LRU'

        return optimized_config

    def predict_future_performance(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """预测未来性能"""
        if not historical_data:
            return {'predicted_hit_rate': 0.5, 'confidence': 0.0}

        avg_hit_rate = sum(d.get('hit_rate', 0) for d in historical_data) / len(historical_data)

        return {
            'predicted_hit_rate': avg_hit_rate,
            'confidence': 0.8,
            'trend': 'stable'
        }

    def _generate_recommendations(self, metrics: Optional[Dict[str, Any]] = None) -> List[str]:
        """生成优化建议"""
        metrics = metrics or (self.metrics_history[-1] if self.metrics_history else {})
        recommendations: List[str] = []

        hit_rate = metrics.get('hit_rate', 0.0)
        if hit_rate < 0.5:
            recommendations.append("考虑增加缓存大小以提高命中率")
        elif hit_rate > 0.9:
            recommendations.append("缓存性能良好，可以考虑减小缓存大小以节省内存")

        response_time = metrics.get('avg_response_time', 0.0)
        if response_time > 100:
            recommendations.append("响应时间较长，考虑优化缓存策略")

        return recommendations if recommendations else ["缓存性能正常"]

    def optimize_cache_size(
        self,
        current_size: int,
        metrics_or_hit_rate: Any = None,
        memory_usage: Optional[float] = None,
        **kwargs,
    ) -> int:
        """优化缓存大小

        支持传入 metrics 字典或(hit_rate, memory_usage)形式的参数，
        以保持与历史测试用例的兼容性。
        """

        if isinstance(metrics_or_hit_rate, dict):
            metrics = metrics_or_hit_rate.copy()
        else:
            metrics = {}
            if metrics_or_hit_rate is not None:
                metrics['hit_rate'] = metrics_or_hit_rate
            if memory_usage is not None:
                metrics['memory_usage'] = memory_usage

        for key in ('hit_rate', 'memory_usage'):
            if key in kwargs and kwargs[key] is not None:
                metrics[key] = kwargs[key]

        hit_rate = metrics['hit_rate'] if 'hit_rate' in metrics else 0.5
        memory_usage_value = metrics['memory_usage'] if 'memory_usage' in metrics else 0.5

        new_size = current_size
        reason = "maintain"

        if hit_rate <= 0.0 and memory_usage_value >= 1.0:
            new_size = MIN_CACHE_SIZE
            reason = "critical_underutilization"
        elif hit_rate >= 1.0 and memory_usage_value <= 0.0:
            new_size = MAX_CACHE_SIZE
            reason = "excellent_with_capacity"
        elif hit_rate < CACHE_HIT_RATE_CRITICAL and memory_usage_value >= CACHE_MEMORY_USAGE_WARNING:
            new_size = max(current_size // 2, MIN_CACHE_SIZE)
            reason = "low_hit_high_memory"
        elif hit_rate < CACHE_HIT_RATE_CRITICAL:
            new_size = min(current_size * 2, MAX_CACHE_SIZE)
            reason = "low_hit_rate"
        elif hit_rate >= 0.9 and memory_usage_value < CACHE_MEMORY_USAGE_WARNING:
            new_size = min(current_size * 2, MAX_CACHE_SIZE)
            reason = "high_hit_low_memory"
        elif hit_rate >= 0.9 and memory_usage_value > CACHE_MEMORY_USAGE_WARNING:
            new_size = max(current_size // 2, MIN_CACHE_SIZE)
            reason = "high_hit_high_memory"

        record = {
            'action': 'cache_size_optimization',
            'old_size': current_size,
            'new_size': new_size,
            'current_size': current_size,
            'target_size': new_size,
            'metrics': metrics,
            'hit_rate': hit_rate,
            'memory_usage': memory_usage_value,
            'reason': reason,
            'timestamp': time.time()
        }
        self._optimization_history.append(record)
        self._size_history.append(new_size)

        if reason != "maintain":
            logger.info(
                "Cache size optimization applied: %s -> %s (%s)",
                current_size,
                new_size,
                reason,
            )

        return new_size

    def suggest_eviction_policy(self, access_patterns: Dict[str, Any]) -> CachePolicy:
        """建议驱逐策略"""
        if not access_patterns:
            return CachePolicy.LRU

        if 'random_access' in access_patterns or 'sequential_access' in access_patterns:
            random_access = access_patterns.get('random_access', 0)
            sequential_access = access_patterns.get('sequential_access', 0)
            if random_access > sequential_access:
                return CachePolicy.LFU
            return CachePolicy.LRU

        if 'frequent_access' in access_patterns:
            return CachePolicy.LFU

        frequencies = access_patterns.get('access_frequencies')
        if frequencies is None:
            numeric_items = {k: v for k, v in access_patterns.items() if isinstance(v, (int, float))}
            frequencies = numeric_items if numeric_items else {}
        if not frequencies:
            return CachePolicy.LRU

        avg_freq = sum(frequencies.values()) / len(frequencies)
        variance = sum((freq - avg_freq) ** 2 for freq in frequencies.values()) / len(frequencies)

        if variance > 1000:  # 高方差
            return CachePolicy.LFU
        else:
            return CachePolicy.LRU

    def get_cache_recommendations(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """获取缓存建议"""
        recommendations: List[str] = []
        warnings: List[str] = []
        performance_improvements: List[str] = []

        hit_rate = metrics.get('hit_rate', 0.5)
        if hit_rate < CACHE_HIT_RATE_WARNING:
            warnings.append("缓存命中率偏低，建议增加缓存容量")
        elif hit_rate > 0.95:
            performance_improvements.append("缓存命中率优秀，可探索智能预取策略")

        response_time = metrics.get('avg_response_time', 0.0)
        if response_time > CACHE_RESPONSE_TIME_WARNING:
            warnings.append("平均响应时间较长，建议优化缓存策略")

        memory_usage = metrics.get('memory_usage', 0.0)
        if memory_usage > CACHE_MEMORY_USAGE_WARNING:
            warnings.append("内存使用率较高，建议调整缓存大小")
        elif memory_usage < 0.5 and hit_rate > 0.9:
            performance_improvements.append("内存充裕且命中率高，可以评估缩减缓存或启用压缩")

        size = metrics.get('size')
        size_suggestion = 'maintain'
        if hit_rate < CACHE_HIT_RATE_CRITICAL:
            size_suggestion = 'increase'
        elif hit_rate > 0.9 and memory_usage < 0.5:
            size_suggestion = 'decrease'

        policy = metrics.get('policy')
        access_pattern = metrics.get('access_pattern', {})
        recommended_policy = None
        if access_pattern:
            recommended_policy = self.suggest_eviction_policy(access_pattern)
        elif isinstance(policy, str):
            try:
                recommended_policy = CachePolicy(policy.lower())
            except ValueError:
                recommended_policy = CachePolicy.LRU
        else:
            recommended_policy = CachePolicy.LRU

        if isinstance(recommended_policy, CachePolicy):
            performance_improvements.append(
                f"建议评估使用 {recommended_policy.value.upper()} 策略优化缓存驱逐")

        return {
            'recommendations': recommendations,
            'warnings': warnings,
            'performance_improvements': performance_improvements,
            'size_optimization': {
                'current_size': size,
                'suggestion': size_suggestion,
            },
            'hit_rate': hit_rate,
            'memory_usage': memory_usage,
            'policy_recommendation': recommended_policy.value if isinstance(recommended_policy, CachePolicy) else recommended_policy,
        }

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return self._optimization_history.copy()

    def clear_optimization_history(self):
        """清除优化历史"""
        self._optimization_history.clear()

    def analyze_access_patterns(self, access_logs: Any) -> Dict[str, Any]:
        """分析访问模式"""
        if not access_logs:
            return {
                'pattern_type': 'unknown',
                'frequency_analysis': {},
                'total_accesses': 0,
                'unique_keys': 0,
                'read_write_ratio': 0.0,
                'hit_rate': 0.0,
                'access_pattern_type': 'unknown',
                'recommendations': [],
                'avg_access_per_key': 0.0,
            }

        if isinstance(access_logs, dict):
            read_ops = int(access_logs.get('read_operations', 0))
            write_ops = int(access_logs.get('write_operations', 0))
            hits = int(access_logs.get('cache_hits', 0))
            misses = int(access_logs.get('cache_misses', 0))
            sequential = int(access_logs.get('sequential_access', 0))
            random_access = int(access_logs.get('random_access', 0))
            total = read_ops + write_ops
            read_write_ratio = read_ops / total if total else 0.0
            total_requests = hits + misses
            hit_rate = hits / total_requests if total_requests else 0.0
            pattern_type = 'sequential' if sequential >= random_access else 'random'
            recommendations: List[str] = []
            if hit_rate < CACHE_HIT_RATE_WARNING:
                recommendations.append("缓存命中率偏低，建议调整缓存策略")
            if write_ops > read_ops:
                recommendations.append("写操作较多，考虑优化写入策略")
            frequencies = {str(k): int(v) for k, v in access_logs.items() if isinstance(v, (int, float))}
            avg_access_per_key = (sum(frequencies.values()) / len(frequencies)) if frequencies else 0.0
            return {
                'pattern_type': 'analyzed',
                'frequency_analysis': frequencies,
                'total_accesses': sum(frequencies.values()),
                'unique_keys': len(frequencies),
                'read_write_ratio': read_write_ratio,
                'hit_rate': hit_rate,
                'access_pattern_type': pattern_type,
                'recommendations': recommendations,
                'avg_access_per_key': avg_access_per_key,
            }

        frequencies: Dict[str, int] = {}
        for log in access_logs:
            if isinstance(log, dict):
                key = log.get('key', 'unknown')
            else:
                key = str(log)
            frequencies[key] = frequencies.get(key, 0) + 1

        total_accesses = sum(frequencies.values())
        avg_access_per_key = total_accesses / len(frequencies) if frequencies else 0.0

        return {
            'pattern_type': 'analyzed',
            'frequency_analysis': frequencies,
            'total_accesses': total_accesses,
            'unique_keys': len(frequencies),
            'read_write_ratio': 0.0,
            'hit_rate': 0.0,
            'access_pattern_type': 'unknown',
            'recommendations': [],
            'avg_access_per_key': avg_access_per_key,
        }

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """获取优化建议"""
        if not self._optimization_history:
            return {
                'size_trend_analysis': {
                    'trend': 'stable',
                    'history': []
                },
                'hit_rate_trend_analysis': {
                    'trend': 'unknown',
                    'history': []
                },
                'overall_recommendations': [],
                'recommendations': []
            }

        size_analysis = self._analyze_size_trend()
        hit_rate_analysis = self._analyze_hit_rate_trend()
        latest_metrics = self.metrics_history[-1] if self.metrics_history else {}
        overall = self._generate_recommendations(latest_metrics)
        reason = self._get_optimization_reason(
            hit_rate_analysis.get('latest_hit_rate', latest_metrics.get('hit_rate', 0.0)),
            hit_rate_analysis.get('latest_memory_usage', latest_metrics.get('memory_usage', 0.5))
        )
        if isinstance(overall, list):
            overall_recommendations = overall + [reason]
        else:
            overall_recommendations = [overall, reason]

        result = {
            'size_trend_analysis': size_analysis,
            'hit_rate_trend_analysis': hit_rate_analysis,
            'overall_recommendations': overall_recommendations,
        }
        result['recommendations'] = overall_recommendations
        return result

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self._optimization_history:
            return {
                'total_optimizations': 0,
                'optimization_count': 0,
                'avg_hit_rate': 0.0,
                'avg_memory_usage': 0.0,
                'optimization_success_rate': 0.0,
                'performance_score': 0.0,
                'avg_improvement': 0.0,
                'last_optimization': None,
                'trend_analysis': 'insufficient_data',
                'policy_effectiveness': 'unknown',
            }

        hit_rates = [item.get('hit_rate', 0.0) for item in self._optimization_history]
        memory_usages = [item.get('memory_usage', 0.0) for item in self._optimization_history]
        improvements = [abs(item['new_size'] - item['old_size']) for item in self._optimization_history]
        total = len(self._optimization_history)
        last = self._optimization_history[-1]

        return {
            'total_optimizations': total,
            'optimization_count': total,
            'avg_hit_rate': sum(hit_rates) / total if hit_rates else 0.0,
            'avg_memory_usage': sum(memory_usages) / total if memory_usages else 0.0,
            'optimization_success_rate': self._calculate_success_rate(),
            'performance_score': self._calculate_performance_score(hit_rates),
            'avg_improvement': sum(improvements) / total if improvements else 0.0,
            'last_optimization': last,
            'trend_analysis': self._analyze_trend(),
            'policy_effectiveness': self._evaluate_policy(),
        }

    def monitor_cache_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """监控缓存性能"""
        self.metrics_history.append(metrics)

        # 简单的监控逻辑
        status = 'normal'
        hit_rate = metrics.get('hit_rate', 0.5)
        response_time = metrics.get('avg_response_time', 0)

        if hit_rate < CACHE_HIT_RATE_CRITICAL or response_time > CACHE_RESPONSE_TIME_CRITICAL:
            status = 'critical'
        elif hit_rate < CACHE_HIT_RATE_WARNING or response_time > CACHE_RESPONSE_TIME_WARNING:
            status = 'warning'

        return {
            'status': status,
            'metrics': metrics,
            'recommendations': self.get_cache_recommendations(metrics)
        }

    def optimize_eviction_policy(self, current_policy: Any, metrics: Optional[Dict[str, Any]] = None) -> str:
        """优化驱逐策略"""
        if isinstance(current_policy, dict) and metrics is None:
            metrics = current_policy
            current_policy = metrics.get('current_policy', 'lru')
        elif metrics is None:
            metrics = {}
        if not isinstance(current_policy, str):
            current_policy = str(current_policy)
        hit_rate = metrics.get('hit_rate', 0.5)
        memory_pressure = metrics.get('memory_pressure', 0.5)
        access_pattern = metrics.get('access_pattern', {})
        random_ratio = 0
        if isinstance(access_pattern, dict) and access_pattern:
            total = sum(access_pattern.values())
            random_ratio = access_pattern.get('random_access', 0) / total if total else 0

        if hit_rate < 0.6 and memory_pressure > 0.8:
            result = 'adaptive'
        elif hit_rate > 0.9:
            result = 'ttl'
        elif random_ratio > 0.5:
            result = 'lfu'
        else:
            result = current_policy
        return str(result).upper()

    def reset_optimization_history(self):
        """重置优化历史"""
        self._optimization_history.clear()
        self.metrics_history.clear()
        self._size_history.clear()

    def _analyze_trend(self) -> str:
        """分析趋势"""
        if len(self.metrics_history) < 2:
            return 'insufficient_data'

        recent = self.metrics_history[-3:]  # 最近3次
        hit_rates = [m.get('hit_rate', 0) for m in recent]

        if hit_rates[-1] > hit_rates[0] + 0.05:
            return 'improving'
        elif hit_rates[-1] < hit_rates[0] - 0.05:
            return 'degrading'
        else:
            return 'stable'

    def _evaluate_policy(self) -> str:
        """评估策略效果"""
        if not self.metrics_history:
            return 'unknown'

        avg_hit_rate = sum(m.get('hit_rate', 0) for m in self.metrics_history) / len(self.metrics_history)

        if avg_hit_rate > 0.8:
            return 'excellent'
        elif avg_hit_rate > 0.6:
            return 'good'
        elif avg_hit_rate > 0.4:
            return 'fair'
        else:
            return 'poor'

    def _analyze_size_trend(self, recommendations: Optional[List[str]] = None, recent_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        history_source = recent_history if recent_history is not None else self._size_history[-5:]
        if history_source and isinstance(history_source[0], dict):
            numeric_history = [
                entry.get('current_size')
                if entry.get('current_size') is not None
                else entry.get('target_size', 0)
                for entry in history_source
            ]
        else:
            numeric_history = history_source
        history = numeric_history[-5:] if len(numeric_history) > 5 else numeric_history
        if len(history) < 2:
            trend = 'stable'
        elif history[-1] > history[0]:
            trend = 'increasing'
        elif history[-1] < history[0]:
            trend = 'decreasing'
        else:
            trend = 'stable'
        message = f"Cache size trend is {trend}."
        if recommendations is not None:
            recommendations.append(message)
        return {
            'trend': trend,
            'history': history,
        }

    def _analyze_hit_rate_trend(self, recommendations: Optional[List[str]] = None, recent_records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        recent = recent_records if recent_records is not None else self._optimization_history[-5:]
        hit_rates = [item.get('hit_rate', 0.0) for item in recent]
        memory_usage = [item.get('memory_usage', 0.0) for item in recent]
        if len(hit_rates) < 2:
            trend = 'stable'
        elif hit_rates[-1] > hit_rates[0]:
            trend = 'improving'
        elif hit_rates[-1] < hit_rates[0]:
            trend = 'degrading'
        else:
            trend = 'stable'
        message = f"Cache hit rate trend is {trend}."
        if recommendations is not None:
            recommendations.append(message)
        return {
            'trend': trend,
            'history': hit_rates,
            'latest_hit_rate': hit_rates[-1] if hit_rates else 0.0,
            'latest_memory_usage': memory_usage[-1] if memory_usage else 0.0,
        }

    def _analyze_cache_size_optimization(self) -> Dict[str, Any]:
        if not self._optimization_history:
            return {
                'total_adjustments': 0,
                'average_change': 0.0,
                'last_adjustment': None,
            }

        changes = [entry.get('new_size', 0) - entry.get('old_size', 0) for entry in self._optimization_history]
        total = len(changes)
        average_change = sum(changes) / total if total else 0.0
        last_entry = self._optimization_history[-1]

        return {
            'total_adjustments': total,
            'average_change': average_change,
            'last_adjustment': last_entry,
        }

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        recent_metrics = self.metrics_history[-5:]
        if not recent_metrics:
            return {
                'trend': 'insufficient_data',
                'hit_rate_trend': 'stable',
                'response_time_trend': 'stable',
                'memory_usage_trend': 'stable',
                'history': []
            }

        def _compute_trend(series: List[float]) -> str:
            if len(series) < 2:
                return 'stable'
            if series[-1] > series[0]:
                return 'increasing'
            if series[-1] < series[0]:
                return 'decreasing'
            return 'stable'

        hit_rates = [metric.get('hit_rate', 0.0) for metric in recent_metrics]
        response_times = [metric.get('avg_response_time', 0.0) for metric in recent_metrics]
        memory_usage = [metric.get('memory_usage', 0.0) for metric in recent_metrics]

        overall_trend = 'stable'
        if any(value > CACHE_MEMORY_USAGE_WARNING for value in memory_usage):
            overall_trend = 'warning'
        if hit_rates and hit_rates[-1] < CACHE_HIT_RATE_CRITICAL:
            overall_trend = 'degrading'

        return {
            'trend': overall_trend,
            'hit_rate_trend': _compute_trend(hit_rates),
            'response_time_trend': _compute_trend(response_times),
            'memory_usage_trend': _compute_trend(memory_usage),
            'history': recent_metrics,
        }

    def _calculate_success_rate(self) -> float:
        if not self._optimization_history:
            return 0.0
        successful = sum(1 for item in self._optimization_history if item.get('new_size') != item.get('old_size'))
        return successful / len(self._optimization_history)

    def _calculate_performance_score(self, hit_rates: List[float]) -> float:
        if not hit_rates:
            return 0.0
        avg_hit = sum(hit_rates) / len(hit_rates)
        return min(1.0, max(0.0, avg_hit))

    def _get_optimization_reason(self, hit_rate: float, memory_usage: float) -> str:
        if hit_rate < CACHE_HIT_RATE_CRITICAL:
            return '命中率偏低，需要扩大缓存或优化策略'
        if memory_usage > CACHE_MEMORY_USAGE_CRITICAL:
            return '内存压力过高，建议回收部分缓存'
        return '缓存命中率与内存占用均处于正常水平'

    def _get_policy_recommendation_reason(self, access_pattern: Dict[str, Any]) -> str:
        random_access = access_pattern.get('random_access', 0)
        sequential_access = access_pattern.get('sequential_access', 0)
        if random_access > sequential_access:
            return '随机访问占比较高，推荐使用LFU策略提升命中率'
        return '顺序访问占比较高，推荐使用LRU策略保持性能'


# 兼容性函数
def handle_cache_exceptions(
    arg: Optional[Any] = None,
    *,
    default_return: Optional[Any] = None,
    log_level: str = "error",
    reraise: bool = False,
):
    """缓存异常处理装饰器，支持多种调用方式。"""

    def _decorate(func: Callable, operation_name: Optional[str]) -> Callable:
        op_name = operation_name or getattr(func, "__name__", "cache_operation")
        fallback = {} if default_return is None else default_return

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - 异常路径
                log_func = getattr(logger, log_level.lower(), logger.error)
                log_func("Cache operation failed during %s: %s", op_name, exc)
                result: Any
                if isinstance(fallback, dict):
                    result = fallback.copy()
                else:
                    result = fallback
                if reraise:
                    raise
                return result

        return wrapper

    if callable(arg):
        return _decorate(arg, None)

    operation_name = None if arg is None else str(arg)

    def decorator(func: Callable) -> Callable:
        return _decorate(func, operation_name)

    return decorator

# 别名兼容性
handle_cache_exception = handle_cache_exceptions
