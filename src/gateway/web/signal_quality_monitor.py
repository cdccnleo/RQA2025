"""
信号质量监控系统

本模块实现信号延迟检测和信号去重机制，满足量化交易系统合规要求：
- QTS-013: 信号延迟检测
- QTS-014: 信号去重机制

功能特性：
- 信号延迟检测与告警
- 信号去重与合并
- 信号质量评估
- 数据源延迟监控
- 信号完整性校验
"""

import hashlib
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
import uuid

from .audit_logger import get_audit_logger, AuditCategory, AuditLevel
from .alert_center import get_alert_center, AlertSeverity


class SignalStatus(Enum):
    """信号状态"""
    PENDING = "pending"           # 待处理
    PROCESSING = "processing"     # 处理中
    PROCESSED = "processed"       # 已处理
    DUPLICATE = "duplicate"       # 重复信号
    DELAYED = "delayed"           # 延迟信号
    EXPIRED = "expired"           # 已过期
    REJECTED = "rejected"         # 已拒绝


@dataclass
class TradingSignal:
    """交易信号"""
    signal_id: str
    strategy_id: str
    symbol: str
    direction: str  # BUY, SELL, HOLD
    timestamp: datetime
    source_timestamp: datetime  # 数据源生成时间
    receive_timestamp: datetime  # 系统接收时间
    price: Optional[float] = None
    volume: Optional[float] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: SignalStatus = SignalStatus.PENDING
    latency_ms: Optional[float] = None
    duplicate_of: Optional[str] = None
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
        if self.latency_ms is None:
            self.latency_ms = (self.receive_timestamp - self.source_timestamp).total_seconds() * 1000
    
    def _calculate_checksum(self) -> str:
        """计算信号校验和"""
        data = f"{self.strategy_id}{self.symbol}{self.direction}{self.timestamp.isoformat()}{self.price}{self.volume}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'signal_id': self.signal_id,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'timestamp': self.timestamp.isoformat(),
            'source_timestamp': self.source_timestamp.isoformat(),
            'receive_timestamp': self.receive_timestamp.isoformat(),
            'price': self.price,
            'volume': self.volume,
            'confidence': self.confidence,
            'status': self.status.value,
            'latency_ms': self.latency_ms,
            'duplicate_of': self.duplicate_of,
            'checksum': self.checksum
        }


@dataclass
class LatencyMetrics:
    """延迟指标"""
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_signals: int = 0
    delayed_signals: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class SignalLatencyMonitor:
    """
    信号延迟监控器
    
    监控信号从生成到处理的延迟，检测延迟异常
    """
    
    DEFAULT_LATENCY_THRESHOLD_MS = 1000  # 默认延迟阈值1秒
    DEFAULT_WINDOW_SIZE = 100  # 滑动窗口大小
    
    def __init__(
        self,
        latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
        window_size: int = DEFAULT_WINDOW_SIZE
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.window_size = window_size
        
        # 延迟历史记录 (按策略)
        self._latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._history_lock = threading.RLock()
        
        # 延迟指标 (按策略)
        self._metrics: Dict[str, LatencyMetrics] = defaultdict(LatencyMetrics)
        self._metrics_lock = threading.RLock()
        
        # 数据源延迟 (按数据源)
        self._source_latency: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # 回调函数
        self._callbacks: List[Callable[[str, TradingSignal, float], None]] = []
        
        # 审计日志
        self._audit = get_audit_logger()
        self._alert = get_alert_center()
    
    def record_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        记录信号延迟
        
        Args:
            signal: 交易信号
        
        Returns:
            延迟分析结果
        """
        latency_ms = signal.latency_ms or 0
        strategy_id = signal.strategy_id
        
        with self._history_lock:
            self._latency_history[strategy_id].append(latency_ms)
        
        # 更新指标
        self._update_metrics(strategy_id, latency_ms)
        
        # 检查是否延迟
        is_delayed = latency_ms > self.latency_threshold_ms
        severity = None
        
        if is_delayed:
            signal.status = SignalStatus.DELAYED
            
            # 确定严重程度
            if latency_ms > self.latency_threshold_ms * 5:
                severity = AlertSeverity.CRITICAL
            elif latency_ms > self.latency_threshold_ms * 2:
                severity = AlertSeverity.HIGH
            else:
                severity = AlertSeverity.MEDIUM
            
            # 触发回调
            for callback in self._callbacks:
                try:
                    callback(strategy_id, signal, latency_ms)
                except Exception as e:
                    self._audit.log(
                        level=AuditLevel.ERROR,
                        category=AuditCategory.SIGNAL_PROCESSING,
                        action="latency_callback_error",
                        message=f"延迟回调错误: {e}",
                        strategy_id=strategy_id
                    )
            
            # 记录审计日志
            self._audit.log(
                level=AuditLevel.WARNING,
                category=AuditCategory.SIGNAL_PROCESSING,
                action="signal_delayed",
                message=f"信号延迟: {latency_ms:.2f}ms",
                strategy_id=strategy_id,
                details={
                    'signal_id': signal.signal_id,
                    'symbol': signal.symbol,
                    'latency_ms': latency_ms,
                    'threshold_ms': self.latency_threshold_ms
                }
            )
            
            # 创建告警
            self._alert.create_alert(
                strategy_id=strategy_id,
                title=f"信号延迟告警 - {signal.symbol}",
                message=f"信号处理延迟 {latency_ms:.2f}ms，超过阈值 {self.latency_threshold_ms}ms",
                severity=severity,
                source="SignalLatencyMonitor",
                metadata={
                    'signal_id': signal.signal_id,
                    'latency_ms': latency_ms,
                    'threshold_ms': self.latency_threshold_ms
                }
            )
        
        return {
            'signal_id': signal.signal_id,
            'latency_ms': latency_ms,
            'is_delayed': is_delayed,
            'threshold_ms': self.latency_threshold_ms,
            'severity': severity.value if severity else None
        }
    
    def _update_metrics(self, strategy_id: str, latency_ms: float):
        """更新延迟指标"""
        with self._metrics_lock:
            metrics = self._metrics[strategy_id]
            
            with self._history_lock:
                history = list(self._latency_history[strategy_id])
            
            if not history:
                return
            
            # 计算统计指标
            sorted_history = sorted(history)
            n = len(sorted_history)
            
            metrics.avg_latency_ms = sum(history) / n
            metrics.max_latency_ms = max(history)
            metrics.min_latency_ms = min(history)
            metrics.p50_latency_ms = sorted_history[int(n * 0.5)]
            metrics.p95_latency_ms = sorted_history[int(n * 0.95)] if n >= 20 else sorted_history[-1]
            metrics.p99_latency_ms = sorted_history[int(n * 0.99)] if n >= 100 else sorted_history[-1]
            metrics.total_signals += 1
            
            if latency_ms > self.latency_threshold_ms:
                metrics.delayed_signals += 1
            
            metrics.last_updated = datetime.now()
    
    def get_metrics(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取延迟指标
        
        Args:
            strategy_id: 策略ID，None则返回全局指标
        
        Returns:
            延迟指标
        """
        with self._metrics_lock:
            if strategy_id:
                metrics = self._metrics.get(strategy_id)
                if not metrics:
                    return {}
                return {
                    'strategy_id': strategy_id,
                    'avg_latency_ms': metrics.avg_latency_ms,
                    'max_latency_ms': metrics.max_latency_ms,
                    'min_latency_ms': metrics.min_latency_ms if metrics.min_latency_ms != float('inf') else 0,
                    'p50_latency_ms': metrics.p50_latency_ms,
                    'p95_latency_ms': metrics.p95_latency_ms,
                    'p99_latency_ms': metrics.p99_latency_ms,
                    'total_signals': metrics.total_signals,
                    'delayed_signals': metrics.delayed_signals,
                    'delayed_ratio': metrics.delayed_signals / max(metrics.total_signals, 1),
                    'last_updated': metrics.last_updated.isoformat()
                }
            else:
                # 全局指标
                all_metrics = list(self._metrics.values())
                if not all_metrics:
                    return {}
                
                total_signals = sum(m.total_signals for m in all_metrics)
                total_delayed = sum(m.delayed_signals for m in all_metrics)
                
                return {
                    'global': True,
                    'avg_latency_ms': sum(m.avg_latency_ms for m in all_metrics) / len(all_metrics),
                    'max_latency_ms': max(m.max_latency_ms for m in all_metrics),
                    'min_latency_ms': min(m.min_latency_ms for m in all_metrics),
                    'total_signals': total_signals,
                    'delayed_signals': total_delayed,
                    'delayed_ratio': total_delayed / max(total_signals, 1),
                    'strategy_count': len(all_metrics)
                }
    
    def record_source_latency(self, source: str, latency_ms: float):
        """记录数据源延迟"""
        self._source_latency[source].append({
            'timestamp': datetime.now(),
            'latency_ms': latency_ms
        })
    
    def get_source_latency(self, source: Optional[str] = None) -> Dict[str, Any]:
        """获取数据源延迟"""
        if source:
            history = list(self._source_latency.get(source, []))
            if not history:
                return {}
            
            latencies = [h['latency_ms'] for h in history]
            return {
                'source': source,
                'avg_latency_ms': sum(latencies) / len(latencies),
                'max_latency_ms': max(latencies),
                'min_latency_ms': min(latencies),
                'record_count': len(latencies)
            }
        else:
            return {
                source: self.get_source_latency(source)
                for source in self._source_latency.keys()
            }
    
    def add_callback(self, callback: Callable[[str, TradingSignal, float], None]):
        """添加延迟回调"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[str, TradingSignal, float], None]):
        """移除延迟回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def set_threshold(self, threshold_ms: float):
        """设置延迟阈值"""
        self.latency_threshold_ms = threshold_ms


class SignalDeduplicator:
    """
    信号去重器
    
    检测并处理重复信号，支持多种去重策略
    """
    
    DEFAULT_DEDUP_WINDOW_SECONDS = 60  # 默认去重窗口60秒
    
    def __init__(
        self,
        dedup_window_seconds: float = DEFAULT_DEDUP_WINDOW_SECONDS
    ):
        self.dedup_window_seconds = dedup_window_seconds
        
        # 信号缓存 (checksum -> signal)
        self._signal_cache: Dict[str, TradingSignal] = {}
        self._cache_lock = threading.RLock()
        
        # 策略最近信号时间
        self._last_signal_time: Dict[str, datetime] = {}
        
        # 去重统计
        self._dedup_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'total': 0,
            'duplicates': 0,
            'merged': 0
        })
        self._stats_lock = threading.Lock()
        
        # 审计日志
        self._audit = get_audit_logger()
    
    def check_duplicate(self, signal: TradingSignal) -> Tuple[bool, Optional[str]]:
        """
        检查信号是否重复
        
        Args:
            signal: 交易信号
        
        Returns:
            (是否重复, 原信号ID)
        """
        with self._cache_lock:
            # 清理过期缓存
            self._cleanup_cache()
            
            # 检查完全重复 (相同checksum)
            if signal.checksum in self._signal_cache:
                original = self._signal_cache[signal.checksum]
                time_diff = (signal.timestamp - original.timestamp).total_seconds()
                
                if time_diff < self.dedup_window_seconds:
                    return True, original.signal_id
            
            # 检查策略+标的+方向的重复
            for cached_signal in self._signal_cache.values():
                if (cached_signal.strategy_id == signal.strategy_id and
                    cached_signal.symbol == signal.symbol and
                    cached_signal.direction == signal.direction):
                    
                    time_diff = (signal.timestamp - cached_signal.timestamp).total_seconds()
                    
                    if time_diff < self.dedup_window_seconds:
                        return True, cached_signal.signal_id
            
            return False, None
    
    def add_signal(self, signal: TradingSignal) -> TradingSignal:
        """
        添加信号到去重器
        
        Args:
            signal: 交易信号
        
        Returns:
            处理后的信号
        """
        is_duplicate, original_id = self.check_duplicate(signal)
        
        with self._stats_lock:
            self._dedup_stats[signal.strategy_id]['total'] += 1
        
        if is_duplicate:
            signal.status = SignalStatus.DUPLICATE
            signal.duplicate_of = original_id
            
            with self._stats_lock:
                self._dedup_stats[signal.strategy_id]['duplicates'] += 1
            
            # 记录审计日志
            self._audit.log(
                level=AuditLevel.INFO,
                category=AuditCategory.SIGNAL_PROCESSING,
                action="signal_duplicate_detected",
                message=f"检测到重复信号",
                strategy_id=signal.strategy_id,
                details={
                    'signal_id': signal.signal_id,
                    'duplicate_of': original_id,
                    'symbol': signal.symbol,
                    'direction': signal.direction
                }
            )
        else:
            # 添加到缓存
            with self._cache_lock:
                self._signal_cache[signal.checksum] = signal
            
            self._last_signal_time[signal.strategy_id] = signal.timestamp
        
        return signal
    
    def merge_signals(
        self,
        signals: List[TradingSignal],
        merge_window_seconds: float = 5.0
    ) -> List[TradingSignal]:
        """
        合并时间窗口内的相似信号
        
        Args:
            signals: 信号列表
            merge_window_seconds: 合并时间窗口
        
        Returns:
            合并后的信号列表
        """
        if not signals:
            return []
        
        # 按策略+标的+方向分组
        groups: Dict[str, List[TradingSignal]] = defaultdict(list)
        for signal in signals:
            key = f"{signal.strategy_id}:{signal.symbol}:{signal.direction}"
            groups[key].append(signal)
        
        merged_signals = []
        
        for key, group in groups.items():
            if len(group) == 1:
                merged_signals.append(group[0])
                continue
            
            # 按时间排序
            group.sort(key=lambda s: s.timestamp)
            
            # 合并时间窗口内的信号
            current_batch = [group[0]]
            
            for signal in group[1:]:
                time_diff = (signal.timestamp - current_batch[0].timestamp).total_seconds()
                
                if time_diff <= merge_window_seconds:
                    current_batch.append(signal)
                else:
                    # 合并当前批次
                    if len(current_batch) > 1:
                        merged = self._merge_batch(current_batch)
                        merged_signals.append(merged)
                        
                        with self._stats_lock:
                            self._dedup_stats[current_batch[0].strategy_id]['merged'] += len(current_batch) - 1
                    else:
                        merged_signals.extend(current_batch)
                    
                    current_batch = [signal]
            
            # 处理最后一批
            if len(current_batch) > 1:
                merged = self._merge_batch(current_batch)
                merged_signals.append(merged)
                
                with self._stats_lock:
                    self._dedup_stats[current_batch[0].strategy_id]['merged'] += len(current_batch) - 1
            else:
                merged_signals.extend(current_batch)
        
        return merged_signals
    
    def _merge_batch(self, signals: List[TradingSignal]) -> TradingSignal:
        """合并一批信号"""
        if not signals:
            raise ValueError("Cannot merge empty signal list")
        
        if len(signals) == 1:
            return signals[0]
        
        # 使用第一个信号作为基础
        base = signals[0]
        
        # 计算平均价格
        prices = [s.price for s in signals if s.price is not None]
        avg_price = sum(prices) / len(prices) if prices else None
        
        # 计算总成交量
        volumes = [s.volume for s in signals if s.volume is not None]
        total_volume = sum(volumes) if volumes else None
        
        # 计算平均置信度
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # 创建合并后的信号
        merged = TradingSignal(
            signal_id=str(uuid.uuid4()),
            strategy_id=base.strategy_id,
            symbol=base.symbol,
            direction=base.direction,
            timestamp=base.timestamp,
            source_timestamp=base.source_timestamp,
            receive_timestamp=signals[-1].receive_timestamp,
            price=avg_price,
            volume=total_volume,
            confidence=avg_confidence,
            metadata={
                'merged': True,
                'merged_count': len(signals),
                'merged_signals': [s.signal_id for s in signals],
                'original_checksums': [s.checksum for s in signals]
            }
        )
        
        # 记录审计日志
        self._audit.log(
            level=AuditLevel.INFO,
            category=AuditCategory.SIGNAL_PROCESSING,
            action="signals_merged",
            message=f"合并了 {len(signals)} 个信号",
            strategy_id=base.strategy_id,
            details={
                'merged_signal_id': merged.signal_id,
                'original_signals': [s.signal_id for s in signals],
                'symbol': base.symbol,
                'direction': base.direction
            }
        )
        
        return merged
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        cutoff = datetime.now() - timedelta(seconds=self.dedup_window_seconds * 2)
        
        to_remove = [
            checksum for checksum, signal in self._signal_cache.items()
            if signal.timestamp < cutoff
        ]
        
        for checksum in to_remove:
            del self._signal_cache[checksum]
    
    def get_stats(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取去重统计
        
        Args:
            strategy_id: 策略ID，None则返回全局统计
        
        Returns:
            统计信息
        """
        with self._stats_lock:
            if strategy_id:
                stats = self._dedup_stats.get(strategy_id, {'total': 0, 'duplicates': 0, 'merged': 0})
                total = stats['total']
                return {
                    'strategy_id': strategy_id,
                    'total_signals': total,
                    'duplicate_signals': stats['duplicates'],
                    'merged_signals': stats['merged'],
                    'duplicate_ratio': stats['duplicates'] / max(total, 1),
                    'effective_signals': total - stats['duplicates'] - stats['merged']
                }
            else:
                # 全局统计
                total = sum(s['total'] for s in self._dedup_stats.values())
                duplicates = sum(s['duplicates'] for s in self._dedup_stats.values())
                merged = sum(s['merged'] for s in self._dedup_stats.values())
                
                return {
                    'global': True,
                    'total_signals': total,
                    'duplicate_signals': duplicates,
                    'merged_signals': merged,
                    'duplicate_ratio': duplicates / max(total, 1),
                    'effective_signals': total - duplicates - merged,
                    'strategy_count': len(self._dedup_stats)
                }
    
    def clear_cache(self):
        """清空缓存"""
        with self._cache_lock:
            self._signal_cache.clear()
        self._last_signal_time.clear()


class SignalQualityMonitor:
    """
    信号质量监控器
    
    整合延迟检测和去重功能，提供统一的信号质量监控
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        latency_threshold_ms: float = 1000,
        dedup_window_seconds: float = 60
    ):
        if self._initialized:
            return
        
        self._initialized = True
        
        # 子监控器
        self.latency_monitor = SignalLatencyMonitor(latency_threshold_ms)
        self.deduplicator = SignalDeduplicator(dedup_window_seconds)
        
        # 信号处理器链
        self._processors: List[Callable[[TradingSignal], TradingSignal]] = []
        
        # 审计日志
        self._audit = get_audit_logger()
        self._alert = get_alert_center()
    
    def process_signal(self, signal: TradingSignal) -> TradingSignal:
        """
        处理信号（延迟检测 + 去重）
        
        Args:
            signal: 原始信号
        
        Returns:
            处理后的信号
        """
        # 1. 延迟检测
        latency_result = self.latency_monitor.record_signal(signal)
        
        # 2. 去重检查
        signal = self.deduplicator.add_signal(signal)
        
        # 3. 执行其他处理器
        for processor in self._processors:
            try:
                signal = processor(signal)
            except Exception as e:
                self._audit.log(
                    level=AuditLevel.ERROR,
                    category=AuditCategory.SIGNAL_PROCESSING,
                    action="signal_processor_error",
                    message=f"信号处理器错误: {e}",
                    strategy_id=signal.strategy_id,
                    error_message=str(e)
                )
        
        return signal
    
    def process_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        批量处理信号
        
        Args:
            signals: 信号列表
        
        Returns:
            处理后的信号列表
        """
        processed = []
        
        for signal in signals:
            processed_signal = self.process_signal(signal)
            if processed_signal.status != SignalStatus.DUPLICATE:
                processed.append(processed_signal)
        
        # 合并相似信号
        merged = self.deduplicator.merge_signals(processed)
        
        return merged
    
    def add_processor(self, processor: Callable[[TradingSignal], TradingSignal]):
        """添加信号处理器"""
        self._processors.append(processor)
    
    def remove_processor(self, processor: Callable[[TradingSignal], TradingSignal]):
        """移除信号处理器"""
        if processor in self._processors:
            self._processors.remove(processor)
    
    def get_quality_report(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取信号质量报告
        
        Args:
            strategy_id: 策略ID，None则返回全局报告
        
        Returns:
            质量报告
        """
        latency_metrics = self.latency_monitor.get_metrics(strategy_id)
        dedup_stats = self.deduplicator.get_stats(strategy_id)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'strategy_id': strategy_id,
            'latency': latency_metrics,
            'deduplication': dedup_stats,
            'quality_score': self._calculate_quality_score(latency_metrics, dedup_stats)
        }
    
    def _calculate_quality_score(
        self,
        latency_metrics: Dict[str, Any],
        dedup_stats: Dict[str, Any]
    ) -> float:
        """计算信号质量评分 (0-100)"""
        score = 100.0
        
        # 延迟扣分
        if latency_metrics:
            delayed_ratio = latency_metrics.get('delayed_ratio', 0)
            score -= delayed_ratio * 30  # 最多扣30分
        
        # 重复率扣分
        if dedup_stats:
            duplicate_ratio = dedup_stats.get('duplicate_ratio', 0)
            score -= duplicate_ratio * 20  # 最多扣20分
        
        return max(0, min(100, score))
    
    def validate_signal_integrity(self, signal: TradingSignal) -> bool:
        """
        验证信号完整性
        
        Args:
            signal: 交易信号
        
        Returns:
            是否通过验证
        """
        # 验证必填字段
        if not all([
            signal.signal_id,
            signal.strategy_id,
            signal.symbol,
            signal.direction,
            signal.timestamp,
            signal.source_timestamp,
            signal.receive_timestamp
        ]):
            return False
        
        # 验证时间顺序
        if signal.source_timestamp > signal.receive_timestamp:
            return False
        
        if signal.timestamp < signal.source_timestamp:
            return False
        
        # 验证校验和
        expected_checksum = signal._calculate_checksum()
        if signal.checksum and signal.checksum != expected_checksum:
            return False
        
        return True


# 全局实例
_quality_monitor: Optional[SignalQualityMonitor] = None


def get_signal_quality_monitor() -> SignalQualityMonitor:
    """获取全局信号质量监控器实例"""
    global _quality_monitor
    if _quality_monitor is None:
        _quality_monitor = SignalQualityMonitor()
    return _quality_monitor
