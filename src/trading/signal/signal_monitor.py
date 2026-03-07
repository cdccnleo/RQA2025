#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号监控器

功能：
- 指标收集
- 告警引擎
- 监控面板API

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class SignalMetrics:
    """信号指标"""
    timestamp: datetime
    total_signals: int = 0
    valid_signals: int = 0
    invalid_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    avg_quality_score: float = 0.0
    avg_risk_score: float = 0.0
    avg_overall_score: float = 0.0


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    condition: str
    threshold: float
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None


class SignalMonitor:
    """
    信号监控器
    
    职责：
    1. 指标收集
    2. 告警引擎
    3. 监控面板API
    """
    
    def __init__(self, max_history: int = 1000):
        """
        初始化信号监控器
        
        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        
        # 信号历史
        self._signal_history: deque = deque(maxlen=max_history)
        
        # 指标历史
        self._metrics_history: deque = deque(maxlen=max_history)
        
        # 告警规则
        self._alert_rules: Dict[str, AlertRule] = {}
        
        # 告警回调
        self._alert_callbacks: List[Callable] = []
        
        # 统计信息
        self._stats = {
            'total_signals': 0,
            'valid_signals': 0,
            'invalid_signals': 0
        }
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 初始化默认告警规则
        self._init_default_rules()
        
        logger.info("信号监控器初始化完成")
    
    def _init_default_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id='low_quality',
                name='低质量信号告警',
                condition='avg_quality_score < 50',
                threshold=50.0
            ),
            AlertRule(
                rule_id='high_risk',
                name='高风险信号告警',
                condition='avg_risk_score > 80',
                threshold=80.0
            ),
            AlertRule(
                rule_id='signal_spike',
                name='信号数量激增告警',
                condition='total_signals > 100',
                threshold=100.0
            )
        ]
        
        for rule in default_rules:
            self._alert_rules[rule.rule_id] = rule
    
    def record_signal(self, signal: Dict[str, Any], validation_result: Optional[Any] = None):
        """
        记录信号
        
        Args:
            signal: 信号数据
            validation_result: 验证结果
        """
        with self._lock:
            # 添加时间戳
            signal_record = {
                **signal,
                'recorded_at': datetime.now()
            }
            
            if validation_result:
                signal_record['validation'] = {
                    'overall_score': validation_result.overall_score,
                    'quality_score': validation_result.quality_score,
                    'risk_score': validation_result.risk_score,
                    'backtest_score': validation_result.backtest_score,
                    'is_valid': validation_result.is_valid
                }
            
            # 添加到历史
            self._signal_history.append(signal_record)
            
            # 更新统计
            self._stats['total_signals'] += 1
            if validation_result:
                if validation_result.is_valid:
                    self._stats['valid_signals'] += 1
                else:
                    self._stats['invalid_signals'] += 1
            
            # 检查告警
            self._check_alerts()
    
    def record_metrics(self, metrics: SignalMetrics):
        """
        记录指标
        
        Args:
            metrics: 信号指标
        """
        with self._lock:
            self._metrics_history.append(metrics)
    
    def calculate_current_metrics(self) -> SignalMetrics:
        """
        计算当前指标
        
        Returns:
            当前信号指标
        """
        with self._lock:
            if not self._signal_history:
                return SignalMetrics(timestamp=datetime.now())
            
            # 只统计最近1小时的信号
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_signals = [
                s for s in self._signal_history
                if s.get('recorded_at', datetime.min) > one_hour_ago
            ]
            
            total = len(recent_signals)
            if total == 0:
                return SignalMetrics(timestamp=datetime.now())
            
            # 统计各类信号
            buy_count = sum(1 for s in recent_signals if s.get('signal_type') == 'buy')
            sell_count = sum(1 for s in recent_signals if s.get('signal_type') == 'sell')
            hold_count = sum(1 for s in recent_signals if s.get('signal_type') == 'hold')
            
            # 统计验证结果
            valid_count = sum(
                1 for s in recent_signals
                if s.get('validation', {}).get('is_valid', False)
            )
            
            # 计算平均分数
            quality_scores = [
                s['validation']['quality_score']
                for s in recent_signals
                if 'validation' in s
            ]
            risk_scores = [
                s['validation']['risk_score']
                for s in recent_signals
                if 'validation' in s
            ]
            overall_scores = [
                s['validation']['overall_score']
                for s in recent_signals
                if 'validation' in s
            ]
            
            return SignalMetrics(
                timestamp=datetime.now(),
                total_signals=total,
                valid_signals=valid_count,
                invalid_signals=total - valid_count,
                buy_signals=buy_count,
                sell_signals=sell_count,
                hold_signals=hold_count,
                avg_quality_score=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                avg_risk_score=sum(risk_scores) / len(risk_scores) if risk_scores else 0.0,
                avg_overall_score=sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            )
    
    def _check_alerts(self):
        """检查告警"""
        metrics = self.calculate_current_metrics()
        
        for rule in self._alert_rules.values():
            if not rule.enabled:
                continue
            
            # 检查冷却时间
            if rule.last_triggered:
                cooldown = timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() - rule.last_triggered < cooldown:
                    continue
            
            # 检查条件
            triggered = False
            
            if rule.condition == 'avg_quality_score < 50':
                if metrics.avg_quality_score < rule.threshold:
                    triggered = True
            
            elif rule.condition == 'avg_risk_score > 80':
                if metrics.avg_risk_score > rule.threshold:
                    triggered = True
            
            elif rule.condition == 'total_signals > 100':
                if metrics.total_signals > rule.threshold:
                    triggered = True
            
            if triggered:
                self._trigger_alert(rule, metrics)
    
    def _trigger_alert(self, rule: AlertRule, metrics: SignalMetrics):
        """
        触发告警
        
        Args:
            rule: 告警规则
            metrics: 当前指标
        """
        rule.last_triggered = datetime.now()
        
        alert_data = {
            'rule_id': rule.rule_id,
            'rule_name': rule.name,
            'condition': rule.condition,
            'threshold': rule.threshold,
            'current_value': self._get_current_value(rule.condition, metrics),
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_signals': metrics.total_signals,
                'valid_signals': metrics.valid_signals,
                'avg_quality_score': metrics.avg_quality_score,
                'avg_risk_score': metrics.avg_risk_score
            }
        }
        
        # 调用告警回调
        for callback in self._alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
        
        logger.warning(f"告警触发: {rule.name}")
    
    def _get_current_value(self, condition: str, metrics: SignalMetrics) -> float:
        """获取当前值"""
        if 'quality_score' in condition:
            return metrics.avg_quality_score
        elif 'risk_score' in condition:
            return metrics.avg_risk_score
        elif 'total_signals' in condition:
            return float(metrics.total_signals)
        return 0.0
    
    def add_alert_rule(self, rule: AlertRule):
        """
        添加告警规则
        
        Args:
            rule: 告警规则
        """
        with self._lock:
            self._alert_rules[rule.rule_id] = rule
        
        logger.info(f"告警规则添加成功: {rule.rule_id}")
    
    def remove_alert_rule(self, rule_id: str):
        """
        移除告警规则
        
        Args:
            rule_id: 规则ID
        """
        with self._lock:
            if rule_id in self._alert_rules:
                del self._alert_rules[rule_id]
        
        logger.info(f"告警规则移除成功: {rule_id}")
    
    def register_alert_callback(self, callback: Callable):
        """
        注册告警回调
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            if callback not in self._alert_callbacks:
                self._alert_callbacks.append(callback)
    
    def unregister_alert_callback(self, callback: Callable):
        """
        注销告警回调
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            if callback in self._alert_callbacks:
                self._alert_callbacks.remove(callback)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        获取监控面板数据
        
        Returns:
            面板数据字典
        """
        with self._lock:
            current_metrics = self.calculate_current_metrics()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': {
                    'total_signals': current_metrics.total_signals,
                    'valid_signals': current_metrics.valid_signals,
                    'invalid_signals': current_metrics.invalid_signals,
                    'buy_signals': current_metrics.buy_signals,
                    'sell_signals': current_metrics.sell_signals,
                    'hold_signals': current_metrics.hold_signals,
                    'avg_quality_score': round(current_metrics.avg_quality_score, 2),
                    'avg_risk_score': round(current_metrics.avg_risk_score, 2),
                    'avg_overall_score': round(current_metrics.avg_overall_score, 2)
                },
                'stats': self._stats,
                'alert_rules': [
                    {
                        'rule_id': rule.rule_id,
                        'name': rule.name,
                        'enabled': rule.enabled,
                        'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
                    }
                    for rule in self._alert_rules.values()
                ],
                'recent_signals': list(self._signal_history)[-10:],
                'history_length': len(self._signal_history)
            }
    
    def get_signal_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取信号历史
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            symbol: 股票代码
            signal_type: 信号类型
            
        Returns:
            信号列表
        """
        with self._lock:
            signals = list(self._signal_history)
        
        # 过滤
        if start_time:
            signals = [
                s for s in signals
                if s.get('recorded_at', datetime.min) >= start_time
            ]
        
        if end_time:
            signals = [
                s for s in signals
                if s.get('recorded_at', datetime.max) <= end_time
            ]
        
        if symbol:
            signals = [s for s in signals if s.get('symbol') == symbol]
        
        if signal_type:
            signals = [s for s in signals if s.get('signal_type') == signal_type]
        
        return signals
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                **self._stats,
                'history_length': len(self._signal_history),
                'metrics_history_length': len(self._metrics_history),
                'alert_rules_count': len(self._alert_rules)
            }
    
    def clear_history(self):
        """清空历史记录"""
        with self._lock:
            self._signal_history.clear()
            self._metrics_history.clear()
        
        logger.info("历史记录已清空")


# 单例实例
_monitor: Optional[SignalMonitor] = None


def get_signal_monitor(max_history: int = 1000) -> SignalMonitor:
    """
    获取信号监控器单例
    
    Args:
        max_history: 最大历史记录数
        
    Returns:
        SignalMonitor实例
    """
    global _monitor
    if _monitor is None:
        _monitor = SignalMonitor(max_history=max_history)
    return _monitor
