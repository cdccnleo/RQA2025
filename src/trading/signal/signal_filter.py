#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号过滤器

功能：
- 基于评分的过滤
- 可配置阈值
- 动态调整

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from .signal_validation_engine import SignalValidationResult

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """过滤器配置"""
    min_overall_score: float = 60.0
    min_quality_score: float = 60.0
    max_risk_score: float = 70.0
    min_backtest_score: float = 50.0
    min_confidence: float = 0.5
    allowed_signal_types: List[str] = None
    blocked_symbols: List[str] = None
    max_signals_per_minute: int = 10


class SignalFilter:
    """
    信号过滤器
    
    职责：
    1. 基于评分的过滤
    2. 可配置阈值
    3. 动态调整
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        初始化信号过滤器
        
        Args:
            config: 过滤器配置
        """
        self.config = config or FilterConfig()
        
        # 信号频率控制
        self._signal_timestamps: List[datetime] = []
        
        # 统计信息
        self._stats = {
            'total_signals': 0,
            'passed_signals': 0,
            'filtered_signals': 0,
            'filter_reasons': {}
        }
        
        logger.info("信号过滤器初始化完成")
    
    def filter_signal(
        self,
        signal: Dict[str, Any],
        validation_result: Optional[SignalValidationResult] = None
    ) -> tuple[bool, str]:
        """
        过滤信号
        
        Args:
            signal: 信号数据
            validation_result: 验证结果
            
        Returns:
            (是否通过, 过滤原因)
        """
        self._stats['total_signals'] += 1
        
        # 1. 检查信号类型
        signal_type = signal.get('signal_type', 'hold')
        if self.config.allowed_signal_types and signal_type not in self.config.allowed_signal_types:
            reason = f"信号类型 {signal_type} 不在允许列表中"
            self._update_filter_stats(reason)
            return False, reason
        
        # 2. 检查股票代码
        symbol = signal.get('symbol', '')
        if self.config.blocked_symbols and symbol in self.config.blocked_symbols:
            reason = f"股票 {symbol} 在黑名单中"
            self._update_filter_stats(reason)
            return False, reason
        
        # 3. 检查置信度
        confidence = signal.get('confidence', 0.0)
        if confidence < self.config.min_confidence:
            reason = f"置信度 {confidence:.2f} 低于阈值 {self.config.min_confidence}"
            self._update_filter_stats(reason)
            return False, reason
        
        # 4. 检查验证结果
        if validation_result:
            # 检查综合评分
            if validation_result.overall_score < self.config.min_overall_score:
                reason = f"综合评分 {validation_result.overall_score:.1f} 低于阈值 {self.config.min_overall_score}"
                self._update_filter_stats(reason)
                return False, reason
            
            # 检查质量评分
            if validation_result.quality_score < self.config.min_quality_score:
                reason = f"质量评分 {validation_result.quality_score:.1f} 低于阈值 {self.config.min_quality_score}"
                self._update_filter_stats(reason)
                return False, reason
            
            # 检查风险评分
            if validation_result.risk_score > self.config.max_risk_score:
                reason = f"风险评分 {validation_result.risk_score:.1f} 高于阈值 {self.config.max_risk_score}"
                self._update_filter_stats(reason)
                return False, reason
            
            # 检查回测评分
            if validation_result.backtest_score < self.config.min_backtest_score:
                reason = f"回测评分 {validation_result.backtest_score:.1f} 低于阈值 {self.config.min_backtest_score}"
                self._update_filter_stats(reason)
                return False, reason
        
        # 5. 检查频率限制
        if not self._check_rate_limit():
            reason = "信号频率超过限制"
            self._update_filter_stats(reason)
            return False, reason
        
        self._stats['passed_signals'] += 1
        return True, "通过"
    
    def _check_rate_limit(self) -> bool:
        """检查频率限制"""
        now = datetime.now()
        
        # 清理过期的信号时间戳
        one_minute_ago = now - __import__('datetime').timedelta(minutes=1)
        self._signal_timestamps = [
            ts for ts in self._signal_timestamps
            if ts > one_minute_ago
        ]
        
        # 检查是否超过限制
        if len(self._signal_timestamps) >= self.config.max_signals_per_minute:
            return False
        
        # 添加当前时间戳
        self._signal_timestamps.append(now)
        return True
    
    def _update_filter_stats(self, reason: str):
        """更新过滤统计"""
        self._stats['filtered_signals'] += 1
        self._stats['filter_reasons'][reason] = self._stats['filter_reasons'].get(reason, 0) + 1
    
    def update_config(self, **kwargs):
        """
        更新配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"过滤器配置更新: {key} = {value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            **self._stats,
            'pass_rate': (
                self._stats['passed_signals'] / self._stats['total_signals']
                if self._stats['total_signals'] > 0 else 0.0
            )
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self._stats = {
            'total_signals': 0,
            'passed_signals': 0,
            'filtered_signals': 0,
            'filter_reasons': {}
        }


# 单例实例
_filter: Optional[SignalFilter] = None


def get_signal_filter(config: Optional[FilterConfig] = None) -> SignalFilter:
    """
    获取信号过滤器单例
    
    Args:
        config: 过滤器配置
        
    Returns:
        SignalFilter实例
    """
    global _filter
    if _filter is None:
        _filter = SignalFilter(config=config)
    return _filter
