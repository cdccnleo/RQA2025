import random
import time
import threading
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
from enum import Enum
import re
from collections import deque

class SamplingStrategyType(Enum):
    """采样策略类型"""
    FIXED_RATE = 1        # 固定比率采样
    LEVEL_BASED = 2       # 基于日志级别
    KEYWORD_FILTER = 3    # 关键字过滤
    COMBINED = 4          # 组合策略
    DYNAMIC = 5           # 动态调整

@dataclass
class SamplingRule:
    """采样规则配置"""
    strategy: SamplingStrategyType
    rate: float = 1.0
    level: Optional[str] = None
    keywords: Optional[List[str]] = None
    condition: Optional[Callable[[Dict], bool]] = None

class LogSampler:
    """增强版日志采样器"""

    def __init__(self,
                 base_rate: float = 1.0,
                 min_rate: float = 0.01,
                 max_rate: float = 1.0,
                 load_window: int = 60):
        """
        初始化采样器

        Args:
            base_rate: 基础采样率(0.0-1.0)
            min_rate: 最小采样率
            max_rate: 最大采样率
            load_window: 负载评估窗口(秒)
        """
        self._base_rate = base_rate
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._load_window = load_window
        self._current_rate = base_rate  # 当前实际使用的采样率

        self._rules: List[SamplingRule] = []
        self._lock = threading.Lock()
        self._load_history = deque(maxlen=load_window)
        self._last_adjust_time = time.time()

        # 默认规则
        self.add_rule(SamplingRule(
            strategy=SamplingStrategyType.FIXED_RATE,
            rate=base_rate
        ))

    def add_rule(self, rule: SamplingRule):
        """添加采样规则"""
        with self._lock:
            self._rules.append(rule)

    def remove_rule(self, index: int):
        """移除采样规则"""
        with self._lock:
            if 0 <= index < len(self._rules):
                self._rules.pop(index)

    def clear_rules(self):
        """清除所有采样规则"""
        with self._lock:
            self._rules.clear()

    def set_base_rate(self, rate: float):
        """设置基础采样率"""
        with self._lock:
            self._base_rate = max(self._min_rate, min(self._max_rate, rate))

    def adjust_for_load(self, current_load: float):
        """根据系统负载调整采样率"""
        now = time.time()
        self._load_history.append((now, current_load))

        # 计算平均负载
        window_start = now - self._load_window
        recent_loads = [load for ts, load in self._load_history if ts >= window_start]
        if not recent_loads:
            return

        avg_load = sum(recent_loads) / len(recent_loads)

        # 动态调整当前采样率
        self._current_rate = self._calculate_dynamic_rate(avg_load)
        self._last_adjust_time = now

    def _calculate_dynamic_rate(self, load: float) -> float:
        """计算动态采样率"""
        # 负载超过0.5时开始降低采样率
        if load > 0.5:
            # 使用二次曲线使采样率下降更快
            scale = min(1.0, (load - 0.5) / 0.5)
            adjusted_rate = self._base_rate - (self._base_rate - self._min_rate) * (scale ** 1.5)
            return max(self._min_rate, min(self._max_rate, adjusted_rate))
        return self._base_rate

    def should_sample(self, record: Union[Dict, str]) -> bool:
        """判断是否采样当前日志记录
        
        Args:
            record: 可以是完整的日志记录字典，或日志级别字符串
        """
        if not self._rules:
            return True

        # 处理简化输入(如直接传入日志级别字符串)
        if isinstance(record, str):
            record = {'level': record}

        with self._lock:
            # 优先检查动态采样规则
            dynamic_rules = [r for r in self._rules if r.strategy == SamplingStrategyType.DYNAMIC]
            if dynamic_rules:
                return self._apply_sampling(dynamic_rules[0], record)
            
            # 检查其他规则
            for rule in self._rules:
                if self._match_rule(rule, record):
                    return self._apply_sampling(rule, record)
            return True

    def _match_rule(self, rule: SamplingRule, record: Dict) -> bool:
        """匹配规则条件"""
        if rule.strategy == SamplingStrategyType.FIXED_RATE:
            return True

        elif rule.strategy == SamplingStrategyType.LEVEL_BASED:
            return record.get('level', '').upper() == rule.level

        elif rule.strategy == SamplingStrategyType.KEYWORD_FILTER:
            msg = str(record.get('message', ''))
            return any(keyword in msg for keyword in rule.keywords)

        elif rule.strategy == SamplingStrategyType.COMBINED:
            return rule.condition(record) if rule.condition else False

        return False

    def _apply_sampling(self, rule: SamplingRule, record: Dict) -> bool:
        """应用采样策略"""
        if rule.strategy == SamplingStrategyType.DYNAMIC:
            # 使用当前调整后的采样率
            return random.random() < self._current_rate

        return random.random() < rule.rate

    @property
    def base_rate(self) -> float:
        """获取当前基础采样率"""
        with self._lock:
            return self._base_rate

    @property
    def rules(self) -> List[SamplingRule]:
        """获取当前配置的采样规则列表(只读)"""
        with self._lock:
            return self._rules.copy()

    def get_current_strategy(self) -> Dict[str, Any]:
        """获取当前采样策略配置"""
        with self._lock:
            return {
                'base_rate': self._base_rate,
                'min_rate': self._min_rate,
                'max_rate': self._max_rate,
                'rules': [
                    {
                        'strategy': rule.strategy.name,
                        'rate': rule.rate,
                        'level': rule.level,
                        'keywords': rule.keywords
                    }
                    for rule in self._rules
                ],
                'last_adjust_time': self._last_adjust_time,
                'current_load': self._load_history[-1][1] if self._load_history else 0
            }

    def filter(self, record) -> bool:
        """
        日志过滤器接口，适配logging.Handler的filter方法
        
        Args:
            record: logging.LogRecord对象
            
        Returns:
            bool: True表示记录日志，False表示过滤
        """
        log_dict = {
            'message': record.getMessage(),
            'level': record.levelname,
            'time': record.created,
            'pathname': record.pathname,
            'lineno': record.lineno,
            'funcName': record.funcName
        }
        return self.should_sample(log_dict)
