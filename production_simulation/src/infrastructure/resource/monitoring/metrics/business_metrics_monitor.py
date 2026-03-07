"""
business_metrics_monitor 模块

提供 business_metrics_monitor 相关功能和接口。
"""

import logging

# 简化的基础组件类，避免复杂的导入依赖
import threading

from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
"""
基础设施层 - 业务指标监控器

专门用于监控业务相关的指标，包括交易指标、模型性能、业务KPI等。
"""


class BaseMonitorComponent:
    """简化的基础监控组件"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._status = "stopped"

    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        try:
            self.config.update(config)
            self._initialized = True
            self._status = "running"
            return True
        except Exception:
            self._status = "error"
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "component": "monitor",
            "status": self._status,
            "initialized": self._initialized,
            "config": self.config
        }

    def shutdown(self) -> None:
        """关闭组件"""
        self._initialized = False
        self._status = "stopped"


class TradingMetricType(Enum):
    """交易指标类型"""
    VOLUME = "volume"           # 交易量
    TURNOVER = "turnover"       # 成交额
    PROFIT_LOSS = "profit_loss"  # 盈亏
    WIN_RATE = "win_rate"       # 胜率
    SHARPE_RATIO = "sharpe_ratio"  # 夏普比率
    MAX_DRAWDOWN = "max_drawdown"  # 最大回撤
    POSITION_SIZE = "position_size"  # 持仓规模
    TRADE_COUNT = "trade_count"  # 交易次数


class ModelMetricType(Enum):
    """模型指标类型"""
    ACCURACY = "accuracy"       # 准确率
    PRECISION = "precision"     # 精确率
    RECALL = "recall"          # 召回率
    F1_SCORE = "f1_score"      # F1分数
    LATENCY = "latency"        # 延迟
    THROUGHPUT = "throughput"  # 吞吐量
    MEMORY_USAGE = "memory_usage"  # 内存使用
    CPU_USAGE = "cpu_usage"    # CPU使用


class BusinessMetricType(Enum):
    """业务指标类型"""
    USER_ACTIVE = "user_active"     # 活跃用户
    REVENUE = "revenue"             # 收入
    COST = "cost"                   # 成本
    PROFIT = "profit"               # 利润
    CONVERSION_RATE = "conversion_rate"  # 转化率
    RETENTION_RATE = "retention_rate"    # 留存率
    CUSTOMER_SATISFACTION = "customer_satisfaction"  # 客户满意度


class BusinessMetricsMonitor(BaseMonitorComponent):
    """业务指标监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 collection_interval: float = 60.0,
                 max_history_size: int = 1000):
        """
        初始化业务指标监控器

        Args:
            config: 监控配置
            collection_interval: 收集间隔（秒）
            max_history_size: 历史记录最大大小
        """
        if config is None:
            config = {}

        # 添加参数到配置中
        config.update({
            'collection_interval': collection_interval,
            'max_history_size': max_history_size
        })

        super().__init__(config)

        # 设置实例属性供测试访问
        self.collection_interval = collection_interval
        self.max_history_size = max_history_size

        # 线程锁
        self._lock = threading.RLock()

        # 业务指标存储
        self._trading_metrics = defaultdict(list)
        self._model_metrics = defaultdict(list)
        self._business_metrics = defaultdict(list)

        # 阈值配置
        self._thresholds = config.get('thresholds', {})

        # 告警规则
        self._alert_rules = config.get('alert_rules', {})

        # 最大指标数量限制
        self._max_metrics = config.get('max_metrics', 10000)

        # 初始化
        self._initialize()

    def _initialize(self):
        """初始化监控器"""
        self.logger = logging.getLogger(__name__)

    def record_metric(self, name, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        记录指标（重写基类方法以支持业务指标分类）

        Args:
            name: 指标名称（字符串或枚举）
            value: 指标值
            tags: 标签
        """
        # 根据指标类型选择记录方法
        if isinstance(name, TradingMetricType):
            self._record_trading_metric(name, value, tags)
        elif isinstance(name, ModelMetricType):
            self._record_model_metric(name, value, tags)
        else:
            self._record_business_metric(str(name), value, tags)

    def _record_trading_metric(self, name: TradingMetricType, value: float,
                               tags: Optional[Dict[str, str]]) -> None:
        """记录交易指标"""
        strategy = tags.get('strategy', 'default') if tags else 'default'
        key = f"{strategy}.{name.value}"

        metric_record = self._create_metric_record(value, tags)
        self._store_metric_record(self._trading_metrics, key, metric_record)
        self._check_alerts(name.value, value, tags)

    def _record_model_metric(self, name: ModelMetricType, value: float,
                             tags: Optional[Dict[str, str]]) -> None:
        """记录模型指标"""
        model = tags.get('model', 'default') if tags else 'default'
        key = f"{model}.{name.value}"

        metric_record = self._create_metric_record(value, tags)
        self._store_metric_record(self._model_metrics, key, metric_record)
        self._check_alerts(name.value, value, tags)

    def _record_business_metric(self, name_str: str, value: float,
                                tags: Optional[Dict[str, str]]) -> None:
        """记录通用业务指标"""
        metric_record = self._create_metric_record(value, tags)
        self._store_metric_record(self._business_metrics, name_str, metric_record)
        self._check_alerts(name_str, value, tags)

    def _create_metric_record(self, value: float, tags: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """创建指标记录"""
        return {
            'value': value,
            'timestamp': datetime.now(),
            'tags': tags or {}
        }

    def _store_metric_record(self, metrics_dict: Dict[str, List],
                             key: str, record: Dict[str, Any]) -> None:
        """存储指标记录"""
        with self._lock:
            if key not in metrics_dict:
                metrics_dict[key] = []

            metrics_dict[key].append(record)

            # 限制存储大小
            if len(metrics_dict[key]) > self._max_metrics:
                metrics_dict[key] = metrics_dict[key][-self._max_metrics:]

    def get_metric_stats(self, metric_name, strategy: str = "test") -> Dict[str, Any]:
        """
        获取指标统计信息

        Args:
            metric_name: 指标名称（字符串或枚举）
            strategy: 策略名称（用于交易和模型指标）

        Returns:
            Dict: 统计信息
        """
        # 调用自己的get_metric方法
        metrics = self.get_metric(metric_name, strategy)
        if not metrics:
            return {}

        values = [m['value'] for m in metrics]
        return {
            'count': len(values),
            'min': min(values) if values else 0,
            'max': max(values) if values else 0,
            'avg': sum(values) / len(values) if values else 0,
            'mean': sum(values) / len(values) if values else 0
        }

    def set_alert_rule(self, metric_name: str, threshold: float, condition: str, alert_handler: Callable) -> None:
        """
        设置告警规则

        Args:
            metric_name: 指标名称
            threshold: 阈值
            condition: 条件 ('above', 'below', 'equals')
            alert_handler: 告警处理器
        """
        if not hasattr(self, '_alert_rules'):
            self._alert_rules = {}

        self._alert_rules[metric_name] = {
            'threshold': threshold,
            'condition': condition,
            'handler': alert_handler
        }

    def _check_alerts(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """检查告警规则"""
        if metric_name in self._alert_rules:
            rule = self._alert_rules[metric_name]
            threshold = rule['threshold']
            condition = rule['condition']

            trigger_alert = False
            if condition == 'above' and value > threshold:
                trigger_alert = True
            elif condition == 'below' and value < threshold:
                trigger_alert = True
            elif condition == 'equals' and abs(value - threshold) < 0.001:
                trigger_alert = True

            if trigger_alert:
                try:
                    rule['handler'](metric_name, value, threshold, tags)
                except Exception as e:
                    self.logger.error(f"告警处理器执行失败: {e}")

    def get_metric(self, name, strategy: str = None) -> List[Dict[str, Any]]:
        """获取指标数据（兼容基类接口）"""
        with self._lock:
            if isinstance(name, TradingMetricType):
                key = f"{strategy or 'default'}.{name.value}"
                return self._trading_metrics.get(key, [])
            elif isinstance(name, ModelMetricType):
                key = f"{strategy or 'default'}.{name.value}"
                return self._model_metrics.get(key, [])
            elif isinstance(name, BusinessMetricType):
                return self._business_metrics.get(name.value, [])
            else:
                return []

    def get_trading_metrics(self, strategy: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """获取交易指标"""
        with self._lock:
            if strategy:
                return {k: v for k, v in self._trading_metrics.items() if k.startswith(f"{strategy}.")}
            return dict(self._trading_metrics)

    def get_model_metrics(self, model: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """获取模型指标"""
        with self._lock:
            if model:
                return {k: v for k, v in self._model_metrics.items() if k.startswith(f"{model}.")}
            return dict(self._model_metrics)

    def get_business_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取业务指标"""
        with self._lock:
            return dict(self._business_metrics)

    def clear_metrics(self):
        """清空所有指标"""
        with self._lock:
            self._trading_metrics.clear()
            self._model_metrics.clear()
            self._business_metrics.clear()

    def get_monitor_stats(self) -> Dict[str, Any]:
        """获取监控器统计信息"""
        with self._lock:
            return {
                'trading_metrics_count': len(self._trading_metrics),
                'model_metrics_count': len(self._model_metrics),
                'business_metrics_count': len(self._business_metrics),
                'alert_rules_count': len(self._alert_rules),
                'total_trading_records': sum(len(records) for records in self._trading_metrics.values()),
                'total_model_records': sum(len(records) for records in self._model_metrics.values()),
                'total_business_records': sum(len(records) for records in self._business_metrics.values())
            }
