#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时风险监控系统

提供实时风险监控、风险指标计算、风险预警等功能
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


class RiskLevel(Enum):

    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):

    """风险类型"""
    POSITION = "position"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    MARKET = "market"
    OPERATIONAL = "operational"


@dataclass
class RiskMetric:

    """风险指标"""
    metric_id: str
    metric_name: str
    value: float
    threshold: float
    risk_level: RiskLevel
    timestamp: datetime
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAlert:

    """风险告警"""
    alert_id: str
    alert_type: RiskType
    severity: RiskLevel
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    resolved: bool = False
    resolved_time: Optional[datetime] = None


class RealTimeMonitor:

    """实时风险监控器"""

    def __init__(self, config: Optional[Dict] = None):

        self.config = config or {}
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts = {}
        self.risk_rules = {}
        self.monitoring_thread = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 初始化风险规则
        self._init_default_rules()

        # 风险指标计算器
        self.risk_calculators = {
            RiskType.POSITION: self._calculate_position_risk,
            RiskType.VOLATILITY: self._calculate_volatility_risk,
            RiskType.LIQUIDITY: self._calculate_liquidity_risk,
            RiskType.CONCENTRATION: self._calculate_concentration_risk,
            RiskType.CORRELATION: self._calculate_correlation_risk,
            RiskType.MARKET: self._calculate_market_risk,
            RiskType.OPERATIONAL: self._calculate_operational_risk
        }

        # 告警处理器
        self.alert_handlers = []

    logger.info("实时风险监控器初始化完成")

    def add_metric(self, metric_data: Dict[str, Any]) -> bool:
        """
        添加风险指标

        Args:
            metric_data: 指标数据，包含 'type', 'value', 'timestamp' 等字段

        Returns:
            bool: 是否成功添加
        """
        try:
            metric_type = metric_data.get('type', 'unknown')
            value = metric_data.get('value', 0)
            timestamp = metric_data.get('timestamp', datetime.now())

            # 添加到历史记录
            self.metrics_history[metric_type].append({
                'value': value,
                'timestamp': timestamp
            })

            # 检查风险规则
            self._check_risk_rules(metric_type, value)

            return True
        except Exception as e:
            logger.error(f"添加指标失败: {e}")
            return False

    def _init_default_rules(self):
        """初始化默认风险规则"""
        self.risk_rules = {
            RiskType.POSITION: {
                "max_position_size": 0.2,  # 单票最大仓位20%
                "max_total_position": 0.8,  # 总仓位最大80%
                "position_concentration": 0.3  # 仓位集中度30%
            },
            RiskType.VOLATILITY: {
                "max_volatility": 0.3,  # 最大波动率30%
                "volatility_threshold": 0.2  # 波动率阈值20%
            },
            RiskType.LIQUIDITY: {
                "min_liquidity": 0.1,  # 最小流动性10%
                "liquidity_threshold": 0.2  # 流动性阈值20%
            },
            RiskType.CONCENTRATION: {
                "max_concentration": 0.3,  # 最大集中度30%
                "concentration_threshold": 0.2  # 集中度阈误0%
            },
            RiskType.CORRELATION: {
                "max_correlation": 0.8,  # 最大相关误0%
                "correlation_threshold": 0.7  # 相关性阈误0%
            },
            RiskType.MARKET: {
                "max_market_risk": 0.5,  # 最大市场风误0%
                "market_risk_threshold": 0.3  # 市场风险阈误0%
            },
            RiskType.OPERATIONAL: {
                "max_operational_risk": 0.1,  # 最大操作风误0%
                "operational_risk_threshold": 0.05  # 操作风险阈误%
            }
        }

    def start_monitoring(self):
        """启动实时监控"""
        if self.running:
            logger.warning("监控已启动")
            return

        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("实时风险监控已启动")

    def stop_monitoring(self):
        """停止实时监控"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("实时风险监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 计算所有风险指标
                self._calculate_all_risks()

                # 检查告警条件
                self._check_alerts()

                # 清理过期告警
                self._cleanup_expired_alerts()

                # 等待下次监控
                time.sleep(self.config.get("monitoring_interval", 5))

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(10)

    def calculate_all_risks(self, portfolio: Dict[str, Dict[str, Any]], market_data: pd.DataFrame) -> Dict[str, float]:
        """计算所有风险类型

        Args:
            portfolio: 投资组合数据
            market_data: 市场数据

        Returns:
            包含7种风险类型的字典
        """
        risk_scores = {}

        # POSITION_RISK - 头寸风险
        risk_scores['POSITION_RISK'] = self._calculate_position_risk_from_data(portfolio)

        # VOLATILITY_RISK - 波动率风险
        risk_scores['VOLATILITY_RISK'] = self._calculate_volatility_risk_from_data(market_data)

        # LIQUIDITY_RISK - 流动性风险
        risk_scores['LIQUIDITY_RISK'] = self._calculate_liquidity_risk_from_data(portfolio, market_data)

        # CONCENTRATION_RISK - 集中度风险
        risk_scores['CONCENTRATION_RISK'] = self._calculate_concentration_risk_from_data(portfolio)

        # CORRELATION_RISK - 相关性风险
        risk_scores['CORRELATION_RISK'] = self._calculate_correlation_risk_from_data(portfolio)

        # MARKET_RISK - 市场风险
        risk_scores['MARKET_RISK'] = self._calculate_market_risk_from_data(market_data)

        # OPERATIONAL_RISK - 操作风险
        risk_scores['OPERATIONAL_RISK'] = self._calculate_operational_risk()

        return risk_scores

    def _calculate_position_risk_from_data(self, portfolio: Dict[str, Dict[str, Any]]) -> float:
        """基于投资组合数据计算头寸风险"""
        if not portfolio:
            return 0.0

        total_value = sum(pos.get('value', pos.get('quantity', 0) * pos.get('price', 0))
                         for pos in portfolio.values())
        if total_value == 0:
            return 0.0

        # 计算最大头寸占比
        max_position = max(pos.get('value', pos.get('quantity', 0) * pos.get('price', 0))
                          for pos in portfolio.values())
        return min(max_position / total_value, 1.0)

    def _calculate_volatility_risk_from_data(self, market_data: pd.DataFrame) -> float:
        """基于市场数据计算波动率风险"""
        if market_data.empty or 'price' not in market_data.columns:
            return 0.0

        prices = market_data['price'].values
        if len(prices) < 2:
            return 0.0

        # 计算收益率波动率
        returns = np.diff(np.log(prices))
        volatility = np.std(returns)
        return min(volatility * 100, 1.0)  # 转换为百分比并限制在0-1范围内

    def _calculate_liquidity_risk_from_data(self, portfolio: Dict[str, Dict[str, Any]], market_data: pd.DataFrame) -> float:
        """基于数据计算流动性风险"""
        if not portfolio:
            return 0.0

        # 简化的流动性风险计算：基于持仓规模和交易量
        total_position_size = sum(abs(pos.get('quantity', 0)) for pos in portfolio.values())

        # 假设市场数据包含交易量信息
        avg_volume = market_data.get('volume', pd.Series([1000000])).mean() if hasattr(market_data, 'get') else 1000000

        if avg_volume == 0:
            return 1.0

        liquidity_ratio = avg_volume / total_position_size if total_position_size > 0 else 1.0
        return max(0.0, min(1.0 - (liquidity_ratio / 100), 1.0))  # 流动性越差风险越高

    def _calculate_concentration_risk_from_data(self, portfolio: Dict[str, Dict[str, Any]]) -> float:
        """基于投资组合数据计算集中度风险"""
        if len(portfolio) < 2:
            return 0.0

        # 简化的集中度风险：基于持仓多样性
        num_positions = len(portfolio)
        diversity_factor = 1.0 / num_positions if num_positions > 0 else 1.0

        # 假设集中度风险与多样性成反比
        return 1.0 - diversity_factor

    def _calculate_correlation_risk_from_data(self, portfolio: Dict[str, Dict[str, Any]]) -> float:
        """基于投资组合数据计算相关性风险"""
        if len(portfolio) < 2:
            return 0.0

        # 简化的相关性风险：基于持仓多样性
        num_positions = len(portfolio)
        diversity_factor = 1.0 / num_positions if num_positions > 0 else 1.0

        # 假设相关性风险与集中度成反比
        return 1.0 - diversity_factor

    def _calculate_market_risk_from_data(self, market_data: pd.DataFrame) -> float:
        """基于市场数据计算市场风险"""
        if market_data.empty:
            return 0.5

        # 简化的市场风险：基于价格波动
        if 'price' in market_data.columns:
            prices = market_data['price'].values
            if len(prices) > 1:
                price_change = (prices[-1] - prices[0]) / prices[0]
                return min(abs(price_change), 1.0)

        return 0.3

    def _calculate_all_risks(self):
        """计算所有风险指标"""
        current_time = datetime.now()

        for risk_type in RiskType:
            try:
                calculator = self.risk_calculators.get(risk_type)
                if calculator:
                    risk_value = calculator()
                    threshold = self._get_threshold(risk_type)
                    risk_level = self._determine_risk_level(risk_value, threshold)

                    metric = RiskMetric(
                        metric_id=f"{risk_type.value}_{current_time.strftime('%Y%m%d_%H%M%S')}",
                        metric_name=risk_type.value,
                        value=risk_value,
                        threshold=threshold,
                        risk_level=risk_level,
                        timestamp=current_time,
                        description=f"{risk_type.value}风险指标"
                    )

                    self.metrics_history[risk_type].append(metric)

            except Exception as e:
                logger.error(f"计算{risk_type.value}风险指标失败: {e}")

    def _calculate_position_risk(self) -> float:
        """计算仓位风险"""
        # 这里应该从实际数据源获取仓位信息
        # 示例实现
        positions = self._get_current_positions()
        if not positions:
            return 0.0

        total_position = sum(abs(pos) for pos in positions.values())
        max_position = max(abs(pos) for pos in positions.values()) if positions else 0

        position_risk = max(
            total_position / self.risk_rules[RiskType.POSITION]["max_total_position"],
            max_position / self.risk_rules[RiskType.POSITION]["max_position_size"]
        )

        return min(position_risk, 1.0)

    def _calculate_volatility_risk(self) -> float:
        """计算波动率风险"""
        # 这里应该从实际数据源获取价格数据
        # 示例实现
        price_data = self._get_price_data()
        if len(price_data) < 2:
            return 0.0

        returns = np.diff(np.log(price_data))
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率

        return min(volatility, 1.0)

    def _calculate_liquidity_risk(self) -> float:
        """计算流动性风险"""
        # 这里应该从实际数据源获取流动性数据
        # 示例实现
        liquidity_data = self._get_liquidity_data()
        if not liquidity_data:
            return 0.0

        avg_liquidity = np.mean(liquidity_data)
        min_liquidity = self.risk_rules[RiskType.LIQUIDITY]["min_liquidity"]

        liquidity_risk = max(0, (min_liquidity - avg_liquidity) / min_liquidity)
        return min(liquidity_risk, 1.0)

    def _calculate_concentration_risk(self) -> float:
        """计算集中度风险"""
        # 这里应该从实际数据源获取持仓数据
        # 示例实现
        positions = self._get_current_positions()
        if not positions:
            return 0.0

        total_value = sum(abs(pos) for pos in positions.values())
        if total_value == 0:
            return 0.0

        concentration = max(abs(pos) / total_value for pos in positions.values())
        max_concentration = self.risk_rules[RiskType.CONCENTRATION]["max_concentration"]

        concentration_risk = concentration / max_concentration
        return min(concentration_risk, 1.0)

    def _calculate_correlation_risk(self) -> float:
        """计算相关性风险"""
        # 这里应该从实际数据源获取相关性数据
        # 示例实现
        correlation_data = self._get_correlation_data()
        if not correlation_data:
            return 0.0

        max_correlation = max(correlation_data) if correlation_data else 0
        correlation_threshold = self.risk_rules[RiskType.CORRELATION]["max_correlation"]

        correlation_risk = max(0, (max_correlation - correlation_threshold) /
                               (1 - correlation_threshold))
        return min(correlation_risk, 1.0)

    def _calculate_market_risk(self) -> float:
        """计算市场风险"""
        # 这里应该从实际数据源获取市场数据
        # 示例实现
        market_data = self._get_market_data()
        if not market_data:
            return 0.0

        # 简化的市场风险计算
        market_risk = np.mean(market_data) if market_data else 0
        return min(market_risk, 1.0)

    def _calculate_operational_risk(self) -> float:
        """计算操作风险"""
        # 这里应该从实际数据源获取操作数据
        # 示例实现
        operational_data = self._get_operational_data()
        if not operational_data:
            return 0.0

        operational_risk = np.mean(operational_data) if operational_data else 0
        return min(operational_risk, 1.0)

    def _get_threshold(self, risk_type: RiskType) -> float:
        """获取风险阈值"""
        rules = self.risk_rules.get(risk_type, {})
        return rules.get("threshold", 0.5)

    def _determine_risk_level(self, value: float, threshold: float) -> RiskLevel:
        """确定风险等级"""
        if value >= threshold * 1.5:
            return RiskLevel.CRITICAL
        elif value >= threshold:
            return RiskLevel.HIGH
        elif value >= threshold * 0.7:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _check_alerts(self):
        """检查告警条件"""
        current_time = datetime.now()

        for risk_type, metrics in self.metrics_history.items():
            if not metrics:
                continue

            latest_metric = metrics[-1]
            threshold = self._get_threshold(risk_type)

            if latest_metric.value >= threshold:
                # 检查是否已有相同告警
                alert_key = f"{risk_type.value}_{latest_metric.metric_id}"
                if alert_key not in self.active_alerts:
                    alert = RiskAlert(
                        alert_id=alert_key,
                        alert_type=risk_type,
                        severity=latest_metric.risk_level,
                        message=f"{risk_type.value}风险超过阈值 {latest_metric.value:.4f} >= {threshold:.4f}",
                        metric_value=latest_metric.value,
                        threshold=threshold,
                        timestamp=current_time
                    )

                    self.active_alerts[alert_key] = alert
                    self._trigger_alert(alert)

    def _trigger_alert(self, alert: RiskAlert):
        """触发告警"""
        logger.warning(f"风险告警: {alert.message}")

        # 调用告警处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器异常 {e}")

    def _cleanup_expired_alerts(self):
        """清理过期告警"""
        current_time = datetime.now()
        expired_alerts = []

        for alert_key, alert in self.active_alerts.items():
            # 清理超过24小时的告警
            if current_time - alert.timestamp > timedelta(hours=24):
                expired_alerts.append(alert_key)

        for alert_key in expired_alerts:
            del self.active_alerts[alert_key]

    def add_alert_handler(self, handler: Callable[[RiskAlert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)

    def get_risk_metrics(self, risk_type: Optional[RiskType] = None,
                         limit: int = 100) -> List[RiskMetric]:
        """获取风险指标"""
        if risk_type:
            metrics = list(self.metrics_history[risk_type])[-limit:]
        else:
            all_metrics = []
            for metrics_list in self.metrics_history.values():
                all_metrics.extend(metrics_list)
            metrics = sorted(all_metrics, key=lambda x: x.timestamp)[-limit:]

        return metrics

    def get_active_alerts(self, risk_type: Optional[RiskType] = None) -> List[RiskAlert]:
        """获取活跃告警"""
        if risk_type:
            return [alert for alert in self.active_alerts.values()
                    if alert.alert_type == risk_type and not alert.resolved]
        else:
            return [alert for alert in self.active_alerts.values()
                    if not alert.resolved]

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].resolved_time = datetime.now()
            logger.info(f"告警已解决 {alert_id}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        summary = {
            "total_metrics": sum(len(metrics) for metrics in self.metrics_history.values()),
            "active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
            "risk_levels": defaultdict(int),
            "latest_metrics": {}
        }

        # 统计风险等级
        for metrics in self.metrics_history.values():
            for metric in metrics:
                summary["risk_levels"][metric.risk_level.value] += 1

        # 获取最新指标
        for risk_type, metrics in self.metrics_history.items():
            if metrics:
                summary["latest_metrics"][risk_type.value] = {
                    "value": metrics[-1].value,
                    "risk_level": metrics[-1].risk_level.value,
                    "timestamp": metrics[-1].timestamp.isoformat()
                }

        return summary

    # 数据获取方法（需要根据实际数据源实现）

    def _get_current_positions(self) -> Dict[str, float]:
        """获取当前持仓"""
        # 示例实现，实际应该从数据源获取
        return {"AAPL": 1000, "GOOGL": 500, "MSFT": 800}

    def _get_price_data(self) -> List[float]:
        """获取价格数据"""
        # 示例实现，实际应该从数据源获取
        return [100.0, 101.0, 99.5, 102.0, 98.5]

    def _get_liquidity_data(self) -> List[float]:
        """获取流动性数据"""
        # 示例实现，实际应该从数据源获取
        return [0.8, 0.7, 0.9, 0.6, 0.8]

    def _get_correlation_data(self) -> List[float]:
        """获取相关性数据"""
        # 示例实现，实际应该从数据源获取
        return [0.3, 0.5, 0.7, 0.4, 0.6]

    def _get_market_data(self) -> List[float]:
        """获取市场数据"""
        # 示例实现，实际应该从数据源获取
        return [0.2, 0.3, 0.1, 0.4, 0.2]

    def _get_operational_data(self) -> List[float]:
        """获取操作数据"""
        # 示例实现，实际应该从数据源获取
        return [0.05, 0.03, 0.07, 0.02, 0.04]


__all__ = [
    'RiskLevel',
    'RiskType',
    'RiskMetric',
    'RiskAlert',
    'RealTimeMonitor'
]
