"""
市场状态自适应监控器
基于基础设施层监控系统，实现市场状态感知和动态采集策略调整
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class MarketRegime(Enum):
    """市场状态枚举"""
    HIGH_VOLATILITY = "high_volatility"  # 高波动
    BULL = "bull"                        # 牛市
    BEAR = "bear"                        # 熊市
    SIDEWAYS = "sideways"                # 横盘
    LOW_LIQUIDITY = "low_liquidity"      # 低流动性


@dataclass
class MarketMetrics:
    """市场指标数据"""
    timestamp: datetime
    volatility: float = 0.0  # 波动率
    trend_strength: float = 0.0  # 趋势强度
    volume_trend: float = 0.0  # 成交量趋势
    market_breadth: float = 0.0  # 市场宽度
    sentiment_score: float = 0.0  # 情绪得分

    # 衍生指标
    volatility_percentile: float = 50.0  # 波动率百分位
    volume_percentile: float = 50.0  # 成交量百分位
    breadth_percentile: float = 50.0  # 市场宽度百分位


@dataclass
class MarketRegimeAnalysis:
    """市场状态分析结果"""
    current_regime: MarketRegime
    confidence: float
    metrics: MarketMetrics
    indicators: Dict[str, Any]
    recommended_actions: List[str]
    analysis_timestamp: datetime


class MarketAdaptiveMonitor:
    """
    市场状态自适应监控器

    基于基础设施层监控系统，实现市场状态感知：
    1. 集成基础设施监控获取市场数据
    2. 计算市场指标和状态识别
    3. 提供动态采集策略建议
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self._market_history: List[MarketMetrics] = []
        self._regime_history: List[MarketRegimeAnalysis] = []
        self._last_update: Optional[datetime] = None

        # 初始化阈值配置
        self._volatility_thresholds = self.config['volatility_thresholds']
        self._trend_thresholds = self.config['trend_thresholds']
        self._volume_thresholds = self.config['volume_thresholds']

        logger.info("市场状态自适应监控器初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'update_interval_seconds': 300,  # 5分钟更新一次
            'history_window_days': 30,       # 历史数据窗口（天）
            'volatility_thresholds': {
                'high': 0.05,    # 5%波动率视为高波动
                'extreme': 0.10  # 10%波动率视为极高波动
            },
            'trend_thresholds': {
                'strong_bull': 0.03,   # 3%日涨幅视为强牛
                'strong_bear': -0.03,  # -3%日跌幅视为强熊
                'sideways_range': 0.02 # 2%日波动视为横盘
            },
            'volume_thresholds': {
                'low_liquidity': 0.3,  # 成交量低于30%分位数视为低流动性
            },
            'analysis_window_days': 7,  # 分析窗口（天）
            'min_data_points': 5,       # 最少数据点要求
        }

    async def get_current_regime(self) -> MarketRegimeAnalysis:
        """
        获取当前市场状态分析

        Returns:
            MarketRegimeAnalysis: 市场状态分析结果
        """
        try:
            # 检查是否需要更新数据
            await self._ensure_data_updated()

            # 获取最新市场指标
            latest_metrics = await self._collect_market_metrics()

            # 分析市场状态
            regime_analysis = await self._analyze_market_regime(latest_metrics)

            # 记录分析历史
            self._regime_history.append(regime_analysis)
            if len(self._regime_history) > 100:  # 保留最近100次分析
                self._regime_history.pop(0)

            logger.info(
                f"市场状态分析完成: {regime_analysis.current_regime.value}, "
                f"置信度: {regime_analysis.confidence:.2f}"
            )

            return regime_analysis

        except Exception as e:
            logger.error(f"获取市场状态失败: {e}", exc_info=True)
            # 返回默认状态
            return self._get_default_regime_analysis()

    async def _ensure_data_updated(self):
        """确保数据是最新的"""
        current_time = datetime.now()

        if (self._last_update is None or
            (current_time - self._last_update).total_seconds() > self.config['update_interval_seconds']):

            logger.debug("更新市场监控数据")
            await self._update_market_data()
            self._last_update = current_time

    async def _update_market_data(self):
        """更新市场监控数据"""
        try:
            # 从基础设施层获取监控数据
            market_data = await self._fetch_market_data_from_infrastructure()

            # 转换为市场指标
            metrics = await self._process_market_data(market_data)

            # 更新历史数据
            self._market_history.append(metrics)
            if len(self._market_history) > self.config['history_window_days'] * 24:  # 按小时保留
                self._market_history = self._market_history[-self.config['history_window_days'] * 24:]

        except Exception as e:
            logger.warning(f"更新市场数据失败: {e}")

    async def _fetch_market_data_from_infrastructure(self) -> Dict[str, Any]:
        """
        从基础设施层获取市场数据

        这里应该集成实际的基础设施监控系统
        """
        try:
            # 临时实现：模拟从基础设施层获取数据
            # 实际应该调用基础设施层的监控API

            # 模拟市场数据
            market_data = {
                'timestamp': datetime.now(),
                'indices': {
                    'sh000001': {  # 上证指数
                        'price': 3200.0,
                        'change_pct': 0.02,
                        'volume': 200000000,
                        'volatility': 0.025
                    },
                    'sz399001': {  # 深证成指
                        'price': 10500.0,
                        'change_pct': -0.01,
                        'volume': 150000000,
                        'volatility': 0.020
                    }
                },
                'market_breadth': 0.55,  # 市场宽度（上涨家数占比）
                'total_volume': 350000000,  # 总成交量
                'sentiment_score': 0.6  # 市场情绪得分
            }

            # TODO: 替换为实际的基础设施层监控调用
            # from src.infrastructure.monitoring import get_market_monitor
            # market_monitor = get_market_monitor()
            # market_data = await market_monitor.get_market_metrics()

            return market_data

        except Exception as e:
            logger.warning(f"获取基础设施层市场数据失败: {e}")
            return self._get_mock_market_data()

    def _get_mock_market_data(self) -> Dict[str, Any]:
        """获取模拟市场数据（开发调试用）"""
        return {
            'timestamp': datetime.now(),
            'indices': {
                'sh000001': {
                    'price': 3200.0,
                    'change_pct': 0.015,
                    'volume': 180000000,
                    'volatility': 0.022
                }
            },
            'market_breadth': 0.52,
            'total_volume': 320000000,
            'sentiment_score': 0.55
        }

    async def _process_market_data(self, raw_data: Dict[str, Any]) -> MarketMetrics:
        """处理原始市场数据，计算指标"""
        try:
            timestamp = raw_data.get('timestamp', datetime.now())

            # 计算波动率（基于指数波动率平均值）
            indices = raw_data.get('indices', {})
            if indices:
                volatilities = [idx.get('volatility', 0) for idx in indices.values()]
                volatility = sum(volatilities) / len(volatilities)
            else:
                volatility = 0.02  # 默认波动率

            # 计算趋势强度（基于指数涨跌幅）
            if indices:
                changes = [idx.get('change_pct', 0) for idx in indices.values()]
                trend_strength = sum(abs(change) for change in changes) / len(changes)
            else:
                trend_strength = 0.01

            # 成交量趋势
            total_volume = raw_data.get('total_volume', 0)
            volume_trend = self._calculate_volume_trend(total_volume)

            # 市场宽度
            market_breadth = raw_data.get('market_breadth', 0.5)

            # 情绪得分
            sentiment_score = raw_data.get('sentiment_score', 0.5)

            # 计算百分位数
            volatility_percentile = self._calculate_percentile(volatility, 'volatility')
            volume_percentile = self._calculate_percentile(total_volume, 'volume')
            breadth_percentile = self._calculate_percentile(market_breadth, 'breadth')

            return MarketMetrics(
                timestamp=timestamp,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_trend=volume_trend,
                market_breadth=market_breadth,
                sentiment_score=sentiment_score,
                volatility_percentile=volatility_percentile,
                volume_percentile=volume_percentile,
                breadth_percentile=breadth_percentile
            )

        except Exception as e:
            logger.error(f"处理市场数据失败: {e}")
            return MarketMetrics(timestamp=datetime.now())

    async def _collect_market_metrics(self) -> MarketMetrics:
        """收集当前市场指标"""
        if self._market_history:
            return self._market_history[-1]
        else:
            # 如果没有历史数据，获取最新数据
            await self._update_market_data()
            return self._market_history[-1] if self._market_history else MarketMetrics(timestamp=datetime.now())

    async def _analyze_market_regime(self, metrics: MarketMetrics) -> MarketRegimeAnalysis:
        """分析市场状态"""
        try:
            # 高波动检测
            if metrics.volatility_percentile > 80:
                regime = MarketRegime.HIGH_VOLATILITY
                confidence = min(0.9, metrics.volatility_percentile / 100)
                indicators = {
                    'volatility_percentile': metrics.volatility_percentile,
                    'volatility_value': metrics.volatility
                }
                actions = [
                    "减少采集频率，避免系统过载",
                    "增加数据质量检查",
                    "考虑暂停非关键数据源采集"
                ]
            # 低流动性检测
            elif metrics.volume_percentile < 20:
                regime = MarketRegime.LOW_LIQUIDITY
                confidence = min(0.8, (100 - metrics.volume_percentile) / 100)
                indicators = {
                    'volume_percentile': metrics.volume_percentile,
                    'volume_trend': metrics.volume_trend
                }
                actions = [
                    "延长采集间隔，减少资源消耗",
                    "优先采集核心数据源",
                    "降低批处理大小"
                ]
            # 趋势分析
            elif metrics.trend_strength > self._trend_thresholds['strong_bull']:
                regime = MarketRegime.BULL
                confidence = min(0.85, metrics.trend_strength / self._trend_thresholds['strong_bull'])
                indicators = {
                    'trend_strength': metrics.trend_strength,
                    'market_breadth': metrics.market_breadth
                }
                actions = [
                    "增加数据采集频率",
                    "扩大采集范围",
                    "提高数据质量要求"
                ]
            elif metrics.trend_strength < self._trend_thresholds['strong_bear']:
                regime = MarketRegime.BEAR
                confidence = min(0.85, abs(metrics.trend_strength) / abs(self._trend_thresholds['strong_bear']))
                indicators = {
                    'trend_strength': metrics.trend_strength,
                    'market_breadth': metrics.market_breadth
                }
                actions = [
                    "保持正常采集频率",
                    "增加风险数据采集",
                    "关注市场情绪指标"
                ]
            else:
                # 检查是否为横盘整理
                if abs(metrics.trend_strength) < self._trend_thresholds['sideways_range']:
                    regime = MarketRegime.SIDEWAYS
                    confidence = 0.7
                    indicators = {
                        'trend_strength': metrics.trend_strength,
                        'volatility': metrics.volatility
                    }
                    actions = [
                        "正常采集频率",
                        "均衡采集各数据源",
                        "优化批处理性能"
                    ]
                else:
                    # 默认牛市
                    regime = MarketRegime.BULL
                    confidence = 0.6
                    indicators = {
                        'trend_strength': metrics.trend_strength,
                        'volatility': metrics.volatility
                    }
                    actions = [
                        "标准采集策略",
                        "监控市场变化"
                    ]

            return MarketRegimeAnalysis(
                current_regime=regime,
                confidence=confidence,
                metrics=metrics,
                indicators=indicators,
                recommended_actions=actions,
                analysis_timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"市场状态分析失败: {e}")
            return self._get_default_regime_analysis()

    def _calculate_volume_trend(self, current_volume: float) -> float:
        """计算成交量趋势"""
        if len(self._market_history) < 2:
            return 0.0

        # 计算最近5个数据点的成交量趋势
        recent_volumes = []
        for metrics in self._market_history[-5:]:
            # 假设成交量存储在某个字段中，这里简化处理
            recent_volumes.append(current_volume)  # 临时使用当前成交量

        if len(recent_volumes) >= 2:
            # 计算趋势（最近值相对于平均值的变化）
            avg_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1])
            latest_volume = recent_volumes[-1]
            return (latest_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0

        return 0.0

    def _calculate_percentile(self, value: float, metric_type: str) -> float:
        """计算百分位数"""
        if len(self._market_history) < self.config['min_data_points']:
            return 50.0  # 默认中位数

        # 获取历史数据
        historical_values = []
        for metrics in self._market_history[-self.config['analysis_window_days'] * 24:]:
            if metric_type == 'volatility':
                historical_values.append(metrics.volatility)
            elif metric_type == 'volume':
                # 这里需要成交量数据，暂时使用固定值
                historical_values.append(300000000)  # 模拟成交量
            elif metric_type == 'breadth':
                historical_values.append(metrics.market_breadth)

        if not historical_values:
            return 50.0

        # 计算百分位数
        sorted_values = sorted(historical_values)
        position = sum(1 for v in sorted_values if v <= value) / len(sorted_values) * 100
        return position

    def _get_default_regime_analysis(self) -> MarketRegimeAnalysis:
        """获取默认的市场状态分析"""
        return MarketRegimeAnalysis(
            current_regime=MarketRegime.SIDEWAYS,
            confidence=0.5,
            metrics=MarketMetrics(timestamp=datetime.now()),
            indicators={},
            recommended_actions=["使用标准采集策略"],
            analysis_timestamp=datetime.now()
        )

    def get_regime_history(self, limit: int = 10) -> List[MarketRegimeAnalysis]:
        """获取市场状态分析历史"""
        return self._regime_history[-limit:] if self._regime_history else []

    def get_market_metrics_history(self, limit: int = 50) -> List[MarketMetrics]:
        """获取市场指标历史"""
        return self._market_history[-limit:] if self._market_history else []

    def get_regime_statistics(self) -> Dict[str, Any]:
        """获取市场状态统计"""
        if not self._regime_history:
            return {}

        stats = {}
        regime_counts = {}

        for analysis in self._regime_history[-100:]:  # 最近100次分析
            regime = analysis.current_regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        total = sum(regime_counts.values())
        stats['regime_distribution'] = {
            regime: count / total for regime, count in regime_counts.items()
        }
        stats['most_common_regime'] = max(regime_counts, key=regime_counts.get)
        stats['analysis_count'] = len(self._regime_history)

        return stats


# 全局实例
_market_monitor = None


def get_market_adaptive_monitor() -> MarketAdaptiveMonitor:
    """获取市场状态自适应监控器实例"""
    global _market_monitor
    if _market_monitor is None:
        _market_monitor = MarketAdaptiveMonitor()
    return _market_monitor


async def get_current_market_regime() -> MarketRegimeAnalysis:
    """便捷函数：获取当前市场状态"""
    monitor = get_market_adaptive_monitor()
    return await monitor.get_current_regime()