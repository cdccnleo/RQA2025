#!/usr/bin/env python3
"""
市场适应性监控服务

根据市场波动自动调整采集策略：
1. 市场波动检测
2. 交易活动度分析
3. 采集策略动态调整
4. 资源分配优化
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from src.infrastructure.logging.core.unified_logger import get_unified_logger
from src.strategy.intelligence.smart_stock_filter import MarketState

logger = get_unified_logger(__name__)


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL = "bull"           # 多头市场
    BEAR = "bear"           # 空头市场
    SIDEWAYS = "sideways"   # 横盘整理
    HIGH_VOLATILITY = "high_volatility"  # 高波动
    LOW_LIQUIDITY = "low_liquidity"      # 低流动性


@dataclass
class AdaptiveStrategy:
    """适应性策略配置"""
    market_regime: MarketRegime
    batch_size_multiplier: float  # 批次大小倍数
    frequency_multiplier: float   # 采集频率倍数
    priority_adjustment: Dict[str, float]  # 优先级调整
    resource_limit: float         # 资源使用限制
    description: str


class MarketAdaptiveMonitor:
    """
    市场适应性监控器

    实时监控市场状态，根据波动和交易活动动态调整数据采集策略
    """

    def __init__(self):
        self.market_state_history = []
        self.market_regime = MarketRegime.SIDEWAYS
        self.adaptive_strategies = self._init_adaptive_strategies()
        self.monitoring_active = False
        self.last_update = datetime.now()

        # 监控参数
        self.monitoring_interval = 60  # 60秒检查一次市场状态
        self.history_window = 300     # 5分钟历史窗口
        self.volatility_threshold = 0.03  # 3%波动阈值
        self.volume_threshold = 0.7   # 70%成交量阈值

    def _init_adaptive_strategies(self) -> Dict[MarketRegime, AdaptiveStrategy]:
        """初始化适应性策略"""
        return {
            MarketRegime.BULL: AdaptiveStrategy(
                market_regime=MarketRegime.BULL,
                batch_size_multiplier=1.2,
                frequency_multiplier=1.5,
                priority_adjustment={
                    'high': 1.0,
                    'medium': 1.2,
                    'low': 1.5
                },
                resource_limit=0.9,
                description="多头市场：增加采集频率，扩大覆盖范围"
            ),

            MarketRegime.BEAR: AdaptiveStrategy(
                market_regime=MarketRegime.BEAR,
                batch_size_multiplier=0.8,
                frequency_multiplier=0.7,
                priority_adjustment={
                    'high': 1.5,
                    'medium': 1.0,
                    'low': 0.5
                },
                resource_limit=0.6,
                description="空头市场：降低采集频率，聚焦核心股票"
            ),

            MarketRegime.SIDEWAYS: AdaptiveStrategy(
                market_regime=MarketRegime.SIDEWAYS,
                batch_size_multiplier=1.0,
                frequency_multiplier=1.0,
                priority_adjustment={
                    'high': 1.0,
                    'medium': 1.0,
                    'low': 1.0
                },
                resource_limit=0.8,
                description="横盘整理：保持标准采集策略"
            ),

            MarketRegime.HIGH_VOLATILITY: AdaptiveStrategy(
                market_regime=MarketRegime.HIGH_VOLATILITY,
                batch_size_multiplier=0.6,
                frequency_multiplier=2.0,
                priority_adjustment={
                    'high': 2.0,
                    'medium': 1.5,
                    'low': 1.0
                },
                resource_limit=0.7,
                description="高波动市场：降低批次大小，提高采集频率"
            ),

            MarketRegime.LOW_LIQUIDITY: AdaptiveStrategy(
                market_regime=MarketRegime.LOW_LIQUIDITY,
                batch_size_multiplier=0.5,
                frequency_multiplier=0.5,
                priority_adjustment={
                    'high': 1.5,
                    'medium': 0.8,
                    'low': 0.3
                },
                resource_limit=0.5,
                description="低流动性市场：大幅降低采集强度，保护资源"
            )
        }

    async def start_monitoring(self):
        """启动市场适应性监控"""
        if self.monitoring_active:
            logger.warning("市场适应性监控已在运行")
            return

        self.monitoring_active = True
        logger.info("启动市场适应性监控服务")

        while self.monitoring_active:
            try:
                await self._update_market_state()
                await self._apply_adaptive_strategy()

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"市场适应性监控循环异常: {e}")
                await asyncio.sleep(self.monitoring_interval)

    def stop_monitoring(self):
        """停止市场适应性监控"""
        self.monitoring_active = False
        logger.info("停止市场适应性监控服务")

    async def _update_market_state(self):
        """更新市场状态"""
        try:
            # 获取当前市场数据
            market_data = await self._get_market_data()

            if market_data:
                market_state = MarketState(
                    volatility_index=market_data.get('volatility_index', 0.02),
                    trading_volume=market_data.get('trading_volume', 1000000000),
                    market_sentiment=market_data.get('market_sentiment', 0.0),
                    sector_rotation=market_data.get('sector_rotation', {}),
                    timestamp=datetime.now()
                )

                # 添加到历史记录
                self.market_state_history.append(market_state)

                # 保持历史窗口大小
                cutoff_time = datetime.now() - timedelta(seconds=self.history_window)
                self.market_state_history = [
                    state for state in self.market_state_history
                    if state.timestamp > cutoff_time
                ]

                # 分析市场状态
                new_regime = self._analyze_market_regime()
                if new_regime != self.market_regime:
                    logger.info(f"市场状态变化: {self.market_regime.value} -> {new_regime.value}")
                    self.market_regime = new_regime
                    self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"更新市场状态失败: {e}")

    async def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """获取市场数据 - 基于真实A股数据"""
        try:
            # 优先从数据库获取最近的A股数据进行分析
            market_data = await self._get_market_data_from_database()

            if market_data:
                logger.debug(f"从数据库获取到市场数据: 波动率={market_data['volatility_index']:.4f}, 成交量={market_data['trading_volume']:,.0f}")
                return market_data

            # 如果数据库查询失败，回退到基于AKShare的实时数据获取
            logger.warning("数据库查询失败，尝试从AKShare获取实时数据")
            market_data = await self._get_market_data_from_akshare()

            if market_data:
                logger.debug(f"从AKShare获取到市场数据: 波动率={market_data['volatility_index']:.4f}, 成交量={market_data['trading_volume']:,.0f}")
                return market_data

            # 如果都失败，使用平滑的模拟数据作为最后后备
            logger.warning("所有数据源都不可用，使用平滑模拟数据")
            return self._get_smooth_simulated_data()

        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            # 发生异常时也使用平滑模拟数据
            return self._get_smooth_simulated_data()

    async def _get_market_data_from_database(self) -> Optional[Dict[str, Any]]:
        """从数据库获取最近的A股市场数据"""
        try:
            # 查询最近的A股主要指数和股票数据
            from src.gateway.web.postgresql_persistence import query_latest_stock_data_from_postgresql

            # 查询已配置的数据源中的股票数据
            # 注意：这里查询的是数据源ID，不是股票代码
            data_sources = ['akshare_stock_a']  # 主要A股数据源

            all_symbols = []
            for source_id in data_sources:
                try:
                    # 从每个数据源查询最近的数据
                    latest_data = query_latest_stock_data_from_postgresql(
                        source_id=source_id,  # 使用数据源ID
                        limit=20,  # 多查询一些数据用于统计
                        data_type='daily'
                    )

                    if latest_data:
                        # 提取唯一的股票代码
                        symbols_in_source = list(set(item.get('symbol', '') for item in latest_data if item.get('symbol')))
                        all_symbols.extend(symbols_in_source[:5])  # 每个数据源最多取5只股票

                except Exception as e:
                    logger.debug(f"查询数据源 {source_id} 失败: {e}")
                    continue

            volatility_data = []
            volume_data = []
            price_changes = []

            # 查询akshare_stock_a数据源中的所有股票数据
            try:
                all_stock_data = query_latest_stock_data_from_postgresql(
                    source_id='akshare_stock_a',  # 主要A股数据源
                    limit=100,  # 查询足够的数据用于统计
                    data_type='daily'
                )

                if all_stock_data:
                    # 按股票代码分组数据
                    stock_data_groups = {}
                    for item in all_stock_data:
                        symbol = item.get('symbol', '')
                        if symbol:
                            if symbol not in stock_data_groups:
                                stock_data_groups[symbol] = []
                            stock_data_groups[symbol].append(item)

                    # 对每只股票计算指标
                    for symbol, symbol_data in list(stock_data_groups.items())[:10]:  # 最多处理10只股票
                        try:
                            symbol_data = sorted(symbol_data, key=lambda x: x.get('date', ''), reverse=True)[:5]  # 最近5天

                            # 计算波动率（日收益率标准差）
                            prices = [float(item.get('close_price', 0)) for item in symbol_data if item.get('close_price')]
                            if len(prices) >= 2:
                                returns = []
                                for i in range(1, len(prices)):
                                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                                    returns.append(ret)

                                if returns:
                                    volatility = np.std(returns)
                                    volatility_data.append(volatility)

                                    # 计算涨跌幅（最新一天相对前一天）
                                    latest_price = prices[0]  # 排序后最新的在前面
                                    prev_price = prices[1]
                                    change_pct = (latest_price - prev_price) / prev_price
                                    price_changes.append(change_pct)

                            # 计算成交量
                            volumes = [float(item.get('volume', 0)) for item in symbol_data if item.get('volume')]
                            if volumes:
                                avg_volume = np.mean(volumes)
                                volume_data.append(avg_volume)

                        except Exception as e:
                            logger.debug(f"处理股票 {symbol} 数据失败: {e}")
                            continue

            except Exception as e:
                logger.error(f"查询A股数据源失败: {e}")
                return None

            if not volatility_data or not volume_data:
                logger.warning("数据库中没有足够的A股数据用于计算市场指标")
                return None

            # 计算综合指标
            avg_volatility = np.mean(volatility_data)
            total_volume = np.sum(volume_data)

            # 计算市场情绪（基于价格变化的平均值）
            avg_sentiment = np.mean(price_changes) if price_changes else 0.0

            # 简单的板块轮动分析（基于不同类型股票的表现）
            sector_rotation = {
                'finance': np.mean([change for change, symbol in zip(price_changes[:3], all_symbols[:3]) if '00' in symbol or '60' in symbol]),  # 银行股
                'consumer': np.mean([change for change, symbol in zip(price_changes[3:], all_symbols[3:]) if '858' in symbol]),  # 白酒股
                'technology': 0.0,  # 暂时设为0
                'healthcare': 0.0,  # 暂时设为0
                'energy': 0.0       # 暂时设为0
            }

            # 处理NaN值
            sector_rotation = {k: float(v) if not np.isnan(v) else 0.0 for k, v in sector_rotation.items()}

            return {
                'volatility_index': float(avg_volatility),
                'trading_volume': float(total_volume),
                'market_sentiment': float(avg_sentiment),
                'sector_rotation': sector_rotation,
                'data_source': 'database_a_stock',
                'sample_count': len(stock_data_groups),
                'time_window': '5_trading_days'
            }

        except Exception as e:
            logger.error(f"从数据库获取市场数据失败: {e}")
            return None

    async def _get_market_data_from_akshare(self) -> Optional[Dict[str, Any]]:
        """从AKShare获取实时A股市场数据"""
        try:
            import akshare as ak

            # 获取主要指数的实时数据
            index_data = ak.stock_zh_index_spot_em()

            if index_data is None or index_data.empty:
                logger.warning("AKShare未返回指数数据")
                return None

            # 筛选主要指数
            major_indices = index_data[
                index_data['代码'].isin(['000001', '399001', '399006'])
            ]

            if major_indices.empty:
                logger.warning("未找到主要指数数据")
                return None

            # 计算波动率（基于涨跌幅的标准差）
            change_pcts = []
            volumes = []

            for _, row in major_indices.iterrows():
                try:
                    change_pct = float(row.get('涨跌幅', '0').replace('%', ''))
                    volume = float(row.get('成交量', '0'))

                    change_pcts.append(change_pct / 100.0)  # 转换为小数
                    volumes.append(volume)
                except (ValueError, TypeError):
                    continue

            if not change_pcts:
                logger.warning("无法计算指数波动率")
                return None

            # 计算综合指标
            avg_volatility = np.std(change_pcts)  # 涨跌幅的标准差作为波动率
            total_volume = np.sum(volumes) if volumes else 0
            avg_sentiment = np.mean(change_pcts)  # 平均涨跌幅作为市场情绪

            # 简单的板块轮动（基于指数表现差异）
            sector_rotation = {
                'technology': change_pcts[0] * 0.8 if len(change_pcts) > 0 else 0.0,  # 创业板指
                'finance': change_pcts[0] * 0.6 if len(change_pcts) > 0 else 0.0,     # 上证指数
                'consumer': change_pcts[1] * 0.7 if len(change_pcts) > 1 else 0.0,    # 深证成指
                'healthcare': change_pcts[2] * 0.5 if len(change_pcts) > 2 else 0.0,  # 创业板指
                'energy': change_pcts[0] * 0.4 if len(change_pcts) > 0 else 0.0       # 上证指数
            }

            return {
                'volatility_index': float(avg_volatility),
                'trading_volume': float(total_volume),
                'market_sentiment': float(avg_sentiment),
                'sector_rotation': sector_rotation,
                'data_source': 'akshare_realtime',
                'sample_count': len(major_indices),
                'time_window': 'realtime'
            }

        except Exception as e:
            logger.error(f"从AKShare获取市场数据失败: {e}")
            return None

    def _get_smooth_simulated_data(self) -> Dict[str, Any]:
        """获取平滑的模拟数据作为后备方案"""
        try:
            # 使用更稳定的模拟数据，避免随机性过大
            # 基于历史平均水平生成

            # 相对稳定的波动率（基于历史A股平均波动率）
            base_volatility = 0.025  # 2.5%的日波动率
            volatility_noise = np.random.normal(0, 0.005)  # 小幅随机噪声
            volatility_index = max(0.005, base_volatility + volatility_noise)

            # 相对稳定的成交量（基于典型A股成交量水平）
            base_volume = 80000000000  # 8000亿基准成交量
            volume_multiplier = np.random.normal(1.0, 0.2)  # 正态分布
            trading_volume = max(10000000000, base_volume * volume_multiplier)  # 至少1000亿

            # 轻微的市场情绪（避免极端值）
            market_sentiment = np.random.normal(0.0, 0.3)  # 均值为0，标准差0.3
            market_sentiment = max(-0.8, min(0.8, market_sentiment))  # 限制范围

            # 温和的板块轮动
            sector_base = np.random.normal(0.0, 0.1)
            sectors = ['technology', 'finance', 'consumer', 'healthcare', 'energy']
            sector_rotation = {
                sector: float(sector_base + np.random.normal(0, 0.05))
                for sector in sectors
            }

            logger.info(f"使用平滑模拟数据: 波动率={volatility_index:.4f}, 成交量={trading_volume:,.0f}亿, 情绪={market_sentiment:.3f}")

            return {
                'volatility_index': float(volatility_index),
                'trading_volume': float(trading_volume),
                'market_sentiment': float(market_sentiment),
                'sector_rotation': sector_rotation,
                'data_source': 'smooth_simulation',
                'sample_count': 0,
                'time_window': 'simulation'
            }

        except Exception as e:
            logger.error(f"生成平滑模拟数据失败: {e}")
            # 返回最基本的默认值
            return {
                'volatility_index': 0.025,
                'trading_volume': 80000000000.0,
                'market_sentiment': 0.0,
                'sector_rotation': {
                    'technology': 0.0, 'finance': 0.0, 'consumer': 0.0,
                    'healthcare': 0.0, 'energy': 0.0
                },
                'data_source': 'default_fallback',
                'sample_count': 0,
                'time_window': 'fallback'
            }

    def _analyze_market_regime(self) -> MarketRegime:
        """分析当前市场状态"""
        try:
            if not self.market_state_history:
                return MarketRegime.SIDEWAYS

            # 获取最近的状态
            recent_states = self.market_state_history[-5:]  # 最近5个状态

            # 计算平均波动率
            avg_volatility = np.mean([state.volatility_index for state in recent_states])

            # 计算平均成交量（相对于基准）
            avg_volume = np.mean([state.trading_volume for state in recent_states])
            volume_ratio = avg_volume / 1000000000  # 相对于100亿的基准

            # 计算平均市场情绪
            avg_sentiment = np.mean([state.market_sentiment for state in recent_states])

            # 基于阈值判断市场状态
            if avg_volatility > self.volatility_threshold * 1.5:
                return MarketRegime.HIGH_VOLATILITY
            elif volume_ratio < self.volume_threshold:
                return MarketRegime.LOW_LIQUIDITY
            elif avg_sentiment > 0.3:
                return MarketRegime.BULL
            elif avg_sentiment < -0.3:
                return MarketRegime.BEAR
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.error(f"市场状态分析失败: {e}")
            return MarketRegime.SIDEWAYS

    async def _apply_adaptive_strategy(self):
        """应用适应性策略"""
        try:
            strategy = self.adaptive_strategies.get(self.market_regime)
            if not strategy:
                return

            # 获取当前调度器
            from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler
            scheduler = get_data_collection_scheduler()

            if not scheduler or not scheduler.is_running():
                return

            # 应用策略调整
            await self._adjust_scheduler_parameters(scheduler, strategy)

            logger.debug(f"应用市场适应性策略: {strategy.description}")

        except Exception as e:
            logger.error(f"应用适应性策略失败: {e}")

    async def _adjust_scheduler_parameters(self, scheduler, strategy: AdaptiveStrategy):
        """调整调度器参数"""
        try:
            # 调整批次大小
            original_batch_size = getattr(scheduler, 'default_batch_size', 50)
            new_batch_size = int(original_batch_size * strategy.batch_size_multiplier)
            new_batch_size = max(5, min(new_batch_size, 200))  # 限制在合理范围内

            # 调整采集频率
            original_interval = getattr(scheduler, 'default_interval', 60)
            new_interval = original_interval / strategy.frequency_multiplier
            new_interval = max(5, min(new_interval, 300))  # 限制在合理范围内

            # 调整优先级权重
            original_priorities = getattr(scheduler, 'priority_multipliers',
                                        {'high': 1.0, 'medium': 1.0, 'low': 1.0})
            new_priorities = {}
            for level, multiplier in original_priorities.items():
                adjustment = strategy.priority_adjustment.get(level, 1.0)
                new_priorities[level] = multiplier * adjustment

            # 应用调整
            scheduler.adjust_parameters(
                batch_size=new_batch_size,
                interval_seconds=new_interval,
                priority_multipliers=new_priorities
            )

            logger.info(
                f"市场适应性调整: 批次大小 {original_batch_size}->{new_batch_size}, "
                f"采集间隔 {original_interval:.1f}->{new_interval:.1f}秒, "
                f"优先级: {new_priorities}"
            )

        except Exception as e:
            logger.error(f"调整调度器参数失败: {e}")

    def get_current_adaptive_config(self) -> Dict[str, Any]:
        """获取当前适应性配置"""
        try:
            strategy = self.adaptive_strategies.get(self.market_regime)

            return {
                'market_regime': self.market_regime.value,
                'strategy_description': strategy.description if strategy else '未知',
                'batch_size_multiplier': strategy.batch_size_multiplier if strategy else 1.0,
                'frequency_multiplier': strategy.frequency_multiplier if strategy else 1.0,
                'priority_adjustment': strategy.priority_adjustment if strategy else {},
                'resource_limit': strategy.resource_limit if strategy else 0.8,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'monitoring_active': self.monitoring_active
            }

        except Exception as e:
            logger.error(f"获取适应性配置失败: {e}")
            return {}

    def get_market_state_summary(self) -> Dict[str, Any]:
        """获取市场状态摘要"""
        try:
            if not self.market_state_history:
                return {}

            recent_states = self.market_state_history[-10:]  # 最近10个状态

            return {
                'current_regime': self.market_regime.value,
                'avg_volatility': np.mean([s.volatility_index for s in recent_states]),
                'avg_volume': np.mean([s.trading_volume for s in recent_states]),
                'avg_sentiment': np.mean([s.market_sentiment for s in recent_states]),
                'data_points': len(self.market_state_history),
                'time_window_seconds': self.history_window
            }

        except Exception as e:
            logger.error(f"获取市场状态摘要失败: {e}")
            return {}


# 全局单例实例
_market_monitor_instance = None

def get_market_adaptive_monitor() -> MarketAdaptiveMonitor:
    """获取市场适应性监控器单例实例"""
    global _market_monitor_instance
    if _market_monitor_instance is None:
        _market_monitor_instance = MarketAdaptiveMonitor()
    return _market_monitor_instance