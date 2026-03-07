"""
事件驱动策略 - 基于市场事件的量化交易策略

本模块提供事件驱动型交易策略，支持：
1. 事件检测和分类（财报、并购、政策、宏观等）
2. 事件影响评估和预测
3. 事件驱动交易信号生成
4. 事件风险管理和仓位控制
5. 事件后效应追踪

作者: 策略团队
创建日期: 2026-02-21
版本: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import pandas as pd
import numpy as np
from collections import defaultdict
import json

from .base_strategy import BaseStrategy, StrategyConfig, Signal, SignalType
from src.trading.risk.risk_manager import RiskManager
from src.common.exceptions import StrategyError


# 配置日志
logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型"""
    EARNINGS = auto()           # 财报事件
    MERGER_ACQUISITION = auto() # 并购事件
    PRODUCT_LAUNCH = auto()     # 产品发布
    REGULATORY = auto()         # 监管政策
    MACRO_ECONOMIC = auto()     # 宏观经济
    SECTOR_ROTATION = auto()    # 板块轮动
    INSIDER_TRADING = auto()    # 内幕交易
    SHARE_BUYBACK = auto()      # 股票回购
    DIVIDEND = auto()           # 分红派息
    ANALYST_RATING = auto()     # 分析师评级


class EventImpact(Enum):
    """事件影响程度"""
    VERY_NEGATIVE = -2    # 非常负面
    NEGATIVE = -1         # 负面
    NEUTRAL = 0           # 中性
    POSITIVE = 1          # 正面
    VERY_POSITIVE = 2     # 非常正面


class EventStatus(Enum):
    """事件状态"""
    PENDING = "pending"           # 待发生
    OCCURRED = "occurred"         # 已发生
    PROCESSING = "processing"     # 处理中
    COMPLETED = "completed"       # 已完成
    CANCELLED = "cancelled"       # 已取消


@dataclass
class MarketEvent:
    """市场事件"""
    event_id: str
    event_type: EventType
    symbol: str
    title: str
    description: str
    expected_time: datetime
    actual_time: Optional[datetime] = None
    impact_prediction: EventImpact = EventImpact.NEUTRAL
    actual_impact: Optional[EventImpact] = None
    confidence: float = 0.5
    status: EventStatus = EventStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_symbols: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "symbol": self.symbol,
            "title": self.title,
            "description": self.description,
            "expected_time": self.expected_time.isoformat(),
            "actual_time": self.actual_time.isoformat() if self.actual_time else None,
            "impact_prediction": self.impact_prediction.name,
            "actual_impact": self.actual_impact.name if self.actual_impact else None,
            "confidence": self.confidence,
            "status": self.status.value,
            "metadata": self.metadata,
            "related_symbols": self.related_symbols
        }


@dataclass
class EventSignal:
    """事件交易信号"""
    signal_id: str
    event: MarketEvent
    signal_type: SignalType
    direction: str  # 'long' or 'short'
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: float = 0.0
    expected_return: float = 0.0
    risk_score: float = 0.0
    time_horizon: int = 1  # 天数
    confidence: float = 0.5
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "signal_id": self.signal_id,
            "event": self.event.to_dict(),
            "signal_type": self.signal_type.value,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "position_size": self.position_size,
            "expected_return": self.expected_return,
            "risk_score": self.risk_score,
            "time_horizon": self.time_horizon,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


@dataclass
class EventBacktestResult:
    """事件回测结果"""
    event_type: EventType
    total_events: int
    successful_trades: int
    failed_trades: int
    avg_return: float
    max_return: float
    min_return: float
    win_rate: float
    sharpe_ratio: float
    avg_holding_period: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_type": self.event_type.name,
            "total_events": self.total_events,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "avg_return": self.avg_return,
            "max_return": self.max_return,
            "min_return": self.min_return,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_holding_period": self.avg_holding_period
        }


class EventDetector:
    """事件检测器"""
    
    def __init__(self):
        """初始化事件检测器"""
        self._event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._detected_events: List[MarketEvent] = []
        
    def register_handler(self, event_type: EventType, handler: Callable):
        """注册事件处理器"""
        self._event_handlers[event_type].append(handler)
    
    async def detect_events(
        self,
        symbol: str,
        data: pd.DataFrame,
        news_data: Optional[List[Dict]] = None
    ) -> List[MarketEvent]:
        """
        检测市场事件
        
        参数:
            symbol: 股票代码
            data: 市场数据
            news_data: 新闻数据
            
        返回:
            List[MarketEvent]: 检测到的事件列表
        """
        events = []
        
        # 检测价格异常波动
        price_events = self._detect_price_anomalies(symbol, data)
        events.extend(price_events)
        
        # 检测成交量异常
        volume_events = self._detect_volume_anomalies(symbol, data)
        events.extend(volume_events)
        
        # 检测技术形态突破
        technical_events = self._detect_technical_breakouts(symbol, data)
        events.extend(technical_events)
        
        # 从新闻中检测事件
        if news_data:
            news_events = self._detect_events_from_news(symbol, news_data)
            events.extend(news_events)
        
        # 存储检测到的事件
        self._detected_events.extend(events)
        
        # 触发事件处理器
        for event in events:
            for handler in self._event_handlers[event.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"事件处理失败: {e}")
        
        return events
    
    def _detect_price_anomalies(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> List[MarketEvent]:
        """检测价格异常"""
        events = []
        
        if len(data) < 20:
            return events
        
        # 计算价格变化
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # 检测大幅波动
        latest_return = data['returns'].iloc[-1]
        avg_volatility = data['volatility'].iloc[-1]
        
        if abs(latest_return) > 3 * avg_volatility:
            event_type = EventType.MACRO_ECONOMIC if abs(latest_return) > 0.05 else EventType.SECTOR_ROTATION
            
            event = MarketEvent(
                event_id=f"price_anomaly_{symbol}_{int(datetime.now().timestamp())}",
                event_type=event_type,
                symbol=symbol,
                title=f"{symbol} 价格异常波动",
                description=f"价格变动: {latest_return:.2%}",
                expected_time=datetime.now(),
                actual_time=datetime.now(),
                impact_prediction=EventImpact.POSITIVE if latest_return > 0 else EventImpact.NEGATIVE,
                confidence=0.7,
                status=EventStatus.OCCURRED,
                metadata={
                    "return": latest_return,
                    "volatility": avg_volatility
                }
            )
            
            events.append(event)
        
        return events
    
    def _detect_volume_anomalies(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> List[MarketEvent]:
        """检测成交量异常"""
        events = []
        
        if len(data) < 20 or 'volume' not in data.columns:
            return events
        
        # 计算成交量均值
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        latest_volume = data['volume'].iloc[-1]
        
        # 检测成交量激增
        if latest_volume > 3 * avg_volume:
            event = MarketEvent(
                event_id=f"volume_spike_{symbol}_{int(datetime.now().timestamp())}",
                event_type=EventType.SECTOR_ROTATION,
                symbol=symbol,
                title=f"{symbol} 成交量异常",
                description=f"成交量激增: {latest_volume/avg_volume:.1f}倍",
                expected_time=datetime.now(),
                actual_time=datetime.now(),
                impact_prediction=EventImpact.NEUTRAL,
                confidence=0.6,
                status=EventStatus.OCCURRED,
                metadata={
                    "volume_ratio": latest_volume / avg_volume
                }
            )
            
            events.append(event)
        
        return events
    
    def _detect_technical_breakouts(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> List[MarketEvent]:
        """检测技术形态突破"""
        events = []
        
        if len(data) < 50:
            return events
        
        # 计算移动平均线
        data['ma20'] = data['close'].rolling(window=20).mean()
        data['ma50'] = data['close'].rolling(window=50).mean()
        
        latest_close = data['close'].iloc[-1]
        latest_ma20 = data['ma20'].iloc[-1]
        latest_ma50 = data['ma50'].iloc[-1]
        prev_close = data['close'].iloc[-2]
        prev_ma20 = data['ma20'].iloc[-2]
        
        # 检测金叉
        if prev_close < prev_ma20 and latest_close > latest_ma20:
            event = MarketEvent(
                event_id=f"golden_cross_{symbol}_{int(datetime.now().timestamp())}",
                event_type=EventType.TECHNICAL_BREAKOUT,
                symbol=symbol,
                title=f"{symbol} 突破20日均线",
                description="价格突破20日均线，技术形态转强",
                expected_time=datetime.now(),
                actual_time=datetime.now(),
                impact_prediction=EventImpact.POSITIVE,
                confidence=0.65,
                status=EventStatus.OCCURRED
            )
            events.append(event)
        
        # 检测死叉
        elif prev_close > prev_ma20 and latest_close < latest_ma20:
            event = MarketEvent(
                event_id=f"death_cross_{symbol}_{int(datetime.now().timestamp())}",
                event_type=EventType.TECHNICAL_BREAKOUT,
                symbol=symbol,
                title=f"{symbol} 跌破20日均线",
                description="价格跌破20日均线，技术形态转弱",
                expected_time=datetime.now(),
                actual_time=datetime.now(),
                impact_prediction=EventImpact.NEGATIVE,
                confidence=0.65,
                status=EventStatus.OCCURRED
            )
            events.append(event)
        
        return events
    
    def _detect_events_from_news(
        self,
        symbol: str,
        news_data: List[Dict]
    ) -> List[MarketEvent]:
        """从新闻中检测事件"""
        events = []
        
        # 关键词映射
        event_keywords = {
            EventType.EARNINGS: ['earnings', 'revenue', 'profit', 'eps', 'beat', 'miss'],
            EventType.MERGER_ACQUISITION: ['merger', 'acquisition', 'buyout', 'takeover'],
            EventType.PRODUCT_LAUNCH: ['launch', 'release', 'unveil', 'announce'],
            EventType.REGULATORY: ['regulation', 'sec', 'fda', 'approval', 'ban'],
            EventType.SHARE_BUYBACK: ['buyback', 'repurchase', 'dividend']
        }
        
        for news in news_data:
            title = news.get('title', '').lower()
            content = news.get('content', '').lower()
            text = title + ' ' + content
            
            for event_type, keywords in event_keywords.items():
                if any(keyword in text for keyword in keywords):
                    # 判断影响方向
                    positive_words = ['beat', 'exceed', 'strong', 'growth', 'surge', 'rally']
                    negative_words = ['miss', 'decline', 'fall', 'weak', 'drop', 'concern']
                    
                    pos_count = sum(1 for word in positive_words if word in text)
                    neg_count = sum(1 for word in negative_words if word in text)
                    
                    if pos_count > neg_count:
                        impact = EventImpact.POSITIVE
                    elif neg_count > pos_count:
                        impact = EventImpact.NEGATIVE
                    else:
                        impact = EventImpact.NEUTRAL
                    
                    event = MarketEvent(
                        event_id=f"news_{event_type.name}_{symbol}_{int(datetime.now().timestamp())}",
                        event_type=event_type,
                        symbol=symbol,
                        title=news.get('title', ''),
                        description=news.get('content', '')[:200],
                        expected_time=datetime.now(),
                        actual_time=datetime.now(),
                        impact_prediction=impact,
                        confidence=0.6,
                        status=EventStatus.OCCURRED,
                        metadata={
                            "source": news.get('source', ''),
                            "url": news.get('url', '')
                        }
                    )
                    
                    events.append(event)
                    break  # 一个新闻只对应一个事件
        
        return events


class EventDrivenStrategy(BaseStrategy):
    """
    事件驱动策略
    
    基于市场事件（财报、并购、政策等）生成交易信号的策略。
    支持事件检测、影响评估、信号生成和风险管理。
    
    使用示例:
        config = StrategyConfig(
            name="EventDriven",
            symbols=["AAPL", "MSFT"],
            initial_capital=100000
        )
        
        strategy = EventDrivenStrategy(config)
        await strategy.initialize()
        
        # 处理市场数据
        signals = await strategy.on_market_data(market_data)
    """
    
    def __init__(self, config: StrategyConfig):
        """
        初始化事件驱动策略
        
        参数:
            config: 策略配置
        """
        super().__init__(config)
        
        self.name = "EventDrivenStrategy"
        self.description = "基于市场事件的量化交易策略"
        
        # 事件检测器
        self._event_detector = EventDetector()
        
        # 事件历史
        self._event_history: List[MarketEvent] = []
        
        # 活跃信号
        self._active_signals: Dict[str, EventSignal] = {}
        
        # 事件影响模型
        self._impact_models: Dict[EventType, Any] = {}
        
        # 回测结果
        self._backtest_results: Dict[EventType, EventBacktestResult] = {}
        
        # 配置参数
        self._max_position_size = 0.1  # 最大仓位10%
        self._max_concurrent_events = 5  # 最大并发事件数
        self._min_confidence = 0.6  # 最小置信度
        
        # 风险管理
        self._risk_manager = RiskManager()
        
        logger.info("事件驱动策略已初始化")
    
    async def initialize(self):
        """初始化策略"""
        await super().initialize()
        
        # 加载历史事件数据
        await self._load_historical_events()
        
        # 训练影响模型
        await self._train_impact_models()
        
        logger.info("事件驱动策略初始化完成")
    
    async def on_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        news_data: Optional[List[Dict]] = None
    ) -> List[Signal]:
        """
        处理市场数据
        
        参数:
            data: 市场数据
            symbol: 股票代码
            news_data: 新闻数据
            
        返回:
            List[Signal]: 交易信号列表
        """
        signals = []
        
        # 检测事件
        events = await self._event_detector.detect_events(symbol, data, news_data)
        
        for event in events:
            # 评估事件影响
            impact = await self._assess_event_impact(event, data)
            
            # 生成交易信号
            if impact.confidence >= self._min_confidence:
                signal = self._generate_signal(event, impact, data)
                
                if signal:
                    signals.append(signal)
                    self._active_signals[signal.signal_id] = signal
                    
                    logger.info(f"生成事件驱动信号: {signal.signal_id} "
                              f"事件: {event.event_type.name} "
                              f"方向: {signal.direction}")
        
        # 更新活跃信号
        await self._update_active_signals(data)
        
        return signals
    
    async def on_event(self, event: MarketEvent):
        """
        处理外部事件
        
        参数:
            event: 市场事件
        """
        logger.info(f"收到外部事件: {event.event_type.name} - {event.title}")
        
        # 存储事件
        self._event_history.append(event)
        
        # 如果事件已发生，评估影响
        if event.status == EventStatus.OCCURRED:
            # 获取相关数据
            data = await self._get_market_data(event.symbol)
            
            if data is not None:
                impact = await self._assess_event_impact(event, data)
                
                # 生成信号
                if impact.confidence >= self._min_confidence:
                    signal = self._generate_signal(event, impact, data)
                    
                    if signal:
                        # 发送信号
                        await self._emit_signal(signal)
    
    async def backtest(
        self,
        historical_events: List[MarketEvent],
        price_data: pd.DataFrame,
        holding_period: int = 5
    ) -> Dict[EventType, EventBacktestResult]:
        """
        回测策略
        
        参数:
            historical_events: 历史事件列表
            price_data: 历史价格数据
            holding_period: 持有期（天数）
            
        返回:
            Dict[EventType, EventBacktestResult]: 回测结果
        """
        results = {}
        
        # 按事件类型分组
        events_by_type: Dict[EventType, List[MarketEvent]] = defaultdict(list)
        for event in historical_events:
            events_by_type[event.event_type].append(event)
        
        # 对每个事件类型进行回测
        for event_type, events in events_by_type.items():
            trades = []
            
            for event in events:
                # 找到事件发生时的价格
                event_time = event.actual_time or event.expected_time
                
                # 简化处理：假设事件发生在某天开盘
                # 实际应该根据具体时间找到对应的价格
                
                # 模拟交易
                entry_price = self._get_price_at_time(price_data, event_time)
                
                if entry_price is None:
                    continue
                
                # 预测方向
                predicted_direction = 1 if event.impact_prediction in [EventImpact.POSITIVE, EventImpact.VERY_POSITIVE] else -1
                
                # 计算持有期收益
                exit_time = event_time + timedelta(days=holding_period)
                exit_price = self._get_price_at_time(price_data, exit_time)
                
                if exit_price is None:
                    continue
                
                actual_return = (exit_price - entry_price) / entry_price * predicted_direction
                
                trades.append({
                    'event_id': event.event_id,
                    'predicted_direction': predicted_direction,
                    'actual_return': actual_return,
                    'success': actual_return > 0
                })
            
            # 计算统计指标
            if trades:
                returns = [t['actual_return'] for t in trades]
                successful = sum(1 for t in trades if t['success'])
                
                result = EventBacktestResult(
                    event_type=event_type,
                    total_events=len(events),
                    successful_trades=successful,
                    failed_trades=len(trades) - successful,
                    avg_return=np.mean(returns),
                    max_return=max(returns),
                    min_return=min(returns),
                    win_rate=successful / len(trades),
                    sharpe_ratio=np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                    avg_holding_period=holding_period
                )
                
                results[event_type] = result
        
        self._backtest_results = results
        
        return results
    
    async def _load_historical_events(self):
        """加载历史事件数据"""
        # 实际实现中从数据库或文件加载
        logger.info("加载历史事件数据")
        # 这里可以加载预定义的事件数据
    
    async def _train_impact_models(self):
        """训练事件影响模型"""
        logger.info("训练事件影响模型")
        
        # 简化实现：使用基于规则的模型
        # 实际生产环境可以使用机器学习模型
        
        for event_type in EventType:
            # 基于历史数据训练
            self._impact_models[event_type] = {
                "avg_impact": 0.02,  # 平均影响2%
                "success_rate": 0.6   # 成功率60%
            }
    
    async def _assess_event_impact(
        self,
        event: MarketEvent,
        data: pd.DataFrame
    ) -> EventSignal:
        """评估事件影响"""
        # 获取模型
        model = self._impact_models.get(event.event_type, {})
        
        # 基于事件类型和历史表现评估
        base_impact = model.get("avg_impact", 0.02)
        base_success_rate = model.get("success_rate", 0.5)
        
        # 根据事件属性调整
        confidence = event.confidence * base_success_rate
        
        # 计算预期收益
        if event.impact_prediction in [EventImpact.POSITIVE, EventImpact.VERY_POSITIVE]:
            expected_return = base_impact * (2 if event.impact_prediction == EventImpact.VERY_POSITIVE else 1)
            direction = 'long'
        elif event.impact_prediction in [EventImpact.NEGATIVE, EventImpact.VERY_NEGATIVE]:
            expected_return = -base_impact * (2 if event.impact_prediction == EventImpact.VERY_NEGATIVE else 1)
            direction = 'short'
        else:
            expected_return = 0
            direction = 'neutral'
        
        # 计算风险分数
        risk_score = self._calculate_risk_score(event, data)
        
        # 创建影响评估结果
        return EventSignal(
            signal_id=f"impact_{event.event_id}",
            event=event,
            signal_type=SignalType.EVENT_DRIVEN,
            direction=direction,
            expected_return=expected_return,
            risk_score=risk_score,
            confidence=confidence,
            reasoning=f"基于{event.event_type.name}事件的影响评估"
        )
    
    def _generate_signal(
        self,
        event: MarketEvent,
        impact: EventSignal,
        data: pd.DataFrame
    ) -> Optional[Signal]:
        """生成交易信号"""
        if impact.direction == 'neutral':
            return None
        
        # 获取当前价格
        current_price = data['close'].iloc[-1] if not data.empty else None
        
        if current_price is None:
            return None
        
        # 计算目标价和止损价
        if impact.direction == 'long':
            target_price = current_price * (1 + abs(impact.expected_return))
            stop_loss = current_price * (1 - abs(impact.expected_return) * 0.5)
        else:
            target_price = current_price * (1 - abs(impact.expected_return))
            stop_loss = current_price * (1 + abs(impact.expected_return) * 0.5)
        
        # 计算仓位大小
        position_size = self._calculate_position_size(impact.risk_score)
        
        return Signal(
            signal_id=impact.signal_id,
            symbol=event.symbol,
            signal_type=impact.signal_type,
            direction=SignalDirection.BUY if impact.direction == 'long' else SignalDirection.SELL,
            strength=impact.confidence,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            position_size=position_size,
            metadata={
                "event_type": event.event_type.name,
                "event_title": event.title,
                "expected_return": impact.expected_return,
                "risk_score": impact.risk_score,
                "time_horizon": impact.time_horizon
            }
        )
    
    def _calculate_position_size(self, risk_score: float) -> float:
        """计算仓位大小"""
        # 基于风险分数调整仓位
        # 风险越高，仓位越小
        risk_adjustment = 1 - risk_score
        position_size = self._max_position_size * risk_adjustment
        
        return max(0.01, position_size)  # 最小1%仓位
    
    def _calculate_risk_score(self, event: MarketEvent, data: pd.DataFrame) -> float:
        """计算风险分数"""
        # 基础风险
        base_risk = 0.3
        
        # 根据事件类型调整
        event_risk = {
            EventType.EARNINGS: 0.4,
            EventType.MERGER_ACQUISITION: 0.5,
            EventType.REGULATORY: 0.6,
            EventType.MACRO_ECONOMIC: 0.3,
            EventType.SECTOR_ROTATION: 0.2
        }
        
        risk = event_risk.get(event.event_type, base_risk)
        
        # 根据市场波动率调整
        if not data.empty and len(data) > 20:
            volatility = data['close'].pct_change().std()
            risk += volatility * 10  # 波动率越高，风险越高
        
        return min(1.0, risk)
    
    async def _update_active_signals(self, data: pd.DataFrame):
        """更新活跃信号"""
        current_time = datetime.now()
        
        for signal_id, signal in list(self._active_signals.items()):
            # 检查是否达到目标或止损
            current_price = data['close'].iloc[-1] if not data.empty else None
            
            if current_price is None:
                continue
            
            should_close = False
            close_reason = ""
            
            if signal.direction == 'long':
                if current_price >= signal.target_price:
                    should_close = True
                    close_reason = "达到目标价"
                elif current_price <= signal.stop_loss:
                    should_close = True
                    close_reason = "触发止损"
            else:  # short
                if current_price <= signal.target_price:
                    should_close = True
                    close_reason = "达到目标价"
                elif current_price >= signal.stop_loss:
                    should_close = True
                    close_reason = "触发止损"
            
            # 检查时间 horizon
            event_time = signal.event.actual_time or signal.event.expected_time
            if (current_time - event_time).days > signal.time_horizon:
                should_close = True
                close_reason = "超过持有期"
            
            if should_close:
                logger.info(f"关闭事件驱动信号: {signal_id}, 原因: {close_reason}")
                del self._active_signals[signal_id]
    
    def _get_price_at_time(
        self,
        price_data: pd.DataFrame,
        target_time: datetime
    ) -> Optional[float]:
        """获取特定时间的价格"""
        # 简化实现：找到最接近的时间
        if price_data.empty:
            return None
        
        # 假设price_data有datetime索引
        try:
            closest_idx = price_data.index.get_loc(target_time, method='nearest')
            return price_data.iloc[closest_idx]['close']
        except Exception:
            return None
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取市场数据"""
        # 实际实现中从数据源获取
        # 这里简化处理
        return None
    
    async def _emit_signal(self, signal: EventSignal):
        """发送信号"""
        # 实际实现中发送到交易系统
        logger.info(f"发送事件驱动信号: {signal.signal_id}")
    
    def get_active_events(self) -> List[MarketEvent]:
        """获取活跃事件"""
        return [signal.event for signal in self._active_signals.values()]
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """获取事件统计"""
        return {
            "total_events_detected": len(self._event_history),
            "active_signals": len(self._active_signals),
            "events_by_type": {
                event_type.name: len([
                    e for e in self._event_history
                    if e.event_type == event_type
                ])
                for event_type in EventType
            },
            "backtest_results": {
                event_type.name: result.to_dict()
                for event_type, result in self._backtest_results.items()
            }
        }


# 便捷函数
def create_earnings_strategy(config: StrategyConfig) -> EventDrivenStrategy:
    """
    创建财报事件策略
    
    参数:
        config: 策略配置
        
    返回:
        EventDrivenStrategy: 财报事件策略
    """
    strategy = EventDrivenStrategy(config)
    
    # 配置专门处理财报事件
    strategy._min_confidence = 0.7
    strategy._max_position_size = 0.15
    
    return strategy


def create_merger_arbitrage_strategy(config: StrategyConfig) -> EventDrivenStrategy:
    """
    创建并购套利策略
    
    参数:
        config: 策略配置
        
    返回:
        EventDrivenStrategy: 并购套利策略
    """
    strategy = EventDrivenStrategy(config)
    
    # 配置专门处理并购事件
    strategy._min_confidence = 0.8
    strategy._max_position_size = 0.2
    
    return strategy
