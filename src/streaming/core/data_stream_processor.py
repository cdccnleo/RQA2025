#!/usr/bin/env python3
"""
实时数据流处理器
处理来自QMT和其他数据源的实时数据流，实现信号生成和决策
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading
import time
import queue

from src.infrastructure.logging.core.interfaces import get_logger

logger = logging.getLogger(__name__)


class DataSourceType(Enum):

    """数据源类型"""
    QMT = "qmt"
    MARKET_DATA = "market_data"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ALTERNATIVE_DATA = "alternative_data"


class SignalType(Enum):

    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class MarketData:

    """市场数据"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    high: float
    low: float
    open: float
    close: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None


@dataclass
class TradingSignal:

    """交易信号"""
    signal_id: str
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime
    reason: str
    strategy_name: str
    parameters: Dict[str, Any]


@dataclass
class ExecutionDecision:

    """执行决策"""
    decision_id: str
    signal_id: str
    symbol: str
    action: str
    quantity: float
    price: Optional[float] = None
    order_type: str = "market"
    timestamp: datetime = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now()


class DataStreamProcessor:

    """实时数据流处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 配置参数
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.processing_interval = self.config.get('processing_interval', 1.0)  # 秒
        self.signal_threshold = self.config.get('signal_threshold', 0.7)  # 信号阈值

        # 数据缓冲区
        self.data_buffer: Dict[str, List[MarketData]] = {}
        self.signal_queue = queue.Queue()
        self.decision_queue = queue.Queue()

        # 策略和处理器
        self.strategies: Dict[str, Any] = {}
        self.indicator_processors: Dict[str, Any] = {}
        self.risk_manager = None

        # 状态控制
        self.running = False
        self.processing_thread = None
        self.signal_thread = None

        # 统计信息
        self.stats = {
            'data_processed': 0,
            'signals_generated': 0,
            'decisions_made': 0,
            'last_update': datetime.now()
        }

        self.logger = get_logger(__name__)

    def start(self):
        """启动处理器"""
        if self.running:
            self.logger.warning("处理器已在运行中")
            return

        self.running = True

        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._data_processing_loop)
        self.signal_thread = threading.Thread(target=self._signal_processing_loop)

        self.processing_thread.daemon = True
        self.signal_thread.daemon = True

        self.processing_thread.start()
        self.signal_thread.start()

        self.logger.info("实时数据流处理器已启动")

    def stop(self):
        """停止处理器"""
        self.running = False

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=5)

        self.logger.info("实时数据流处理器已停止")

    def add_market_data(self, data: MarketData):
        """添加市场数据"""
        symbol = data.symbol

        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []

        # 添加数据到缓冲区
        self.data_buffer[symbol].append(data)

        # 保持缓冲区大小
        if len(self.data_buffer[symbol]) > self.buffer_size:
            self.data_buffer[symbol] = self.data_buffer[symbol][-self.buffer_size:]

        self.stats['data_processed'] += 1

    def register_strategy(self, name: str, strategy: Any):
        """注册策略"""
        self.strategies[name] = strategy
        self.logger.info(f"策略已注册: {name}")

    def register_indicator_processor(self, name: str, processor: Any):
        """注册指标处理器"""
        self.indicator_processors[name] = processor
        self.logger.info(f"指标处理器已注册: {name}")

    def set_risk_manager(self, risk_manager: Any):
        """设置风险管理器"""
        self.risk_manager = risk_manager
        self.logger.info("风险管理器已设置")

    def get_latest_data(self, symbol: str, n: int = 100) -> List[MarketData]:
        """获取最新的数据"""
        if symbol not in self.data_buffer:
            return []

        return self.data_buffer[symbol][-n:]

    def get_signal_queue(self) -> queue.Queue:
        """获取信号队列"""
        return self.signal_queue

    def get_decision_queue(self) -> queue.Queue:
        """获取决策队列"""
        return self.decision_queue

    def _data_processing_loop(self):
        """数据处理循环"""
        while self.running:
            try:
                # 处理每个标的的数据
                for symbol in list(self.data_buffer.keys()):
                    data_list = self.get_latest_data(symbol, 100)

                    if len(data_list) < 10:  # 需要足够的数据
                        continue

                    # 计算技术指标
                    self._calculate_indicators(symbol, data_list)

                    # 生成信号
                    self._generate_signals(symbol, data_list)

                time.sleep(self.processing_interval)

            except Exception as e:
                self.logger.error(f"数据处理循环异常: {e}")
                time.sleep(1)

    def _signal_processing_loop(self):
        """信号处理循环"""
        while self.running:
            try:
                # 从队列获取信号
                try:
                    signal = self.signal_queue.get(timeout=1)
                    decision = self._process_signal(signal)

                    if decision:
                        self.decision_queue.put(decision)
                        self.stats['decisions_made'] += 1

                    self.signal_queue.task_done()

                except queue.Empty:
                    continue

            except Exception as e:
                self.logger.error(f"信号处理循环异常: {e}")
                time.sleep(1)

    def _calculate_indicators(self, symbol: str, data_list: List[MarketData]):
        """计算技术指标"""
        if not data_list:
            return

        # 转换为DataFrame
        df = self._convert_to_dataframe(data_list)

        # 对每个指标处理器计算指标
        for name, processor in self.indicator_processors.items():
            try:
                # 计算指标
                result_df = processor.calculate_indicators(df)

                # 存储计算结果（可以用于后续分析）
                self._store_indicator_result(symbol, name, result_df)

            except Exception as e:
                self.logger.error(f"计算指标失败 {name}: {e}")

    def _generate_signals(self, symbol: str, data_list: List[MarketData]):
        """生成交易信号"""
        if not data_list:
            return

        latest_data = data_list[-1]

        # 对每个策略生成信号
        for strategy_name, strategy in self.strategies.items():
            try:
                # 准备数据
                df = self._convert_to_dataframe(data_list)

                # 生成信号
                signal_data = strategy.generate_signal(df)

                if signal_data and signal_data.get('signal') != 'HOLD':
                    signal_type = self._convert_signal_type(signal_data.get('signal'))

                    if signal_type:
                        signal = TradingSignal(
                            signal_id=self._generate_signal_id(),
                            symbol=symbol,
                            signal_type=signal_type,
                            confidence=signal_data.get('confidence', 0.5),
                            price=latest_data.price,
                            timestamp=datetime.now(),
                            reason=signal_data.get('reason', ''),
                            strategy_name=strategy_name,
                            parameters=signal_data
                        )

                        # 只有当信号强度超过阈值时才加入队列
                        if signal.confidence >= self.signal_threshold:
                            self.signal_queue.put(signal)
                            self.stats['signals_generated'] += 1

            except Exception as e:
                self.logger.error(f"生成信号失败 {strategy_name}: {e}")

    def _process_signal(self, signal: TradingSignal) -> Optional[ExecutionDecision]:
        """处理交易信号"""
        try:
            # 风险检查
            if self.risk_manager:
                risk_result = self.risk_manager.check_order_risk(
                    signal.symbol, 100, signal.price, {}  # 简化风险检查
                )

                if not risk_result.get('approved', False):
                    self.logger.warning(
                        f"信号 {signal.signal_id} 未通过风险检查: {risk_result.get('reason')}")
                    return None

            # 生成执行决策
            decision = ExecutionDecision(
                decision_id=self._generate_decision_id(),
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                action=signal.signal_type.value,
                quantity=self._calculate_position_size(signal),
                price=signal.price,
                order_type="market" if signal.signal_type in [
                    SignalType.BUY, SignalType.SELL] else "limit"
            )

            self.logger.info(
                f"生成执行决策: {decision.decision_id} for {signal.symbol} {decision.action}")
            return decision

        except Exception as e:
            self.logger.error(f"处理信号失败 {signal.signal_id}: {e}")
            return None

    def _convert_to_dataframe(self, data_list: List[MarketData]) -> pd.DataFrame:
        """转换为DataFrame"""
        data = {
            'timestamp': [d.timestamp for d in data_list],
            'open': [d.open for d in data_list],
            'high': [d.high for d in data_list],
            'low': [d.low for d in data_list],
            'close': [d.close for d in data_list],
            'volume': [d.volume for d in data_list]
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def _convert_signal_type(self, signal_str: str) -> Optional[SignalType]:
        """转换信号类型"""
        mapping = {
            'BUY': SignalType.BUY,
            'SELL': SignalType.SELL,
            'HOLD': SignalType.HOLD,
            'STRONG_BUY': SignalType.STRONG_BUY,
            'STRONG_SELL': SignalType.STRONG_SELL
        }
        return mapping.get(signal_str.upper())

    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """计算仓位大小"""
        # 简化计算，实际应该考虑风险管理和账户情况
        base_quantity = 100

        if signal.signal_type == SignalType.STRONG_BUY:
            return base_quantity * 1.5
        elif signal.signal_type == SignalType.STRONG_SELL:
            return base_quantity * 1.5
        elif signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            return base_quantity
        else:
            return 0

    def _store_indicator_result(self, symbol: str, processor_name: str, result_df: pd.DataFrame):
        """存储指标计算结果"""
        # 这里可以实现结果缓存逻辑

    def _generate_signal_id(self) -> str:
        """生成信号ID"""
        return f"SIG_{datetime.now().strftime('%Y % m % d % H % M % S % f')}"

    def _generate_decision_id(self) -> str:
        """生成决策ID"""
        return f"DEC_{datetime.now().strftime('%Y % m % d % H % M % S % f')}"

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

    def clear_buffers(self):
        """清空缓冲区"""
        self.data_buffer.clear()
        self.logger.info("数据缓冲区已清空")
