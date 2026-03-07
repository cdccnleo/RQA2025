#!/usr/bin/env python3
"""
RQA2025基础交易策略实现
创建基础策略框架和常用交易策略
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TradingStrategy(ABC):
    """交易策略抽象基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.config.get('name', self.__class__.__name__)
        self.logger = logging.getLogger(self.name)

        # 策略参数
        self.max_position = self.config.get('max_position', 100)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)

        # 策略状态
        self.position = 0
        self.entry_price = 0.0
        self.signals = []

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成交易信号

        Args:
            data: 包含价格数据的DataFrame

        Returns:
            交易信号字典
        """

    def calculate_position_size(self, price: float, signal: str) -> int:
        """计算仓位大小"""
        if signal in ['BUY', 'SELL']:
            # 基于风险计算仓位大小
            risk_amount = self.max_position * self.risk_per_trade
            position_size = int(risk_amount / price)
            return min(position_size, self.max_position)
        return 0

    def execute_signal(self, signal: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """执行交易信号"""
        signal_type = signal.get('signal', 'HOLD')
        quantity = self.calculate_position_size(current_price, signal_type)

        execution = {
            'timestamp': datetime.now(),
            'signal': signal_type,
            'price': current_price,
            'quantity': quantity,
            'reason': signal.get('reason', ''),
            'strategy': self.name,
            'success': quantity > 0
        }

        if signal_type == 'BUY' and self.position <= 0:
            self.position = quantity
            self.entry_price = current_price
            execution['action'] = 'OPEN_LONG'
        elif signal_type == 'SELL' and self.position >= 0:
            self.position = -quantity
            self.entry_price = current_price
            execution['action'] = 'OPEN_SHORT'
        elif signal_type == 'CLOSE' and self.position != 0:
            execution['action'] = 'CLOSE_POSITION'
            execution['pnl'] = (current_price - self.entry_price) * abs(self.position)
            self.position = 0
            self.entry_price = 0.0
        else:
            execution['action'] = 'NO_ACTION'
            execution['quantity'] = 0

        self.signals.append(execution)
        return execution

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取策略性能统计"""
        if not self.signals:
            return {'total_signals': 0, 'win_rate': 0.0, 'total_pnl': 0.0}

        successful_signals = [s for s in self.signals if s.get('success', False)]
        profitable_signals = [s for s in successful_signals if s.get('pnl', 0) > 0]

        total_pnl = sum(s.get('pnl', 0) for s in successful_signals)

        return {
            'total_signals': len(self.signals),
            'successful_signals': len(successful_signals),
            'profitable_signals': len(profitable_signals),
            'win_rate': len(profitable_signals) / len(successful_signals) if successful_signals else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(successful_signals) if successful_signals else 0
        }


class MovingAverageCrossoverStrategy(TradingStrategy):
    """均线交叉策略"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.short_period = self.config.get('short_period', 5)
        self.long_period = self.config.get('long_period', 20)

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成均线交叉信号"""
        try:
            if len(data) < self.long_period:
                return {'signal': 'HOLD', 'reason': '数据不足'}

            # 计算移动平均线
            data_copy = data.copy()
            data_copy['sma_short'] = data_copy['close'].rolling(window=self.short_period).mean()
            data_copy['sma_long'] = data_copy['close'].rolling(window=self.long_period).mean()

            # 获取最新数据
            latest = data_copy.iloc[-1]
            prev = data_copy.iloc[-2] if len(data_copy) > 1 else latest

            current_short = latest['sma_short']
            current_long = latest['sma_long']
            prev_short = prev['sma_short']
            prev_long = prev['sma_long']

            # 生成信号
            if pd.notna(current_short) and pd.notna(current_long) and pd.notna(prev_short) and pd.notna(prev_long):
                # 金叉：短期均线上穿长期均线
                if prev_short <= prev_long and current_short > current_long:
                    return {
                        'signal': 'BUY',
                        'reason': f'金叉信号: {self.short_period}日线穿过{self.long_period}日线',
                        'short_ma': current_short,
                        'long_ma': current_long
                    }
                # 死叉：短期均线下穿长期均线
                elif prev_short >= prev_long and current_short < current_long:
                    return {
                        'signal': 'SELL',
                        'reason': f'死叉信号: {self.short_period}日线跌破{self.long_period}日线',
                        'short_ma': current_short,
                        'long_ma': current_long
                    }

            return {'signal': 'HOLD', 'reason': '无明确交叉信号'}

        except Exception as e:
            self.logger.error(f"均线交叉策略信号生成失败: {e}")
            return {'signal': 'HOLD', 'reason': f'错误: {str(e)}'}


class RSIStrategy(TradingStrategy):
    """RSI超买超卖策略"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.period = self.config.get('period', 14)
        self.overbought_level = self.config.get('overbought', 70)
        self.oversold_level = self.config.get('oversold', 30)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not rsi.empty else 50

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成RSI信号"""
        try:
            if len(data) < self.period + 1:
                return {'signal': 'HOLD', 'reason': '数据不足'}

            # 计算RSI
            rsi = self.calculate_rsi(data['close'], self.period)

            # 生成信号
            if rsi <= self.oversold_level:
                return {
                    'signal': 'BUY',
                    'reason': f'RSI超卖信号: {rsi:.2f} <= {self.oversold_level}',
                    'rsi': rsi,
                    'level': 'oversold'
                }
            elif rsi >= self.overbought_level:
                return {
                    'signal': 'SELL',
                    'reason': f'RSI超买信号: {rsi:.2f} >= {self.overbought_level}',
                    'rsi': rsi,
                    'level': 'overbought'
                }
            else:
                return {
                    'signal': 'HOLD',
                    'reason': f'RSI中性: {rsi:.2f}',
                    'rsi': rsi,
                    'level': 'neutral'
                }

        except Exception as e:
            self.logger.error(f"RSI策略信号生成失败: {e}")
            return {'signal': 'HOLD', 'reason': f'错误: {str(e)}'}


class BollingerBandsStrategy(TradingStrategy):
    """布林带策略"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.period = self.config.get('period', 20)
        self.std_dev = self.config.get('std_dev', 2)

    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算布林带"""
        close = data['close']
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)

        return {
            'upper': upper_band.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower_band.iloc[-1]
        }

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成布林带信号"""
        try:
            if len(data) < self.period:
                return {'signal': 'HOLD', 'reason': '数据不足'}

            # 计算布林带
            bands = self.calculate_bollinger_bands(data)
            current_price = data['close'].iloc[-1]

            # 生成信号
            if current_price <= bands['lower']:
                return {
                    'signal': 'BUY',
                    'reason': f'价格触及下轨: {current_price:.2f} <= {bands["lower"]:.2f}',
                    'price': current_price,
                    'bands': bands
                }
            elif current_price >= bands['upper']:
                return {
                    'signal': 'SELL',
                    'reason': f'价格触及上轨: {current_price:.2f} >= {bands["upper"]:.2f}',
                    'price': current_price,
                    'bands': bands
                }
            else:
                return {
                    'signal': 'HOLD',
                    'reason': f'价格在中轨附近: {current_price:.2f}',
                    'price': current_price,
                    'bands': bands
                }

        except Exception as e:
            self.logger.error(f"布林带策略信号生成失败: {e}")
            return {'signal': 'HOLD', 'reason': f'错误: {str(e)}'}


class MACDStrategy(TradingStrategy):
    """MACD策略"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.fast_period = self.config.get('fast_period', 12)
        self.slow_period = self.config.get('slow_period', 26)
        self.signal_period = self.config.get('signal_period', 9)

    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算MACD"""
        close = data['close']

        # 计算指数移动平均
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        # 计算MACD线
        macd_line = fast_ema - slow_ema

        # 计算信号线
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # 计算直方图
        histogram = macd_line - signal_line

        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成MACD信号"""
        try:
            if len(data) < self.slow_period:
                return {'signal': 'HOLD', 'reason': '数据不足'}

            # 计算MACD
            macd_data = self.calculate_macd(data)

            # 获取前一个周期的数据
            if len(data) > 1:
                prev_data = data.iloc[:-1]
                prev_macd = self.calculate_macd(prev_data)

                # 金叉：MACD线上穿信号线
                if (prev_macd['macd'] <= prev_macd['signal'] and
                        macd_data['macd'] > macd_data['signal']):
                    return {
                        'signal': 'BUY',
                        'reason': f'MACD金叉: {macd_data["macd"]:.4f} > {macd_data["signal"]:.4f}',
                        'macd_data': macd_data
                    }
                # 死叉：MACD线下穿信号线
                elif (prev_macd['macd'] >= prev_macd['signal'] and
                      macd_data['macd'] < macd_data['signal']):
                    return {
                        'signal': 'SELL',
                        'reason': f'MACD死叉: {macd_data["macd"]:.4f} < {macd_data["signal"]:.4f}',
                        'macd_data': macd_data
                    }

            return {
                'signal': 'HOLD',
                'reason': f'MACD无交叉信号',
                'macd_data': macd_data
            }

        except Exception as e:
            self.logger.error(f"MACD策略信号生成失败: {e}")
            return {'signal': 'HOLD', 'reason': f'错误: {str(e)}'}


class StrategyManager:
    """策略管理器"""

    def __init__(self):
        self.strategies = {}
        self.active_strategies = []

    def register_strategy(self, strategy: TradingStrategy):
        """注册策略"""
        self.strategies[strategy.name] = strategy
        self.active_strategies.append(strategy.name)
        logger.info(f"策略已注册: {strategy.name}")

    def get_strategy(self, name: str) -> Optional[TradingStrategy]:
        """获取策略"""
        return self.strategies.get(name)

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """生成所有策略的信号"""
        signals = {}

        for name, strategy in self.strategies.items():
            if name in self.active_strategies:
                try:
                    signal = strategy.generate_signal(data)
                    signals[name] = signal
                except Exception as e:
                    logger.error(f"策略 {name} 信号生成失败: {e}")
                    signals[name] = {'signal': 'ERROR', 'reason': str(e)}

        return signals

    def execute_signals(self, signals: Dict[str, Dict[str, Any]],
                        current_price: float) -> Dict[str, Dict[str, Any]]:
        """执行所有策略的信号"""
        executions = {}

        for strategy_name, signal in signals.items():
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                try:
                    execution = strategy.execute_signal(signal, current_price)
                    executions[strategy_name] = execution
                except Exception as e:
                    logger.error(f"策略 {strategy_name} 信号执行失败: {e}")
                    executions[strategy_name] = {'error': str(e)}

        return executions

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能总结"""
        summary = {
            'total_strategies': len(self.strategies),
            'active_strategies': len(self.active_strategies),
            'strategy_performance': {}
        }

        for name, strategy in self.strategies.items():
            if name in self.active_strategies:
                summary['strategy_performance'][name] = strategy.get_performance_stats()

        return summary


def create_trading_strategies():
    """创建交易策略"""
    print("📈 创建基础交易策略...")

    try:
        # 创建策略目录
        import os
        os.makedirs("src/trading/strategies/basic", exist_ok=True)

        # 创建基础策略框架
        base_strategy = '''#!/usr/bin/env python3
"""
基础交易策略框架
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """基础策略类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.config.get('name', self.__class__.__name__)
        self.logger = logging.getLogger(self.name)

        # 策略参数
        self.max_position = self.config.get('max_position', 100)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)

        # 策略状态
        self.position = 0
        self.entry_price = 0.0
        self.signals = []
        self.trades = []

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        pass

    def calculate_position_size(self, price: float, signal: str) -> int:
        """计算仓位大小"""
        if signal in ['BUY', 'SELL']:
            risk_amount = self.max_position * self.risk_per_trade
            position_size = int(risk_amount / price)
            return min(position_size, self.max_position)
        return 0

    def execute_signal(self, signal: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """执行交易信号"""
        signal_type = signal.get('signal', 'HOLD')
        quantity = self.calculate_position_size(current_price, signal_type)

        execution = {
            'timestamp': datetime.now(),
            'signal': signal_type,
            'price': current_price,
            'quantity': quantity,
            'reason': signal.get('reason', ''),
            'strategy': self.name,
            'success': quantity > 0
        }

        if signal_type == 'BUY' and self.position <= 0:
            self.position = quantity
            self.entry_price = current_price
            execution['action'] = 'OPEN_LONG'
            self.trades.append(execution)
        elif signal_type == 'SELL' and self.position >= 0:
            self.position = -quantity
            self.entry_price = current_price
            execution['action'] = 'OPEN_SHORT'
            self.trades.append(execution)
        elif signal_type == 'CLOSE' and self.position != 0:
            execution['action'] = 'CLOSE_POSITION'
            execution['pnl'] = (current_price - self.entry_price) * abs(self.position)
            self.trades.append(execution)
            self.position = 0
            self.entry_price = 0.0
        else:
            execution['action'] = 'NO_ACTION'
            execution['quantity'] = 0

        self.signals.append(execution)
        return execution

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取策略性能统计"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }

        closed_trades = [t for t in self.trades if t.get('action') == 'CLOSE_POSITION']
        profitable_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]

        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)

        return {
            'total_trades': len(self.trades),
            'closed_trades': len(closed_trades),
            'profitable_trades': len(profitable_trades),
            'win_rate': len(profitable_trades) / len(closed_trades) if closed_trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(closed_trades) if closed_trades else 0
        }
'''

        # 创建具体策略
        ma_strategy = '''#!/usr/bin/env python3
"""
均线交叉策略
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略 - 均线交叉"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.short_period = self.config.get('short_period', 5)
        self.long_period = self.config.get('long_period', 20)

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成均线交叉信号"""
        try:
            if len(data) < self.long_period:
                return {'signal': 'HOLD', 'reason': '数据不足'}

            # 计算移动平均线
            data_copy = data.copy()
            data_copy['sma_short'] = data_copy['close'].rolling(window=self.short_period).mean()
            data_copy['sma_long'] = data_copy['close'].rolling(window=self.long_period).mean()

            # 获取最新数据
            latest = data_copy.iloc[-1]
            prev = data_copy.iloc[-2] if len(data_copy) > 1 else latest

            current_short = latest['sma_short']
            current_long = latest['sma_long']
            prev_short = prev['sma_short']
            prev_long = prev['sma_long']

            # 生成信号
            if pd.notna(current_short) and pd.notna(current_long) and pd.notna(prev_short) and pd.notna(prev_long):
                # 金叉：短期均线上穿长期均线
                if prev_short <= prev_long and current_short > current_long:
                    return {
                        'signal': 'BUY',
                        'reason': f'金叉信号: {self.short_period}日线穿过{self.long_period}日线',
                        'short_ma': current_short,
                        'long_ma': current_long
                    }
                # 死叉：短期均线下穿长期均线
                elif prev_short >= prev_long and current_short < current_long:
                    return {
                        'signal': 'SELL',
                        'reason': f'死叉信号: {self.short_period}日线跌破{self.long_period}日线',
                        'short_ma': current_short,
                        'long_ma': current_long
                    }

            return {'signal': 'HOLD', 'reason': '无明确交叉信号'}

        except Exception as e:
            self.logger.error(f"均线交叉策略信号生成失败: {e}")
            return {'signal': 'HOLD', 'reason': f'错误: {str(e)}'}
'''

        rsi_strategy = '''#!/usr/bin/env python3
"""
RSI策略
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略 - RSI"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.period = self.config.get('period', 14)
        self.overbought_level = self.config.get('overbought', 70)
        self.oversold_level = self.config.get('oversold', 30)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not rsi.empty else 50

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成RSI信号"""
        try:
            if len(data) < self.period + 1:
                return {'signal': 'HOLD', 'reason': '数据不足'}

            # 计算RSI
            rsi = self.calculate_rsi(data['close'], self.period)

            # 生成信号
            if rsi <= self.oversold_level:
                return {
                    'signal': 'BUY',
                    'reason': f'RSI超卖信号: {rsi:.2f} <= {self.oversold_level}',
                    'rsi': rsi,
                    'level': 'oversold'
                }
            elif rsi >= self.overbought_level:
                return {
                    'signal': 'SELL',
                    'reason': f'RSI超买信号: {rsi:.2f} >= {self.overbought_level}',
                    'rsi': rsi,
                    'level': 'overbought'
                }
            else:
                return {
                    'signal': 'HOLD',
                    'reason': f'RSI中性: {rsi:.2f}',
                    'rsi': rsi,
                    'level': 'neutral'
                }

        except Exception as e:
            self.logger.error(f"RSI策略信号生成失败: {e}")
            return {'signal': 'HOLD', 'reason': f'错误: {str(e)}'}
'''

        # 创建策略文件
        with open("src/trading/strategies/basic/__init__.py", 'w', encoding='utf-8') as f:
            f.write('''"""
基础交易策略
"""

from .base_strategy import BaseStrategy
from .trend_following_strategy import TrendFollowingStrategy
from .mean_reversion_strategy import MeanReversionStrategy

__all__ = [
    'BaseStrategy',
    'TrendFollowingStrategy',
    'MeanReversionStrategy'
]
''')

        with open("src/trading/strategies/basic/base_strategy.py", 'w', encoding='utf-8') as f:
            f.write(base_strategy)

        with open("src/trading/strategies/basic/trend_following_strategy.py", 'w', encoding='utf-8') as f:
            f.write(ma_strategy)

        with open("src/trading/strategies/basic/mean_reversion_strategy.py", 'w', encoding='utf-8') as f:
            f.write(rsi_strategy)

        print("   ✅ 基础交易策略创建完成")

        return True

    except Exception as e:
        print(f"   ❌ 创建基础交易策略失败: {e}")
        return False


def create_strategy_backtester():
    """创建策略回测框架"""
    print("\n📊 创建策略回测框架...")

    try:
        backtester = '''#!/usr/bin/env python3
"""
策略回测框架
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BacktestResult:
    """回测结果"""

    def __init__(self):
        self.trades = []
        self.portfolio_value = []
        self.signals = []
        self.metrics = {}

    def add_trade(self, trade: Dict[str, Any]):
        """添加交易"""
        self.trades.append(trade)

    def add_portfolio_value(self, timestamp: datetime, value: float):
        """添加投资组合价值"""
        self.portfolio_value.append({'timestamp': timestamp, 'value': value})

    def add_signal(self, signal: Dict[str, Any]):
        """添加信号"""
        self.signals.append(signal)

    def calculate_metrics(self) -> Dict[str, Any]:
        """计算回测指标"""
        if not self.trades:
            return {'total_trades': 0, 'total_return': 0.0}

        # 计算基础指标
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.get('action') == 'CLOSE_POSITION']

        if not closed_trades:
            return {
                'total_trades': total_trades,
                'closed_trades': 0,
                'total_return': 0.0,
                'win_rate': 0.0
            }

        # 计算胜率和总收益
        profitable_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)

        # 计算最大回撤
        if self.portfolio_value:
            values = [pv['value'] for pv in self.portfolio_value]
            peak = values[0]
            max_drawdown = 0

            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)

        # 计算夏普比率 (简化版)
        if len(values) > 1:
            returns = pd.Series(values).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # 年化
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        self.metrics = {
            'total_trades': total_trades,
            'closed_trades': len(closed_trades),
            'profitable_trades': len(profitable_trades),
            'win_rate': len(profitable_trades) / len(closed_trades),
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(closed_trades),
            'max_drawdown': max_drawdown if 'max_drawdown' in locals() else 0,
            'sharpe_ratio': sharpe_ratio,
            'total_return': (values[-1] / values[0] - 1) if 'values' in locals() and len(values) > 1 else 0
        }

        return self.metrics

class StrategyBacktester:
    """策略回测器"""

    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.results = BacktestResult()

    def run_backtest(self, strategy, data: pd.DataFrame,
                    transaction_cost: float = 0.001) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 交易策略实例
            data: 历史数据
            transaction_cost: 交易成本

        Returns:
            回测结果
        """
        logger.info(f"开始回测策略: {strategy.name}")
        logger.info(f"初始资金: {self.initial_balance:.2f}")
        logger.info(f"数据长度: {len(data)}")

        # 重置状态
        self.current_balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.results = BacktestResult()

        # 回测循环
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            if len(current_data) < 30:  # 跳过前30个数据点
                continue

            current_price = current_data['close'].iloc[-1]
            current_timestamp = current_data.index[-1] if isinstance(current_data.index, pd.DatetimeIndex) else current_data['timestamp'].iloc[-1]

            # 生成交易信号
            signal = strategy.generate_signal(current_data)
            self.results.add_signal(signal)

            # 执行信号
            if signal.get('signal') in ['BUY', 'SELL']:
                execution = self._execute_trade(signal, current_price, current_timestamp, transaction_cost)
                if execution:
                    self.results.add_trade(execution)

            # 计算投资组合价值
            portfolio_value = self.current_balance + (self.position * current_price)
            self.results.add_portfolio_value(current_timestamp, portfolio_value)

        # 计算最终指标
        self.results.calculate_metrics()

        logger.info(f"回测完成，交易次数: {len(self.results.trades)}")
        logger.info(f"最终资金: {self.current_balance:.2f}")

        return self.results

    def _execute_trade(self, signal: Dict[str, Any], price: float,
                      timestamp: datetime, transaction_cost: float) -> Optional[Dict[str, Any]]:
        """执行交易"""
        signal_type = signal.get('signal')
        quantity = signal.get('quantity', 1)

        # 计算交易成本
        cost = price * quantity * transaction_cost

        if signal_type == 'BUY' and self.position <= 0:
            # 开多仓
            if self.current_balance >= (price * quantity + cost):
                self.position = quantity
                self.entry_price = price
                self.current_balance -= (price * quantity + cost)

                return {
                    'timestamp': timestamp,
                    'action': 'OPEN_LONG',
                    'price': price,
                    'quantity': quantity,
                    'cost': cost,
                    'balance_after': self.current_balance
                }

        elif signal_type == 'SELL' and self.position >= 0:
            # 开空仓
            if self.current_balance >= cost:
                self.position = -quantity
                self.entry_price = price
                self.current_balance -= cost

                return {
                    'timestamp': timestamp,
                    'action': 'OPEN_SHORT',
                    'price': price,
                    'quantity': quantity,
                    'cost': cost,
                    'balance_after': self.current_balance
                }

        elif signal_type in ['BUY', 'SELL'] and self.position != 0:
            # 平仓
            pnl = (price - self.entry_price) * abs(self.position)
            self.current_balance += (price * abs(self.position) + pnl - cost)

            result = {
                'timestamp': timestamp,
                'action': 'CLOSE_POSITION',
                'price': price,
                'quantity': abs(self.position),
                'pnl': pnl,
                'cost': cost,
                'balance_after': self.current_balance
            }

            self.position = 0
            self.entry_price = 0.0
            return result

        return None

    def plot_results(self, results: BacktestResult, save_path: str = None):
        """绘制回测结果"""
        try:
            import matplotlib.pyplot as plt

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 1. 投资组合价值曲线
            if results.portfolio_value:
                dates = [pv['timestamp'] for pv in results.portfolio_value]
                values = [pv['value'] for pv in results.portfolio_value]
                ax1.plot(dates, values)
                ax1.set_title('Portfolio Value Over Time')
                ax1.set_ylabel('Value')
                ax1.grid(True)

            # 2. 交易信号
            if results.signals:
                signal_data = pd.DataFrame(results.signals)
                buy_signals = signal_data[signal_data['signal'] == 'BUY']
                sell_signals = signal_data[signal_data['signal'] == 'SELL']

                if not buy_signals.empty:
                    ax2.scatter(buy_signals.index, [1] * len(buy_signals), color='green', marker='^', label='BUY')
                if not sell_signals.empty:
                    ax2.scatter(sell_signals.index, [1] * len(sell_signals), color='red', marker='v', label='SELL')

                ax2.set_title('Trading Signals')
                ax2.legend()

            # 3. 收益分布
            if results.trades:
                closed_trades = [t for t in results.trades if t.get('action') == 'CLOSE_POSITION']
                if closed_trades:
                    pnls = [t.get('pnl', 0) for t in closed_trades]
                    ax3.hist(pnls, bins=20, alpha=0.7)
                    ax3.set_title('P&L Distribution')
                    ax3.set_xlabel('P&L')
                    ax3.set_ylabel('Frequency')

            # 4. 累计收益
            if results.portfolio_value:
                values = [pv['value'] for pv in results.portfolio_value]
                cumulative_returns = [(v - self.initial_balance) / self.initial_balance for v in values]
                dates = [pv['timestamp'] for pv in results.portfolio_value]
                ax4.plot(dates, cumulative_returns)
                ax4.set_title('Cumulative Returns')
                ax4.set_ylabel('Return')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"回测结果图表已保存到: {save_path}")

            plt.show()

        except ImportError:
            logger.warning("matplotlib未安装，无法绘制图表")
        except Exception as e:
            logger.error(f"绘制回测结果失败: {e}")

        '''

        # 创建回测框架
        with open("src/trading/backtesting/backtester.py", 'w', encoding='utf-8') as f:
            f.write(backtester)

        print("   ✅ 策略回测框架创建完成")

        return True

    except Exception as e:
        print(f"   ❌ 创建策略回测框架失败: {e}")
        return False


def run_sample_backtest():
    """运行示例回测"""
    print("\n🧪 运行示例回测...")

    try:
        # 导入必要的组件
        from src.data.loaders import StockDataLoader
        from src.trading.strategies.basic import TrendFollowingStrategy
        from src.trading.backtesting.backtester import StrategyBacktester

        # 加载数据
        loader = StockDataLoader({'data_source': 'mock'})
        data = loader.load_data('000001', '2024-01-01', '2024-12-31')

        if not data.empty and len(data) > 100:
            # 创建策略
            strategy = TrendFollowingStrategy({
                'short_period': 5,
                'long_period': 20,
                'max_position': 1000
            })

            # 创建回测器
            backtester = StrategyBacktester(initial_balance=100000)

            # 运行回测
            results = backtester.run_backtest(strategy, data, transaction_cost=0.001)

            # 显示结果
            metrics = results.calculate_metrics()
            print("📊 回测结果:")
            print(f"   调试信息 - trades数量: {len(results.trades)}")
            if results.trades:
                print(f"   调试信息 - 第一个trade: {results.trades[0]}")
                print(
                    f"   调试信息 - trades action类型: {[t.get('action', 'UNKNOWN') for t in results.trades[:5]]}")
            print(f"   调试信息 - metrics键: {list(metrics.keys())}")
            print(f"   调试信息 - metrics内容: {metrics}")
            print(f"   总交易次数: {metrics['total_trades']}")
            print(f"   胜率: {metrics['win_rate']:.2%}")
            if 'total_pnl' in metrics:
                print(f"   总收益: {metrics['total_pnl']:.2f}")
            else:
                print("   总收益: N/A (无平仓交易)")
            if 'max_drawdown' in metrics:
                print(f"   最大回撤: {metrics['max_drawdown']:.2%}")
            else:
                print("   最大回撤: N/A")
            if 'sharpe_ratio' in metrics:
                print(f"   夏普比率: {metrics['sharpe_ratio']:.2f}")
            else:
                print("   夏普比率: N/A")

            # 尝试绘制结果
            try:
                backtester.plot_results(results, save_path='backtest_results.png')
            except:
                print("   ⚠️ 无法绘制图表")

            return True
        else:
            print("   ❌ 数据不足，无法运行回测")
            return False

    except Exception as e:
        print(f"   ❌ 运行示例回测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_strategies():
    """测试基础交易策略"""
    print("\n🧪 测试基础交易策略...")

    try:
        # 导入策略
        from src.trading.strategies.basic import TrendFollowingStrategy, MeanReversionStrategy
        from src.data.loaders import StockDataLoader

        # 加载测试数据
        loader = StockDataLoader({'data_source': 'mock'})
        data = loader.load_data('000001', '2024-01-01', '2024-01-31')

        if not data.empty and len(data) > 30:
            # 测试均线交叉策略
            ma_strategy = TrendFollowingStrategy({
                'short_period': 5,
                'long_period': 20
            })

            signal = ma_strategy.generate_signal(data)
            print(f"   ✅ 均线交叉策略信号: {signal.get('signal', 'UNKNOWN')}")

            # 测试RSI策略
            rsi_strategy = MeanReversionStrategy({
                'period': 14,
                'overbought': 70,
                'oversold': 30
            })

            signal = rsi_strategy.generate_signal(data)
            print(f"   ✅ RSI策略信号: {signal.get('signal', 'UNKNOWN')}")

            return True
        else:
            print("   ❌ 测试数据不足")
            return False

    except Exception as e:
        print(f"   ❌ 测试基础交易策略失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("📈 RQA2025基础交易策略实现")

    results = {
        'create_strategies': False,
        'create_backtester': False,
        'test_strategies': False,
        'sample_backtest': False
    }

    # 1. 创建交易策略
    results['create_strategies'] = create_trading_strategies()

    # 2. 创建策略回测框架
    results['create_backtester'] = create_strategy_backtester()

    # 3. 测试基础交易策略
    results['test_strategies'] = test_basic_strategies()

    # 4. 运行示例回测
    results['sample_backtest'] = run_sample_backtest()

    # 总结
    successful = sum(results.values())
    total = len(results)

    print(f"\n📊 基础交易策略实现总结:")
    print(f"   成功: {successful}/{total}")
    print(".1f")

    for task, success in results.items():
        status = "✅" if success else "❌"
        task_name = {
            'create_strategies': '创建交易策略',
            'create_backtester': '创建回测框架',
            'test_strategies': '测试基础策略',
            'sample_backtest': '运行示例回测'
        }.get(task, task)
        print(f"   {status} {task_name}")

    if successful == total:
        print("\n🎉 基础交易策略实现全部完成！")
        print("   现在支持多种交易策略和完整的回测框架")
    elif successful >= 2:
        print("\n👍 基础交易策略实现大部分完成")
        print("   核心功能已经实现")
    else:
        print("\n⚠️ 基础交易策略实现需要进一步完善")


if __name__ == "__main__":
    main()
