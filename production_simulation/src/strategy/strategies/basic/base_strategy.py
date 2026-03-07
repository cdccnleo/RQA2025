#!/usr/bin/env python3
"""
基础交易策略框架
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
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
