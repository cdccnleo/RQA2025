import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from prometheus_client import Counter, Gauge, Histogram

@dataclass
class BacktestMetrics:
    """回测监控指标"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0

class BacktestMonitor:
    """回测监控器"""

    def __init__(self):
        """初始化回测监控器"""
        # 初始化metrics属性
        self.metrics = BacktestMetrics()
        
        # 初始化Prometheus指标
        self._trade_counter = Counter(
            'backtest_trades_total',
            'Total number of trades',
            ['status']
        )
        self._pnl_gauge = Gauge(
            'backtest_pnl',
            'Current PnL value'
        )
        self._drawdown_gauge = Gauge(
            'backtest_max_drawdown',
            'Maximum drawdown'
        )
        self._sharpe_gauge = Gauge(
            'backtest_sharpe_ratio',
            'Sharpe ratio'
        )
        
        # 初始化基础指标
        self.metrics.update({
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_trade_duration': 0.0
        })

    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        记录交易数据
        
        Args:
            trade_data: 交易数据字典
        """
        # 更新内部指标
        self.metrics.total_trades += 1
        
        if trade_data.get('success', False):
            self.metrics.successful_trades += 1
            self._trade_counter.labels(status='success').inc()
        else:
            self.metrics.failed_trades += 1
            self._trade_counter.labels(status='failed').inc()
        
        # 更新PnL
        pnl = trade_data.get('pnl', 0.0)
        self.metrics.total_pnl += pnl
        self._pnl_gauge.set(self.metrics.total_pnl)
        
        # 更新胜率
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.successful_trades / self.metrics.total_trades

    def record_portfolio(self, portfolio_data: Dict[str, Any]) -> None:
        """
        记录投资组合数据
        
        Args:
            portfolio_data: 投资组合数据字典
        """
        # 更新最大回撤
        drawdown = portfolio_data.get('drawdown', 0.0)
        if drawdown > self.metrics.max_drawdown:
            self.metrics.max_drawdown = drawdown
            self._drawdown_gauge.set(drawdown)
        
        # 更新夏普比率
        sharpe = portfolio_data.get('sharpe_ratio', 0.0)
        self.metrics.sharpe_ratio = sharpe
        self._sharpe_gauge.set(sharpe)
        
        # 更新平均交易时长
        avg_duration = portfolio_data.get('avg_trade_duration', 0.0)
        self.metrics.avg_trade_duration = avg_duration

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取监控指标
        
        Returns:
            指标字典
        """
        return {
            'total_trades': self.metrics.total_trades,
            'successful_trades': self.metrics.successful_trades,
            'failed_trades': self.metrics.failed_trades,
            'total_pnl': self.metrics.total_pnl,
            'max_drawdown': self.metrics.max_drawdown,
            'sharpe_ratio': self.metrics.sharpe_ratio,
            'win_rate': self.metrics.win_rate,
            'avg_trade_duration': self.metrics.avg_trade_duration
        }

    def reset_metrics(self) -> None:
        """重置所有指标"""
        self.metrics = BacktestMetrics()
        self._pnl_gauge.set(0.0)
        self._drawdown_gauge.set(0.0)
        self._sharpe_gauge.set(0.0)
