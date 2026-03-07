# -*- coding: utf-8 -*-
"""
执行结果模块
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class ExecutionResultStatus(Enum):
    """执行结果状态枚举"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ExecutionResult:
    """执行结果类"""

    # 基本信息
    execution_id: str
    symbol: str
    order_id: Optional[str] = None

    # 结果状态
    status: ExecutionResultStatus = ExecutionResultStatus.SUCCESS

    # 执行统计
    requested_quantity: float = 0.0
    executed_quantity: float = 0.0
    average_price: Optional[float] = None
    total_cost: float = 0.0

    # 时间信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # 交易记录
    trades: List[Dict[str, Any]] = field(default_factory=list)

    # 性能指标
    execution_time: float = 0.0  # 秒
    slippage: float = 0.0  # 滑点
    market_impact: float = 0.0  # 市场影响

    # 错误信息
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """初始化后处理"""
        if self.start_time is None:
            self.start_time = datetime.now()

    def add_trade(self, trade: Dict[str, Any]):
        """添加交易记录"""
        self.trades.append(trade)

        # 更新统计信息
        if 'quantity' in trade and 'price' in trade:
            qty = trade['quantity']
            price = trade['price']

            # 更新已执行数量和总成本
            self.executed_quantity += qty
            self.total_cost += qty * price

            # 更新平均价格
            if self.executed_quantity > 0:
                self.average_price = self.total_cost / self.executed_quantity

    def calculate_metrics(self):
        """计算性能指标"""
        if self.end_time and self.start_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()

        # 计算执行率
        if self.requested_quantity > 0:
            execution_rate = self.executed_quantity / self.requested_quantity
            if execution_rate < 1.0:
                self.status = ExecutionResultStatus.PARTIAL
        else:
            execution_rate = 0.0

        # 这里可以添加更复杂的指标计算
        # 比如滑点计算、市场影响评估等

        return {
            "execution_rate": execution_rate,
            "execution_time": self.execution_time,
            "average_price": self.average_price,
            "total_cost": self.total_cost
        }

    def mark_completed(self, status: ExecutionResultStatus = ExecutionResultStatus.SUCCESS):
        """标记完成"""
        self.status = status
        self.end_time = datetime.now()
        self.calculate_metrics()

    def add_error(self, error: str):
        """添加错误信息"""
        self.errors.append(error)
        self.status = ExecutionResultStatus.FAILED

    def add_warning(self, warning: str):
        """添加警告信息"""
        self.warnings.append(warning)

    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        metrics = self.calculate_metrics()

        return {
            "execution_id": self.execution_id,
            "symbol": self.symbol,
            "status": self.status.value,
            "execution_rate": metrics["execution_rate"],
            "average_price": self.average_price,
            "total_cost": self.total_cost,
            "execution_time": self.execution_time,
            "trade_count": len(self.trades),
            "errors": self.errors,
            "warnings": self.warnings
        }

    def is_successful(self) -> bool:
        """检查是否成功"""
        return self.status == ExecutionResultStatus.SUCCESS and len(self.errors) == 0

    def is_partial(self) -> bool:
        """检查是否部分执行"""
        return self.status == ExecutionResultStatus.PARTIAL
