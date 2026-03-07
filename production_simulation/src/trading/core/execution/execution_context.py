# -*- coding: utf-8 -*-
"""
执行上下文模块
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class ExecutionPhase(Enum):
    """执行阶段枚举"""
    PRE_EXECUTION = "pre_execution"
    EXECUTING = "executing"
    POST_EXECUTION = "post_execution"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutionContext:
    """执行上下文类"""

    # 基本信息
    execution_id: str
    symbol: str
    order_id: Optional[str] = None

    # 执行参数
    quantity: float = 0.0
    price: Optional[float] = None
    side: str = "buy"

    # 执行配置
    execution_strategy: str = "market"
    time_limit: Optional[int] = None  # 秒
    max_slippage: float = 0.01

    # 执行状态
    phase: ExecutionPhase = ExecutionPhase.PRE_EXECUTION
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # 执行结果
    executed_quantity: float = 0.0
    executed_price: Optional[float] = None
    total_cost: float = 0.0

    # 附加信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """初始化后处理"""
        if self.start_time is None:
            self.start_time = datetime.now()

    def update_progress(self, executed_qty: float, executed_price: float):
        """更新执行进度"""
        self.executed_quantity = executed_qty
        self.executed_price = executed_price
        if executed_price and executed_qty:
            self.total_cost = executed_price * executed_qty

    def mark_completed(self):
        """标记执行完成"""
        self.phase = ExecutionPhase.COMPLETED
        self.end_time = datetime.now()

    def mark_failed(self, error: str):
        """标记执行失败"""
        self.phase = ExecutionPhase.FAILED
        self.end_time = datetime.now()
        self.errors.append(error)

    def is_completed(self) -> bool:
        """检查是否完成"""
        return self.phase in [ExecutionPhase.COMPLETED, ExecutionPhase.FAILED]

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "execution_id": self.execution_id,
            "symbol": self.symbol,
            "total_quantity": self.quantity,
            "executed_quantity": self.executed_quantity,
            "execution_rate": self.executed_quantity / self.quantity if self.quantity > 0 else 0,
            "average_price": self.executed_price,
            "total_cost": self.total_cost,
            "phase": self.phase.value,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
            "errors": self.errors
        }
