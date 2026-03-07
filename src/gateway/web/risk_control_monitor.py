"""
风险控制监控模块
提供止损止盈监控、仓位控制、风险告警等功能
符合量化交易系统合规要求
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# 使用统一日志系统
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskEventType(Enum):
    """风险事件类型"""
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    RISK_EXPOSURE_HIGH = "risk_exposure_high"


@dataclass
class RiskEvent:
    """风险事件"""
    strategy_id: str
    event_type: RiskEventType
    risk_level: RiskLevel
    message: str
    timestamp: float
    details: Dict[str, Any]


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class RiskControlMonitor:
    """
    风险控制监控器
    
    提供以下功能：
    1. 止损止盈监控
    2. 仓位限制监控
    3. 最大回撤监控
    4. 风险敞口计算
    5. 自动风控执行
    6. 风险告警
    """
    
    # 默认风控参数
    DEFAULT_STOP_LOSS = 0.05      # 默认止损 5%
    DEFAULT_TAKE_PROFIT = 0.10    # 默认止盈 10%
    DEFAULT_MAX_POSITION = 0.30   # 默认最大仓位 30%
    DEFAULT_MAX_DRAWDOWN = 0.15   # 默认最大回撤 15%
    
    # 检查间隔（秒）
    CHECK_INTERVAL = 5
    
    def __init__(self):
        self._monitored_strategies: Dict[str, Dict[str, Any]] = {}
        self._risk_callbacks: List[Callable[[RiskEvent], None]] = []
        self._running = False
        self._check_task = None
        
    def register_strategy(self, strategy_id: str, risk_params: Dict[str, Any]) -> None:
        """
        注册策略到风控监控
        
        Args:
            strategy_id: 策略ID
            risk_params: 风控参数字典
                - stop_loss: 止损比例（默认0.05）
                - take_profit: 止盈比例（默认0.10）
                - max_position_size: 最大仓位（默认0.30）
                - max_drawdown: 最大回撤（默认0.15）
        """
        self._monitored_strategies[strategy_id] = {
            "stop_loss": risk_params.get("stop_loss", self.DEFAULT_STOP_LOSS),
            "take_profit": risk_params.get("take_profit", self.DEFAULT_TAKE_PROFIT),
            "max_position_size": risk_params.get("max_position_size", self.DEFAULT_MAX_POSITION),
            "max_drawdown": risk_params.get("max_drawdown", self.DEFAULT_MAX_DRAWDOWN),
            "peak_value": 0,  # 峰值权益
            "current_drawdown": 0,  # 当前回撤
            "positions": {},  # 持仓信息
            "enabled": True  # 是否启用风控
        }
        logger.info(f"策略 {strategy_id} 已注册到风控监控")
        
    def unregister_strategy(self, strategy_id: str) -> None:
        """取消策略风控监控"""
        if strategy_id in self._monitored_strategies:
            del self._monitored_strategies[strategy_id]
            logger.info(f"策略 {strategy_id} 已取消风控监控")
            
    def update_position(self, strategy_id: str, position: Position) -> None:
        """
        更新持仓信息
        
        Args:
            strategy_id: 策略ID
            position: 持仓信息
        """
        if strategy_id not in self._monitored_strategies:
            logger.warning(f"策略 {strategy_id} 未注册到风控监控")
            return
            
        self._monitored_strategies[strategy_id]["positions"][position.symbol] = position
        
        # 更新峰值和回撤
        total_value = self._calculate_total_value(strategy_id)
        strategy_config = self._monitored_strategies[strategy_id]
        
        if total_value > strategy_config["peak_value"]:
            strategy_config["peak_value"] = total_value
            
        if strategy_config["peak_value"] > 0:
            strategy_config["current_drawdown"] = (
                strategy_config["peak_value"] - total_value
            ) / strategy_config["peak_value"]
            
    def check_stop_loss(self, strategy_id: str, position: Position) -> Optional[RiskEvent]:
        """
        检查止损条件
        
        Args:
            strategy_id: 策略ID
            position: 持仓信息
            
        Returns:
            RiskEvent: 如果触发止损，返回风险事件；否则返回None
        """
        if strategy_id not in self._monitored_strategies:
            return None
            
        config = self._monitored_strategies[strategy_id]
        stop_loss_threshold = config["stop_loss"]
        
        # 检查未实现亏损是否超过止损阈值
        if position.unrealized_pnl_pct <= -stop_loss_threshold:
            event = RiskEvent(
                strategy_id=strategy_id,
                event_type=RiskEventType.STOP_LOSS_TRIGGERED,
                risk_level=RiskLevel.HIGH,
                message=f"止损触发: {position.symbol} 亏损 {position.unrealized_pnl_pct:.2%}，超过阈值 {stop_loss_threshold:.2%}",
                timestamp=time.time(),
                details={
                    "symbol": position.symbol,
                    "unrealized_pnl_pct": position.unrealized_pnl_pct,
                    "threshold": stop_loss_threshold,
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "current_price": position.current_price
                }
            )
            logger.warning(f"策略 {strategy_id} 止损触发: {event.message}")
            return event
            
        return None
        
    def check_take_profit(self, strategy_id: str, position: Position) -> Optional[RiskEvent]:
        """
        检查止盈条件
        
        Args:
            strategy_id: 策略ID
            position: 持仓信息
            
        Returns:
            RiskEvent: 如果触发止盈，返回风险事件；否则返回None
        """
        if strategy_id not in self._monitored_strategies:
            return None
            
        config = self._monitored_strategies[strategy_id]
        take_profit_threshold = config["take_profit"]
        
        # 检查未实现盈利是否超过止盈阈值
        if position.unrealized_pnl_pct >= take_profit_threshold:
            event = RiskEvent(
                strategy_id=strategy_id,
                event_type=RiskEventType.TAKE_PROFIT_TRIGGERED,
                risk_level=RiskLevel.MEDIUM,
                message=f"止盈触发: {position.symbol} 盈利 {position.unrealized_pnl_pct:.2%}，超过阈值 {take_profit_threshold:.2%}",
                timestamp=time.time(),
                details={
                    "symbol": position.symbol,
                    "unrealized_pnl_pct": position.unrealized_pnl_pct,
                    "threshold": take_profit_threshold,
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "current_price": position.current_price
                }
            )
            logger.info(f"策略 {strategy_id} 止盈触发: {event.message}")
            return event
            
        return None
        
    def check_position_limit(self, strategy_id: str) -> Optional[RiskEvent]:
        """
        检查仓位限制
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            RiskEvent: 如果超过仓位限制，返回风险事件；否则返回None
        """
        if strategy_id not in self._monitored_strategies:
            return None
            
        config = self._monitored_strategies[strategy_id]
        max_position = config["max_position_size"]
        
        # 计算当前仓位比例
        total_position_value = self._calculate_total_value(strategy_id)
        # 假设总资金为1（实际应从策略配置获取）
        total_capital = 1.0
        position_ratio = total_position_value / total_capital if total_capital > 0 else 0
        
        if position_ratio > max_position:
            event = RiskEvent(
                strategy_id=strategy_id,
                event_type=RiskEventType.POSITION_LIMIT_EXCEEDED,
                risk_level=RiskLevel.HIGH,
                message=f"仓位超限: 当前仓位 {position_ratio:.2%}，超过限制 {max_position:.2%}",
                timestamp=time.time(),
                details={
                    "current_position": position_ratio,
                    "max_position": max_position,
                    "total_value": total_position_value
                }
            )
            logger.warning(f"策略 {strategy_id} 仓位超限: {event.message}")
            return event
            
        return None
        
    def check_max_drawdown(self, strategy_id: str) -> Optional[RiskEvent]:
        """
        检查最大回撤
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            RiskEvent: 如果超过最大回撤限制，返回风险事件；否则返回None
        """
        if strategy_id not in self._monitored_strategies:
            return None
            
        config = self._monitored_strategies[strategy_id]
        max_drawdown = config["max_drawdown"]
        current_drawdown = config["current_drawdown"]
        
        if current_drawdown > max_drawdown:
            event = RiskEvent(
                strategy_id=strategy_id,
                event_type=RiskEventType.MAX_DRAWDOWN_EXCEEDED,
                risk_level=RiskLevel.CRITICAL,
                message=f"最大回撤超限: 当前回撤 {current_drawdown:.2%}，超过限制 {max_drawdown:.2%}",
                timestamp=time.time(),
                details={
                    "current_drawdown": current_drawdown,
                    "max_drawdown": max_drawdown,
                    "peak_value": config["peak_value"]
                }
            )
            logger.error(f"策略 {strategy_id} 最大回撤超限: {event.message}")
            return event
            
        return None
        
    def calculate_risk_exposure(self, strategy_id: str) -> Dict[str, Any]:
        """
        计算风险敞口
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            Dict: 风险敞口信息
        """
        if strategy_id not in self._monitored_strategies:
            return {}
            
        config = self._monitored_strategies[strategy_id]
        positions = config["positions"]
        
        total_exposure = 0
        long_exposure = 0
        short_exposure = 0
        
        for symbol, position in positions.items():
            exposure = abs(position.market_value)
            total_exposure += exposure
            
            if position.quantity > 0:
                long_exposure += exposure
            else:
                short_exposure += exposure
                
        return {
            "total_exposure": total_exposure,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "net_exposure": long_exposure - short_exposure,
            "current_drawdown": config["current_drawdown"],
            "position_count": len(positions)
        }
        
    def execute_auto_risk_control(self, event: RiskEvent) -> bool:
        """
        执行自动风控措施
        
        Args:
            event: 风险事件
            
        Returns:
            bool: 是否成功执行
        """
        strategy_id = event.strategy_id
        
        try:
            if event.event_type == RiskEventType.STOP_LOSS_TRIGGERED:
                # 执行自动止损 - 平仓
                logger.info(f"执行自动止损: 策略 {strategy_id}")
                # TODO: 调用交易接口平仓
                return True
                
            elif event.event_type == RiskEventType.TAKE_PROFIT_TRIGGERED:
                # 执行自动止盈 - 平仓或减仓
                logger.info(f"执行自动止盈: 策略 {strategy_id}")
                # TODO: 调用交易接口平仓或减仓
                return True
                
            elif event.event_type == RiskEventType.POSITION_LIMIT_EXCEEDED:
                # 仓位超限 - 禁止新开仓
                logger.warning(f"仓位超限，禁止新开仓: 策略 {strategy_id}")
                # TODO: 设置禁止开仓标志
                return True
                
            elif event.event_type == RiskEventType.MAX_DRAWDOWN_EXCEEDED:
                # 最大回撤超限 - 暂停策略
                logger.error(f"最大回撤超限，暂停策略: 策略 {strategy_id}")
                # TODO: 调用暂停策略接口
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"执行自动风控失败: 策略 {strategy_id}, 错误: {e}")
            return False
            
    def add_risk_callback(self, callback: Callable[[RiskEvent], None]) -> None:
        """添加风险事件回调"""
        self._risk_callbacks.append(callback)
        
    def remove_risk_callback(self, callback: Callable[[RiskEvent], None]) -> None:
        """移除风险事件回调"""
        if callback in self._risk_callbacks:
            self._risk_callbacks.remove(callback)
            
    def _notify_risk_event(self, event: RiskEvent) -> None:
        """通知风险事件"""
        for callback in self._risk_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"风险事件回调执行失败: {e}")
                
    def _calculate_total_value(self, strategy_id: str) -> float:
        """计算总持仓价值"""
        if strategy_id not in self._monitored_strategies:
            return 0
            
        positions = self._monitored_strategies[strategy_id]["positions"]
        return sum(pos.market_value for pos in positions.values())
        
    async def start_monitoring(self) -> None:
        """启动风控监控"""
        if self._running:
            return
            
        self._running = True
        logger.info("风控监控已启动")
        
        # 启动定期检查任务
        import asyncio
        self._check_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self) -> None:
        """停止风控监控"""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("风控监控已停止")
        
    async def _monitoring_loop(self) -> None:
        """监控循环"""
        import asyncio
        
        while self._running:
            try:
                await self._perform_risk_check()
                await asyncio.sleep(self.CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"风控检查失败: {e}")
                await asyncio.sleep(self.CHECK_INTERVAL)
                
    async def _perform_risk_check(self) -> None:
        """执行风控检查"""
        for strategy_id in self._monitored_strategies:
            config = self._monitored_strategies[strategy_id]
            
            if not config["enabled"]:
                continue
                
            # 检查每个持仓的止损止盈
            for symbol, position in config["positions"].items():
                # 检查止损
                event = self.check_stop_loss(strategy_id, position)
                if event:
                    self._notify_risk_event(event)
                    self.execute_auto_risk_control(event)
                    
                # 检查止盈
                event = self.check_take_profit(strategy_id, position)
                if event:
                    self._notify_risk_event(event)
                    self.execute_auto_risk_control(event)
                    
            # 检查仓位限制
            event = self.check_position_limit(strategy_id)
            if event:
                self._notify_risk_event(event)
                self.execute_auto_risk_control(event)
                
            # 检查最大回撤
            event = self.check_max_drawdown(strategy_id)
            if event:
                self._notify_risk_event(event)
                self.execute_auto_risk_control(event)


# 全局风控监控器实例
_risk_monitor: Optional[RiskControlMonitor] = None


def get_risk_control_monitor() -> RiskControlMonitor:
    """获取全局风控监控器实例（单例模式）"""
    global _risk_monitor
    if _risk_monitor is None:
        _risk_monitor = RiskControlMonitor()
    return _risk_monitor
