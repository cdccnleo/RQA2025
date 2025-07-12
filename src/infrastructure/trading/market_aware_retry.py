from datetime import datetime, time, timedelta
from typing import Dict, Optional, List
import holidays
import pytz
from enum import Enum, auto

class MarketPhase(Enum):
    """交易时段阶段"""
    PRE_OPEN = auto()       # 开盘前
    MORNING = auto()        # 上午交易
    LUNCH_BREAK = auto()    # 午间休市
    AFTERNOON = auto()      # 下午交易
    CLOSED = auto()         # 收盘后

class MarketAwareRetryHandler:
    """A股交易时段感知的重试处理器"""

    # A股标准交易时间（可配置覆盖）
    DEFAULT_TRADING_HOURS = {
        "morning_open": time(9, 30),
        "morning_close": time(11, 30),
        "afternoon_open": time(13, 0),
        "afternoon_close": time(15, 0),
        "pre_open": time(9, 15)  # 集合竞价开始
    }

    def __init__(self,
                 trading_hours: Optional[Dict] = None,
                 holidays_calendar: Optional[List[datetime]] = None,
                 timezone: str = "Asia/Shanghai"):
        """
        Args:
            trading_hours: 自定义交易时间配置
            holidays_calendar: 自定义节假日列表
            timezone: 时区设置
        """
        self.trading_hours = trading_hours or self.DEFAULT_TRADING_HOURS
        self.timezone = pytz.timezone(timezone)

        # 初始化中国节假日（可缓存）
        self.holidays = holidays_calendar or holidays.China()

        # 重试策略配置
        self.base_retry_interval = 5  # 基础重试间隔(秒)
        self.max_retry_attempts = 3   # 最大重试次数
        self.current_attempt = 0

    def get_market_phase(self, dt: Optional[datetime] = None) -> MarketPhase:
        """
        获取当前市场阶段
        Args:
            dt: 指定时间，默认当前时间
        Returns:
            MarketPhase枚举值
        """
        dt = dt or datetime.now(self.timezone)
        dt_time = dt.time()

        # 检查节假日
        if dt.date() in self.holidays:
            return MarketPhase.CLOSED

        # 检查周末
        if dt.weekday() >= 5:  # 周六周日
            return MarketPhase.CLOSED

        # 开盘前
        if dt_time < self.trading_hours["pre_open"]:
            return MarketPhase.PRE_OPEN

        # 上午交易
        if (self.trading_hours["morning_open"] <= dt_time < self.trading_hours["morning_close"]):
            return MarketPhase.MORNING

        # 午间休市
        if (self.trading_hours["morning_close"] <= dt_time < self.trading_hours["afternoon_open"]):
            return MarketPhase.LUNCH_BREAK

        # 下午交易
        if (self.trading_hours["afternoon_open"] <= dt_time < self.trading_hours["afternoon_close"]):
            return MarketPhase.AFTERNOON

        return MarketPhase.CLOSED

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """检查市场是否开市"""
        phase = self.get_market_phase(dt)
        return phase in (MarketPhase.MORNING, MarketPhase.AFTERNOON)

    def next_market_open_time(self, dt: Optional[datetime] = None) -> datetime:
        """
        计算下一个开市时间
        Args:
            dt: 参考时间，默认当前时间
        Returns:
            下一个开市时间(datetime)
        """
        dt = dt or datetime.now(self.timezone)
        phase = self.get_market_phase(dt)

        if phase == MarketPhase.PRE_OPEN:
            return dt.replace(
                hour=self.trading_hours["morning_open"].hour,
                minute=self.trading_hours["morning_open"].minute,
                second=0
            )

        elif phase == MarketPhase.MORNING:
            return dt.replace(
                hour=self.trading_hours["afternoon_open"].hour,
                minute=self.trading_hours["afternoon_open"].minute,
                second=0
            )

        elif phase == MarketPhase.LUNCH_BREAK:
            return dt.replace(
                hour=self.trading_hours["afternoon_open"].hour,
                minute=self.trading_hours["afternoon_open"].minute,
                second=0
            )

        else:  # CLOSED 或 AFTERNOON
            next_day = dt + timedelta(days=1)
            while next_day.date() in self.holidays or next_day.weekday() >= 5:
                next_day += timedelta(days=1)

            return next_day.replace(
                hour=self.trading_hours["morning_open"].hour,
                minute=self.trading_hours["morning_open"].minute,
                second=0
            )

    def should_retry(self, last_failure_time: Optional[datetime] = None) -> bool:
        """判断是否应该继续重试"""
        if self.current_attempt >= self.max_retry_attempts:
            return False

        self.current_attempt += 1
        return True

    def get_retry_delay(self, last_failure_time: Optional[datetime] = None) -> float:
        """
        获取下次重试的延迟时间（秒）
        Args:
            last_failure_time: 上次失败时间
        Returns:
            延迟秒数
        """
        last_failure_time = last_failure_time or datetime.now(self.timezone)

        if not self.is_market_open():
            next_open = self.next_market_open_time(last_failure_time)
            return (next_open - last_failure_time).total_seconds()

        # 指数退避策略
        return min(self.base_retry_interval * (2 ** (self.current_attempt - 1)), 60)

    def reset_attempts(self):
        """重置重试计数器"""
        self.current_attempt = 0

    def register_holiday(self, date: datetime):
        """注册节假日"""
        self.holidays.append(date.date())

    def update_trading_hours(self, new_hours: Dict):
        """更新交易时间配置"""
        self.trading_hours.update(new_hours)

class SmartOrderRetry:
    """智能订单重试管理器"""

    def __init__(self, retry_handler: MarketAwareRetryHandler):
        self.retry_handler = retry_handler
        self.pending_orders = {}

    def submit_order(self, order: Dict) -> Dict:
        """提交订单（带重试逻辑）"""
        result = self._try_execute(order)

        if not result["success"] and self.retry_handler.should_retry():
            delay = self.retry_handler.get_retry_delay()
            order_id = order.get("order_id")
            self.pending_orders[order_id] = {
                "order": order,
                "next_retry": datetime.now() + timedelta(seconds=delay)
            }

        return result

    def check_pending_orders(self):
        """检查待重试订单"""
        current_time = datetime.now()
        to_retry = []

        for order_id, data in self.pending_orders.items():
            if current_time >= data["next_retry"]:
                to_retry.append((order_id, data["order"]))

        for order_id, order in to_retry:
            del self.pending_orders[order_id]
            self.submit_order(order)

    def _try_execute(self, order: Dict) -> Dict:
        """尝试执行订单"""
        # 实际执行逻辑
        return {"success": True, "order_id": order.get("order_id")}

# 兼容测试导出
MarketAwareRetry = MarketAwareRetryHandler
