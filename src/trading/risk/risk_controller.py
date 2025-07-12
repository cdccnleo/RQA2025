"""统一风控控制器"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Dict, List, Optional, Set
from collections import defaultdict
from src.features.processors.feature_engineer import FeatureEngineer
from ..fpga.fpga_accelerator import FpgaManager
from ..execution.order import Order

logger = logging.getLogger(__name__)

@dataclass
class DynamicRiskParams:
    """动态风险参数"""
    volatility_adjusted: bool = False
    liquidity_adjusted: bool = False
    market_sentiment: float = 0.0  # -1到1之间
    adjustment_factors: Dict[str, float] = field(default_factory=lambda: {
        "position": 1.0,
        "loss": 1.0,
        "concentration": 1.0
    })

    def update_params(self, market_data: Dict):
        """根据市场数据更新参数"""
        # 根据波动率调整
        if self.volatility_adjusted:
            vol = market_data.get("volatility", 0)
            self.adjustment_factors["position"] = max(0.5, 1 - vol * 2)
            self.adjustment_factors["loss"] = max(0.3, 1 - vol * 3)

        # 根据流动性调整
        if self.liquidity_adjusted:
            liq = market_data.get("liquidity", 1)
            self.adjustment_factors["concentration"] = min(1.5, liq * 1.2)

@dataclass
class RiskConfig:
    """风控配置"""
    max_position: float = 0.2  # 单票最大仓位
    max_daily_loss: float = 0.05  # 单日最大亏损
    a_share_rules: bool = True  # 是否启用A股特定规则
    use_fpga: bool = True  # 是否使用FPGA加速
    circuit_breaker_levels: List[float] = field(
        default_factory=lambda: [0.05, 0.07, 0.10]
    )  # 熔断级别
    enable_dynamic_risk: bool = False  # 是否启用动态风控

class MarketMonitor(threading.Thread):
    """实时市场监控"""
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.subscribers = set()
        self.market_data = {}
        self.interval = 5  # 监控间隔(秒)
        self.logger = logging.getLogger(f"{__name__}.MarketMonitor")

    def run(self):
        """启动监控线程"""
        self.running = True
        while self.running:
            try:
                self._collect_market_data()
                self._notify_subscribers()
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"市场监控异常: {e}")

    def _collect_market_data(self):
        """收集市场数据"""
        # 实现实际的市场数据收集逻辑
        self.market_data = {
            "volatility": 0.1,  # 示例数据
            "liquidity": 0.8,   # 示例数据
            "sentiment": 0.5    # 示例数据
        }

    def _notify_subscribers(self):
        """通知订阅者"""
        for subscriber in self.subscribers:
            try:
                subscriber.update(self.market_data)
            except Exception as e:
                self.logger.error(f"通知订阅者失败: {e}")

    def subscribe(self, subscriber):
        """添加订阅者"""
        self.subscribers.add(subscriber)

    def stop(self):
        """停止监控"""
        self.running = False

class RiskController:
    """统一风控控制器"""

    def __init__(self, feature_engine: FeatureEngineer, config: Optional[RiskConfig] = None):
        self.engine = feature_engine
        self.config = config or RiskConfig()
        self.fpga_manager = FpgaManager()
        self.market_monitor = MarketMonitor()
        self.dynamic_params = DynamicRiskParams()
        
        if self.config.enable_dynamic_risk:
            self.market_monitor.subscribe(self.dynamic_params)
            self.market_monitor.start()

        # 初始化A股特定规则
        if self.config.a_share_rules:
            self._init_a_share_rules()

        # 风控状态
        self.daily_pnl = 0.0
        self.position_limits = {}

    def _init_a_share_rules(self):
        """初始化A股特定规则"""
        self.price_limit_rules = {
            "ST": 0.05,    # ST股票涨跌幅限制5%
            "normal": 0.1,  # 普通股票10%
            "688": 0.2      # 科创板20%
        }

        self.t1_settlement = True  # T+1结算制度
        self.margin_rules = {
            "maintenance_ratio": 1.3,  # 维持担保比例130%
            "concentration_limit": 0.3  # 单票集中度限制30%
        }

    def _check_a_share_restrictions(self, order: Order) -> bool:
        """检查A股交易限制"""
        if not self.config.a_share_rules:
            return True

        symbol = order.symbol

        # 检查涨跌停限制
        if symbol.startswith("688"):
            limit = self.price_limit_rules["688"]
        elif symbol.startswith(("ST", "*ST")):
            limit = self.price_limit_rules["ST"]
        else:
            limit = self.price_limit_rules["normal"]

        # 实际实现中会检查当前价格是否接近涨跌停
        # 这里简化为总是允许
        return True

    def _check_with_fpga(self, order: Order, market_data: Dict) -> Optional[bool]:
        """使用FPGA加速风控检查"""
        if not self.config.use_fpga:
            return None

        try:
            # 获取FPGA加速器
            fpga_accelerator = self.fpga_manager.get_accelerator("RISK_FPGA")
            if fpga_accelerator and fpga_accelerator.health_monitor.is_healthy():
                # 准备FPGA输入数据
                fpga_input = np.array([
                    order.price,
                    order.quantity,
                    market_data.get("volatility", 0),
                    market_data.get("liquidity", 0)
                ], dtype=np.float32)

                # 调用FPGA检查 (模拟)
                return True  # 实际实现中会从FPGA获取结果

        except Exception as e:
            logger.error(f"FPGA risk check failed: {str(e)}")

        return None

    def check_order(self, order: Order, market_data: Dict) -> bool:
        """检查订单是否符合风控规则"""
        # 尝试使用FPGA加速
        fpga_result = self._check_with_fpga(order, market_data)
        if fpga_result is not None:
            return fpga_result

        # 软件路径
        # 1. 检查基础风控规则
        if order.quantity <= 0:
            return False

        # 2. 检查仓位限制
        if order.symbol in self.position_limits:
            if order.quantity + self.position_limits[order.symbol] > self.config.max_position:
                return False

        # 3. 检查A股特定规则
        if not self._check_a_share_restrictions(order):
            return False

        # 4. 检查熔断状态
        if self.is_circuit_breaker_triggered(market_data):
            return False

        return True

    def is_circuit_breaker_triggered(self, market_data: Dict) -> bool:
        """检查是否触发熔断"""
        market_drop = market_data.get("market_drop", 0)
        for level in self.config.circuit_breaker_levels:
            if market_drop >= level:
                return True
        return False

    def update_pnl(self, pnl: float):
        """更新当日盈亏"""
        self.daily_pnl += pnl

        # 检查单日亏损限制
        if self.daily_pnl < -self.config.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl}")
            # 实际实现中会触发相应措施

    def update_position(self, symbol: str, quantity: float):
        """更新持仓信息"""
        self.position_limits[symbol] = quantity

    def batch_check(self, orders: List[Order], market_data: List[Dict]) -> List[bool]:
        """批量风控检查"""
        results = []

        for order, md in zip(orders, market_data):
            try:
                results.append(self.check_order(order, md))
            except Exception as e:
                logger.error(f"Risk check failed for order {order.id}: {str(e)}")
                results.append(False)

        return results

class ChinaRiskController(RiskController):
    """A股特定风控控制器"""

    def __init__(self, feature_engine: FeatureEngineer, config: Optional[RiskConfig] = None):
        super().__init__(feature_engine, config)
        self.config.a_share_rules = True  # 强制启用A股规则

        # 初始化A股特定风控参数
        self._init_a_share_risk_params()

    def _init_a_share_risk_params(self):
        """初始化A股特定风控参数"""
        self.policy_risk_symbols = set()  # 政策风险股票池
        self.margin_blacklist = set()  # 融资融券黑名单
        self.northbound_flow_threshold = 0.05  # 北向资金异常波动阈值

    def check_policy_risk(self, symbol: str) -> bool:
        """检查政策风险"""
        return symbol not in self.policy_risk_symbols

    def liquidity_check(self, order: Order) -> bool:
        """流动性检查"""
        # 简化为总是通过
        # 实际实现中会检查市场深度和流动性
        return True

    def check_margin_requirements(self, positions: Dict[str, float]) -> bool:
        """检查融资融券要求"""
        total_value = sum(positions.values())
        if total_value == 0:
            return True

        # 检查维持担保比例
        maintenance_ratio = self._calculate_maintenance_ratio(positions)
        if maintenance_ratio < self.margin_rules["maintenance_ratio"]:
            return False

        # 检查单票集中度
        max_position = max(positions.values())
        if max_position / total_value > self.margin_rules["concentration_limit"]:
            return False

        return True

    def _calculate_maintenance_ratio(self, positions: Dict[str, float]) -> float:
        """计算维持担保比例"""
        # 简化为1.5
        # 实际实现中会计算(现金+证券市值)/(融资余额+融券市值)
        return 1.5

    def check_northbound_flow(self, flow_data: Dict) -> bool:
        """检查北向资金流向"""
        net_flow = flow_data.get("net_flow", 0)
        if abs(net_flow) > self.northbound_flow_threshold:
            return False
        return True
