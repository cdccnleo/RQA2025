"""实时交易信号生成引擎"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from ..features.feature_engine import FeatureEngine
from ..fpga.fpga_accelerator import FpgaManager

logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    """信号生成配置"""
    min_confidence: float = 0.7  # 最小置信度阈值
    max_position: float = 0.1    # 单信号最大仓位比例
    a_share_rules: bool = True   # 是否启用A股特定规则
    use_fpga: bool = True        # 是否使用FPGA加速
    cool_down_period: int = 5    # 相同信号冷却期(分钟)

class SignalGenerator:
    """实时交易信号生成引擎"""

    def __init__(self, feature_engine: FeatureEngine, config: Optional[SignalConfig] = None):
        self.engine = feature_engine
        self.config = config or SignalConfig()
        self.fpga_manager = FpgaManager()
        self.last_signal_time = {}  # symbol -> last signal time

        # 初始化A股特定规则
        if self.config.a_share_rules:
            self._init_a_share_rules()

    def _init_a_share_rules(self):
        """初始化A股特定规则"""
        self.price_limit_rules = {
            "ST": 0.05,    # ST股票涨跌幅限制5%
            "normal": 0.1,  # 普通股票10%
            "688": 0.2      # 科创板20%
        }

        self.t1_settlement = True  # T+1结算制度
        self.trade_restrictions = {
            "ST": ["no_intraday"],  # ST股票禁止日内交易
            "688": ["price_limit"]  # 科创板价格限制
        }

    def _check_a_share_restrictions(self, symbol: str, signal_type: str) -> bool:
        """检查A股交易限制"""
        if not self.config.a_share_rules:
            return True

        # 检查涨跌停限制
        if signal_type == "BUY":
            if symbol.startswith("688"):
                limit = self.price_limit_rules["688"]
            elif symbol.startswith(("ST", "*ST")):
                limit = self.price_limit_rules["ST"]
            else:
                limit = self.price_limit_rules["normal"]

            # 实际实现中会检查当前价格是否接近涨跌停
            # 这里简化为总是允许
            return True

        # 检查其他限制
        if symbol.startswith(("ST", "*ST")) and "no_intraday" in self.trade_restrictions["ST"]:
            return False

        return True

    def _check_cool_down(self, symbol: str) -> bool:
        """检查信号冷却期"""
        if symbol not in self.last_signal_time:
            return True

        last_time = self.last_signal_time[symbol]
        elapsed = (time.time() - last_time) / 60  # 转换为分钟

        return elapsed >= self.config.cool_down_period

    def _generate_with_fpga(self, features: Dict[str, float]) -> Optional[Dict]:
        """使用FPGA加速生成信号"""
        if not self.config.use_fpga:
            return None

        try:
            # 获取FPGA加速器
            fpga_accelerator = self.fpga_manager.get_accelerator("SIGNAL_FPGA")
            if fpga_accelerator and fpga_accelerator.health_monitor.is_healthy():
                # 准备FPGA输入数据
                fpga_input = np.array([
                    features.get("momentum", 0),
                    features.get("sentiment", 0),
                    features.get("order_flow", 0)
                ], dtype=np.float32)

                # 调用FPGA计算 (模拟)
                fpga_result = {
                    "signal": "BUY",  # 实际实现中会从FPGA获取
                    "confidence": 0.8,
                    "target_price": features.get("price", 0) * 1.01,
                    "position": self.config.max_position
                }

                return fpga_result

        except Exception as e:
            logger.error(f"FPGA signal generation failed: {str(e)}")

        return None

    def generate(self, symbol: str, features: Dict[str, float]) -> Optional[Dict]:
        """生成交易信号

        Args:
            symbol: 标的代码
            features: 特征字典

        Returns:
            Dict: 信号字典，包含以下字段:
                - signal: 信号类型(BUY/SELL/HOLD)
                - confidence: 置信度(0-1)
                - target_price: 目标价格
                - position: 建议仓位比例
        """
        # 检查冷却期
        if not self._check_cool_down(symbol):
            return None

        # 检查A股限制
        if not self._check_a_share_restrictions(symbol, "BUY"):
            return None

        # 尝试使用FPGA加速
        fpga_signal = self._generate_with_fpga(features)
        if fpga_signal is not None:
            self.last_signal_time[symbol] = time.time()
            return fpga_signal

        # 软件路径
        signal = {
            "signal": "BUY",
            "confidence": 0.75,
            "target_price": features.get("price", 0) * 1.01,
            "position": self.config.max_position * 0.8
        }

        # 应用置信度阈值
        if signal["confidence"] < self.config.min_confidence:
            signal["signal"] = "HOLD"

        self.last_signal_time[symbol] = time.time()
        return signal

    def batch_generate(self, symbols: List[str], features_list: List[Dict[str, float]]) -> List[Optional[Dict]]:
        """批量生成交易信号"""
        signals = []

        for symbol, features in zip(symbols, features_list):
            try:
                signal = self.generate(symbol, features)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Failed to generate signal for {symbol}: {str(e)}")
                signals.append(None)

        return signals

class ChinaSignalGenerator(SignalGenerator):
    """A股特定信号生成器"""

    def __init__(self, feature_engine: FeatureEngine, config: Optional[SignalConfig] = None):
        super().__init__(feature_engine, config)
        self.config.a_share_rules = True  # 强制启用A股规则

        # 初始化A股特定特征
        self._init_a_share_features()

    def _init_a_share_features(self):
        """初始化A股特定特征"""
        self.margin_features = ["margin_ratio", "short_balance"]
        self.dragon_board_features = ["institutional_net_buy", "hot_money_flow"]

    def generate(self, symbol: str, features: Dict[str, float]) -> Optional[Dict]:
        """生成A股特定交易信号"""
        # 添加A股特定特征检查
        features["margin_effect"] = (
            features.get("margin_ratio", 0) * 0.3 +
            features.get("short_balance", 0) * 0.7
        )

        features["dragon_effect"] = (
            features.get("institutional_net_buy", 0) * 0.6 +
            features.get("hot_money_flow", 0) * 0.4
        )

        # 调用父类生成信号
        signal = super().generate(symbol, features)

        if signal and signal["signal"] != "HOLD":
            # 应用A股特定调整
            signal["position"] = self._adjust_position_for_a_share(signal["position"], features)

        return signal

    def _adjust_position_for_a_share(self, position: float, features: Dict[str, float]) -> float:
        """根据A股特性调整仓位"""
        # 降低ST股票仓位
        if features.get("is_st", False):
            return position * 0.5

        # 根据融资融券数据调整
        margin_effect = features.get("margin_effect", 0)
        if margin_effect > 0.5:
            return min(position * 1.2, self.config.max_position)
        elif margin_effect < -0.5:
            return position * 0.7

        # 根据龙虎榜数据调整
        dragon_effect = features.get("dragon_effect", 0)
        if dragon_effect > 0.8:
            return min(position * 1.3, self.config.max_position)

        return position
