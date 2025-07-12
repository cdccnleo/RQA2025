# ---------------- src\core\trading\strategies\factory.py ----------------

from typing import Dict, Type, Any
import importlib
import pandas as pd
import numpy as np
import backtrader as bt
from src.infrastructure.config.paths import path_config
from src.infrastructure.utils.tools import time_execution
from src.trading.strategies.enhanced import EnhancedTradingStrategy
from src.trading.strategies.china.star_market_strategy import StarMarketStrategy
from src.trading.strategies.china.base_strategy import ChinaMarketStrategy
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)  # 自动继承全局配置


class StrategyFactory:
    """策略工厂类，负责动态创建和管理交易策略

    属性：
        STRATEGY_REGISTRY (Dict): 策略注册表，存储策略名称与类的映射
    """

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type) -> None:
        """注册自定义策略类

        参数:
            name: 策略名称标识
            strategy_class: 策略类
        """
        cls.STRATEGY_REGISTRY[name] = strategy_class
        logger.info(f"策略 '{name}' 注册成功")

    @classmethod
    def get_strategy(cls, name: str, *args, **kwargs) -> Any:
        """通过名称获取策略实例

        参数:
            name: 策略名称
            args: 传递给策略构造函数的位置参数
            kwargs: 传递给策略构造函数的关键字参数

        返回:
            策略实例

        异常:
            ValueError: 当策略名称未注册时抛出
        """
        if name not in cls.STRATEGY_REGISTRY:
            raise ValueError(f"未注册的策略名称: {name}")
        return cls.STRATEGY_REGISTRY[name](*args, **kwargs)

    @classmethod
    def load_strategy_from_config(cls, config: Dict) -> Any:
        """从配置字典加载策略实例

        参数:
            config: 包含策略配置的字典，需包含 'name' 键

        返回:
            策略实例

        异常:
            KeyError: 当配置中缺少 'name' 键时抛出
        """
        if 'name' not in config:
            raise KeyError("策略配置必须包含 'name' 键")
        strategy_name = config['name']
        params = config.get('params', {})
        return cls.get_strategy(strategy_name, **params)

    @classmethod
    def dynamic_load_strategy(cls, module_name: str, class_name: str) -> Type:
        """动态加载策略类

        参数:
            module_name: 模块名称
            class_name: 类名称

        返回:
            策略类

        异常:
            ImportError: 当模块或类不存在时抛出
        """
        try:
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            if not issubclass(strategy_class, bt.Strategy):
                raise TypeError(f"类 {class_name} 不是 bt.Strategy 的子类")
            return strategy_class
        except (ImportError, AttributeError, TypeError) as e:
            logger.error(f"动态加载策略失败: {str(e)}")
            raise

    @classmethod
    def create_and_attach(cls, strategy_name: str, cerebro: bt.Cerebro, **kwargs) -> Any:
        """创建策略实例并附加到 Cerebro 引擎

        参数:
            strategy_name: 策略名称
            cerebro: Backtrader 引擎实例
            kwargs: 传递给策略构造函数的参数

        返回:
            策略实例
        """
        strategy_class = cls.STRATEGY_REGISTRY.get(strategy_name)
        if not strategy_class:
            logger.warning(f"未找到策略 {strategy_name}，使用默认策略")
            strategy_class = EnhancedTradingStrategy
        strategy = strategy_class(**kwargs)
        cerebro.addstrategy(strategy)
        return strategy

    @staticmethod
    def create(strategy_name: str, config: dict) -> bt.Strategy:
        # 注册策略以便动态加载
        STRATEGY_REGISTRY = {
            "moving_average": MovingAverageStrategy,
            "rsi": RSIStrategy,
            "mean_reversion": MeanReversionStrategy,
            "momentum": MomentumStrategy,
            "multi_factor": MultiFactorStrategy,
            "predictive": PredictiveStrategy
        }

    @classmethod
    def perform_sensitivity_analysis(cls, strategy_class: Type, data: pd.DataFrame, params_grid: Dict,
                                     **kwargs) -> pd.DataFrame:
        """执行策略参数敏感性分析

        Args:
            strategy_class: 策略类
            data: 回测数据
            params_grid: 参数网格，例如{'param1': [value1, value2], 'param2': [value1, value2]}
            kwargs: 其他传递给回测引擎的参数

        Returns:
            包含不同参数组合下策略性能指标的DataFrame
        """
        from sklearn.model_selection import ParameterGrid

        results = []
        param_grid = ParameterGrid(params_grid)

        for params in param_grid:
            cerebro = bt.Cerebro()
            cerebro.addstrategy(strategy_class, **params)
            cerebro.adddata(bt.feeds.PandasData(dataname=data))
            cerebro.broker.setcash(kwargs.get('initial_cash', 1e6))
            cerebro.broker.setcommission(kwargs.get('commission', 0.001))

            # 运行回测
            cerebro.run()

            # 获取分析结果
            analyzer = cerebro.runstrats()[0][0].analyzers.getbyname('backtest_analyzer')
            performance = analyzer.get_analysis()

            # 记录结果
            result = {**params, **performance}
            results.append(result)

        return pd.DataFrame(results)


class MovingAverageStrategy(bt.Strategy):
    """基于移动平均线的交易策略

    参数:
        ma_period (int): 移动平均线窗口期，默认20天
    """

    params = (
        ('ma_period', 20),
    )

    def __init__(self):
        # 初始化移动平均线
        self.ma = bt.indicators.SMA(self.data.close, period=self.params.ma_period)

    def next(self):
        # 如果收盘价上穿移动平均线，发送买入信号
        if self.data.close[0] > self.ma[0] and self.data.close[-1] <= self.ma[-1]:
            self.buy()
        # 如果收盘价下穿移动平均线，发送卖出信号
        elif self.data.close[0] < self.ma[0] and self.data.close[-1] >= self.ma[-1]:
            self.sell()


class RSIStrategy(bt.Strategy):
    """基于RSI指标的交易策略

    参数:
        rsi_period (int): RSI计算周期，默认14天
        overbought (int): 超买阈值，默认70
        oversold (int): 超卖阈值，默认30
    """

    params = (
        ('rsi_period', 14),
        ('overbought', 70),
        ('oversold', 30),
    )

    def __init__(self):
        # 初始化RSI指标
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

    def next(self):
        # 如果RSI低于超卖阈值，发送买入信号
        if self.rsi[0] < self.params.oversold:
            self.buy()
        # 如果RSI高于超买阈值，发送卖出信号
        elif self.rsi[0] > self.params.overbought:
            self.sell()


class MeanReversionStrategy(bt.Strategy):
    """均值回归策略

    参数:
        window (int): 均值窗口期，默认20天
        threshold (float): 均值偏离阈值，默认1.5
    """

    params = (
        ('window', 20),
        ('threshold', 1.5),
    )

    def __init__(self):
        # 计算均值和标准差
        self.mean = bt.indicators.SMA(self.data.close, period=self.params.window)
        self.std = bt.indicators.StdDev(self.data.close, period=self.params.window)

    def next(self):
        # 计算当前价格与均值的偏离度
        z_score = (self.data.close[0] - self.mean[0]) / self.std[0]

        # 如果价格显著低于均值，买入
        if z_score < -self.params.threshold:
            self.buy()
        # 如果价格显著高于均值，卖出
        elif z_score > self.params.threshold:
            self.sell()


class MomentumStrategy(bt.Strategy):
    """动量策略

    参数:
        window (int): 动量计算窗口期，默认20天
        threshold (float): 动量阈值，默认0.05（5%）
    """

    params = (
        ('window', 20),
        ('threshold', 0.05),
    )

    def __init__(self):
        # 计算动量（当前价格相对于窗口期初的价格变化）
        self.momentum = self.data.close / self.data.close(-self.params.window) - 1

    def next(self):
        # 如果动量超过阈值，买入
        if self.momentum[0] > self.params.threshold:
            self.buy()
        # 如果动量低于负阈值，卖出
        elif self.momentum[0] < -self.params.threshold:
            self.sell()


class MultiFactorStrategy(EnhancedTradingStrategy):
    """多因子策略，继承自增强型交易策略

    参数:
        factors (list): 因子列表，默认["momentum", "value", "quality"]
    """

    params = (
        ('factors', ["momentum", "value", "quality"]),
    )

    def __init__(self):
        super().__init__()
        # 动态加载因子数据
        self._load_factors()

    def _load_factors(self):
        """加载多因子数据"""
        logger.info("加载多因子数据...")
        # 这里可以根据实际需求加载因子数据
        self.factor_data = pd.DataFrame({
            "momentum": np.random.randn(len(self.data)),
            "value": np.random.randn(len(self.data)),
            "quality": np.random.randn(len(self.data))
        })

    def next(self):
        # 计算因子得分
        factor_scores = self.factor_data.iloc[self.datas[0].datetime.date()]
        # 综合得分为各因子得分的加权和
        composite_score = factor_scores[self.params.factors].mean()

        # 根据综合得分生成交易信号
        if composite_score > 0:
            self.buy()
        elif composite_score < 0:
            self.sell()


class PredictiveStrategy(bt.Strategy):
    """基于预测信号的交易策略

    参数:
        signal_name (str): 预测信号列名，默认"prediction"
    """

    params = (
        ('signal_name', 'prediction'),
    )

    def __init__(self):
        # 加载预测信号
        self._load_signals()

    def _load_signals(self):
        """加载预测信号"""
        logger.info("加载预测信号...")
        # 假设预测信号已经加载到数据中
        self.signals = self.data.getline(self.params.signal_name)

    def next(self):
        # 根据预测信号生成交易信号
        if self.signals[0] > 0:
            self.buy()
        elif self.signals[0] < 0:
            self.sell()


STRATEGY_REGISTRY = {
    "enhanced": EnhancedTradingStrategy,
    "moving_average": MovingAverageStrategy,
    "rsi": RSIStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "multi_factor": MultiFactorStrategy,
    "predictive": PredictiveStrategy,
    "china_market": ChinaMarketStrategy,
    "star_market": StarMarketStrategy
}
