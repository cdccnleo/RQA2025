class BaseStrategy:

    """最小可用基础策略类，便于回测与测试用例调用"""

    def __init__(self, name="BaseStrategy"):

        self.name = name
        self.current_positions = {}
        self.portfolio_history = [1000000.0]
        self.trade_history = []

    def initialize(self, initial_capital):

        self.portfolio_history = [initial_capital]
        self.current_positions = {}
        self.trade_history = []

    def generate_signals(self, market_data):
        """生成信号，默认全0（无操作）"""
        return [0 for _ in range(len(market_data))] if hasattr(market_data, '__len__') else [0]

    def execute_trades(self, signals, current_date):
        """执行交易，简单记录"""
        # 假设每次信号都买入1手
        for idx, sig in enumerate(signals):
            if sig != 0:
                self.trade_history.append(type('Trade', (), {'profit': 1.0})())
                self.portfolio_history.append(self.portfolio_history[-1] + 1.0)
            else:
                self.portfolio_history.append(self.portfolio_history[-1])
