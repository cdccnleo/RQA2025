# 读取文件
with open('tests/unit/core/foundation/test_unified_exceptions.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换BusinessLogicError方法
old_business_logic = '''    def test_business_logic_error(self):
        """测试业务逻辑错误"""
        exc = BusinessLogicError(
            message="Business rule violation",
            rule_name="max_trade_limit",
            current_value=100000,
            limit_value=50000
        )

        assert exc.error_type == "BUSINESS_LOGIC_ERROR"
        assert exc.context["operation"] == "trade_execution"
        assert exc.context["entity_id"] == "order_123"
'''

new_business_logic = '''    def test_business_logic_error(self):
        """测试业务逻辑错误"""
        exc = BusinessLogicError(
            message="Business rule violation",
            operation="trade_execution",
            entity_id="order_123"
        )

        assert exc.message == "Business rule violation"
        assert exc.error_type == "BUSINESS_LOGIC_ERROR"
        assert exc.context["operation"] == "trade_execution"
        assert exc.context["entity_id"] == "order_123"

'''

content = content.replace(old_business_logic, new_business_logic)

# 替换TradingError方法
old_trading = '''    def test_trading_error(self):
        """测试交易错误"""
        exc = TradingError(
            message="Order execution failed",
            order_id="ORD_001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            reason="INSUFFICIENT_FUNDS"
        )

        assert exc.message == "Order execution failed"
        assert exc.error_type == "TRADING_ERROR"
        assert exc.context["order_id"] == "ORD_001"
        assert exc.context["symbol"] == "AAPL"
        assert exc.side == "BUY"
        assert exc.quantity == 100

'''

new_trading = '''    def test_trading_error(self):
        """测试交易错误"""
        exc = TradingError(
            message="Order execution failed",
            order_id="ORD_001",
            symbol="AAPL"
        )

        assert exc.message == "Order execution failed"
        assert exc.error_type == "TRADING_ERROR"
        assert exc.context["order_id"] == "ORD_001"
        assert exc.context["symbol"] == "AAPL"

'''

content = content.replace(old_trading, new_trading)

# 替换RiskError方法
old_risk = '''    def test_risk_error(self):
        """测试风险错误"""
        exc = RiskError(
            message="Risk limit exceeded",
            risk_type="POSITION_SIZE",
            current_value=150000,
            limit_value=100000,
            threshold_percentage=50.0
        )

        assert exc.message == "Risk limit exceeded"
        assert exc.error_type == "RISK_ERROR"
        assert exc.context["risk_type"] == "POSITION_SIZE"
        assert exc.context["threshold"] == 100000.0

'''

new_risk = '''    def test_risk_error(self):
        """测试风险错误"""
        exc = RiskError(
            message="Risk limit exceeded",
            risk_type="POSITION_SIZE",
            threshold=100000.0
        )

        assert exc.message == "Risk limit exceeded"
        assert exc.error_type == "RISK_ERROR"
        assert exc.context["risk_type"] == "POSITION_SIZE"
        assert exc.context["threshold"] == 100000.0

'''

content = content.replace(old_risk, new_risk)

# 替换StrategyError方法
old_strategy = '''    def test_strategy_error(self):
        """测试策略错误"""
        exc = StrategyError(
            message="Strategy execution error",
            strategy_id="STRAT_001",
            strategy_name="MeanReversion",
            error_phase="SIGNAL_GENERATION",
            retry_count=2
        )

        assert exc.message == "Strategy execution error"
        assert exc.error_type == "STRATEGY_ERROR"
        assert exc.context["strategy_id"] == "STRAT_001"
        assert exc.context["signal"] == "BUY_SIGNAL"

'''

new_strategy = '''    def test_strategy_error(self):
        """测试策略错误"""
        exc = StrategyError(
            message="Strategy execution error",
            strategy_id="STRAT_001",
            signal="BUY_SIGNAL"
        )

        assert exc.message == "Strategy execution error"
        assert exc.error_type == "STRATEGY_ERROR"
        assert exc.context["strategy_id"] == "STRAT_001"
        assert exc.context["signal"] == "BUY_SIGNAL"

'''

content = content.replace(old_strategy, new_strategy)

# 写入文件
with open('tests/unit/core/foundation/test_unified_exceptions.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed all exception constructor calls and assertions')

