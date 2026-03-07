# 修复异常测试断言
with open('tests/unit/core/foundation/test_unified_exceptions.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复TradingError断言
content = content.replace('assert exc.order_id == "ORD_001"', 'assert exc.context["order_id"] == "ORD_001"')
content = content.replace('assert exc.symbol == "AAPL"', 'assert exc.context["symbol"] == "AAPL"')

# 修复RiskError断言
content = content.replace('assert exc.risk_type == "POSITION_SIZE"', 'assert exc.context["risk_type"] == "POSITION_SIZE"')
content = content.replace('assert exc.threshold == 100000.0', 'assert exc.context["threshold"] == 100000.0')

# 修复StrategyError断言
content = content.replace('assert exc.strategy_id == "STRAT_001"', 'assert exc.context["strategy_id"] == "STRAT_001"')
content = content.replace('assert exc.signal == "BUY_SIGNAL"', 'assert exc.context["signal"] == "BUY_SIGNAL"')

# 写入文件
with open('tests/unit/core/foundation/test_unified_exceptions.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed exception test assertions')

