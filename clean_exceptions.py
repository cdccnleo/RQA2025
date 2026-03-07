# 清理异常测试参数
with open('tests/unit/core/foundation/test_unified_exceptions.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义要移除的行
to_remove = [
    'side="BUY",',
    'quantity=100,',
    'reason="INSUFFICIENT_FUNDS"',
    'current_value=150000,',
    'limit_value=100000,',
    'threshold_percentage=50.0',
    'strategy_name="MeanReversion",',
    'error_phase="SIGNAL_GENERATION",',
    'retry_count=2',
    'assert exc.side == "BUY"',
    'assert exc.quantity == 100',
    'assert exc.current_value == 150000',
    'assert exc.limit_value == 100000',
    'assert exc.threshold_percentage == 50.0',
    'assert exc.strategy_name == "MeanReversion"',
    'assert exc.error_phase == "SIGNAL_GENERATION"',
    'assert exc.retry_count == 2'
]

# 逐个移除
for item in to_remove:
    content = content.replace(item, '')

# 写入文件
with open('tests/unit/core/foundation/test_unified_exceptions.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Cleaned up exception test parameters')

