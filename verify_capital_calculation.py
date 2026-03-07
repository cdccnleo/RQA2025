#!/usr/bin/env python3
"""
验证资金计算逻辑
"""
import sys
sys.path.insert(0, '/app')
from src.gateway.web.backtest_persistence import load_backtest_result
from decimal import Decimal

# 获取最新的回测记录
backtest_id = 'backtest_model_model_job_1771237605_20260216_182646_1771323986'
result = load_backtest_result(backtest_id)

if not result:
    print('回测记录未找到')
    sys.exit(1)

print('=' * 80)
print('💰 资金计算逻辑验证报告')
print('=' * 80)
print()

# 基本信息
initial_capital = float(result.get('initial_capital', 100000))
final_capital = float(result.get('final_capital', 0))

print(f"初始资金: ¥{initial_capital:,.2f}")
print(f"记录的最终资金: ¥{final_capital:,.2f}")
print()

# 获取交易记录
trades = result.get('trades', [])
print(f"交易记录数: {len(trades)}")
print()

# 按时间排序
trades_sorted = sorted(trades, key=lambda x: x.get('timestamp', ''))

# 模拟前端资金计算
available_capital = initial_capital
print('【逐笔交易资金计算】')
print()

for i, trade in enumerate(trades_sorted):
    trade_type = trade.get('type', '')
    price = float(trade.get('price', 0))
    quantity = float(trade.get('quantity', 0))
    cost = float(trade.get('cost', 0))
    pnl = float(trade.get('pnl', 0))
    
    capital_before = available_capital
    
    if trade_type in ['buy', 'buy_forced']:
        # 买入：扣除成本和手续费
        total_cost = (price * quantity) + cost
        available_capital -= total_cost
        change = -total_cost
    elif trade_type in ['sell', 'sell_forced']:
        # 卖出：增加卖出金额，扣除手续费
        total_proceeds = (price * quantity) - cost
        available_capital += total_proceeds
        change = total_proceeds
    else:
        change = 0
    
    print(f"交易{i+1}: {trade.get('timestamp', '')[:10]} {trade.get('symbol', '')} {trade_type}")
    print(f"  价格: ¥{price:,.2f}, 数量: {quantity}, 手续费: ¥{cost:,.2f}, 盈亏: ¥{pnl:,.2f}")
    print(f"  资金变化: ¥{change:+,.2f}")
    print(f"  交易前: ¥{capital_before:,.2f} → 交易后: ¥{available_capital:,.2f}")
    print()

print('【验证结果】')
print(f"计算的最终资金: ¥{available_capital:,.2f}")
print(f"记录的最终资金: ¥{final_capital:,.2f}")

difference = abs(available_capital - final_capital)
if difference < 0.01:
    print(f"✅ 资金计算正确！差异: ¥{difference:,.2f}")
else:
    print(f"❌ 资金计算有误！差异: ¥{difference:,.2f}")
    print()
    print('可能的原因:')
    print('1. 交易记录不完整')
    print('2. 手续费计算不准确')
    print('3. 强制平仓处理有误')
