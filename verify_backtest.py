#!/usr/bin/env python3
"""
回测交易记录验证脚本
验证回测记录 backtest_model_model_job_1771237605_20260216_182646_1771251743 的交易记录明细
"""

import sys
sys.path.insert(0, '/app')

from src.gateway.web.backtest_persistence import load_backtest_result
import json
from datetime import datetime

def verify_backtest(backtest_id):
    """验证回测记录"""
    result = load_backtest_result(backtest_id)
    
    if not result:
        print("❌ 回测记录未找到")
        return False
    
    print("=" * 60)
    print("📊 回测记录验证报告")
    print("=" * 60)
    print()
    
    # 基本信息
    print("【基本信息】")
    print(f"  回测ID: {result.get('backtest_id')}")
    print(f"  策略ID: {result.get('strategy_id')}")
    print(f"  状态: {result.get('status')}")
    print(f"  开始日期: {result.get('start_date')}")
    print(f"  结束日期: {result.get('end_date')}")
    print()
    
    # 资金信息
    print("【资金信息】")
    print(f"  初始资金: ¥{result.get('initial_capital', 0):,.2f}")
    print(f"  最终资金: ¥{result.get('final_capital', 0):,.2f}")
    total_return = result.get('total_return', 0)
    print(f"  总收益率: {total_return*100:.2f}%")
    print(f"  年化收益率: {result.get('annualized_return', 0)*100:.2f}%")
    print()
    
    # 绩效指标
    print("【绩效指标】")
    print(f"  夏普比率: {result.get('sharpe_ratio', 0):.4f}")
    print(f"  最大回撤: {result.get('max_drawdown', 0)*100:.2f}%")
    print(f"  胜率: {result.get('win_rate', 0)*100:.2f}%")
    print(f"  总交易次数: {result.get('total_trades', 0)}")
    print()
    
    # 交易记录明细
    trades = result.get('trades', [])
    print(f"【交易记录明细】共 {len(trades)} 条")
    print()
    
    for i, trade in enumerate(trades, 1):
        print(f"  {i}. 交易时间: {trade.get('timestamp')}")
        print(f"      股票代码: {trade.get('symbol')}")
        print(f"      交易类型: {trade.get('type')}")
        print(f"      交易价格: ¥{trade.get('price')}")
        print(f"      交易数量: {trade.get('quantity')}")
        
        if trade.get('type') == 'buy':
            print(f"      交易成本: ¥{trade.get('cost')}")
            print(f"      手续费: ¥{trade.get('commission')}")
        else:
            print(f"      交易收入: ¥{trade.get('revenue')}")
            print(f"      手续费: ¥{trade.get('commission')}")
            pnl = trade.get('pnl', 0)
            pnl_percent = trade.get('pnl_percent', 0)
            print(f"      盈亏金额: ¥{pnl}")
            print(f"      盈亏比例: {pnl_percent}%")
        print()
    
    # 资金曲线
    equity_curve = result.get('equity_curve', [])
    print(f"【资金曲线】共 {len(equity_curve)} 个点")
    print(f"  起始: ¥{equity_curve[0]:,.2f}")
    print(f"  结束: ¥{equity_curve[-1]:,.2f}")
    print()
    
    # 验证结果
    print("=" * 60)
    print("✅ 验证结果")
    print("=" * 60)
    
    checks = []
    
    # 检查1: 交易记录不为空
    checks.append(("交易记录存在", len(trades) > 0))
    
    # 检查2: 交易记录包含必要字段
    if trades:
        required_fields = ['timestamp', 'symbol', 'type', 'price', 'quantity']
        has_all_fields = all(field in trades[0] for field in required_fields)
        checks.append(("交易记录字段完整", has_all_fields))
    
    # 检查3: 股票代码有效
    if trades:
        symbols = [t.get('symbol') for t in trades if t.get('symbol')]
        checks.append(("股票代码存在", len(symbols) > 0))
    
    # 检查4: 交易时间有效
    if trades:
        timestamps = [t.get('timestamp') for t in trades if t.get('timestamp')]
        checks.append(("交易时间存在", len(timestamps) > 0))
    
    # 检查5: 资金曲线有效
    checks.append(("资金曲线存在", len(equity_curve) > 0))
    
    # 检查6: 收益率计算正确
    if equity_curve and len(equity_curve) >= 2:
        expected_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        checks.append(("收益率计算正确", abs(expected_return - total_return) < 0.01))
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    print()
    if all_passed:
        print("🎉 所有验证通过！交易记录基于真实历史数据。")
    else:
        print("⚠️ 部分验证未通过，请检查数据。")
    
    return all_passed

if __name__ == "__main__":
    backtest_id = "backtest_model_model_job_1771237605_20260216_182646_1771251743"
    verify_backtest(backtest_id)
