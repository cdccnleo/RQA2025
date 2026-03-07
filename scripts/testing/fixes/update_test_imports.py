#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量更新测试文件导入路径脚本
用于修复因目录结构优化导致的导入路径问题
"""

import os

# 导入路径映射
IMPORT_MAPPINGS = {
    # 已删除的模块 -> 新的正确路径
    'from src.risk_control import': 'from src.risk.risk_engine import',
    'from src.signal.signal_generator import': 'from src.features.signal_generator import',
    'from src.settlement.settlement_engine import': 'from src.trading.settlement.settlement_engine import',
    'from src.compliance.regulatory_compliance import': 'from src.infrastructure.compliance.regulatory_compliance import',
    'from src.strategy.base_strategy import': 'from src.trading.strategies.base_strategy import',
    'from src.live_trading.broker_adapter import': 'from src.trading.broker.broker_adapter import',
    'from src.execution.smart_execution import': 'from src.trading.execution.smart_execution import',
    'from src.backtesting.backtest_engine import': 'from src.backtest.backtest_engine import',
}

# 需要更新的测试文件列表
TEST_FILES = [
    'tests/unit/integration/test_fpga_integration.py',
    'tests/unit/integration/test_order_executor.py',
    'tests/unit/integration/test_signal_generator.py',
    'tests/unit/integration/test_settlement_engine.py',
    'tests/unit/compliance/test_regulatory_interface.py',
    'tests/unit/strategy/test_strategy.py',
    'tests/unit/live_trading/test_broker_adapter.py',
]


def update_imports_in_file(file_path):
    """更新单个文件中的导入路径"""
    if not os.path.exists(file_path):
        print(f"⚠️  文件不存在: {file_path}")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        updated = False

        # 应用导入映射
        for old_import, new_import in IMPORT_MAPPINGS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                updated = True
                print(f"  ✅ 更新: {old_import} -> {new_import}")

        # 如果内容有变化，写回文件
        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ 文件已更新: {file_path}")
            return True
        else:
            print(f"  ℹ️  无需更新: {file_path}")
            return False

    except Exception as e:
        print(f"  ❌ 更新失败: {file_path} - {e}")
        return False


def main():
    """主函数"""
    print("🔄 开始批量更新测试文件导入路径...")
    print("=" * 60)

    success_count = 0
    total_count = len(TEST_FILES)

    for test_file in TEST_FILES:
        print(f"\n📁 处理文件: {test_file}")
        if update_imports_in_file(test_file):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"📊 更新完成统计:")
    print(f"  ✅ 成功更新: {success_count}/{total_count}")
    print(f"  ❌ 失败数量: {total_count - success_count}")

    if success_count == total_count:
        print("🎉 所有文件更新成功!")
    else:
        print("⚠️  部分文件更新失败，请检查错误信息")


if __name__ == "__main__":
    main()
