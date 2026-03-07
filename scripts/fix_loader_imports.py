#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复数据层加载器和适配器导入问题的脚本
"""

import re
from pathlib import Path


def create_loader_alias_file():
    """创建加载器别名文件"""

    loader_aliases = """
# 数据加载器别名文件
# 为保持向后兼容性，提供常用别名

from .stock_loader import StockDataLoader as StockLoader
from .crypto_loader import CryptoDataLoader as CryptoLoader
from .bond_loader import BondDataLoader as BondLoader
from .forex_loader import ForexDataLoader as ForexLoader
from .macro_loader import MacroDataLoader as MacroLoader
from .commodity_loader import CommodityDataLoader as CommodityLoader
from .options_loader import OptionsDataLoader as OptionsLoader
from .index_loader import IndexDataLoader as IndexLoader
from .financial_loader import FinancialDataLoader as FinancialLoader
from .news_loader import FinancialNewsLoader as NewsLoader
from .batch_loader import BatchDataLoader as BatchLoader

# 保持向后兼容性
__all__ = [
    'StockLoader', 'CryptoLoader', 'BondLoader', 'ForexLoader',
    'MacroLoader', 'CommodityLoader', 'OptionsLoader', 'IndexLoader',
    'FinancialLoader', 'NewsLoader', 'BatchLoader'
]
"""

    alias_file = Path("src/data/loader/__init__.py")
    if not alias_file.exists():
        alias_file.write_text(loader_aliases, encoding='utf-8')
        print("✅ 创建了加载器别名文件")
    else:
        content = alias_file.read_text(encoding='utf-8')
        if "StockLoader" not in content:
            # 追加别名
            with open(alias_file, 'a', encoding='utf-8') as f:
                f.write("\n" + loader_aliases)
            print("✅ 更新了加载器别名文件")


def create_adapter_alias_file():
    """创建适配器别名文件"""

    adapter_aliases = """
# 数据适配器别名文件
# 为保持向后兼容性，提供常用别名

from .china.adapter import ChinaDataAdapter as ChinaStockAdapter
from .miniqmt.adapter import MiniQMTAdapter
from .miniqmt.miniqmt_data_adapter import MiniQMTDataAdapter
from .news import NewsDataAdapter, NewsSentimentAdapter
from .macro import MacroEconomicAdapter

# 中国市场适配器别名
from .china import ChinaStockAdapter, MarginTradingAdapter

# 保持向后兼容性
__all__ = [
    'ChinaStockAdapter', 'MiniQMTAdapter', 'MiniQMTDataAdapter',
    'NewsDataAdapter', 'NewsSentimentAdapter', 'MacroEconomicAdapter',
    'MarginTradingAdapter'
]
"""

    alias_file = Path("src/data/adapters/__init__.py")
    if not alias_file.exists():
        alias_file.write_text(adapter_aliases, encoding='utf-8')
        print("✅ 创建了适配器别名文件")
    else:
        content = alias_file.read_text(encoding='utf-8')
        if "ChinaStockAdapter" not in content:
            # 追加别名
            with open(alias_file, 'a', encoding='utf-8') as f:
                f.write("\n" + adapter_aliases)
            print("✅ 更新了适配器别名文件")


def fix_remaining_import_issues():
    """修复剩余的导入问题"""

    data_src_path = Path("src/data")

    # 需要修复的文件和对应的类名映射
    loader_class_mapping = {
        'StockDataLoader': 'StockLoader',
        'CryptoDataLoader': 'CryptoLoader',
        'BondDataLoader': 'BondLoader',
        'ForexDataLoader': 'ForexLoader',
        'MacroDataLoader': 'MacroLoader',
        'CommodityDataLoader': 'CommodityLoader',
        'OptionsDataLoader': 'OptionsLoader',
        'IndexDataLoader': 'IndexLoader',
        'FinancialDataLoader': 'FinancialLoader',
        'FinancialNewsLoader': 'NewsLoader',
        'BatchDataLoader': 'BatchLoader'
    }

    adapter_class_mapping = {
        'ChinaDataAdapter': 'ChinaStockAdapter',
        'MiniQMTDataAdapter': 'MiniQMTAdapter'
    }

    fixed_files = 0

    # 遍历测试文件，查找导入问题
    test_files = [
        "tests/unit/data/test_stock_adapter.py",
        "tests/unit/data/test_crypto_adapter_basic.py",
        "tests/unit/data/test_china_stock_adapter_basic.py"
    ]

    for test_file_path in test_files:
        test_file = Path(test_file_path)
        if test_file.exists():
            content = test_file.read_text(encoding='utf-8')

            # 修复类名导入问题
            for old_class, new_class in {**loader_class_mapping, **adapter_class_mapping}.items():
                # 查找 from ... import OldClass 模式
                pattern = rf'from\s+[\w\.]+\s+import\s+{re.escape(old_class)}'
                if re.search(pattern, content):
                    # 替换为新类名
                    new_content = re.sub(
                        rf'(\s+){re.escape(old_class)}(\s+)',
                        rf'\1{new_class}\2',
                        content
                    )
                    test_file.write_text(new_content, encoding='utf-8')
                    fixed_files += 1
                    print(f"✅ 修复测试文件: {test_file.name}")

    return fixed_files


def main():
    """主函数"""
    print("🔧 开始修复数据层加载器和适配器导入问题...")
    print("=" * 60)

    # 创建别名文件
    create_loader_alias_file()
    create_adapter_alias_file()

    # 修复测试文件中的导入问题
    fixed_count = fix_remaining_import_issues()

    print("\n📊 修复统计:")
    print(f"✅ 创建/更新了别名文件")
    print(f"✅ 修复了 {fixed_count} 个测试文件")
    print("\n🎯 导入问题修复完成!")
    print("\n📝 别名文件说明:")
    print("   - src/data/loader/__init__.py: 加载器别名")
    print("   - src/data/adapters/__init__.py: 适配器别名")
    print("\n🔄 现在可以使用统一的类名进行导入")


if __name__ == "__main__":
    main()
