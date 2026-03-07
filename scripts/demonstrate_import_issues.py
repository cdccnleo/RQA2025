#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示绝对路径导入和相对导入的区别
"""

import os
import sys

# 添加src路径（模拟pytest配置）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("🔍 导入路径问题演示")
print("=" * 50)

print("1️⃣ 当前Python路径:")
for i, path in enumerate(sys.path[:5]):  # 只显示前5个
    print(f"   {i}: {path}")

print()
print("2️⃣ 测试绝对路径导入:")
try:
    # 这种导入在某些环境下会失败
    from src.infrastructure.cache.strategies.cache_strategy_manager import LRUStrategy
    print("   ✅ 绝对路径导入成功")
except ImportError as e:
    print(f"   ❌ 绝对路径导入失败: {e}")

print()
print("3️⃣ 测试相对路径导入:")
try:
    # 这种导入更稳定
    from infrastructure.cache.strategies.cache_strategy_manager import LRUStrategy
    print("   ✅ 相对路径导入成功")
    print(f"   📦 LRUStrategy类: {LRUStrategy}")
except ImportError as e:
    print(f"   ❌ 相对路径导入失败: {e}")

print()
print("4️⃣ 路径解析对比:")
print("   绝对路径: src.infrastructure.cache.strategies.cache_strategy_manager")
print("   相对路径: infrastructure.cache.strategies.cache_strategy_manager")
print("   差异: 移除了前缀 'src.'")

print()
print("🎯 为什么相对导入更好:")
print("   • 不依赖于src目录在Python路径中的位置")
print("   • 更符合Python的包结构约定")
print("   • 在不同环境中更稳定")
print("   • pytest测试发现机制更友好")
print("   • 避免了路径配置问题")

print()
print("📋 修复策略:")
print("   • 将所有测试文件中的 'from src.infrastructure.' 改为 'from infrastructure.'")
print("   • 保持pytest.ini中的pythonpath = src配置")
print("   • 确保src目录下有正确的__init__.py文件")
