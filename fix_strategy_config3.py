#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Read file
with open('tests/unit/strategy/test_unified_strategy_service_simple.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and fix the broken function
# Remove the corrupted part in test_list_strategies
corrupted_part = '''        """测试列出策略"""
                status=StrategyStatus.CREATED,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )'''

if corrupted_part in content:
    content = content.replace(corrupted_part, '        """测试列出策略"""')

# Write back
with open('tests/unit/strategy/test_unified_strategy_service_simple.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Corrupted content removed')
