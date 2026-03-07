#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Read file
with open('tests/unit/strategy/test_unified_strategy_service_simple.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix the broken line
for i, line in enumerate(lines):
    if 'risk_limits={"max_loss": 0.02}, strategy_service):' in line:
        # Fix this line
        lines[i] = '            risk_limits={"max_loss": 0.02}\n'
        lines[i+1] = '        )\n'
        lines[i+2] = '\n'
        lines[i+3] = '        # Create strategy\n'
        lines[i+4] = '        result = strategy_service.create_strategy(config)\n'
        lines[i+5] = '        assert result is True\n'
        lines[i+6] = '\n'
        lines[i+7] = '        # Get strategy\n'
        lines[i+8] = '        retrieved = strategy_service.get_strategy("test_001")\n'
        lines[i+9] = '        assert retrieved is not None\n'
        lines[i+10] = '        assert retrieved.strategy_id == "test_001"\n'
        lines[i+11] = '\n'
        lines[i+12] = '    def test_list_strategies(self, strategy_service):\n'
        lines[i+13] = '        """测试列出策略"""\n'
        break

# Write back
with open('tests/unit/strategy/test_unified_strategy_service_simple.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('File syntax error fixed')
