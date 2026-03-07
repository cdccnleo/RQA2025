#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# Read file
with open('tests/unit/strategy/test_unified_strategy_service_simple.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all StrategyConfig with name parameter
content = re.sub(
    r'name="([^"]*)",\s*description="[^"]*",\s*strategy_type=([^,]+),\s*parameters=([^,]+),\s*symbols=([^,]+),\s*status=[^,]+,\s*created_at=[^,]+,\s*updated_at=[^,]+',
    r'strategy_name="\1",\n            strategy_type=\2,\n            parameters=\3,\n            symbols=\4,\n            timeframe="1d",\n            risk_limits={"max_loss": 0.02}',
    content,
    flags=re.MULTILINE | re.DOTALL
)

# Write back
with open('tests/unit/strategy/test_unified_strategy_service_simple.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('StrategyConfig fixes applied successfully')
