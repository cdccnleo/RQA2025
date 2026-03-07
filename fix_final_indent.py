#!/usr/bin/env python3

with open('src/data/integration/enhanced_data_integration_modules/utilities.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修复缩进
lines[1783] = '        except Exception as e:\n'
lines[1784] = '            logger.error(f"缓存预热失败: {e}")\n'
lines[1785] = "            self._cache_warming_status['is_warming'] = False\n"

with open('src/data/integration/enhanced_data_integration_modules/utilities.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('修复了最终的缩进问题')
