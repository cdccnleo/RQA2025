#!/usr/bin/env python3

with open('src/data/integration/enhanced_data_integration_modules/utilities.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修复2690-2703行的缩进
lines[2689] = '        while True:\n'
lines[2690] = '            try:\n'
lines[2691] = '                # 更新性能指标\n'
lines[2692] = '                self._update_performance_metrics()\n'
lines[2694] = '                # 自适应调整\n'
lines[2695] = '                self._adaptive_adjustment()\n'
lines[2697] = '                # 缓存预热检查\n'
lines[2698] = '                self._check_cache_warming()\n'
lines[2700] = '                time.sleep(30)  # 每30秒检查一次\n'
lines[2701] = '            except Exception as e:\n'
lines[2702] = '                logger.error(f"性能监控错误: {e}")\n'
lines[2703] = '                time.sleep(60)\n'

with open('src/data/integration/enhanced_data_integration_modules/utilities.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('修复了2690-2703行的缩进')
