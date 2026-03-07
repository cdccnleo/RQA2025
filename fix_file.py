#!/usr/bin/env python3
import sys

with open('src/infrastructure/logging/services/hot_reload_service.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找并修改return语句
for i, line in enumerate(lines):
    if '"monitored_files": len(self._file_timestamps)' in line:
        lines[i] = line.rstrip() + ',\n            "last_check": self._last_check\n'
        break

with open('src/infrastructure/logging/services/hot_reload_service.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('File modified successfully')