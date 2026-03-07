#!/usr/bin/env python3
import os

# 修复测试文件
with open('tests/unit/infrastructure/logging/services/test_hot_reload_service.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复断言
content = content.replace("assert status['watched_files_count'] == 2", "assert status['monitored_files'] == 2")

with open('tests/unit/infrastructure/logging/services/test_hot_reload_service.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('File updated successfully')
