#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终语法修复脚本 - 修复所有残留的语法错误
"""

import re
from pathlib import Path

def final_fix(file_path):
    """最终语法修复"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 修复1: WriteResult缺少execution_time后的闭括号
    content = re.sub(
        r'(execution_time=[^)]+)\)(\s*\n\s*except)',
        r'\1\n                )\2',
        content
    )
    
    # 修复2: SQL语句缺少闭括号 
    content = re.sub(
        r'(VALUES \(\?, \?, \?\)",\s*\(measurement, field_set, tag_set\))\s*\n\s*affected_rows',
        r'\1\n                    )\n\n                affected_rows',
        content
    )
    
    # 修复3: HealthCheckResult错误参数
    # status -> is_healthy
    content = re.sub(r'status=ConnectionStatus\.CONNECTED', 'is_healthy=True', content)
    content = re.sub(r'status=ConnectionStatus\.ERROR', 'is_healthy=False', content)
    content = re.sub(r'status=ConnectionStatus\.DISCONNECTED', 'is_healthy=False', content)
    
    # 移除不存在的参数
    content = re.sub(r',?\s*error_count=\d+,?', '', content)
    content = re.sub(r',?\s*active_connections=\d+,?', '', content)
    content = re.sub(r',?\s*total_connections=\d+,?', '', content)
    content = re.sub(r',?\s*last_check_time=[^,)]+,?', '', content)
    
    # 修复4: 确保HealthCheckResult有message参数
    content = re.sub(
        r'HealthCheckResult\(\s*is_healthy=(True|False),\s*response_time=([^,)]+)(?:,\s*details=([^)]+))?\s*\)',
        r'HealthCheckResult(is_healthy=\1, response_time=\2, message="", details=\3 if \3 else None)',
        content
    )
    
    # 修复5: 清理多余的逗号
    content = re.sub(r',\s*,', ',', content)
    content = re.sub(r',\s*\)', ')', content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    files = [
        Path("src/infrastructure/utils/adapters/sqlite_adapter.py"),
        Path("src/infrastructure/utils/adapters/influxdb_adapter.py"),
        Path("src/infrastructure/utils/adapters/postgresql_adapter.py"),
        Path("src/infrastructure/utils/adapters/redis_adapter.py"),
    ]
    
    print("🔧 最终语法修复...")
    for file_path in files:
        if file_path.exists():
            if final_fix(file_path):
                print(f"  ✅ {file_path.name}")
    print("✅ 完成")

if __name__ == "__main__":
    main()



