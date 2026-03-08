#!/usr/bin/env python3
"""修复剩余的导入路径"""
import os
import re

# 需要更新的文件列表
files_to_update = [
    "src/gateway/web/datasource_routes.py",
]

# 替换规则
replacements = [
    # 从 distributed/coordinator 导入
    (r'from src\.infrastructure\.distributed\.coordinator\.unified_scheduler import', 
     'from src.core.orchestration.scheduler import'),
]

for file_path in files_to_update:
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        continue
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已更新: {file_path}")
        else:
            print(f"⏭️ 无需更新: {file_path}")
            
    except Exception as e:
        print(f"❌ 更新失败 {file_path}: {e}")

print("\n🎉 导入路径更新完成!")
