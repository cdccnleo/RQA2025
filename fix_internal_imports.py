#!/usr/bin/env python3
"""修复core/orchestration目录内的导入路径"""
import os
import re

# 需要更新的文件列表
files_to_update = [
    "src/core/orchestration/business_process/app_startup_listener.py",
    "src/core/orchestration/business_process/service_governance.py",
    "src/core/orchestration/business_process/service_scheduler.py",
    "src/core/orchestration/business_process/data_collection_orchestrator.py",
    "src/core/orchestration/distributed_scheduler.py",
    "src/core/orchestration/incremental_collection_persistence.py",
    "src/core/orchestration/historical_data_scheduler.py",
    "src/core/orchestration/historical_data_acquisition_service.py",
    "src/core/orchestration/data_complement_scheduler.py",
    "src/core/orchestration/ai_driven_optimizer.py",
    "src/core/orchestration/ai_driven_scheduler.py",
]

# 替换规则 - 修复内部导入
replacements = [
    # 修复从infrastructure.orchestration的导入
    (r'from src\.infrastructure\.orchestration\.', 'from src.core.orchestration.'),
    # 修复相对导入 - business_process内部
    (r'from \.\.\.\.infrastructure\.orchestration\.business_process\.', 'from src.core.orchestration.business_process.'),
    (r'from \.\.\.infrastructure\.orchestration\.business_process\.', 'from src.core.orchestration.business_process.'),
    (r'from \.\.infrastructure\.orchestration\.business_process\.', 'from src.core.orchestration.business_process.'),
    # 修复相对导入 - orchestration根目录
    (r'from \.\.\.\.infrastructure\.orchestration\.', 'from src.core.orchestration.'),
    (r'from \.\.\.infrastructure\.orchestration\.', 'from src.core.orchestration.'),
    (r'from \.\.infrastructure\.orchestration\.', 'from src.core.orchestration.'),
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

print("\n🎉 内部导入路径更新完成!")
