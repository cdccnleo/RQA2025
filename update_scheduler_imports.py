#!/usr/bin/env python3
"""批量更新统一调度器导入路径"""
import os
import re

# 需要更新的文件列表
files_to_update = [
    "src/gateway/web/datasource_routes.py",
    "src/gateway/web/api.py",
    "src/gateway/web/training_job_executor.py",
    "src/gateway/web/model_training_routes.py",
    "src/gateway/web/inference_service.py",
    "src/gateway/web/feature_task_executor.py",
    "src/gateway/web/feature_engineering_service.py",
    "src/gateway/web/data_collection_service.py",
    "src/gateway/web/data_collection_scheduler_manager.py",
    "src/features/distributed/worker_executor.py",
    "src/infrastructure/orchestration/business_process/app_startup_listener.py",
]

# 替换规则
replacements = [
    # 从 distributed/coordinator 导入
    (r'from src\.infrastructure\.distributed\.coordinator\.unified_scheduler import', 
     'from src.core.orchestration.scheduler import'),
    # 从 orchestration/scheduler 导入
    (r'from src\.infrastructure\.orchestration\.scheduler import', 
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
