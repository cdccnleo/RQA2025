#!/usr/bin/env python3
"""批量更新业务编排器导入路径"""
import os
import re

# 需要更新的文件列表
files_to_update = [
    "src/gateway/web/datasource_routes.py",
    "src/gateway/web/api.py",
    "src/gateway/web/trading_execution_service.py",
    "src/gateway/web/trading_execution_routes.py",
    "src/gateway/web/strategy_execution_routes.py",
    "src/gateway/web/risk_control_service.py",
    "src/gateway/web/risk_control_routes.py",
    "src/gateway/web/model_training_routes.py",
    "src/gateway/web/health_routes.py",
    "src/gateway/web/feature_engineering_routes.py",
    "src/gateway/web/data_collectors.py",
    "src/gateway/web/data_collection_api.py",
    "src/gateway/web/backtest_routes.py",
    "src/gateway/web/architecture_service.py",
    "src/gateway/web/architecture_routes.py",
    "src/gateway/api/historical_collection_monitor_api.py",
    "src/gateway/api/historical_collection_websocket.py",
    "src/core/core_optimization/optimizations/short_term_optimizations.py",
    "src/strategy/core/strategy_service.py",
    "src/ml/core/ml_core.py",
    "src/data/loader/stock_loader.py",
    "src/data/loader/batch_loader.py",
    "src/core/core_optimization/components/test_file_generator.py",
    "src/infrastructure/monitoring/services/market_adaptive_monitor.py",
]

# 替换规则
replacements = [
    # 从 infrastructure.orchestration 导入
    (r'from src\.infrastructure\.orchestration\.business\.event_system import', 
     'from src.core.orchestration.business.event_system import'),
    (r'from src\.infrastructure\.orchestration\.business_process\.app_startup_listener import', 
     'from src.core.orchestration.business_process.app_startup_listener import'),
    (r'from src\.infrastructure\.orchestration\.business_process\.data_collection_orchestrator import', 
     'from src.core.orchestration.business_process.data_collection_orchestrator import'),
    (r'from src\.infrastructure\.orchestration\.business_process\.data_collection_state_machine import', 
     'from src.core.orchestration.business_process.data_collection_state_machine import'),
    (r'from src\.infrastructure\.orchestration\.business_process\.service_scheduler import', 
     'from src.core.orchestration.business_process.service_scheduler import'),
    (r'from src\.infrastructure\.orchestration\.business_process\.service_governance import', 
     'from src.core.orchestration.business_process.service_governance import'),
    (r'from src\.infrastructure\.orchestration\.business_process import', 
     'from src.core.orchestration.business_process import'),
    (r'from src\.infrastructure\.orchestration\.pool\.process_instance_pool import', 
     'from src.core.orchestration.pool.process_instance_pool import'),
    (r'from src\.infrastructure\.orchestration\.ai_driven_scheduler import', 
     'from src.core.orchestration.ai_driven_scheduler import'),
    (r'from src\.infrastructure\.orchestration\.ai_driven_optimizer import', 
     'from src.core.orchestration.ai_driven_optimizer import'),
    (r'from src\.infrastructure\.orchestration\.distributed_scheduler import', 
     'from src.core.orchestration.distributed_scheduler import'),
    (r'from src\.infrastructure\.orchestration\.historical_data_scheduler import', 
     'from src.core.orchestration.historical_data_scheduler import'),
    (r'from src\.infrastructure\.orchestration\.historical_data_acquisition_service import', 
     'from src.core.orchestration.historical_data_acquisition_service import'),
    (r'from src\.infrastructure\.orchestration\.data_complement_scheduler import', 
     'from src.core.orchestration.data_complement_scheduler import'),
    (r'from src\.infrastructure\.orchestration\.realtime_data_processor import', 
     'from src.core.orchestration.realtime_data_processor import'),
    (r'from src\.infrastructure\.orchestration import', 
     'from src.core.orchestration import'),
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
