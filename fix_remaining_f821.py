#!/usr/bin/env python3
"""
修复剩余的F821错误
"""

from pathlib import Path

# 修复列表
fixes = [
    # 修复 missing Any import
    ('src/backtest/portfolio/optimized_portfolio_optimizer.py', 'from typing import Any, Dict, List, Optional, Tuple, Union', 'from typing import'),
    
    # 修复 missing uuid import  
    ('src/core/boundary/core/unified_service_manager.py', 'import uuid', 'import uuid'),
    
    # 修复 missing timedelta import
    ('src/data/adapters/professional/level2_market_data_adapter.py', 'from datetime import datetime, timedelta', 'from datetime import'),
    ('src/infrastructure/orchestration/scheduler/worker_manager.py', 'from datetime import datetime, timedelta', 'from datetime import'),
    
    # 修复 missing datetime import
    ('src/data/loader/postgresql_loader.py', 'from datetime import datetime', 'from datetime import'),
    ('src/infrastructure/integration/adapters/features_adapter.py', 'from datetime import datetime', 'from datetime import'),
    ('src/infrastructure/integration/adapters/risk_adapter.py', 'from datetime import datetime', 'from datetime import'),
    ('src/gateway/web/scheduler_routes.py', 'from datetime import datetime', 'from datetime import'),
    
    # 修复 missing time import
    ('src/core/core_services/api/api_models.py', 'import time', 'import time'),
    
    # 修复 missing asyncio import
    ('src/core/core_optimization/components/testing_enhancer.py', 'import asyncio', 'import asyncio'),
    ('src/gateway/web/risk_control_monitor.py', 'import asyncio', 'import asyncio'),
    ('src/infrastructure/distributed/coordinator/queue_engine.py', 'import asyncio', 'import asyncio'),
    
    # 修复 missing json import
    ('src/gateway/web/strategy_lifecycle_routes.py', 'import json', 'import json'),
    
    # 修复 missing threading import
    ('src/infrastructure/orchestration/distributed_scheduler.py', 'import threading', 'import threading'),
    ('src/infrastructure/testing/integration/health_monitor.py', 'import threading', 'import threading'),
    
    # 修复 missing Protocol import
    ('src/core/core_optimization/optimizations/long_term_optimizations.py', 'from typing import Protocol', 'from typing import Protocol'),
    ('src/core/core_optimization/optimizations/short_term_optimizations.py', 'from typing import Protocol', 'from typing import Protocol'),
    
    # 修复 missing queue import
    ('src/infrastructure/testing/integration/integration_tester.py', 'import queue', 'import queue'),
    ('src/infrastructure/health/integration/distributed_test_runner.py', 'from queue import Queue', 'from queue import Queue'),
    
    # 修复 missing functools import
    ('src/infrastructure/security/audit/advanced_audit_logger.py', 'import functools', 'import functools'),
    
    # 修复 missing date import
    ('src/infrastructure/orchestration/historical_data_scheduler.py', 'from datetime import date', 'from datetime import date'),
]

fixed_count = 0
error_files = []

for file_path, import_line, check_pattern in fixes:
    try:
        file = Path(file_path)
        if not file.exists():
            error_files.append(f"文件不存在: {file_path}")
            continue
        
        content = file.read_text(encoding='utf-8')
        
        # 检查是否已存在该导入
        if check_pattern in content:
            continue
        
        # 在文件开头找到合适的位置插入导入
        lines = content.split('\n')
        import_idx = 0
        
        # 跳过文档字符串
        in_docstring = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    import_idx = i + 1
                else:
                    in_docstring = True
            elif not in_docstring and (line.startswith('import ') or line.startswith('from ')):
                import_idx = i + 1
        
        # 插入导入行
        lines.insert(import_idx, import_line)
        file.write_text('\n'.join(lines), encoding='utf-8')
        fixed_count += 1
        print(f"✅ 已修复: {file_path}")
        
    except Exception as e:
        error_files.append(f"{file_path}: {str(e)}")

print(f"\n📝 完成修复 {fixed_count} 个文件")
if error_files:
    print(f"⚠️  {len(error_files)} 个文件出错:")
    for err in error_files[:10]:
        print(f"   - {err}")
