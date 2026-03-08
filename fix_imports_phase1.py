#!/usr/bin/env python3
"""
Phase 1: 批量修复F821导入错误脚本
修复最常见的导入问题
"""

import re
from pathlib import Path

# 修复列表: (文件路径, 需要添加的导入行)
fixes = [
    # 标准库导入修复
    ('src/core/boundary/core/unified_service_manager.py', 'import uuid\n'),
    ('src/infrastructure/orchestration/scheduler/worker_manager.py', 'from datetime import timedelta\n'),
    ('src/infrastructure/orchestration/historical_data_scheduler.py', 'from datetime import date\n'),
    ('src/infrastructure/orchestration/distributed_scheduler.py', 'import threading\n'),
    ('src/gateway/web/scheduler_routes.py', 'from datetime import datetime\n'),
    ('src/gateway/web/strategy_lifecycle_routes.py', 'import json\n'),
    ('src/gateway/web/risk_control_monitor.py', 'import asyncio\n'),
    ('src/data/adapters/professional/level2_market_data_adapter.py', 'from datetime import timedelta\n'),
    ('src/data/loader/postgresql_loader.py', 'from datetime import datetime\n'),
    ('src/infrastructure/integration/adapters/features_adapter.py', 'from datetime import datetime\n'),
    ('src/infrastructure/integration/adapters/risk_adapter.py', 'from datetime import datetime\n'),
    ('src/core/core_optimization/components/testing_enhancer.py', 'import asyncio\n'),
    ('src/infrastructure/distributed/coordinator/queue_engine.py', 'import asyncio\n'),
    ('src/infrastructure/resource/resource_manager.py', 'import traceback\n'),
    ('src/infrastructure/resource/unified_monitor_adapter.py', 'import traceback\n'),
    ('src/infrastructure/security/audit/advanced_audit_logger.py', 'import functools\n'),
    ('src/infrastructure/testing/integration/health_monitor.py', 'import threading\n'),
    ('src/infrastructure/testing/integration/integration_tester.py', 'import queue\nimport threading\n'),
    ('src/infrastructure/health/integration/distributed_test_runner.py', 'from queue import Queue\n'),
    
    # numpy导入修复
    ('src/core/core_optimization/monitoring/ai_performance_optimizer.py', 'import numpy as np\n'),
    ('src/features/core/engine.py', 'import numpy as np\n'),
    ('src/infrastructure/testing/integration/health_monitor.py', 'import numpy as np\n'),
    ('src/infrastructure/testing/integration/integration_tester.py', 'import numpy as np\n'),
    
    # typing导入修复
    ('src/core/core_optimization/optimizations/long_term_optimizations.py', 'from typing import Protocol\n'),
    ('src/core/core_optimization/optimizations/short_term_optimizations.py', 'from typing import Protocol\n'),
]

fixed_count = 0
error_files = []

for file_path, import_line in fixes:
    try:
        file = Path(file_path)
        if not file.exists():
            # 尝试在src目录下查找
            file = Path('src') / file_path.replace('src/', '')
            if not file.exists():
                error_files.append(f"文件不存在: {file_path}")
                continue
        
        content = file.read_text(encoding='utf-8')
        
        # 检查是否已存在该导入
        import_name = import_line.strip().replace('import ', '').replace('from ', '')
        if import_name.split()[0] in content:
            continue
        
        # 在文件开头找到合适的位置插入导入
        lines = content.split('\n')
        import_idx = 0
        
        # 跳过文档字符串
        in_docstring = False
        for i, line in enumerate(lines):
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    import_idx = i + 1
                else:
                    in_docstring = True
            elif not in_docstring and (line.startswith('import ') or line.startswith('from ')):
                import_idx = i + 1
        
        # 插入导入行
        lines.insert(import_idx, import_line.strip())
        file.write_text('\n'.join(lines), encoding='utf-8')
        fixed_count += 1
        print(f"✅ 已修复: {file_path}")
        
    except Exception as e:
        error_files.append(f"{file_path}: {str(e)}")

print(f"\n📝 Phase 1: 完成修复 {fixed_count} 个文件")
if error_files:
    print(f"⚠️  {len(error_files)} 个文件出错:")
    for err in error_files[:10]:  # 只显示前10个错误
        print(f"   - {err}")
