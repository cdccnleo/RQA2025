#!/usr/bin/env python3
"""
修复最后的F821导入错误
"""

from pathlib import Path

# 修复列表: (文件路径, 导入行)
fixes = [
    # timedelta imports
    ('src/data/adapters/professional/level2_market_data_adapter.py', 'from datetime import timedelta'),
    
    # asyncio imports
    ('src/gateway/web/risk_control_monitor.py', 'import asyncio'),
    
    # json imports
    ('src/gateway/web/strategy_lifecycle_routes.py', 'import json'),
    
    # time imports
    ('src/gateway/web/data_source_config_manager.py', 'import time'),
    
    # dataclasses imports
    ('src/infrastructure/async/core/async_data_processor.py', 'from dataclasses import dataclass, field'),
]

fixed_count = 0
error_files = []

for file_path, import_line in fixes:
    try:
        file = Path(file_path)
        if not file.exists():
            error_files.append(f"文件不存在: {file_path}")
            continue
        
        content = file.read_text(encoding='utf-8')
        
        # 检查是否已存在该导入
        import_name = import_line.replace('import ', '').replace('from ', '').split()[0]
        if import_name in content and 'import' in content:
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
    for err in error_files:
        print(f"   - {err}")
