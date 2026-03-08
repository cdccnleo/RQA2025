#!/usr/bin/env python3
"""
批量修复F821导入错误脚本
"""

import re
from pathlib import Path

# 定义需要修复的文件和对应的导入
typing_fixes = {
    'src/backtest/portfolio/optimized_portfolio_optimizer.py': ['from typing import Any, Dict, List, Optional, Tuple'],
    'src/infrastructure/integration/fallback_services.py': ['from typing import Dict, Any, List, Optional'],
    'src/infrastructure/distributed/consul_service_discovery.py': ['from typing import Callable, Dict, Any, List, Optional'],
}

# 修复1: backtest/portfolio/optimized_portfolio_optimizer.py
file1 = Path('src/backtest/portfolio/optimized_portfolio_optimizer.py')
if file1.exists():
    content = file1.read_text(encoding='utf-8')
    # 检查是否已有typing导入
    if 'from typing import' not in content and 'import typing' not in content:
        # 在文件开头添加导入
        lines = content.split('\n')
        import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_idx = i + 1
        lines.insert(import_idx, 'from typing import Any, Dict, List, Optional, Tuple')
        file1.write_text('\n'.join(lines), encoding='utf-8')
        print(f"✅ 已修复: {file1}")

# 修复2: infrastructure/integration/fallback_services.py
file2 = Path('src/infrastructure/integration/fallback_services.py')
if file2.exists():
    content = file2.read_text(encoding='utf-8')
    if 'from typing import' not in content and 'import typing' not in content:
        lines = content.split('\n')
        import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_idx = i + 1
        lines.insert(import_idx, 'from typing import Dict, Any, List, Optional')
        file2.write_text('\n'.join(lines), encoding='utf-8')
        print(f"✅ 已修复: {file2}")

# 修复3: infrastructure/distributed/consul_service_discovery.py
file3 = Path('src/infrastructure/distributed/consul_service_discovery.py')
if file3.exists():
    content = file3.read_text(encoding='utf-8')
    if 'from typing import' not in content and 'import typing' not in content:
        lines = content.split('\n')
        import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_idx = i + 1
        lines.insert(import_idx, 'from typing import Callable, Dict, Any, List, Optional')
        file3.write_text('\n'.join(lines), encoding='utf-8')
        print(f"✅ 已修复: {file3}")

print("\n📝 Phase 1: 基础typing导入修复完成")
