#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复监控系统测试阻塞问题
移除或减少time.sleep调用，优化测试性能
"""

import re
from pathlib import Path


# 需要修复的文件和修复策略
FIXES = {
    "test_component_bus.py": [
        # 将1.1秒睡眠改为0.01秒或移除
        (r"time\.sleep\(1\.1\)", "time.sleep(0.01)"),
        (r"time\.sleep\(0\.1\)", "pass  # time.sleep(0.01)"),
    ],
    "test_component_bus_boundary_conditions.py": [
        (r"time\.sleep\(0\.1\)", "pass  # time.sleep(0.01)"),
    ],
    "test_performance_monitor.py": [
        (r"time\.sleep\(0\.01\)", "pass  # Removed sleep"),
        (r"time\.sleep\(0\.1\)", "pass  # Removed sleep"),
    ],
}


def fix_file(file_path, replacements):
    """修复单个文件"""
    print(f"修复: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 已修复")
        return True
    else:
        print(f"  ⏭️  无需修复")
        return False


def main():
    """主函数"""
    base_path = Path("tests/unit/infrastructure/monitoring")
    
    print("="*80)
    print("修复监控系统测试阻塞问题")
    print("="*80)
    
    fixed_count = 0
    
    for filename, replacements in FIXES.items():
        file_path = base_path / filename
        if file_path.exists():
            if fix_file(file_path, replacements):
                fixed_count += 1
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    print(f"\n{'='*80}")
    print(f"修复完成！共修复 {fixed_count} 个文件")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

