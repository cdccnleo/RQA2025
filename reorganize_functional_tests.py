#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重组functional测试文件到对应的unit测试目录
"""

import shutil
from pathlib import Path


def reorganize_tests():
    """重组测试文件"""
    project_root = Path(__file__).parent
    functional_dir = project_root / "tests" / "functional"
    unit_dir = project_root / "tests" / "unit"
    
    # 定义映射关系
    mappings = {
        # functional子目录 -> unit目标目录
        "data": "data",
        "features": "features",
        "infrastructure": "infrastructure",
        "ml": "ml",
        "monitoring": "monitoring",
        "risk": "risk",
        "security": "security",
        "strategy": "strategy",
        "trading": "trading",
        "logging": "infrastructure/logging",  # logging归到infrastructure下
    }
    
    moved_files = []
    errors = []
    
    print("="*80)
    print("开始重组functional测试文件")
    print("="*80)
    
    # 处理每个映射
    for func_subdir, unit_subdir in mappings.items():
        source_dir = functional_dir / func_subdir
        target_dir = unit_dir / unit_subdir
        
        if not source_dir.exists():
            print(f"⏭️  跳过: {func_subdir} (源目录不存在)")
            continue
        
        # 确保目标目录存在
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动所有Python测试文件
        py_files = list(source_dir.glob("*.py"))
        
        if not py_files:
            print(f"⏭️  跳过: {func_subdir} (无Python文件)")
            continue
        
        print(f"\n📁 处理: {func_subdir} -> {unit_subdir}")
        print(f"   找到 {len(py_files)} 个文件")
        
        for py_file in py_files:
            if py_file.name == "__init__.py":
                continue
            
            target_file = target_dir / py_file.name
            
            try:
                # 如果目标文件已存在，重命名
                if target_file.exists():
                    # 添加_functional后缀
                    base_name = py_file.stem
                    if not base_name.endswith('_functional'):
                        new_name = f"{base_name}_functional.py"
                        target_file = target_dir / new_name
                
                # 移动文件
                shutil.copy2(py_file, target_file)
                moved_files.append((py_file, target_file))
                print(f"   ✅ {py_file.name} -> {target_file.relative_to(project_root)}")
                
            except Exception as e:
                errors.append((py_file, str(e)))
                print(f"   ❌ {py_file.name}: {e}")
    
    # 处理顶层文件（data/infrastructure相关）
    top_level_files = {
        "test_migrator_functional.py": "infrastructure",
        "test_query_executor_functional.py": "data",
        "test_write_manager_functional.py": "data",
    }
    
    print(f"\n📁 处理顶层文件")
    for filename, target_module in top_level_files.items():
        source_file = functional_dir / filename
        if source_file.exists():
            target_dir = unit_dir / target_module
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / filename
            
            try:
                shutil.copy2(source_file, target_file)
                moved_files.append((source_file, target_file))
                print(f"   ✅ {filename} -> {target_file.relative_to(project_root)}")
            except Exception as e:
                errors.append((source_file, str(e)))
                print(f"   ❌ {filename}: {e}")
    
    # 打印总结
    print("\n" + "="*80)
    print("重组完成总结")
    print("="*80)
    print(f"✅ 成功移动: {len(moved_files)} 个文件")
    if errors:
        print(f"❌ 失败: {len(errors)} 个文件")
        for source, error in errors:
            print(f"   - {source.name}: {error}")
    
    # 生成移动记录
    record_file = project_root / "test_reorganization_record.txt"
    with open(record_file, 'w', encoding='utf-8') as f:
        f.write("测试文件重组记录\n")
        f.write("="*80 + "\n")
        f.write(f"日期: 2025-11-02\n")
        f.write(f"成功移动: {len(moved_files)} 个文件\n\n")
        
        for source, target in moved_files:
            f.write(f"{source.relative_to(project_root)} -> {target.relative_to(project_root)}\n")
    
    print(f"\n📝 移动记录已保存到: {record_file}")
    
    return moved_files, errors


if __name__ == "__main__":
    moved, errors = reorganize_tests()
    print(f"\n🎉 重组完成！移动了 {len(moved)} 个文件")
    if not errors:
        print("✅ 所有文件移动成功！")

