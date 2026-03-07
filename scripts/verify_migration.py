#!/usr/bin/env python3
"""
迁移验证脚本

验证Phase 3迁移的完成情况，检查：
1. 适配器是否已迁移到UnifiedBusinessAdapter
2. 组件是否使用新的BaseComponent
3. 导入路径是否正确
4. 向后兼容性是否保持

创建时间: 2025-11-03
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def analyze_file_imports(file_path: Path) -> Dict[str, any]:
    """分析文件的导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {'error': str(e)}
    
    analysis = {
        'file': str(file_path.relative_to(PROJECT_ROOT)),
        'imports': [],
        'uses_base_component': False,
        'uses_base_adapter': False,
        'uses_unified_business_adapter': False,
        'uses_old_imports': False,
        'class_definitions': []
    }
    
    # 检查导入
    import_patterns = {
        'BaseComponent': r'from\s+src\.core\.foundation\.base_component\s+import',
        'BaseAdapter': r'from\s+src\.core\.foundation\.base_adapter\s+import',
        'UnifiedBusinessAdapter': r'from\s+src\.core\.integration\.unified_business_adapters\s+import',
        'OldBaseBusinessAdapter': r'from\s+[.\w]+business_adapters\s+import\s+BaseBusinessAdapter',
        'OldComponentFactory': r'class\s+ComponentFactory',
    }
    
    for key, pattern in import_patterns.items():
        if re.search(pattern, content):
            if 'Component' in key or 'Adapter' in key:
                analysis['imports'].append(key)
                if 'BaseComponent' in key:
                    analysis['uses_base_component'] = True
                elif 'BaseAdapter' in key:
                    analysis['uses_base_adapter'] = True
                elif 'UnifiedBusinessAdapter' in key:
                    analysis['uses_unified_business_adapter'] = True
                elif 'Old' in key:
                    analysis['uses_old_imports'] = True
    
    # 检查类定义
    class_pattern = r'class\s+(\w+)\s*\([^)]*\):'
    classes = re.findall(class_pattern, content)
    analysis['class_definitions'] = classes
    
    return analysis


def check_adapters() -> Dict[str, any]:
    """检查适配器迁移状态"""
    adapter_dir = PROJECT_ROOT / 'src' / 'core' / 'integration' / 'adapters'
    
    adapter_files = [
        adapter_dir / 'trading_adapter.py',
        adapter_dir / 'risk_adapter.py',
        adapter_dir / 'security_adapter.py',
    ]
    
    results = {
        'total': len(adapter_files),
        'migrated': 0,
        'using_old': 0,
        'files': []
    }
    
    for file_path in adapter_files:
        if not file_path.exists():
            continue
        
        analysis = analyze_file_imports(file_path)
        
        if analysis['uses_unified_business_adapter']:
            results['migrated'] += 1
            status = '✅ 已迁移'
        elif analysis.get('uses_old_imports'):
            results['using_old'] += 1
            status = '⚠️ 使用旧导入'
        else:
            status = '❓ 未知状态'
        
        results['files'].append({
            'file': analysis['file'],
            'status': status,
            'imports': analysis['imports']
        })
    
    return results


def check_components() -> Dict[str, any]:
    """检查组件迁移状态"""
    component_dirs = [
        PROJECT_ROOT / 'src' / 'core' / 'container',
        PROJECT_ROOT / 'src' / 'core' / 'integration' / 'middleware',
        PROJECT_ROOT / 'src' / 'core' / 'orchestration' / 'business_process',
    ]
    
    component_files = []
    for dir_path in component_dirs:
        if dir_path.exists():
            component_files.extend(dir_path.glob('*_components.py'))
            component_files.extend(dir_path.glob('refactored_*_components.py'))
    
    results = {
        'total': 0,
        'migrated': 0,
        'refactored_available': 0,
        'files': []
    }
    
    for file_path in component_files:
        if not file_path.exists():
            continue
        
        analysis = analyze_file_imports(file_path)
        
        is_refactored = 'refactored' in file_path.name
        
        if is_refactored:
            results['refactored_available'] += 1
            status = '✅ 重构版本可用'
        elif analysis['uses_base_component']:
            results['migrated'] += 1
            status = '✅ 已迁移'
        else:
            results['total'] += 1
            status = '⏸️ 待迁移'
        
        results['files'].append({
            'file': analysis['file'],
            'status': status,
            'is_refactored': is_refactored
        })
    
    return results


def generate_verification_report(adapter_results: Dict, component_results: Dict) -> str:
    """生成验证报告"""
    report = []
    report.append("=" * 70)
    report.append("Phase 3 迁移验证报告")
    report.append("=" * 70)
    report.append("")
    
    # 适配器状态
    report.append("📦 适配器迁移状态")
    report.append("-" * 70)
    report.append(f"总文件数: {adapter_results['total']}")
    report.append(f"✅ 已迁移: {adapter_results['migrated']}")
    report.append(f"⚠️ 使用旧导入: {adapter_results['using_old']}")
    report.append("")
    
    for file_info in adapter_results['files']:
        report.append(f"  {file_info['status']}: {file_info['file']}")
        if file_info['imports']:
            report.append(f"    导入: {', '.join(file_info['imports'])}")
    report.append("")
    
    # 组件状态
    report.append("🧩 组件迁移状态")
    report.append("-" * 70)
    report.append(f"总文件数: {component_results['total']}")
    report.append(f"✅ 已迁移: {component_results['migrated']}")
    report.append(f"✅ 重构版本可用: {component_results['refactored_available']}")
    report.append("")
    
    for file_info in component_results['files']:
        report.append(f"  {file_info['status']}: {file_info['file']}")
    report.append("")
    
    # 总体统计
    report.append("=" * 70)
    report.append("总体进度")
    report.append("=" * 70)
    
    total_files = adapter_results['total'] + component_results['total']
    migrated_files = adapter_results['migrated'] + component_results['migrated']
    refactored_available = component_results['refactored_available']
    
    if total_files > 0:
        progress = (migrated_files / total_files) * 100
        report.append(f"迁移进度: {migrated_files}/{total_files} ({progress:.1f}%)")
    else:
        report.append(f"重构版本可用: {refactored_available} 个")
        report.append("所有组件已提供重构版本，可以开始替换")
    
    report.append("")
    
    return "\n".join(report)


def main():
    """主函数"""
    print("🔍 开始验证迁移状态...")
    print()
    
    # 检查适配器
    print("检查适配器...")
    adapter_results = check_adapters()
    
    # 检查组件
    print("检查组件...")
    component_results = check_components()
    
    # 生成报告
    report = generate_verification_report(adapter_results, component_results)
    print(report)
    
    # 保存报告
    report_file = PROJECT_ROOT / 'test_logs' / 'phase3_migration_verification.txt'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 验证报告已保存到: {report_file.relative_to(PROJECT_ROOT)}")
    print()
    
    # 总结
    if adapter_results['migrated'] == adapter_results['total']:
        print("✅ 所有适配器已成功迁移！")
    else:
        print(f"⚠️  {adapter_results['using_old']} 个适配器仍使用旧导入")
    
    if component_results['refactored_available'] > 0:
        print(f"✅ {component_results['refactored_available']} 个重构版本可用，可以开始替换原始文件")


if __name__ == '__main__':
    main()

