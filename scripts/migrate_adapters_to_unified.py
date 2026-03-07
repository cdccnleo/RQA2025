#!/usr/bin/env python3
"""
适配器迁移验证脚本

验证现有适配器与新UnifiedBusinessAdapter的兼容性
并更新导入路径

创建时间: 2025-11-03
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_adapter_files() -> List[Path]:
    """查找所有需要迁移的适配器文件"""
    adapter_dir = PROJECT_ROOT / 'src' / 'core' / 'integration' / 'adapters'
    
    adapter_files = [
        adapter_dir / 'trading_adapter.py',
        adapter_dir / 'risk_adapter.py',
        adapter_dir / 'security_adapter.py',
    ]
    
    return [f for f in adapter_files if f.exists()]


def analyze_adapter_file(file_path: Path) -> Dict[str, any]:
    """分析适配器文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    analysis = {
        'file': str(file_path.relative_to(PROJECT_ROOT)),
        'lines': len(content.splitlines()),
        'imports': [],
        'uses_base_business_adapter': False,
        'uses_unified_business_adapter': False,
        'class_name': None,
        'migration_needed': False
    }
    
    # 检查导入
    import_patterns = [
        r'from\s+\.business_adapters\s+import\s+([^\n]+)',
        r'from\s+src\.core\.integration\.business_adapters\s+import\s+([^\n]+)',
        r'from\s+src\.core\.integration\.unified_business_adapters\s+import\s+([^\n]+)',
    ]
    
    for pattern in import_patterns:
        matches = re.findall(pattern, content)
        if matches:
            analysis['imports'].extend([m.strip() for m in matches])
    
    # 检查是否使用BaseBusinessAdapter
    if 'BaseBusinessAdapter' in content:
        analysis['uses_base_business_adapter'] = True
        
        # 检查类定义
        class_match = re.search(r'class\s+(\w+Adapter)\s*\([^)]*BaseBusinessAdapter', content)
        if class_match:
            analysis['class_name'] = class_match.group(1)
            analysis['migration_needed'] = True
    
    # 检查是否已经使用UnifiedBusinessAdapter
    if 'UnifiedBusinessAdapter' in content:
        analysis['uses_unified_business_adapter'] = True
        analysis['migration_needed'] = False
    
    return analysis


def generate_migration_report(analyses: List[Dict]) -> str:
    """生成迁移报告"""
    report = []
    report.append("=" * 70)
    report.append("适配器迁移分析报告")
    report.append("=" * 70)
    report.append("")
    
    for analysis in analyses:
        report.append(f"文件: {analysis['file']}")
        report.append(f"  行数: {analysis['lines']}")
        report.append(f"  类名: {analysis['class_name'] or '未找到'}")
        report.append(f"  使用BaseBusinessAdapter: {analysis['uses_base_business_adapter']}")
        report.append(f"  使用UnifiedBusinessAdapter: {analysis['uses_unified_business_adapter']}")
        report.append(f"  需要迁移: {'是' if analysis['migration_needed'] else '否'}")
        report.append(f"  导入: {', '.join(analysis['imports']) if analysis['imports'] else '无'}")
        report.append("")
    
    # 统计
    total_files = len(analyses)
    files_to_migrate = sum(1 for a in analyses if a['migration_needed'])
    total_lines = sum(a['lines'] for a in analyses)
    
    report.append("=" * 70)
    report.append("统计信息")
    report.append("=" * 70)
    report.append(f"总文件数: {total_files}")
    report.append(f"需要迁移: {files_to_migrate}")
    report.append(f"总代码行数: {total_lines}")
    report.append(f"平均文件大小: {total_lines // total_files if total_files > 0 else 0} 行")
    report.append("")
    
    return "\n".join(report)


def update_adapter_imports(file_path: Path) -> Tuple[bool, str]:
    """更新适配器文件的导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 替换导入
    # 从: from .business_adapters import BaseBusinessAdapter, BusinessLayerType
    # 到: from src.core.integration.unified_business_adapters import UnifiedBusinessAdapter, BusinessLayerType
    
    patterns = [
        (
            r'from\s+\.business_adapters\s+import\s+BaseBusinessAdapter',
            'from src.core.integration.unified_business_adapters import UnifiedBusinessAdapter'
        ),
        (
            r'from\s+src\.core\.integration\.business_adapters\s+import\s+BaseBusinessAdapter',
            'from src.core.integration.unified_business_adapters import UnifiedBusinessAdapter'
        ),
    ]
    
    for old_pattern, new_import in patterns:
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_import, content)
            
            # 更新类继承
            content = re.sub(
                r'class\s+(\w+Adapter)\s*\([^)]*BaseBusinessAdapter',
                r'class \1(UnifiedBusinessAdapter',
                content
            )
    
    # 检查是否有变化
    changed = content != original_content
    
    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "导入已更新"
    else:
        return False, "无需更新（可能已使用UnifiedBusinessAdapter）"


def main():
    """主函数"""
    print("🔍 开始分析适配器文件...")
    print()
    
    # 查找文件
    adapter_files = find_adapter_files()
    print(f"找到 {len(adapter_files)} 个适配器文件")
    print()
    
    # 分析文件
    analyses = []
    for file_path in adapter_files:
        print(f"分析: {file_path.name}")
        analysis = analyze_adapter_file(file_path)
        analyses.append(analysis)
    
    print()
    
    # 生成报告
    report = generate_migration_report(analyses)
    print(report)
    
    # 保存报告
    report_file = PROJECT_ROOT / 'test_logs' / 'adapter_migration_analysis.txt'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 分析报告已保存到: {report_file.relative_to(PROJECT_ROOT)}")
    print()
    
    # 询问是否更新
    files_to_migrate = [a for a in analyses if a['migration_needed']]
    
    if files_to_migrate:
        print(f"发现 {len(files_to_migrate)} 个文件需要迁移")
        print("开始更新导入...")
        print()
        
        for analysis in files_to_migrate:
            file_path = PROJECT_ROOT / analysis['file']
            success, message = update_adapter_imports(file_path)
            
            if success:
                print(f"✅ {file_path.name}: {message}")
            else:
                print(f"ℹ️  {file_path.name}: {message}")
        
        print()
        print("✅ 迁移完成！")
    else:
        print("✅ 所有文件已使用UnifiedBusinessAdapter，无需迁移")


if __name__ == '__main__':
    main()

