#!/usr/bin/env python3
"""
核心服务层代码分析器 - 简化独立版本

专门用于核心服务层的代码审查和重复检测

创建时间: 2025-11-03
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib

PROJECT_ROOT = Path(r'C:\PythonProject\RQA2025')
TARGET_PATH = PROJECT_ROOT / 'src' / 'core'


def analyze_core_layer():
    """分析核心服务层"""
    print("🤖 AI智能化代码分析器 - 核心服务层审查")
    print("=" * 70)
    print()
    
    py_files = [f for f in TARGET_PATH.rglob('*.py') if '__pycache__' not in str(f)]
    
    print(f"📊 扫描代码文件...")
    print(f"   找到 {len(py_files)} 个Python文件")
    
    # 基础统计
    total_lines = 0
    by_module = defaultdict(lambda: {'files': 0, 'lines': 0})
    large_files = []
    functions = []
    classes = []
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
            
            total_lines += len(lines)
            rel_path = py_file.relative_to(TARGET_PATH)
            
            # 按模块统计
            if len(rel_path.parts) > 0:
                module = rel_path.parts[0]
                by_module[module]['files'] += 1
                by_module[module]['lines'] += len(lines)
            
            # 大文件
            if len(lines) > 1000:
                large_files.append({
                    'file': str(rel_path),
                    'lines': len(lines)
                })
            
            # 提取函数和类
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('def '):
                    functions.append({
                        'file': str(rel_path),
                        'line': i + 1,
                        'name': stripped.split('(')[0].replace('def ', '')
                    })
                elif stripped.startswith('class '):
                    classes.append({
                        'file': str(rel_path),
                        'line': i + 1,
                        'name': stripped.split('(')[0].split(':')[0].replace('class ', '')
                    })
        
        except Exception as e:
            continue
    
    print(f"✅ 基础扫描完成")
    print(f"   总行数: {total_lines:,}")
    print(f"   平均文件大小: {total_lines//len(py_files)} 行")
    print(f"   函数数: {len(functions):,}")
    print(f"   类数: {len(classes):,}")
    print()
    
    # 检测重复模式
    print(f"🔍 检测代码重复模式...")
    
    duplicates = detect_duplicate_patterns(py_files)
    
    print(f"✅ 重复检测完成")
    print(f"   发现 {len(duplicates)} 组重复模式")
    print()
    
    # 生成报告
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'analyzer': 'AI智能化代码分析器 v2.0',
            'target_path': 'src/core',
            'analysis_type': '深度代码审查 - 重点检查代码重复'
        },
        'statistics': {
            'total_files': len(py_files),
            'total_lines': total_lines,
            'avg_file_size': total_lines // len(py_files),
            'large_files_count': len(large_files),
            'functions_count': len(functions),
            'classes_count': len(classes)
        },
        'module_distribution': {
            k: v for k, v in sorted(
                by_module.items(),
                key=lambda x: x[1]['lines'],
                reverse=True
            )
        },
        'large_files': sorted(large_files, key=lambda x: x['lines'], reverse=True),
        'duplicate_patterns': duplicates,
        'quality_assessment': assess_quality(total_lines, len(py_files), duplicates),
        'recommendations': generate_recommendations(duplicates, large_files)
    }
    
    return report


def detect_duplicate_patterns(py_files):
    """检测重复模式"""
    patterns = []
    
    # 检查文件名模式
    name_groups = defaultdict(list)
    for f in py_files:
        base_name = f.stem
        # 移除常见后缀
        pattern = base_name
        for suffix in ['_components', '_adapter', '_manager', '_service', '_interface']:
            pattern = pattern.replace(suffix, '')
        
        if pattern and pattern != '__init__':
            name_groups[pattern].append(str(f.relative_to(PROJECT_ROOT / 'src' / 'core')))
    
    # 找出重复的文件名模式
    for pattern, files in name_groups.items():
        if len(files) > 1:
            patterns.append({
                'type': '文件名模式重复',
                'pattern': pattern,
                'count': len(files),
                'files': files
            })
    
    # 检查ComponentFactory重复
    factory_files = []
    for f in py_files:
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            if 'class ComponentFactory:' in content:
                # 排除合法的定义（base_component.py）
                if 'base_component.py' not in str(f):
                    factory_files.append(str(f.relative_to(PROJECT_ROOT / 'src' / 'core')))
        except:
            continue
    
    if factory_files:
        patterns.append({
            'type': 'ComponentFactory类重复',
            'pattern': 'ComponentFactory',
            'count': len(factory_files),
            'files': factory_files,
            'estimated_duplicate_lines': len(factory_files) * 30
        })
    
    return patterns


def assess_quality(total_lines, total_files, duplicates):
    """评估质量"""
    # 计算重复率
    duplicate_lines = sum(
        d.get('estimated_duplicate_lines', 0)
        for d in duplicates
    )
    
    duplication_rate = (duplicate_lines / total_lines * 100) if total_lines > 0 else 0
    
    # 质量评分
    if duplication_rate < 1:
        quality_score = 9.5
    elif duplication_rate < 3:
        quality_score = 8.0
    elif duplication_rate < 5:
        quality_score = 7.0
    else:
        quality_score = 6.0
    
    return {
        'estimated_duplicate_lines': duplicate_lines,
        'duplication_rate': round(duplication_rate, 2),
        'quality_score': quality_score,
        'grade': 'A' if quality_score >= 9 else 'B' if quality_score >= 8 else 'C'
    }


def generate_recommendations(duplicates, large_files):
    """生成建议"""
    recommendations = []
    
    if duplicates:
        for dup in duplicates:
            recommendations.append({
                'priority': 'P0' if dup['count'] > 5 else 'P1',
                'title': f"消除{dup['type']}",
                'description': f"发现{dup['count']}处{dup['pattern']}重复",
                'impact': '代码重复',
                'solution': '使用统一基类或重构为单一实现'
            })
    
    if large_files:
        for lf in large_files[:5]:
            recommendations.append({
                'priority': 'P1',
                'title': f"拆分超大文件",
                'description': f"{lf['file']} ({lf['lines']}行)",
                'impact': '可维护性',
                'solution': '按功能拆分为多个小文件'
            })
    
    return recommendations


def main():
    """主函数"""
    report = analyze_core_layer()
    
    # 保存JSON报告
    output_file = PROJECT_ROOT / 'test_logs' / '核心服务层AI代码审查报告.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 显示摘要
    print("=" * 70)
    print("📊 核心服务层代码审查报告")
    print("=" * 70)
    print()
    print(f"分析文件: {report['statistics']['total_files']} 个")
    print(f"总代码行: {report['statistics']['total_lines']:,} 行")
    print(f"平均文件大小: {report['statistics']['avg_file_size']} 行")
    print(f"函数数量: {report['statistics']['functions_count']:,}")
    print(f"类数量: {report['statistics']['classes_count']:,}")
    print()
    
    print(f"📈 质量评估:")
    qa = report['quality_assessment']
    print(f"  代码重复率: {qa['duplication_rate']}%")
    print(f"  质量评分: {qa['quality_score']}/10")
    print(f"  质量等级: {qa['grade']}")
    print()
    
    if report['duplicate_patterns']:
        print(f"🔍 发现代码重复:")
        for dup in report['duplicate_patterns']:
            print(f"  • {dup['type']}: {dup['count']} 处")
            if 'estimated_duplicate_lines' in dup:
                print(f"    预估重复行数: {dup['estimated_duplicate_lines']}")
    else:
        print("✅ 未发现明显代码重复")
    print()
    
    if report['large_files']:
        print(f"⚠️  超大文件 ({len(report['large_files'])} 个):")
        for lf in report['large_files'][:5]:
            print(f"  • {lf['file']}: {lf['lines']} 行")
    print()
    
    print(f"💡 建议事项: {len(report['recommendations'])} 项")
    for rec in report['recommendations'][:5]:
        print(f"  {rec['priority']}: {rec['title']}")
    print()
    
    print(f"📄 详细报告已保存到: {output_file.relative_to(PROJECT_ROOT)}")
    print()
    print("=" * 70)
    print("✅ AI代码审查完成")
    print("=" * 70)


if __name__ == '__main__':
    main()

















