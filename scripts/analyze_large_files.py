#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 大文件分析脚本

分析项目中的大文件，为代码重构做准备
"""

import os
import sys
from pathlib import Path
from datetime import datetime


def analyze_large_files():
    """分析项目中的大文件"""
    print("🔍 RQA2025 大文件分析")
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    # 定义大文件的阈值
    LARGE_FILE_THRESHOLD = 1000  # 1000行

    print(f"分析阈值: {LARGE_FILE_THRESHOLD} 行")
    print()

    large_files = []

    # 遍历所有Python文件
    for py_file in project_root.rglob("*.py"):
        # 排除venv目录和__pycache__目录
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                line_count = len(lines)

                if line_count >= LARGE_FILE_THRESHOLD:
                    large_files.append({
                        'file_path': py_file,
                        'line_count': line_count,
                        'relative_path': py_file.relative_to(project_root)
                    })

        except Exception as e:
            print(f"分析文件 {py_file} 时出错: {e}")

    # 按行数排序
    large_files.sort(key=lambda x: x['line_count'], reverse=True)

    print(f"📊 发现 {len(large_files)} 个大文件 (>={LARGE_FILE_THRESHOLD}行)")
    print("-" * 60)

    # 显示大文件列表
    for i, file_info in enumerate(large_files, 1):
        print("2d")
    print("-" * 60)

    # 统计分析
    print("📈 统计分析:")
    print(f"  总大文件数: {len(large_files)}")

    if large_files:
        avg_lines = sum(f['line_count'] for f in large_files) / len(large_files)
        max_lines = max(f['line_count'] for f in large_files)
        min_lines = min(f['line_count'] for f in large_files)

        print(".1f" print(f"  最大文件行数: {max_lines}")
        print(f"  最小文件行数: {min_lines}")

        # 按目录分组统计
        dir_stats={}
        for file_info in large_files:
            dir_name=str(file_info['relative_path']).split('/')[0].split('\\')[0]
            if dir_name not in dir_stats:
                dir_stats[dir_name]=[]
            dir_stats[dir_name].append(file_info['line_count'])

        print("
📁 按目录分组: " for dir_name, line_counts in sorted(dir_stats.items()):
            count=len(line_counts)
            avg_lines=sum(line_counts) / count
            print("2d")

    # 生成重构建议
    print("
💡 重构建议: "    print(" -" * 30)

    if len(large_files) == 0:
        print("✅ 没有发现大文件，代码结构良好!")
        return True

    # 优先级排序
    priority_files=[]
    for file_info in large_files:
        priority_score=file_info['line_count']
        if 'mobile' in str(file_info['relative_path']).lower():
            priority_score += 2000  # 移动端文件优先级更高
        if 'trading' in str(file_info['relative_path']).lower():
            priority_score += 1000  # 交易相关文件优先级更高

        priority_files.append({
            'file_info': file_info,
            'priority_score': priority_score
        })

    priority_files.sort(key=lambda x: x['priority_score'], reverse=True)

    print("🔥 建议优先重构的文件:")
    for i, item in enumerate(priority_files[:5], 1):
        file_info=item['file_info']
        print("d"
    # 生成详细的重构计划
    print("
📋 详细重构计划: "    print(" -" * 30)

    for i, item in enumerate(priority_files, 1):
        file_info=item['file_info']
        file_path=file_info['relative_path']

        print(f"\n{i}. {file_path}")
        print(f"   行数: {file_info['line_count']}")

        # 根据文件类型和位置给出具体建议
        if 'mobile' in str(file_path).lower():
            print("   建议: 拆分为UI组件、业务逻辑、数据管理三个模块")
        elif 'trading' in str(file_path).lower():
            print("   建议: 按功能拆分为订单处理、风险控制、策略执行模块")
        elif 'scheduler' in str(file_path).lower():
            print("   建议: 拆分为任务调度器、执行引擎、监控组件")
        elif 'manager' in str(file_path).lower():
            print("   建议: 按职责分离为配置管理器、状态管理器、协调器")
        elif 'engine' in str(file_path).lower():
            print("   建议: 按层次拆分为接口层、业务层、数据访问层")
        else:
            print("   建议: 按单一职责原则拆分为多个功能模块")

    # 保存分析报告
    report={
        "analysis_timestamp": datetime.now().isoformat(),
        "large_file_threshold": LARGE_FILE_THRESHOLD,
        "total_large_files": len(large_files),
        "files": [
            {
                "path": str(f['relative_path']),
                "lines": f['line_count']
            } for f in large_files
        ],
        "priority_files": [
            {
                "path": str(item['file_info']['relative_path']),
                "lines": item['file_info']['line_count'],
                "priority_score": item['priority_score']
            } for item in priority_files[:5]
        ]
    }

    import json
    report_file=f"large_files_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📁 详细分析报告已保存: {report_file}")

    return len(large_files) > 0

if __name__ == "__main__":
    needs_refactoring=analyze_large_files()

    print("
🎯 分析总结: " if needs_refactoring:
        print("⚠️  发现大文件需要重构")
        print("📋 请按照上述建议逐步进行代码重构")
        print("🚀 重构完成后将显著提升代码可维护性")
    else:
        print("✅ 代码结构良好，无需大规模重构")

    exit(0 if not needs_refactoring else 1)
