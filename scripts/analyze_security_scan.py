#!/usr/bin/env python3
"""
RQA2025安全扫描结果分析脚本

分析bandit安全扫描结果，生成安全报告
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any


def analyze_security_scan(scan_file: str) -> Dict[str, Any]:
    """分析安全扫描结果"""
    with open(scan_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计基本信息
    total_files = len(set(error['filename'] for error in data.get('errors', [])))
    total_issues = len(data.get('results', []))

    print(f"📊 安全扫描分析结果")
    print(f"总文件数: {total_files}")
    print(f"安全问题数: {total_issues}")

    # 按严重程度统计
    severity_counts = Counter()
    confidence_counts = Counter()
    cwe_counts = Counter()
    test_counts = Counter()

    issues_by_file = defaultdict(list)
    issues_by_severity = defaultdict(list)

    for issue in data.get('results', []):
        severity = issue.get('issue_severity', 'UNKNOWN')
        confidence = issue.get('issue_confidence', 'UNKNOWN')
        test_id = issue.get('test_id', 'UNKNOWN')
        filename = issue.get('filename', 'UNKNOWN')

        severity_counts[severity] += 1
        confidence_counts[confidence] += 1
        test_counts[test_id] += 1

        if 'issue_cwe' in issue:
            cwe_id = issue['issue_cwe'].get('id', 'UNKNOWN')
            cwe_counts[cwe_id] += 1

        issues_by_file[filename].append(issue)
        issues_by_severity[severity].append(issue)

    # 输出统计结果
    print(f"\n🔥 安全问题严重程度分布:")
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count} 个")

    print(f"\n🎯 安全问题置信度分布:")
    for confidence, count in confidence_counts.items():
        print(f"  {confidence}: {count} 个")

    print(f"\n🚨 Top 10 安全问题类型:")
    for test_id, count in test_counts.most_common(10):
        print(f"  {test_id}: {count} 个")

    print(f"\n📁 受影响文件最多的Top 10:")
    file_issue_counts = [(file, len(issues)) for file, issues in issues_by_file.items()]
    file_issue_counts.sort(key=lambda x: x[1], reverse=True)

    for filename, count in file_issue_counts[:10]:
        print(f"  {filename}: {count} 个问题")

    # 识别高危问题
    high_priority_issues = []

    # 高严重程度问题
    for issue in issues_by_severity.get('HIGH', []):
        high_priority_issues.append({
            'type': 'high_severity',
            'issue': issue,
            'reason': f"高严重程度安全问题: {issue.get('test_name', 'UNKNOWN')}"
        })

    # SQL注入相关问题
    for issue in data.get('results', []):
        if 'sql' in issue.get('issue_text', '').lower() or 'injection' in issue.get('issue_text', '').lower():
            high_priority_issues.append({
                'type': 'sql_injection',
                'issue': issue,
                'reason': f"SQL注入风险: {issue.get('issue_text', '')}"
            })

    # 硬编码密码
    for issue in data.get('results', []):
        if 'password' in issue.get('issue_text', '').lower() and 'hardcoded' in issue.get('issue_text', '').lower():
            high_priority_issues.append({
                'type': 'hardcoded_password',
                'issue': issue,
                'reason': f"硬编码密码: {issue.get('issue_text', '')}"
            })

    print(f"\n🚨 高优先级安全问题 ({len(high_priority_issues)} 个):")
    for i, issue_info in enumerate(high_priority_issues[:10], 1):
        issue = issue_info['issue']
        print(
            f"{i}. {issue_info['type'].upper()}: {issue.get('filename', 'UNKNOWN')}:{issue.get('line_number', 'UNKNOWN')}")
        print(f"   {issue_info['reason']}")

    # 生成修复建议
    recommendations = []

    if severity_counts.get('HIGH', 0) > 0:
        recommendations.append("🔴 紧急修复高严重程度安全问题")

    if 'B102' in test_counts:  # exec used
        recommendations.append("🟡 替换exec()调用为更安全的方法")

    if 'B105' in test_counts:  # hardcoded password
        recommendations.append("🟡 移除硬编码密码，使用环境变量或配置管理")

    if 'B104' in test_counts:  # hardcoded bind all interfaces
        recommendations.append("🟡 生产环境避免绑定所有接口，使用具体IP")

    if 'B404' in test_counts:  # subprocess import
        recommendations.append("🟡 检查subprocess使用是否安全，避免shell注入")

    if total_issues > 50:
        recommendations.append("🟠 安全问题较多，建议分批修复，从高严重程度开始")

    print(f"\n💡 修复建议:")
    for rec in recommendations:
        print(f"  {rec}")

    # 生成报告摘要
    report = {
        'summary': {
            'total_files': total_files,
            'total_issues': total_issues,
            'severity_distribution': dict(severity_counts),
            'confidence_distribution': dict(confidence_counts),
            'top_test_types': dict(test_counts.most_common(5)),
            'most_affected_files': file_issue_counts[:5]
        },
        'high_priority_issues': high_priority_issues[:20],  # 前20个高优先级问题
        'recommendations': recommendations,
        'issues_by_severity': {
            severity: len(issues) for severity, issues in issues_by_severity.items()
        }
    }

    return report


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python analyze_security_scan.py <scan_results.json>")
        sys.exit(1)

    scan_file = sys.argv[1]
    if not Path(scan_file).exists():
        print(f"❌ 文件不存在: {scan_file}")
        sys.exit(1)

    try:
        report = analyze_security_scan(scan_file)

        # 保存详细报告
        output_file = Path(scan_file).stem + "_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📄 详细分析报告已保存到: {output_file}")

        # 输出安全评估结论
        total_issues = report['summary']['total_issues']
        high_severity = report['summary']['severity_distribution'].get('HIGH', 0)

        if high_severity > 0:
            print(f"\n❌ 安全评估: 存在 {high_severity} 个高严重程度问题，需要立即修复")
        elif total_issues > 100:
            print(f"\n⚠️ 安全评估: 安全问题较多 ({total_issues} 个)，建议逐步修复")
        elif total_issues > 50:
            print(f"\n🟡 安全评估: 存在 {total_issues} 个安全问题，可接受但需改进")
        else:
            print(f"\n✅ 安全评估: 安全状况良好 ({total_issues} 个问题)")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
