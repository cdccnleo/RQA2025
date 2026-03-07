import json

# 读取分析结果
with open('analysis_results.json', encoding='utf-8') as f:
    data = json.load(f)

# 打印总体统计
print("=== AI智能化代码分析结果 ===")
print(f"总体评分: {data['overall_score']:.3f}")
print(f"质量评分: {data['quality_score']:.3f}")
print(f"风险等级: {data['risk_assessment']['overall_risk']}")
print(f"总文件数: {data['metrics']['total_files']}")
print(f"总代码行: {data['metrics']['total_lines']}")
print(f"识别模式: {data['metrics']['total_patterns']}")
print(f"重构机会: {data['metrics']['refactor_opportunities']}")
print(f"自动化机会: {data['risk_assessment']['automated_opportunities']}")
print(f"手动机会: {data['risk_assessment']['manual_opportunities']}")

# 风险分解
print("\n=== 风险分解 ===")
for risk_level, count in data['risk_assessment']['risk_breakdown'].items():
    print(f"{risk_level}: {count}")

# 严重性分解
print("\n=== 严重性分解 ===")
for severity, count in data['risk_assessment']['severity_breakdown'].items():
    print(f"{severity}: {count}")

# 高严重性机会
high_severity = [opp for opp in data['opportunities'] if opp['severity'] == 'high']
print(f"\n=== 高严重性机会 ===")
print(f"高严重性机会数量: {len(high_severity)}")

print("\n前20个高严重性机会:")
for i, opp in enumerate(high_severity[:20]):
    print(f"{i+1}. {opp['title']}")
    print(f"   文件: {opp['file_path']}:{opp['line_number']}")
    print(f"   复杂度: {opp['code_snippet']}")
    print(f"   建议: {opp['suggested_fix']}")
    print()

# 中等严重性机会
medium_severity = [opp for opp in data['opportunities'] if opp['severity'] == 'medium']
print(f"\n=== 中等严重性机会 ===")
print(f"中等严重性机会数量: {len(medium_severity)}")

print("\n前20个中等严重性机会:")
for i, opp in enumerate(medium_severity[:20]):
    print(f"{i+1}. {opp['title']}")
    print(f"   文件: {opp['file_path']}:{opp['line_number']}")
    print(f"   复杂度: {opp['code_snippet']}")
    print(f"   建议: {opp['suggested_fix']}")
    print()

# 组织结构分析
print("\n=== 组织结构分析 ===")
org_analysis = data['organization_analysis']
print(f"组织质量评分: {org_analysis['metrics']['quality_score']:.3f}")
print(f"总文件数: {org_analysis['metrics']['total_files']}")
print(f"总代码行: {org_analysis['metrics']['total_lines']}")
print(f"平均文件大小: {org_analysis['metrics']['avg_file_size']:.1f}")
print(f"最大文件大小: {org_analysis['metrics']['max_file_size']}")
print(f"最大文件: {org_analysis['metrics']['largest_file']}")
print(f"问题数量: {org_analysis['issues_count']}")
print(f"建议数量: {org_analysis['recommendations_count']}")

# 文档同步检查
print("\n=== 文档同步检查 ===")
doc_sync = data.get('documentation_sync')
if doc_sync:
    print(f"文档同步成功: {doc_sync['success']}")
    if 'issues_found' in doc_sync:
        print(f"发现问题: {len(doc_sync['issues_found'])}")
        if doc_sync['issues_found']:
            print("前5个文档问题:")
            for i, issue in enumerate(doc_sync['issues_found'][:5]):
                print(f"  {i+1}. {issue.get('description', 'N/A')} ({issue.get('severity', 'N/A')})")
else:
    print("未执行文档同步检查")

# 按文件类型统计机会
print("\n=== 按文件类型统计机会 ===")
file_types = {}
for opp in data['opportunities']:
    file_path = opp['file_path']
    if '.' in file_path:
        ext = file_path.split('.')[-1]
        if ext not in file_types:
            file_types[ext] = 0
        file_types[ext] += 1

print("前10个文件类型的机会分布:")
sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)
for ext, count in sorted_types[:10]:
    print(f"  {ext}: {count}")

# 按目录统计机会
print("\n=== 按目录统计机会 ===")
directories = {}
for opp in data['opportunities']:
    file_path = opp['file_path']
    if '\\' in file_path:
        dir_name = file_path.split('\\')[1]  # 获取第一级目录
        if dir_name not in directories:
            directories[dir_name] = 0
        directories[dir_name] += 1

print("前10个目录的机会分布:")
sorted_dirs = sorted(directories.items(), key=lambda x: x[1], reverse=True)
for dir_name, count in sorted_dirs[:10]:
    print(f"  {dir_name}: {count}")
