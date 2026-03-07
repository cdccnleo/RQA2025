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
