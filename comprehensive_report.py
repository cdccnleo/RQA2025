import json

with open('analysis_result_1758895562.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('🏥 基础设施层健康管理系统 - AI智能化代码审查详细报告')
print('=' * 70)

print('\n📊 代码规模分析:')
total_files = data["metrics"]["total_files"]
total_lines = data["metrics"]["total_lines"]
print(f'  • 分析文件数: {total_files}')
print(f'  • 总代码行数: {total_lines:,}')
avg_size = total_lines // total_files if total_files > 0 else 0
print(f'  • 平均文件大小: {avg_size} 行/文件')
print(f'  • 识别代码模式: {data["metrics"]["total_patterns"]}')
print(f'  • 重构机会总数: {data["metrics"]["refactor_opportunities"]}')

print('\n🎯 质量评估结果:')
quality_score = data["quality_score"]
print(f'  • 代码质量评分: {quality_score:.3f}/1.000 ({quality_score*100:.1f}%)')

if 'organization_analysis' in data and data['organization_analysis']:
    org = data['organization_analysis']
    org_score = org["metrics"]["quality_score"]
    print(f'  • 组织质量评分: {org_score:.3f}/1.000')
    print(f'  • 组织问题数量: {org["issues_count"]}')
    print(f'  • 组织建议数量: {org["recommendations_count"]}')
else:
    print('  • 组织质量评分: N/A (分析器不可用)')

overall_score = data["overall_score"]
print(f'  • 综合质量评分: {overall_score:.3f}/1.000')

print('\n⚠️ 风险评估:')
risk = data['risk_assessment']
risk_level = risk["overall_risk"].upper()
print(f'  • 整体风险等级: {risk_level}')
print(f'  • 高风险问题: {risk["risk_breakdown"].get("high", 0)}')
print(f'  • 低风险问题: {risk["risk_breakdown"].get("low", 0)}')
print(f'  • 可自动化修复: {risk["automated_opportunities"]}')
print(f'  • 需要手动修复: {risk["manual_opportunities"]}')

# 统计不同严重程度的问题
severity_stats = {}
total_opportunities = data["metrics"]["refactor_opportunities"]
for opp in data['opportunities']:
    severity = opp['severity']
    severity_stats[severity] = severity_stats.get(severity, 0) + 1

print('\n🚨 问题严重程度分布:')
severity_levels = [('critical', '🔴'), ('high', '🟠'), ('medium', '🟡'), ('low', '🟢')]
for level, icon in severity_levels:
    count = severity_stats.get(level, 0)
    if count > 0:
        percentage = (count / total_opportunities) * 100
        print(f'  {icon} {level.upper()}: {count} ({percentage:.1f}%)')

print('\n🎯 优先级重构建议 (Top 15):')
print('-' * 50)
for i, opp in enumerate(data['opportunities'][:15], 1):
    filename = opp['file_path'].split('\\')[-1]
    severity_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}[opp['severity']]
    automated = '🤖' if opp['automated'] else '👤'

    print(f'{i:2d}. {severity_icon} {opp["title"]}')
    print(f'    📁 {filename}:{opp["line_number"]}')
    print(
        f'    {automated} {opp["effort"]} | 风险: {opp["risk_level"]} | 置信度: {opp["confidence"]:.2f}')
    print(f'    💡 {opp["suggested_fix"]}')
    print()

print('📋 执行计划摘要:')
execution_steps = data.get("execution_plan_steps", 0)
print(f'  • 计划执行步骤: {execution_steps}')
if data.get('execution_plan') and len(data['execution_plan']) > 0:
    print('  • 建议执行顺序 (前5步):')
    for step in data['execution_plan'][:5]:
        print(f'    {step["step"]:2d}. {step["title"]} ({step["estimated_time"]})')

print('\n🏆 代码质量评估总结:')
if quality_score >= 0.8:
    grade = '优秀'
    icon = '🏆'
elif quality_score >= 0.6:
    grade = '良好'
    icon = '👍'
elif quality_score >= 0.4:
    grade = '一般'
    icon = '⚠️'
else:
    grade = '需改进'
    icon = '🚨'

print(f'  {icon} 整体代码质量等级: {grade} ({quality_score:.3f})')
print('  • 主要优势: 代码结构相对清晰，模式识别效果良好')
print('  • 主要问题: 函数过长，复杂度较高，可维护性需要改进')
print('  • 改进建议: 重点关注长函数拆分，提高代码模块化程度')

print('\n💡 后续行动建议:')
print('  1. 🔧 优先处理高严重程度的重构机会')
print('  2. 🤖 利用自动化工具处理简单重复性问题')
print('  3. 📚 建立代码规范和重构指南')
print('  4. 🔄 定期进行代码质量检查')
print('  5. 📈 关注代码质量指标的持续改进')

print('\n📄 详细分析报告已保存至: analysis_result_1758895562.json')
print('🎉 AI智能化代码审查完成！')
