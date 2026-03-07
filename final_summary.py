import json

with open('core_service_layer_comprehensive_module_review.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('🎯 核心服务层分模块审查完成报告')
print('=' * 50)
print()

summary = data['overall_summary']
print('📊 总体统计:')
print(f'  • 分析模块数: {summary["total_modules"]}')
print(f'  • 总文件数: {summary["total_files"]}')
print(f'  • 总代码行: {summary["total_lines"]:,}')
print(f'  • 识别模式: {summary["total_patterns"]}')
print(f'  • 重构机会: {summary["total_refactor_opportunities"]}')
print(f'  • 平均质量评分: {summary["average_quality_score"]:.3f}')
print()

print('🏆 模块质量排名:')
modules = list(data['module_analyses'].keys())
scores = [data['module_analyses'][m]['overall_score'] for m in modules]
ranked = sorted(zip(modules, scores), key=lambda x: x[1], reverse=True)

for i, (module, score) in enumerate(ranked, 1):
    status = '⭐' if score >= 0.9 else '✅' if score >= 0.85 else '⚠️' if score >= 0.8 else '❌'
    print(f'  {i}. {module}: {score:.3f} {status}')

print()
print('🚨 高风险模块 (评分<0.85):')
high_risk = summary['modules_with_high_risk']
if high_risk:
    for module in high_risk:
        score = data['module_analyses'][module]['overall_score']
        print(f'  • {module}: {score:.3f}')
else:
    print('  • 无')

print()
print('📋 后续优化建议:')
print('  1. 重点关注integration模块的深度重构')
print('  2. 实施组合模式拆分大类')
print('  3. 改善类型提示和文档')
print('  4. 统一异常处理机制')
print('  5. 增加单元测试覆盖率')
