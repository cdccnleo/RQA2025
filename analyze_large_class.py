"""
分析大类结构并制定重构计划
"""

import re
from pathlib import Path


def analyze_class_structure(file_path, class_name):
    """分析类的结构"""

    print(f'🔍 分析类: {class_name}')
    print('=' * 60)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到类的开始位置
    class_pattern = rf'class {class_name}:'
    class_match = re.search(class_pattern, content)
    if not class_match:
        print(f'未找到类 {class_name}')
        return

    class_start = class_match.start()

    # 找到类的结束位置（下一个class或def在同一缩进级别）
    lines = content[class_start:].split('\n')

    # 分析类中的方法
    methods = []
    current_method = None
    method_lines = []
    indent_level = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 跳过空行和注释
        if not stripped or stripped.startswith('#'):
            if current_method:
                method_lines.append(line)
            continue

        # 计算缩进级别
        if stripped.startswith(('def ', 'class ')):
            line_indent = len(line) - len(line.lstrip())

            if current_method and indent_level is not None and line_indent <= indent_level:
                # 方法结束
                methods.append({
                    'name': current_method,
                    'lines': method_lines.copy(),
                    'line_count': len(method_lines)
                })
                current_method = None
                method_lines = []

            # 开始新方法或类
            if stripped.startswith('def '):
                current_method = stripped.split('(')[0].replace('def ', '')
                indent_level = line_indent
                method_lines = [line]
            elif stripped.startswith('class ') and current_method:
                # 遇到嵌套类，结束当前方法
                methods.append({
                    'name': current_method,
                    'lines': method_lines.copy(),
                    'line_count': len(method_lines)
                })
                current_method = None
                method_lines = []
        elif current_method:
            method_lines.append(line)

    # 处理最后一个方法
    if current_method and method_lines:
        methods.append({
            'name': current_method,
            'lines': method_lines,
            'line_count': len(method_lines)
        })

    print(f'📊 方法数量: {len(methods)}')

    # 按大小排序
    sorted_methods = sorted(methods, key=lambda x: x['line_count'], reverse=True)

    print('\n🏆 方法大小排名 (前10名):')
    for i, method in enumerate(sorted_methods[:10], 1):
        print(f'{i:2d}. {method["name"]:<35} {method["line_count"]:3d}行')

    # 分析方法职责分类
    categories = {
        'initialization': [],
        'metrics_management': [],
        'data_recording': [],
        'export_generation': [],
        'grafana_integration': [],
        'utility': []
    }

    for method in methods:
        name = method['name'].lower()

        if name.startswith('_init') or name in ['__init__']:
            categories['initialization'].append(method)
        elif 'metric' in name or 'prometheus' in name or 'gauge' in name or 'counter' in name:
            categories['metrics_management'].append(method)
        elif 'record' in name or 'update' in name or 'set' in name:
            categories['data_recording'].append(method)
        elif 'export' in name or 'generate' in name or 'render' in name:
            categories['export_generation'].append(method)
        elif 'grafana' in name or 'dashboard' in name:
            categories['grafana_integration'].append(method)
        else:
            categories['utility'].append(method)

    print('\n🏗️ 方法职责分类:')
    total_lines = 0
    for category, methods_in_cat in categories.items():
        if methods_in_cat:
            cat_lines = sum(m['line_count'] for m in methods_in_cat)
            total_lines += cat_lines
            print(f'  • {category}: {len(methods_in_cat):2d}个方法, {cat_lines:3d}行')

    print(f'\n📏 总结: {len(methods)}个方法, {total_lines}行代码')

    # 提出重构建议
    print('\n💡 重构建议:')

    if len(categories['initialization']) > 3:
        print('  • 初始化方法过多，建议提取MetricsInitializer类')

    if len(categories['metrics_management']) > 5:
        print('  • 指标管理方法过多，建议提取MetricsManager类')

    if len(categories['grafana_integration']) > 3:
        print('  • Grafana集成方法过多，建议提取GrafanaManager类')

    if len(categories['data_recording']) > 5:
        print('  • 数据记录方法过多，建议提取DataRecorder类')

    print('\n🎯 建议的拆分方案:')
    print('  1. MetricsInitializer - 处理所有初始化逻辑')
    print('  2. MetricsManager - 处理指标创建和管理')
    print('  3. DataRecorder - 处理数据记录和更新')
    print('  4. GrafanaManager - 处理Grafana集成')
    print('  5. ExportManager - 处理数据导出')
    print('  6. PrometheusExporter - 协调器类，组合上述组件')

    return methods, categories


def create_refactor_plan():
    """创建重构计划"""

    print('\n📋 大类重构实施计划')
    print('=' * 80)

    # 分析所有大类
    large_classes = [
        ('src/infrastructure/health/integration/prometheus_exporter.py', 'HealthCheckPrometheusExporter'),
        ('src/infrastructure/health/ml/inference_engine.py', 'AsyncInferenceEngine'),
        ('src/infrastructure/health/monitoring/automation_monitor.py', 'AutomationMonitor'),
        ('src/infrastructure/health/integration/prometheus_integration.py', 'PrometheusIntegration'),
        ('src/infrastructure/health/database/database_health_monitor.py', 'DatabaseHealthMonitor'),
        ('src/infrastructure/health/monitoring/application_monitor_metrics.py',
         'ApplicationMonitorMetricsMixin'),
        ('src/infrastructure/health/monitoring/performance_monitor.py', 'PerformanceMonitor'),
        ('src/infrastructure/health/monitoring/system_metrics_collector.py', 'SystemMetricsCollector')
    ]

    plans = []

    for file_path, class_name in large_classes:
        if Path(file_path).exists():
            print(f'\n🔧 分析 {class_name}:')
            try:
                methods, categories = analyze_class_structure(file_path, class_name)
                plan = {
                    'class': class_name,
                    'file': file_path,
                    'methods': len(methods),
                    'categories': {k: len(v) for k, v in categories.items()},
                    'priority': 'high' if len(methods) > 20 else 'medium'
                }
                plans.append(plan)
            except Exception as e:
                print(f'  ❌ 分析失败: {e}')
        else:
            print(f'  ⚠️ 文件不存在: {file_path}')

    # 生成优先级排序的重构计划
    print('\n📅 重构优先级计划:')
    print('=' * 50)

    high_priority = [p for p in plans if p['priority'] == 'high']
    medium_priority = [p for p in plans if p['priority'] == 'medium']

    print('🚨 高优先级 (方法数>20):')
    for i, plan in enumerate(high_priority, 1):
        print(f'  {i}. {plan["class"]} ({plan["methods"]}个方法)')

    print('\n🟡 中优先级 (方法数15-20):')
    for i, plan in enumerate(medium_priority, 1):
        print(f'  {i}. {plan["class"]} ({plan["methods"]}个方法)')

    print('\n⏰ 实施时间表:')
    print('第1周: 重构HealthCheckPrometheusExporter (最高优先级)')
    print('第2周: 重构AsyncInferenceEngine和AutomationMonitor')
    print('第3周: 重构剩余大类，完成代码重构')
    print('第4周: 系统集成测试和性能优化')

    print('\n🎯 成功指标:')
    print('• 所有大类行数控制在300行以内')
    print('• 每个类的方法数不超过15个')
    print('• 保持原有API接口兼容性')
    print('• 提升代码可维护性和可测试性')

    return plans


if __name__ == '__main__':
    plans = create_refactor_plan()
