"""
基础设施层资源管理系统代码组织分析脚本
"""

import re
from pathlib import Path


def analyze_resource_system():
    """分析资源管理系统代码组织"""

    resource_dir = Path('src/infrastructure/resource')
    files = list(resource_dir.glob('*.py'))

    print('基础设施层资源管理系统代码组织分析')
    print('=' * 60)

    # 统计文件信息
    file_info = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                classes = len(re.findall(r'^class \w+', content, re.MULTILINE))
                functions = len(re.findall(r'^def \w+', content, re.MULTILINE))
                imports = len(re.findall(r'^(from|import) ', content, re.MULTILINE))

                file_info.append({
                    'name': file_path.name,
                    'lines': len(lines),
                    'classes': classes,
                    'functions': functions,
                    'imports': imports
                })
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # 按行数排序
    file_info.sort(key=lambda x: x['lines'], reverse=True)

    print("总文件数:", len(files))
    print()
    print('文件统计 (按行数降序):')
    print('-' * 60)
    for info in file_info:
        print("{:<30} {:>4}行 {:>2}类 {:>2}函数 {:>2}导入".format(
            info['name'], info['lines'], info['classes'],
            info['functions'], info['imports']
        ))

    total_lines = sum(info['lines'] for info in file_info)
    total_classes = sum(info['classes'] for info in file_info)
    total_functions = sum(info['functions'] for info in file_info)

    print()
    print('汇总统计:')
    print('- 总代码行数:', total_lines)
    print('- 总类数:', total_classes)
    print('- 总函数数:', total_functions)
    print('- 平均每文件行数: {:.1f}'.format(total_lines/len(files)))

    # 分析文件职责分布
    print()
    print('文件职责分析:')
    print('-' * 60)

    # 按功能分组
    functional_groups = {
        '监控相关': ['business_metrics_monitor.py', 'monitor_components.py', 'monitoring_alert_system.py', 'monitoringservice.py', 'system_monitor.py', 'unified_monitor_adapter.py'],
        '资源管理': ['resource_manager.py', 'resource_components.py', 'resource_api.py', 'resource_optimization.py'],
        '资源池': ['pool_components.py', 'quota_components.py'],
        '调度': ['task_scheduler.py'],
        'GPU管理': ['gpu_manager.py'],
        '仪表板': ['resource_dashboard.py'],
        '装饰器': ['decorators.py'],
        '接口': ['interfaces.py'],
        '基础': ['base.py']
    }

    for group_name, group_files in functional_groups.items():
        group_lines = sum(info['lines'] for info in file_info if info['name'] in group_files)
        group_classes = sum(info['classes'] for info in file_info if info['name'] in group_files)
        print("{:<12} {:>4}行 {:>2}类 {:>1}文件".format(
            group_name, group_lines, group_classes, len(group_files)
        ))


if __name__ == '__main__':
    analyze_resource_system()
