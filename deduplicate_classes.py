#!/usr/bin/env python3
"""
基础设施层配置管理类定义去重工具
"""

import os


def deduplicate_classes():
    """执行类定义去重"""

    print('=== 🗑️ Phase 1.1: 实施类定义去重 ===')
    print()

    # 制定去重策略
    deduplication_plan = {
        'StrategyType': {'keep': 'interfaces/unified_interface.py', 'remove': 'core/strategy_base.py'},
        'ConfigSourceType': {'keep': 'interfaces/unified_interface.py', 'remove': 'core/strategy_base.py'},
        'ConfigFormat': {'keep': 'interfaces/unified_interface.py', 'remove': 'core/strategy_base.py'},
        'IConfigStrategy': {'keep': 'interfaces/unified_interface.py', 'remove': 'core/strategy_base.py'},
        'ConfigLoaderStrategy': {'keep': 'interfaces/unified_interface.py', 'remove': 'core/strategy_base.py'},

        'ServiceMeshType': {'keep': 'environment/cloud_native_configs.py', 'remove': 'environment/cloud_native_enhanced.py'},
        'CloudProvider': {'keep': 'environment/cloud_native_configs.py', 'remove': 'environment/cloud_native_enhanced.py'},
        'ScalingPolicy': {'keep': 'environment/cloud_native_configs.py', 'remove': 'environment/cloud_native_enhanced.py'},
        'ServiceMeshConfig': {'keep': 'environment/cloud_native_configs.py', 'remove': 'environment/cloud_native_enhanced.py'},
        'MultiCloudConfig': {'keep': 'environment/cloud_native_configs.py', 'remove': 'environment/cloud_native_enhanced.py'},
        'AutoScalingConfig': {'keep': 'environment/cloud_native_configs.py', 'remove': 'environment/cloud_native_enhanced.py'},
        'CloudNativeMonitoringConfig': {'keep': 'environment/cloud_native_configs.py', 'remove': 'environment/cloud_native_enhanced.py'},

        'ServiceMeshManager': {'keep': 'environment/cloud_service_mesh.py', 'remove': 'environment/cloud_native_enhanced.py'},

        'AlertManager': {'keep': 'monitoring/dashboard_alerts.py', 'remove': 'monitoring/performance_monitor_dashboard.py'},
        'InMemoryAlertManager': {'keep': 'monitoring/dashboard_alerts.py', 'remove': 'monitoring/performance_monitor_dashboard.py'},
        'MetricsCollector': {'keep': 'monitoring/dashboard_collectors.py', 'remove': 'monitoring/performance_monitor_dashboard.py'},
        'InMemoryMetricsCollector': {'keep': 'monitoring/dashboard_collectors.py', 'remove': 'monitoring/performance_monitor_dashboard.py'},
        'UnifiedMonitoringManager': {'keep': 'monitoring/dashboard_manager.py', 'remove': 'monitoring/performance_monitor_dashboard.py'},

        'UnifiedConfigManager': {'keep': 'core/config_manager_complete.py', 'remove': 'core/config_manager_core.py'},
    }

    print(f'📋 去重计划制定完成: 总共需要处理 {len(deduplication_plan)} 个重复类')
    print()

    # 开始执行去重
    config_dir = 'src/infrastructure/config'
    processed_count = 0

    for class_name, plan in deduplication_plan.items():
        print(f'🔄 处理 {class_name}:')
        keep_path = os.path.join(config_dir, plan['keep'])
        remove_path = os.path.join(config_dir, plan['remove'])

        print(f'   ✅ 保留: {plan["keep"]}')
        print(f'   🗑️  删除: {plan["remove"]} 中的定义')

        # 检查文件是否存在
        if not os.path.exists(keep_path):
            print(f'   ❌ 保留文件不存在: {keep_path}')
            continue
        if not os.path.exists(remove_path):
            print(f'   ❌ 删除文件不存在: {remove_path}')
            continue

        # 读取要删除的文件
        try:
            with open(remove_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            new_lines = []
            skip_lines = 0
            in_class = False
            class_found = False

            for i, line in enumerate(lines):
                stripped = line.strip()

                # 查找类定义开始
                if stripped.startswith(f'class {class_name}'):
                    print(f'   🔍 找到类定义在第{i+1}行，开始删除')
                    in_class = True
                    skip_lines = 1
                    class_found = True

                    # 计算类的缩进级别
                    indent_level = len(line) - len(line.lstrip())

                    # 继续查找类定义结束
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j]
                        next_stripped = next_line.strip()

                        if next_stripped == '':
                            skip_lines += 1
                            continue

                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent_level and next_stripped and not next_stripped.startswith('#'):
                            break

                        skip_lines += 1

                    print(f'   📏 需删除 {skip_lines} 行')
                    continue

                # 跳过类定义相关的行
                if skip_lines > 0:
                    skip_lines -= 1
                    continue

                new_lines.append(line)

            if class_found:
                # 写回文件
                new_content = '\n'.join(new_lines)
                with open(remove_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                print(f'   ✅ {class_name} 定义已从 {plan["remove"]} 中删除')
                processed_count += 1
            else:
                print(f'   ⚠️  未找到 {class_name} 的定义')

        except Exception as e:
            print(f'   ❌ 处理失败: {e}')

        print()

    print(f'🎯 类定义去重完成！成功处理 {processed_count}/{len(deduplication_plan)} 个重复类')

    return processed_count


if __name__ == '__main__':
    deduplicate_classes()
