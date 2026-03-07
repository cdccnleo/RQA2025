#!/usr/bin/env python3
"""
配置管理导出检查脚本
检查各个目录的__init__.py导出情况
"""

import os
import ast


def analyze_init_file(init_path):
    """分析__init__.py文件的导出"""
    if not os.path.exists(init_path):
        return {'exports': [], 'error': '文件不存在'}

    try:
        with open(init_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        # 查找__all__列表
        all_list = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            for item in node.value.elts:
                                if isinstance(item, ast.Str):
                                    all_list.append(item.s)

        return {'exports': all_list, 'error': None}
    except Exception as e:
        return {'exports': [], 'error': str(e)}


def check_directory_exports(config_dir):
    """检查配置目录的导出情况"""
    print('🔍 配置管理各目录导出情况检查')
    print('=' * 70)

    # 定义要检查的目录
    dirs_to_check = [
        ('core', '核心组件'),
        ('loaders', '加载器组件'),
        ('mergers', '合并器'),
        ('monitoring', '监控面板'),
        ('services', '服务组件'),
        ('storage', '存储模块'),
        ('version', '版本管理'),
        ('interfaces', '接口定义'),
        ('utils', '工具函数')
    ]

    total_issues = 0

    for dirname, desc in dirs_to_check:
        dir_path = os.path.join(config_dir, dirname)
        init_path = os.path.join(dir_path, '__init__.py')

        if os.path.exists(dir_path):
            print(f'\n📁 {desc} ({dirname})')
            print('-' * 50)

            # 分析__init__.py导出
            result = analyze_init_file(init_path)
            if result['error']:
                print(f'   ❌ __init__.py 错误: {result["error"]}')
                total_issues += 1
            else:
                print(f'   📄 __init__.py 导出: {len(result["exports"])} 项')
                if result['exports']:
                    print(f'   导出列表: {result["exports"][:3]}', end='')
                    if len(result['exports']) > 3:
                        print(f' ... 等{len(result["exports"])}项')
                    else:
                        print()

            # 检查实际文件
            actual_files = []
            for f in os.listdir(dir_path):
                if f.endswith('.py') and f != '__init__.py':
                    actual_files.append(f[:-3])

            if actual_files:
                print(f'   📂 实际文件数: {len(actual_files)}')
                print(f'   文件列表: {actual_files}')

                # 检查导出完整性
                if result['exports']:
                    missing = []
                    for file in actual_files:
                        # 检查文件是否在导出列表中
                        found = False
                        for export in result['exports']:
                            if file in export or export in file:
                                found = True
                                break
                        if not found:
                            missing.append(file)

                    if missing:
                        print(f'   ⚠️  可能未导出的文件: {missing}')
                        total_issues += len(missing)
                    else:
                        print('   ✅ 导出完整')
                else:
                    print('   ⚠️  无__all__导出列表')
                    total_issues += 1
            else:
                print('   📂 无其他Python文件')
        else:
            print(f'\n📁 {desc} ({dirname}) - ❌ 目录不存在')
            total_issues += 1

    print('\n' + '=' * 70)
    print(f'📊 检查总结:')
    print(f'   • 检查目录数: {len(dirs_to_check)}')
    print(f'   • 发现问题数: {total_issues}')

    if total_issues == 0:
        print('   ✅ 所有目录导出正常')
    else:
        print(f'   ⚠️  发现 {total_issues} 个导出问题需要处理')

    return total_issues


if __name__ == '__main__':
    config_dir = 'src/infrastructure/config'
    check_directory_exports(config_dir)
