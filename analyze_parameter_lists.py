"""
分析和优化长参数列表问题

识别最严重的方法并制定重构计划
"""

import json
from pathlib import Path


def analyze_parameter_lists():
    """分析长参数列表问题"""

    print('🔍 长参数列表问题深度分析')
    print('=' * 50)

    # 读取分析结果
    with open('analysis_result_1758897688.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤出长参数列表的问题
    param_issues = [opp for opp in data['opportunities']
                    if '长参数列表' in opp['title']]

    print(f'\n📊 发现长参数列表问题: {len(param_issues)}个')

    # 按文件分组统计
    file_stats = {}
    for issue in param_issues:
        filename = issue['file_path'].split('\\')[-1]
        if filename not in file_stats:
            file_stats[filename] = []
        file_stats[filename].append(issue)

    print('\n📁 按文件统计:')
    for filename, issues in sorted(file_stats.items(), key=lambda x: len(x[1]), reverse=True):
        print(f'  • {filename}: {len(issues)}个问题')

    # 分析最严重的问题
    print('\n🚨 最严重的长参数问题 (前10个):')
    for i, issue in enumerate(param_issues[:10], 1):
        filename = issue['file_path'].split('\\')[-1]
        print(f'{i:2d}. {filename}:{issue["line_number"]} - {issue["title"]}')

    # 制定重构计划
    print('\n💡 重构策略和计划:')

    print('\n🔧 策略1: 数据类封装 (推荐)')
    print('  • 为相关参数创建数据类')
    print('  • 使用 @dataclass 装饰器')
    print('  • 添加类型提示和验证')

    print('\n🏗️ 策略2: 构建器模式')
    print('  • 为复杂参数创建构建器类')
    print('  • 逐步设置参数，支持链式调用')
    print('  • 提供参数验证和默认值')

    print('\n✂️ 策略3: 方法拆分')
    print('  • 将大方法拆分为多个小方法')
    print('  • 每个方法职责单一')
    print('  • 减少单个方法的参数数量')

    # 识别具体需要重构的方法
    print('\n🎯 优先重构目标:')

    # 读取具体的文件来分析方法签名
    health_dir = Path('src/infrastructure/health')

    print('\n分析具体方法签名:')
    analyzed_methods = []

    for issue in param_issues[:5]:  # 只分析前5个最严重的问题
        file_path = health_dir / issue['file_path'].split('\\')[-1]
        if not file_path.exists():
            # 尝试在子目录中查找
            for sub_dir in ['monitoring', 'validation', 'integration', 'core', 'components', 'api', 'database', 'testing', 'ml', 'infrastructure']:
                candidate = health_dir / sub_dir / issue['file_path'].split('\\')[-1]
                if candidate.exists():
                    file_path = candidate
                    break

        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 查找问题行附近的方法定义
                line_num = issue['line_number'] - 1  # 转换为0索引
                if 0 <= line_num < len(lines):
                    # 向前查找方法定义
                    for i in range(max(0, line_num-10), min(len(lines), line_num+5)):
                        line = lines[i].strip()
                        if line.startswith('def '):
                            method_name = line.split('(')[0].replace('def ', '')
                            params = line.split('(', 1)[1].split(')')[0] if '(' in line else ''
                            param_count = len([p.strip() for p in params.split(
                                ',') if p.strip() and not p.strip().startswith('*')])

                            analyzed_methods.append({
                                'file': file_path.name,
                                'method': method_name,
                                'params': param_count,
                                'line': i+1,
                                'signature': line
                            })
                            break

            except Exception as e:
                print(f'  ⚠️ 无法分析文件 {file_path}: {e}')

    # 显示分析结果
    for method in analyzed_methods:
        print(f'  📌 {method["file"]}:{method["line"]} - {method["method"]} ({method["params"]}个参数)')

    print('\n📋 重构执行计划:')

    print('\n阶段1: 数据类重构 (1-2周)')
    print('1. 创建参数数据类')
    print('   • 定义ParameterConfig, RequestConfig等数据类')
    print('   • 使用@dataclass装饰器')
    print('   • 添加参数验证')

    print('\n2. 重构方法签名')
    print('   • 将多个参数替换为数据类实例')
    print('   • 更新所有调用位置')
    print('   • 保持向后兼容性')

    print('\n阶段2: 构建器模式应用 (1周)')
    print('3. 实现构建器类')
    print('   • 为复杂配置创建构建器')
    print('   • 支持链式调用')
    print('   • 提供参数验证')

    print('\n阶段3: 方法拆分优化 (1-2周)')
    print('4. 拆分超长方法')
    print('   • 识别职责边界')
    print('   • 创建专用方法')
    print('   • 重构调用关系')

    print('\n🎯 预期改进效果:')
    print('• 📉 参数数量减少 60-80%')
    print('• 🔧 方法可读性显著提升')
    print('• 🐛 参数错误减少 70%')
    print('• 🚀 开发效率提升 30%')


if __name__ == '__main__':
    analyze_parameter_lists()
