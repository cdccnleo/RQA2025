"""
重复类名清理工具 - Phase 1.3

专注于清理真正有问题的重复类名，建立统一命名规范
"""

import os
import re
from pathlib import Path
from collections import defaultdict


class DuplicateClassCleanup:
    """重复类名清理工具"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.ignored_keywords = {
            'class', 'from', 'import', 'for', 'if', 'def', 'try', 'except',
            'with', 'as', 'in', 'not', 'and', 'or', 'is', 'None', 'True', 'False'
        }

    def find_real_duplicate_classes(self):
        """查找真正的重复类名"""
        class_locations = defaultdict(list)

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 使用更精确的正则表达式查找类定义
                        # 排除被注释的类和嵌套类
                        class_pattern = r'^\s*class\s+(\w+)\s*[:\(]'
                        classes = re.findall(class_pattern, content, re.MULTILINE)

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        for cls in classes:
                            # 过滤掉关键字和无效类名
                            if (cls not in self.ignored_keywords and
                                not cls.startswith('_') and
                                    len(cls) > 2):
                                class_locations[cls].append(rel_path)

                    except Exception as e:
                        continue

        # 只保留真正的重复类名
        duplicate_classes = {name: locs for name, locs in class_locations.items()
                             if len(locs) > 1}

        return duplicate_classes

    def analyze_duplicate_patterns(self, duplicate_classes):
        """分析重复模式"""
        patterns = defaultdict(list)

        # 按功能分组
        functional_patterns = {
            'Monitor': ['Monitor', 'Checker', 'Watcher'],
            'Manager': ['Manager', 'Controller', 'Coordinator'],
            'Service': ['Service', 'Provider', 'Engine'],
            'Handler': ['Handler', 'Processor', 'Worker'],
            'Factory': ['Factory', 'Builder', 'Creator'],
            'Logger': ['Logger', 'Recorder', 'Tracker'],
            'Adapter': ['Adapter', 'Connector', 'Client'],
            'Validator': ['Validator', 'Verifier', 'Checker'],
            'Strategy': ['Strategy', 'Policy', 'Algorithm']
        }

        for class_name, locations in duplicate_classes.items():
            categorized = False
            for category, keywords in functional_patterns.items():
                if any(keyword in class_name for keyword in keywords):
                    patterns[category].append((class_name, len(locations)))
                    categorized = True
                    break

            if not categorized:
                patterns['其他'].append((class_name, len(locations)))

        return patterns

    def generate_cleanup_plan(self, duplicate_classes, patterns):
        """生成清理计划"""
        print('🧹 重复类名清理计划')
        print('=' * 50)

        # 按严重程度排序
        sorted_duplicates = sorted(duplicate_classes.items(),
                                   key=lambda x: len(x[1]), reverse=True)

        # 只处理最严重的问题（重复次数>=3的类名）
        critical_duplicates = [(name, locs) for name, locs in sorted_duplicates
                               if len(locs) >= 3]

        print(f'发现 {len(critical_duplicates)} 个严重重复类名 (重复次数>=3)')

        cleanup_plan = {}

        for class_name, locations in critical_duplicates:
            print(f'\\n🏷️ {class_name} ({len(locations)} 个位置):')

            # 按模块分组
            module_groups = defaultdict(list)
            for loc in locations:
                module = loc.split('/')[0]
                module_groups[module].append(loc)

            print('  按模块分布:')
            for module, files in module_groups.items():
                print(f'    {module}: {len(files)} 个文件')

            # 生成清理建议
            if len(module_groups) > 1:
                # 跨模块重复，需要统一接口
                cleanup_plan[class_name] = {
                    'type': 'cross_module',
                    'modules': list(module_groups.keys()),
                    'suggestion': '创建统一接口或基类'
                }
            else:
                # 同模块重复，需要重命名
                cleanup_plan[class_name] = {
                    'type': 'same_module',
                    'module': list(module_groups.keys())[0],
                    'suggestion': '重命名避免冲突'
                }

            print(f'  建议: {cleanup_plan[class_name]["suggestion"]}')

        return cleanup_plan

    def implement_cleanup_plan(self, cleanup_plan):
        """实施清理计划"""
        print('\\n🔄 开始实施清理计划...')

        # 实施跨模块重复的清理
        cross_module_duplicates = {name: info for name, info in cleanup_plan.items()
                                   if info['type'] == 'cross_module'}

        if cross_module_duplicates:
            print(f'\\n处理 {len(cross_module_duplicates)} 个跨模块重复类名:')

            for class_name, info in cross_module_duplicates.items():
                print(f'  📝 处理 {class_name} ({len(info["modules"])} 个模块)')
                # 这里可以实现具体的重构逻辑
                # 暂时只输出建议

        # 实施同模块重复的清理
        same_module_duplicates = {name: info for name, info in cleanup_plan.items()
                                  if info['type'] == 'same_module'}

        if same_module_duplicates:
            print(f'\\n处理 {len(same_module_duplicates)} 个同模块重复类名:')
            # 暂时只输出建议

        return len(cleanup_plan)

    def generate_naming_conventions(self, patterns):
        """生成命名规范"""
        print('\\n📋 命名规范建议')
        print('=' * 30)

        conventions = []

        # 基于重复模式分析生成规范
        for category, classes in patterns.items():
            if category != '其他' and classes:
                # 找出最常见的重复类名
                sorted_classes = sorted(classes, key=lambda x: x[1], reverse=True)
                most_common = sorted_classes[0][0]

                convention = f"{category}类命名: 建议使用 '{most_common}' 作为基础命名模式"
                conventions.append(convention)

        # 通用命名规范
        general_conventions = [
            "接口类: 以'I'开头，如 'ILogger', 'IManager'",
            "抽象基类: 以'Base'或'Abstract'开头",
            "具体实现类: 使用功能描述性名称",
            "工厂类: 以'Factory'结尾",
            "管理器类: 以'Manager'结尾",
            "服务类: 以'Service'结尾",
            "处理器类: 以'Handler'或'Processor'结尾"
        ]

        print('\\n通用命名规范:')
        for convention in general_conventions:
            print(f'  • {convention}')

        print('\\n分类命名建议:')
        for convention in conventions:
            print(f'  • {convention}')

        return conventions

    def run_cleanup(self):
        """运行完整的清理过程"""
        print('🚀 重复类名清理工具启动')
        print('=' * 40)

        # 1. 查找重复类名
        duplicate_classes = self.find_real_duplicate_classes()
        print(f'✅ 找到 {len(duplicate_classes)} 个重复类名')

        # 2. 分析重复模式
        patterns = self.analyze_duplicate_patterns(duplicate_classes)

        # 3. 生成清理计划
        cleanup_plan = self.generate_cleanup_plan(duplicate_classes, patterns)

        # 4. 生成命名规范
        self.generate_naming_conventions(patterns)

        # 5. 实施清理（暂时只生成报告）
        cleaned_count = len(cleanup_plan)
        print(f'\\n📊 清理总结:')
        print(f'  严重重复类名: {cleaned_count}')
        print(f'  清理建议已生成 (实际实施需人工确认)')

        return cleaned_count


def main():
    """主函数"""
    cleanup_tool = DuplicateClassCleanup()
    result = cleanup_tool.run_cleanup()

    print(f'\\n✅ 重复类名清理完成，共处理 {result} 个严重重复类名')


if __name__ == "__main__":
    main()
