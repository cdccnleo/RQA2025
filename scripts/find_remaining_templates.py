#!/usr/bin/env python3
"""
查找剩余的模板文件

搜索项目中剩余的handler_*.py、processor_*.py、strategy_*.py等文件
"""

import os
import re
from pathlib import Path
from collections import defaultdict


class TemplateFileFinder:
    """模板文件查找器"""

    def __init__(self):
        self.templates_found = defaultdict(list)
        self.patterns = {
            'handler_templates': r'handler_\d+\.py$',
            'processor_templates': r'processor_\d+\.py$',
            'strategy_templates': r'strategy_\d+\.py$',
            'cache_templates': r'cache_\d+\.py$',
            'service_templates': r'service_\d+\.py$',
            'client_templates': r'client_\d+\.py$',
            'optimizer_templates': r'optimizer_\d+\.py$',
            'manager_templates': r'manager_\d+\.py$',
            'adapter_templates': r'adapter_\d+\.py$'
        }

    def find_template_files(self):
        """查找所有模板文件"""
        print("🔍 查找剩余的模板文件...")
        print("="*60)

        # 排除目录
        exclude_dirs = {
            '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            'backup', 'backups', 'temp', 'tmp', 'build', 'dist',
            'test', 'tests', 'testing'
        }

        total_files = 0

        for root, dirs, files in os.walk('.'):
            # 移除需要排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    self._check_file_pattern(file_path)
                    total_files += 1

                    if total_files % 1000 == 0:
                        print(f"   已扫描 {total_files} 个文件...")

        return self.templates_found

    def _check_file_pattern(self, file_path: Path):
        """检查文件是否匹配模板模式"""
        filename = file_path.name

        for pattern_name, pattern in self.patterns.items():
            if re.match(pattern, filename):
                try:
                    size_kb = file_path.stat().st_size / 1024
                    self.templates_found[pattern_name].append({
                        'path': str(file_path),
                        'size_kb': round(size_kb, 1),
                        'name': filename
                    })
                except Exception as e:
                    print(f"   ⚠️  处理文件失败 {file_path}: {e}")

    def generate_report(self):
        """生成报告"""
        print("\n📊 剩余模板文件报告:")
        print("="*60)

        total_files = 0
        total_size = 0

        # 按文件数量排序
        sorted_templates = sorted(
            self.templates_found.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for pattern_name, files in sorted_templates:
            if files:
                file_count = len(files)
                pattern_size = sum(f['size_kb'] for f in files)
                total_files += file_count
                total_size += pattern_size

                print(f"\n📁 {pattern_name}: {file_count}个文件 ({pattern_size:.1f} KB)")

                # 显示文件位置分布
                locations = defaultdict(int)
                for file_info in files:
                    dir_path = Path(file_info['path']).parent
                    locations[str(dir_path)] += 1

                for location, count in sorted(locations.items()):
                    print(f"   📂 {location}: {count}个文件")

                # 显示前几个文件作为示例
                for i, file_info in enumerate(files[:3]):
                    print(f"      - {file_info['name']} ({file_info['size_kb']} KB)")
                if file_count > 3:
                    print(f"      ... 还有 {file_count - 3} 个文件")

        print(f"\n🎯 总计发现: {total_files}个模板文件 ({total_size:.1f} KB)")
        print(f"   📈 优化潜力: {total_size * 0.9:.1f} KB (90%减少)")
        # 优化优先级建议
        print("\n🚨 优化优先级建议:")
        print("="*60)

        high_priority = []
        medium_priority = []
        low_priority = []

        for pattern_name, files in sorted_templates:
            count = len(files)
            if count >= 20:
                high_priority.append((pattern_name, count, sum(f['size_kb'] for f in files)))
            elif count >= 10:
                medium_priority.append((pattern_name, count, sum(f['size_kb'] for f in files)))
            else:
                low_priority.append((pattern_name, count, sum(f['size_kb'] for f in files)))

        if high_priority:
            print("🚨 高优先级 (立即处理):")
            for pattern_name, count, size in high_priority:
                print(f"   - {pattern_name}: {count}个文件 ({size:.1f} KB)")
        if medium_priority:
            print("\n📋 中优先级 (近期处理):")
            for pattern_name, count, size in medium_priority:
                print(f"   - {pattern_name}: {count}个文件 ({size:.1f} KB)")
        if low_priority:
            print("\n🔄 低优先级 (长期规划):")
            for pattern_name, count, size in low_priority:
                print(f"   - {pattern_name}: {count}个文件 ({size:.1f} KB)")
        return {
            'total_files': total_files,
            'total_size_kb': total_size,
            'templates': dict(self.templates_found)
        }


def main():
    """主函数"""
    finder = TemplateFileFinder()
    finder.find_template_files()
    result = finder.generate_report()

    print("\n🎯 查找完成！")
    print(f"发现 {result['total_files']} 个模板文件需要优化")
    print(f"预计可节省 {result['total_size_kb'] * 0.9:.1f} KB空间")
    if result['total_files'] > 0:
        print("\n💡 建议按优先级顺序处理这些文件")
        print("   1. 使用通用优化器处理高优先级的模板")
        print("   2. 创建专门的优化脚本处理复杂情况")
        print("   3. 验证优化后的功能完整性")


if __name__ == "__main__":
    main()
