#!/usr/bin/env python3
"""
简化版模板文件扫描器

快速识别项目中的模板文件
"""

import os
import re
from pathlib import Path
from collections import defaultdict


class SimpleTemplateScanner:
    """简化版模板扫描器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.template_patterns = {
            'cache_templates': r'cache_\d+\.py$',
            'client_templates': r'client_\d+\.py$',
            'service_templates': r'service_\d+\.py$',
            'strategy_templates': r'strategy_\d+\.py$',
            'optimizer_templates': r'optimizer_\d+\.py$',
            'manager_templates': r'manager_\d+\.py$',
            'handler_templates': r'handler_\d+\.py$',
            'controller_templates': r'controller_\d+\.py$',
            'processor_templates': r'processor_\d+\.py$',
            'adapter_templates': r'adapter_\d+\.py$'
        }
        self.found_templates = defaultdict(list)

    def scan_for_templates(self):
        """扫描模板文件"""
        print("🔍 开始扫描模板文件...")
        print("="*60)

        for root, dirs, files in os.walk(self.project_root):
            # 跳过不需要的目录
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', 'node_modules', '.venv', 'venv',
                'backup', 'backups', 'temp', 'tmp', 'build', 'dist'
            }]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    self._check_file_pattern(file_path)

        return self.found_templates

    def _check_file_pattern(self, file_path: Path):
        """检查文件是否匹配模板模式"""
        filename = file_path.name

        for pattern_name, pattern in self.template_patterns.items():
            if re.match(pattern, filename):
                try:
                    size_kb = file_path.stat().st_size / 1024
                    self.found_templates[pattern_name].append({
                        'path': str(file_path),
                        'size_kb': round(size_kb, 1),
                        'name': filename
                    })
                except Exception as e:
                    print(f"   ⚠️  处理文件失败 {file_path}: {e}")

    def generate_summary(self):
        """生成摘要报告"""
        print("\n📊 扫描结果摘要:")
        print("="*60)

        total_files = 0
        total_size = 0

        for pattern_name, files in self.found_templates.items():
            if files:
                file_count = len(files)
                pattern_size = sum(f['size_kb'] for f in files)
                total_files += file_count
                total_size += pattern_size

                print(f"   📁 {pattern_name}: {file_count}个文件 ({pattern_size:.1f} KB)")

                # 显示前几个文件作为示例
                for i, file_info in enumerate(files[:3]):
                    print(f"      - {file_info['name']} ({file_info['size_kb']} KB)")
                if file_count > 3:
                    print(f"      ... 还有 {file_count - 3} 个文件")

        print(f"\n   🎯 总计发现: {total_files}个模板文件 ({total_size:.1f} KB)")
        print(f"   📈 优化潜力: {total_size * 0.9:.1f} KB (90%减少)")

        return {
            'total_files': total_files,
            'total_size_kb': total_size,
            'templates': dict(self.found_templates)
        }

    def show_optimization_priority(self):
        """显示优化优先级"""
        print("\n🚨 优化优先级建议:")
        print("="*60)

        # 按文件数量排序
        sorted_patterns = sorted(
            self.found_templates.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        for i, (pattern_name, files) in enumerate(sorted_patterns, 1):
            count = len(files)
            size = sum(f['size_kb'] for f in files)

            if count >= 20:
                priority = "🚨 高优先级"
            elif count >= 10:
                priority = "📋 中优先级"
            else:
                priority = "🔄 低优先级"

            print(f"   {i}. {priority} - {pattern_name}")
            print(f"      文件数: {count}个, 大小: {size:.1f} KB")


def main():
    """主函数"""
    project_root = os.getcwd()

    if not os.path.exists(project_root):
        print("❌ 项目目录不存在")
        return

    scanner = SimpleTemplateScanner(project_root)
    scanner.scan_for_templates()
    result = scanner.generate_summary()
    scanner.show_optimization_priority()

    if result['total_files'] > 0:
        print(f"\n🎯 发现 {result['total_files']} 个模板文件需要优化!")
        print(f"💡 建议优先处理前3个类型，预计可节省 {result['total_size_kb'] * 0.9:.1f} KB空间")
    else:
        print("\n✅ 未发现明显的模板文件重复问题")


if __name__ == "__main__":
    main()
