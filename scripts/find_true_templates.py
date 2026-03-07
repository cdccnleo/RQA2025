#!/usr/bin/env python3
"""
精确模板文件查找器

专门查找真正的模板文件，避免误报
模板文件的特征：
1. 数字编号的文件名（如 component_1.py, component_2.py）
2. 文件大小相同或相近
3. 内容高度相似但不完全相同
4. 通常包含类定义和简单的业务逻辑
"""

import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import List
from difflib import SequenceMatcher


class TrueTemplateFinder:
    """精确模板文件查找器"""

    def __init__(self):
        self.potential_templates = []
        self.content_hashes = defaultdict(list)
        self.size_groups = defaultdict(list)
        self.similar_content_groups = defaultdict(list)

    def calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def read_file_content(self, file_path: Path) -> str:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def is_template_filename(self, filename: str) -> bool:
        """判断是否是模板文件名"""
        # 排除已知的非模板文件
        exclude_patterns = [
            '__init__.py',
            '__pycache__',
            'conftest.py',
            'setup.py',
            'test_*.py',
            '*_test.py',
            'config.py',
            'settings.py',
            'constants.py',
            'interfaces.py',
            'base.py',
            'utils.py',
            'helpers.py',
            'factory.py',
            'manager.py',
            'service.py',
            'engine.py',
            'client.py',
            'server.py',
            'api.py',
            'main.py',
            'app.py',
            'index.py'
        ]

        # 检查是否匹配排除模式
        for pattern in exclude_patterns:
            if re.match(pattern.replace('*', '.*'), filename):
                return False

        # 检查是否包含数字编号
        number_patterns = [
            r'.*_(\d+)\.py$',      # component_1.py
            r'.*(\d+)\.py$',       # 1.py (不常见)
            r'.*_copy(\d*)\.py$',  # file_copy1.py
            r'.*_v(\d+)\.py$',     # file_v1.py
            r'.*_version(\d+)\.py$'  # file_version1.py
        ]

        for pattern in number_patterns:
            if re.match(pattern, filename):
                return True

        return False

    def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        if not content1 or not content2:
            return 0.0

        # 标准化内容
        def normalize_content(content):
            # 移除注释
            content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
            # 移除空行
            content = re.sub(r'\n\s*\n', '\n', content)
            # 移除多余空格
            content = re.sub(r'\s+', ' ', content)
            return content.strip()

        norm1 = normalize_content(content1)
        norm2 = normalize_content(content2)

        if not norm1 or not norm2:
            return 0.0

        # 使用序列匹配器计算相似度
        matcher = SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()

    def find_similar_content_files(self, files: List[Path], similarity_threshold: float = 0.8):
        """查找内容相似的文件"""
        print("🔍 分析文件内容相似度...")

        # 读取所有文件内容
        file_contents = {}
        for file_path in files:
            content = self.read_file_content(file_path)
            if content and len(content) > 100:  # 只分析有意义的内容
                file_contents[file_path] = content

        similar_groups = []

        # 两两比较文件内容
        processed = set()
        for file1, content1 in file_contents.items():
            if file1 in processed:
                continue

            group = [file1]
            processed.add(file1)

            for file2, content2 in file_contents.items():
                if file2 in processed:
                    continue

                similarity = self.calculate_content_similarity(content1, content2)
                if similarity >= similarity_threshold:
                    group.append(file2)
                    processed.add(file2)

            if len(group) > 1:
                similar_groups.append({
                    'files': group,
                    'similarity': similarity,
                    'file_count': len(group)
                })

        return similar_groups

    def scan_for_true_templates(self):
        """扫描真正的模板文件"""
        print("🚀 开始精确模板文件扫描...")
        print("="*60)

        # 扫描src目录
        print("📁 扫描src目录文件...")
        all_files = []
        for root, dirs, files in os.walk('src'):
            # 排除不需要的目录
            dirs[:] = [d for d in dirs if d not in [
                '__pycache__', '.git', 'node_modules', '.venv', 'venv',
                'backup', 'backups', 'temp', 'tmp', 'build', 'dist',
                'test', 'tests', 'testing', 'scripts'
            ]]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    all_files.append(file_path)

        print(f"   发现 {len(all_files)} 个Python文件")

        # 筛选可能的模板文件
        print("🔍 筛选可能的模板文件...")
        potential_templates = []
        for file_path in all_files:
            if self.is_template_filename(file_path.name):
                potential_templates.append(file_path)

        print(f"   发现 {len(potential_templates)} 个可能的模板文件")

        if not potential_templates:
            print("   ⚠️  未发现可能的模板文件")
            return []

        # 分析文件大小分布
        print("📏 分析文件大小分布...")
        size_counter = Counter()
        for file_path in potential_templates:
            try:
                size = file_path.stat().st_size
                size_counter[size] += 1
            except Exception as e:
                print(f"   ⚠️  读取文件大小失败 {file_path}: {e}")

        # 找出常见的大小
        common_sizes = [size for size, count in size_counter.items() if count > 1]
        print(f"   发现 {len(common_sizes)} 个常见文件大小")

        # 按大小分组
        size_groups = defaultdict(list)
        for file_path in potential_templates:
            try:
                size = file_path.stat().st_size
                if size in common_sizes:
                    size_groups[size].append(file_path)
            except Exception:
                pass

        # 筛选出有多个文件的组
        multi_file_groups = {size: paths for size, paths in size_groups.items() if len(paths) > 1}
        print(f"   发现 {len(multi_file_groups)} 个包含多个文件的组")

        # 分析内容相似度
        true_templates = []
        for size, files in multi_file_groups.items():
            if len(files) >= 2:
                print(f"   分析大小为 {size} 字节的 {len(files)} 个文件...")

                # 检查内容相似度
                similar_groups = self.find_similar_content_files(files, similarity_threshold=0.7)

                for group in similar_groups:
                    if len(group['files']) >= 2:
                        true_templates.extend(group['files'])
                        print(
                            f"      ✅ 发现模板组: {len(group['files'])} 个文件 (相似度: {group['similarity']:.2f})")

        # 去重
        true_templates = list(set(true_templates))

        print("\n📊 扫描结果汇总:")
        print(f"   📋 可能的模板文件: {len(potential_templates)} 个")
        print(f"   📏 多个文件的组: {len(multi_file_groups)} 个")
        print(f"   ✅ 确认的模板文件: {len(true_templates)} 个")

        return true_templates

    def generate_template_report(self, templates: List[Path]) -> str:
        """生成模板文件报告"""
        report = []

        report.append("# RQA2025 精确模板文件扫描报告")
        report.append("")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## 发现的模板文件")
        report.append("")
        report.append(f"共发现 {len(templates)} 个确认的模板文件")
        report.append("")

        # 按目录分组
        dir_groups = defaultdict(list)
        for template in templates:
            dir_groups[template.parent].append(template)

        for directory, files in dir_groups.items():
            report.append(f"### {directory}")
            report.append(f"文件数量: {len(files)}")
            report.append("")

            for file in sorted(files):
                try:
                    size = file.stat().st_size
                    report.append(f"- {file.name} ({size} 字节)")
                except Exception:
                    report.append(f"- {file.name} (读取失败)")

            report.append("")

        return "\n".join(report)


def main():
    """主函数"""
    finder = TrueTemplateFinder()
    templates = finder.scan_for_true_templates()

    if templates:
        # 生成报告
        report = finder.generate_template_report(templates)
        with open('reports/TRUE_TEMPLATES_REPORT.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📋 模板文件报告已保存到: reports/TRUE_TEMPLATES_REPORT.md")

        # 显示前20个模板文件
        print("\n🎯 前20个发现的模板文件:")
        for i, template in enumerate(templates[:20], 1):
            try:
                size = template.stat().st_size
                print("2d")
            except Exception:
                print("2d")

        if len(templates) > 20:
            print(f"   ... 还有 {len(templates) - 20} 个文件")
    else:
        print("\n❌ 未发现确认的模板文件")

    print("\n🎉 精确模板文件扫描完成！")
    print(f"共发现 {len(templates)} 个确认的模板文件")


if __name__ == "__main__":
    main()
