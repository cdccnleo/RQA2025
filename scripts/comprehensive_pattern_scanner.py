#!/usr/bin/env python3
"""
综合项目重复文件模式扫描器

扫描整个项目，识别各种可能的重复文件模式，包括：
- 数字编号的文件 (如 component_1.py, component_2.py)
- 相似命名的文件
- 内容重复的文件
- 模板化的文件结构
"""

import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict


class ComprehensivePatternScanner:
    """综合项目重复文件模式扫描器"""

    def __init__(self):
        self.all_files = []
        self.pattern_groups = defaultdict(list)
        self.content_hashes = defaultdict(list)
        self.size_groups = defaultdict(list)
        self.similar_files = defaultdict(list)

    def scan_project_files(self):
        """扫描项目中的所有文件"""
        print("🔍 扫描项目文件...")
        print("="*60)

        # 排除目录
        exclude_dirs = {
            '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            'backup', 'backups', 'temp', 'tmp', 'build', 'dist',
            'test', 'tests', 'testing', '.pytest_cache', '.cache',
            '.mypy_cache', 'node_modules', '.tox', '.coverage'
        }

        # 排除文件
        exclude_files = {
            '.gitignore', '.gitattributes', 'requirements.txt',
            'setup.py', 'pyproject.toml', 'Makefile', 'Dockerfile',
            'README.md', 'CHANGELOG.md', 'LICENSE'
        }

        for root, dirs, files in os.walk('.'):
            # 移除需要排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file in exclude_files or file.startswith('.'):
                    continue

                file_path = Path(root) / file
                try:
                    # 获取文件信息
                    file_stat = file_path.stat()
                    file_info = {
                        'path': file_path,
                        'name': file_path.name,
                        'directory': file_path.parent,
                        'size': file_stat.st_size,
                        'extension': file_path.suffix,
                        'relative_path': str(file_path.relative_to('.'))
                    }
                    self.all_files.append(file_info)
                except (OSError, ValueError):
                    continue

        print(f"   📁 发现 {len(self.all_files)} 个项目文件")

    def analyze_numerical_patterns(self):
        """分析数字编号的文件模式"""
        print("🔢 分析数字编号文件模式...")

        # 匹配数字编号的模式
        patterns = [
            (r'(.+)_(\d+)\.py$', '数字编号Python文件'),
            (r'(.+)(\d+)\.py$', '数字后缀Python文件'),
            (r'(.+)_(\d+)\.(.+)$', '数字编号通用文件'),
            (r'(.+)(\d+)\.(.+)$', '数字后缀通用文件'),
        ]

        numerical_files = defaultdict(list)

        for file_info in self.all_files:
            filename = file_info['name']
            for pattern, description in patterns:
                match = re.match(pattern, filename)
                if match:
                    base_name = match.group(1)
                    number = match.group(2)
                    key = f"{base_name}_{description}"

                    numerical_files[key].append({
                        **file_info,
                        'base_name': base_name,
                        'number': int(number),
                        'pattern_type': description
                    })
                    break

        # 筛选出有多个文件的模式
        for pattern, files in numerical_files.items():
            if len(files) >= 3:  # 至少3个文件才算模式
                self.pattern_groups[f"数字模式_{pattern}"] = files

        print(f"   🔢 发现 {len(self.pattern_groups)} 个数字编号模式")

    def analyze_similar_names(self):
        """分析相似命名的文件"""
        print("📝 分析相似命名的文件...")

        # 按目录分组文件
        dir_files = defaultdict(list)
        for file_info in self.all_files:
            dir_files[file_info['directory']].append(file_info)

        # 在每个目录中查找相似文件
        for directory, files in dir_files.items():
            if len(files) < 2:
                continue

            # 按文件名的相似性分组
            name_groups = defaultdict(list)

            for file_info in files:
                # 提取文件名特征
                name = file_info['name']
                # 移除数字
                name_no_numbers = re.sub(r'\d+', '', name)
                # 移除常见后缀
                name_clean = re.sub(r'\.(py|txt|md|json|yaml|yml)$', '', name_no_numbers)
                # 移除下划线和连字符
                name_base = re.sub(r'[_-]', '', name_clean).lower()

                if len(name_base) > 3:  # 只考虑有意义的名称
                    name_groups[name_base].append(file_info)

            # 保存相似文件组
            for base_name, similar_files in name_groups.items():
                if len(similar_files) >= 3:
                    self.similar_files[f"目录_{directory.name}_{base_name}"] = similar_files

        print(f"   📝 发现 {len(self.similar_files)} 个相似命名文件组")

    def analyze_file_sizes(self):
        """分析文件大小相同的文件"""
        print("📏 分析文件大小相同的文件...")

        # 按文件大小分组
        for file_info in self.all_files:
            if file_info['size'] > 1024:  # 只考虑大于1KB的文件
                self.size_groups[file_info['size']].append(file_info)

        # 筛选出多个文件大小相同的组
        for size, files in self.size_groups.items():
            if len(files) >= 3:  # 至少3个文件大小相同
                self.pattern_groups[f"相同大小_{size}字节"] = files

        print(
            f"   📏 发现 {len([k for k in self.pattern_groups.keys() if k.startswith('相同大小')])} 个相同大小文件组")

    def analyze_content_duplicates(self):
        """分析内容重复的文件"""
        print("📄 分析内容重复的文件...")

        # 计算文件内容哈希（只对小文件）
        for file_info in self.all_files:
            if 100 < file_info['size'] < 10240:  # 100字节到10KB的文件
                try:
                    with open(file_info['path'], 'rb') as f:
                        content = f.read()
                        content_hash = hashlib.md5(content).hexdigest()
                        self.content_hashes[content_hash].append(file_info)
                except (IOError, OSError):
                    continue

        # 筛选出内容完全相同的文件组
        for content_hash, files in self.content_hashes.items():
            if len(files) >= 2:  # 至少2个文件内容相同
                self.pattern_groups[f"内容重复_{content_hash[:8]}"] = files

        print(
            f"   📄 发现 {len([k for k in self.pattern_groups.keys() if k.startswith('内容重复')])} 个内容重复文件组")

    def analyze_template_patterns(self):
        """分析模板文件模式"""
        print("🔧 分析模板文件模式...")

        # 已知的模板模式 - 扩展版本
        template_patterns = [
            # 基础组件模式
            r'.*component.*\.py$',
            r'.*service.*\.py$',
            r'.*handler.*\.py$',
            r'.*processor.*\.py$',
            r'.*manager.*\.py$',
            r'.*controller.*\.py$',
            r'.*factory.*\.py$',
            r'.*strategy.*\.py$',
            r'.*adapter.*\.py$',
            r'.*client.*\.py$',
            r'.*cache.*\.py$',
            r'.*optimizer.*\.py$',

            # 交易相关模式
            r'.*account.*\.py$',
            r'.*balance.*\.py$',
            r'.*capital.*\.py$',
            r'.*fund.*\.py$',
            r'.*margin.*\.py$',
            r'.*position.*\.py$',
            r'.*portfolio.*\.py$',
            r'.*trading.*\.py$',
            r'.*order.*\.py$',
            r'.*execution.*\.py$',

            # 数据处理模式
            r'.*analyzer.*\.py$',
            r'.*validator.*\.py$',
            r'.*checker.*\.py$',
            r'.*monitor.*\.py$',
            r'.*tracker.*\.py$',
            r'.*collector.*\.py$',
            r'.*filter.*\.py$',
            r'.*transformer.*\.py$',

            # 基础设施模式
            r'.*config.*\.py$',
            r'.*logger.*\.py$',
            r'.*metrics.*\.py$',
            r'.*storage.*\.py$',
            r'.*repository.*\.py$',
            r'.*connector.*\.py$',
            r'.*provider.*\.py$',
            r'.*source.*\.py$',

            # 数字编号模式 (最常见的模板模式)
            r'.*\d+\.py$',  # 任何包含数字的文件名
        ]

        template_files = []

        for file_info in self.all_files:
            filename = file_info['name']
            for pattern in template_patterns:
                if re.match(pattern, filename.lower()):
                    template_files.append(file_info)
                    break

        if template_files:
            self.pattern_groups["模板文件模式"] = template_files

        print(f"   🔧 发现 {len(template_files)} 个可能的模板文件")

    def analyze_function_patterns(self):
        """分析函数级别的重复模式"""
        print("🔍 分析函数级别的重复模式...")

        # 读取Python文件内容，分析函数模式
        python_files = [f for f in self.all_files if f['extension'] == '.py']

        function_patterns = defaultdict(list)

        for file_info in python_files:
            if file_info['size'] > 10240:  # 跳过大文件
                continue

            try:
                with open(file_info['path'], 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取函数定义
                function_defs = re.findall(r'def\s+(\w+)\s*\([^)]*\)\s*:', content)

                for func_name in function_defs:
                    # 标准化函数名
                    func_name_clean = func_name.lower().replace('_', '')
                    function_patterns[func_name_clean].append({
                        **file_info,
                        'function_name': func_name
                    })

            except (IOError, OSError, UnicodeDecodeError):
                continue

        # 筛选出重复的函数名模式
        for func_pattern, files in function_patterns.items():
            if len(files) >= 5:  # 至少5个文件有相同的函数名
                self.pattern_groups[f"函数模式_{func_pattern}"] = files

        print(
            f"   🔍 发现 {len([k for k in self.pattern_groups.keys() if k.startswith('函数模式')])} 个函数重复模式")

    def generate_report(self):
        """生成扫描报告"""
        print("\n📊 综合扫描结果报告")
        print("="*60)

        total_patterns = len(self.pattern_groups) + len(self.similar_files)
        print(f"📈 发现 {total_patterns} 个潜在重复模式")

        if self.pattern_groups:
            print("\n🔢 数字编号模式:")
            for pattern_name, files in self.pattern_groups.items():
                if pattern_name.startswith('数字模式'):
                    print(f"   • {pattern_name}: {len(files)} 个文件")

            print("\n📏 相同大小文件组:")
            for pattern_name, files in self.pattern_groups.items():
                if pattern_name.startswith('相同大小'):
                    size_kb = files[0]['size'] / 1024
                    print(f"   • {pattern_name}: {len(files)} 个文件 ({size_kb:.1f} KB)")
            print("\n📄 内容重复文件组:")
            for pattern_name, files in self.pattern_groups.items():
                if pattern_name.startswith('内容重复'):
                    print(f"   • {pattern_name}: {len(files)} 个文件")

            print("\n🔧 模板文件模式:")
            for pattern_name, files in self.pattern_groups.items():
                if pattern_name == "模板文件模式":
                    print(f"   • {pattern_name}: {len(files)} 个文件")

            print("\n🔍 函数重复模式:")
            for pattern_name, files in self.pattern_groups.items():
                if pattern_name.startswith('函数模式'):
                    print(f"   • {pattern_name}: {len(files)} 个文件")

        if self.similar_files:
            print("\n📝 相似命名文件组:")
            for group_name, files in self.similar_files.items():
                print(f"   • {group_name}: {len(files)} 个文件")

        return {
            'pattern_groups': dict(self.pattern_groups),
            'similar_files': dict(self.similar_files),
            'total_patterns': total_patterns,
            'total_files_analyzed': len(self.all_files)
        }

    def run_comprehensive_scan(self):
        """运行综合扫描"""
        print("🚀 开始项目综合重复模式扫描...")
        print("="*80)

        try:
            # 1. 扫描项目文件
            self.scan_project_files()

            # 2. 分析各种模式
            self.analyze_numerical_patterns()
            self.analyze_similar_names()
            self.analyze_file_sizes()
            self.analyze_content_duplicates()
            self.analyze_template_patterns()
            self.analyze_function_patterns()

            # 3. 生成报告
            result = self.generate_report()

            print("\n" + "="*80)
            print("✅ 项目综合重复模式扫描完成！")
            print("="*80)

            return result

        except Exception as e:
            print(f"\n❌ 扫描过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    scanner = ComprehensivePatternScanner()
    result = scanner.run_comprehensive_scan()

    if result:
        print("\n🎉 扫描成功完成！")
        print(f"共发现 {result['total_patterns']} 个潜在重复模式")
        print(f"分析了 {result['total_files_analyzed']} 个项目文件")
    else:
        print("\n❌ 扫描失败！")


if __name__ == "__main__":
    main()
