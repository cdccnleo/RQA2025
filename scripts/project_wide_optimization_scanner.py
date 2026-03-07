#!/usr/bin/env python3
"""
项目级代码重复问题扫描器

扫描整个项目，识别所有重复的模板文件和类似的代码重复问题
"""

import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class ProjectOptimizationScanner:
    """项目优化扫描器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.file_patterns = {
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
        self.optimization_targets = {}
        self.code_duplicates = {}

    def scan_project_for_templates(self):
        """扫描项目中的模板文件"""
        print("🔍 开始扫描项目模板文件...")
        print("="*60)

        for root, dirs, files in os.walk(self.project_root):
            # 跳过某些目录
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', 'node_modules', '.venv', 'venv',
                'backup', 'backups', 'temp', 'tmp', 'build', 'dist'
            }]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    self._analyze_file(file_path)

        return self.optimization_targets

    def _analyze_file(self, file_path: Path):
        """分析单个文件"""
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 获取文件大小
            size_kb = file_path.stat().st_size / 1024

            # 分析文件类型
            for pattern_name, pattern in self.file_patterns.items():
                if re.match(pattern, file_path.name):
                    if pattern_name not in self.optimization_targets:
                        self.optimization_targets[pattern_name] = {
                            'files': [],
                            'total_size_kb': 0,
                            'locations': set()
                        }

                    self.optimization_targets[pattern_name]['files'].append({
                        'path': str(file_path),
                        'size_kb': size_kb,
                        'content_hash': hashlib.md5(content.encode()).hexdigest(),
                        'line_count': len(content.split('\n'))
                    })

                    self.optimization_targets[pattern_name]['total_size_kb'] += size_kb
                    self.optimization_targets[pattern_name]['locations'].add(str(file_path.parent))

                    break

        except Exception as e:
            print(f"   ⚠️  分析文件失败 {file_path}: {e}")

    def find_similar_files(self):
        """查找内容相似的文件"""
        print("\n🔍 分析文件内容相似性...")

        self.code_duplicates = defaultdict(list)

        for pattern_name, data in self.optimization_targets.items():
            if len(data['files']) > 1:
                print(f"   📊 分析{pattern_name}: {len(data['files'])}个文件")

                # 按内容哈希分组
                hash_groups = defaultdict(list)
                for file_info in data['files']:
                    hash_groups[file_info['content_hash']].append(file_info)

                # 找出重复的组
                for hash_value, files in hash_groups.items():
                    if len(files) > 1:
                        self.code_duplicates[pattern_name].append({
                            'hash': hash_value,
                            'files': files,
                            'count': len(files),
                            'total_size_kb': sum(f['size_kb'] for f in files)
                        })

    def generate_optimization_report(self):
        """生成优化报告"""
        print("\n📊 生成项目优化报告...")
        print("="*60)

        report_content = f"""# 项目级代码重复问题分析报告

## 📊 扫描概览

### 扫描时间
- **执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **扫描范围**: {self.project_root}
- **文件类型**: Python (.py)

### 发现的模板文件类型
"""

        total_files = 0
        total_size_kb = 0
        total_duplicates = 0

        for pattern_name, data in self.optimization_targets.items():
            if data['files']:
                total_files += len(data['files'])
                total_size_kb += data['total_size_kb']

                report_content += f"\n#### {pattern_name} ({len(data['files'])}个文件)\n"
                report_content += f"- **总大小**: {data['total_size_kb']:.1f} KB\n"
                report_content += f"- **分布位置**: {len(data['locations'])}个目录\n"

                for location in sorted(data['locations']):
                    files_in_location = [f for f in data['files'] if f['path'].startswith(location)]
                    report_content += f"  - `{location}`: {len(files_in_location)}个文件\n"

                # 列出文件详情
                for file_info in sorted(data['files'], key=lambda x: x['path']):
                    report_content += f"  - `{file_info['path']}` ({file_info['size_kb']:.1f} KB, {file_info['line_count']}行)\n"

        report_content += f"""
### 重复文件分析
"""

        for pattern_name, duplicates in self.code_duplicates.items():
            if duplicates:
                report_content += f"\n#### {pattern_name} 重复文件组\n"

                for i, duplicate in enumerate(duplicates, 1):
                    total_duplicates += duplicate['count']
                    report_content += f"\n**重复组 {i}** ({duplicate['count']}个完全相同的文件):\n"
                    report_content += f"- 总大小: {duplicate['total_size_kb']:.1f} KB\n"
                    report_content += "- 文件列表:\n"

                    for file_info in duplicate['files']:
                        report_content += f"  - `{file_info['path']}`\n"

        report_content += f"""
## 📈 优化建议

### 总体优化统计
- **发现的模板文件**: {total_files}个
- **总文件大小**: {total_size_kb:.1f} KB
- **完全重复的文件**: {total_duplicates}个
- **潜在优化空间**: {total_size_kb * 0.9:.1f} KB (90%减少)

### 优先级优化建议

#### 🚨 高优先级 (立即处理)
"""

        high_priority = []
        for pattern_name, data in self.optimization_targets.items():
            if len(data['files']) >= 10:  # 10个以上文件的类型
                high_priority.append(pattern_name)

        for pattern_name in high_priority:
            data = self.optimization_targets[pattern_name]
            duplicate_count = sum(len(group['files'])
                                  for group in self.code_duplicates.get(pattern_name, []))
            report_content += f"- **{pattern_name}**: {len(data['files'])}个文件，{duplicate_count}个重复，节省{data['total_size_kb'] * 0.9:.1f} KB\n"

        report_content += f"""
#### 📋 中优先级 (近期处理)
"""

        medium_priority = []
        for pattern_name, data in self.optimization_targets.items():
            if 5 <= len(data['files']) < 10:  # 5-10个文件的类型
                medium_priority.append(pattern_name)

        for pattern_name in medium_priority:
            data = self.optimization_targets[pattern_name]
            duplicate_count = sum(len(group['files'])
                                  for group in self.code_duplicates.get(pattern_name, []))
            report_content += f"- **{pattern_name}**: {len(data['files'])}个文件，{duplicate_count}个重复，节省{data['total_size_kb'] * 0.9:.1f} KB\n"

        report_content += f"""
#### 🔄 低优先级 (长期规划)
"""

        low_priority = []
        for pattern_name, data in self.optimization_targets.items():
            if len(data['files']) < 5:  # 少于5个文件的类型
                low_priority.append(pattern_name)

        for pattern_name in low_priority:
            data = self.optimization_targets[pattern_name]
            duplicate_count = sum(len(group['files'])
                                  for group in self.code_duplicates.get(pattern_name, []))
            report_content += f"- **{pattern_name}**: {len(data['files'])}个文件，{duplicate_count}个重复，节省{data['total_size_kb'] * 0.9:.1f} KB\n"

        report_content += f"""
## 🏭 优化实施策略

### 统一组件工厂模式
```python
# 示例：统一服务组件工厂
from infrastructure.services.service_components import ServiceComponentFactory

# 创建服务组件
service = ServiceComponentFactory.create_component(3)
result = service.process_request({"data": "test"})
```

### 向后兼容性保证
```python
# 兼容旧代码
from infrastructure.services.service_components import create_service_component_3
service = create_service_component_3()
```

### 实施步骤
1. **备份所有原始文件**
2. **按类型分组优化** (从高优先级开始)
3. **创建统一工厂类**
4. **更新__init__.py文件**
5. **测试功能完整性**
6. **更新文档和使用指南**

## 📋 目录优化清单
"""

        # 按目录整理需要优化的文件
        directory_optimization = defaultdict(list)
        for pattern_name, data in self.optimization_targets.items():
            for file_info in data['files']:
                dir_path = Path(file_info['path']).parent
                directory_optimization[str(dir_path)].append({
                    'pattern': pattern_name,
                    'file': file_info
                })

        for directory, files in sorted(directory_optimization.items()):
            report_content += f"\n### {directory}\n"
            pattern_counts = defaultdict(int)
            total_size = 0

            for file_info in files:
                pattern_counts[file_info['pattern']] += 1
                total_size += file_info['file']['size_kb']

            for pattern, count in pattern_counts.items():
                report_content += f"- **{pattern}**: {count}个文件\n"

            report_content += f"- **总计**: {len(files)}个文件，{total_size:.1f} KB\n"

        report_content += f"""
---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**扫描器版本**: 1.0.0
**优化目标**: 发现并解决项目级代码重复问题
"""

        # 保存报告
        report_file = self.project_root / "PROJECT_WIDE_OPTIMIZATION_REPORT.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 优化报告已生成: {report_file}")
        return report_file

    def run_full_scan(self):
        """运行完整扫描"""
        print("🚀 开始项目级代码重复问题扫描...")
        print("="*60)

        # 扫描模板文件
        self.scan_project_for_templates()

        # 分析相似文件
        self.find_similar_files()

        # 生成报告
        report_file = self.generate_optimization_report()

        print("\n" + "="*60)
        print("✅ 项目级扫描完成！")
        print("="*60)

        # 输出摘要
        total_template_files = sum(len(data['files'])
                                   for data in self.optimization_targets.values())
        total_duplicate_files = sum(
            sum(len(group['files']) for group in duplicates)
            for duplicates in self.code_duplicates.values()
        )

        print("\n📊 扫描结果摘要:")
        print(f"   🔍 发现模板文件类型: {len(self.optimization_targets)}种")
        print(f"   📁 总模板文件数: {total_template_files}个")
        print(f"   🔄 完全重复文件数: {total_duplicate_files}个")
        print(f"   📈 优化潜力: {total_duplicate_files / max(total_template_files, 1) * 100:.1f}%")
        print(f"   📄 详细报告: {report_file}")

        return {
            'total_template_files': total_template_files,
            'total_duplicate_files': total_duplicate_files,
            'optimization_targets': self.optimization_targets,
            'code_duplicates': self.code_duplicates,
            'report_file': str(report_file)
        }


def main():
    """主函数"""
    project_root = os.getcwd()  # 当前工作目录

    if not os.path.exists(project_root):
        print("❌ 项目目录不存在")
        return

    scanner = ProjectOptimizationScanner(project_root)
    result = scanner.run_full_scan()

    if result:
        print("\n🎉 项目级优化扫描成功完成！")
        print(f"共发现 {result['total_template_files']} 个模板文件")
        print(f"其中 {result['total_duplicate_files']} 个为重复文件")
    else:
        print("\n❌ 项目级优化扫描失败！")


if __name__ == "__main__":
    main()
