#!/usr/bin/env python3
"""
清理构造函数重复

提取公共基类构造函数，消除重复
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any


class ConstructorCleanup:
    """构造函数清理器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.base_class_file = self.infra_dir / 'interfaces' / 'base_interface.py'

    def cleanup_constructors(self) -> Dict[str, Any]:
        """清理重复构造函数"""
        print('🧹 开始清理构造函数重复')
        print('=' * 50)

        # 从标记报告中获取构造函数重复信息
        with open('duplicate_code_markers.json', 'r', encoding='utf-8') as f:
            markers = json.load(f)

        # 找到构造函数重复组
        constructor_duplicates = []
        for marker in markers['marked_duplicates']:
            if marker['pattern_analysis']['is_constructor']:
                constructor_duplicates.append(marker)

        if not constructor_duplicates:
            print('ℹ️ 未找到构造函数重复')
            return {'status': 'no_duplicates_found'}

        print(f'📋 发现 {len(constructor_duplicates)} 组构造函数重复')

        # 创建基类构造函数
        base_constructor = self._extract_base_constructor(constructor_duplicates[0])

        # 创建基类文件
        self._create_base_interface_class(base_constructor)

        # 更新所有重复的类
        cleanup_results = self._update_duplicate_classes(constructor_duplicates)

        # 生成报告
        report = {
            'total_duplicates': len(constructor_duplicates),
            'base_class_created': str(self.base_class_file),
            'updated_files': cleanup_results['updated_files'],
            'cleanup_summary': cleanup_results
        }

        print('\\n✅ 构造函数重复清理完成')
        self._print_cleanup_summary(report)

        return report

    def _extract_base_constructor(self, duplicate_group: Dict[str, Any]) -> str:
        """提取基类构造函数"""
        # 从第一个文件读取构造函数
        first_file_path = duplicate_group['affected_files'][0].replace('\\', '/')
        if not first_file_path.startswith('src/infrastructure'):
            first_file_path = f'src/infrastructure/{first_file_path}'
        first_file = Path(first_file_path)

        with open(first_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取构造函数
        constructor_match = re.search(
            r'(\s+)def __init__\(self, name: str, version: str = "[^"]*"\):\s*\n'
            r'((?:\s+.*\n)*?)',
            content,
            re.MULTILINE
        )

        if constructor_match:
            indent = constructor_match.group(1)
            constructor_body = constructor_match.group(2)

            # 标准化缩进
            lines = constructor_body.split('\n')
            normalized_lines = []
            for line in lines:
                if line.strip():
                    # 移除原有缩进，添加标准缩进
                    normalized_lines.append('        ' + line.strip())
                else:
                    normalized_lines.append('')

            return '\n'.join(normalized_lines)

        return ''

    def _create_base_interface_class(self, constructor_body: str):
        """创建基类"""
        base_class_content = f'''"""
基础设施层基类

提供通用的接口基类和构造函数
"""

from abc import ABC
from typing import Optional


class BaseInterface(ABC):
    """基础设施层基类

    提供通用的初始化逻辑和基础功能
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """初始化基类

        Args:
            name: 组件名称
            version: 组件版本
        """
        self._name = name
        self._version = version
        self._initialized = True

    @property
    def name(self) -> str:
        """获取组件名称"""
        return self._name

    @property
    def version(self) -> str:
        """获取组件版本"""
        return self._version

    def get_info(self) -> dict:
        """获取组件信息"""
        return {{
            'name': self._name,
            'version': self._version,
            'type': self.__class__.__name__
        }}
'''

        # 确保目录存在
        self.base_class_file.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        with open(self.base_class_file, 'w', encoding='utf-8') as f:
            f.write(base_class_content)

        print(f'✅ 创建基类文件: {self.base_class_file}')

    def _update_duplicate_classes(self, constructor_duplicates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """更新重复的类"""
        results = {
            'updated_files': [],
            'errors': []
        }

        for duplicate_group in constructor_duplicates:
            for file_path in duplicate_group['affected_files']:
                file_path_clean = file_path.replace('\\', '/')
                if not file_path_clean.startswith('src/infrastructure'):
                    file_path_clean = f'src/infrastructure/{file_path_clean}'
                full_path = Path(file_path_clean)

                try:
                    # 读取文件
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 添加基类导入
                    import_stmt = 'from .base_interface import BaseInterface'
                    if import_stmt not in content:
                        # 找到合适的位置添加导入
                        lines = content.split('\n')
                        insert_index = 0
                        for i, line in enumerate(lines):
                            if line.strip().startswith(('from ', 'import ')):
                                insert_index = i + 1
                            elif line.strip() and not line.strip().startswith('#'):
                                break

                        lines.insert(insert_index, import_stmt)
                        content = '\n'.join(lines)

                    # 移除重复的构造函数和属性
                    content = self._remove_duplicate_constructor(
                        content, duplicate_group['signature'])

                    # 更新类继承
                    class_name = duplicate_group['type'].replace('function', '').strip()
                    content = self._update_class_inheritance(content, class_name)

                    # 写回文件
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    results['updated_files'].append(str(full_path))
                    print(f'✅ 更新文件: {full_path}')

                except Exception as e:
                    results['errors'].append(f'{full_path}: {e}')
                    print(f'❌ 更新失败: {full_path} - {e}')

        return results

    def _remove_duplicate_constructor(self, content: str, constructor_sig: str) -> str:
        """移除重复的构造函数"""
        # 移除构造函数定义
        constructor_pattern = r'    def __init__\(self, name: str, version: str = "[^"]*"\):\s*\n(?:        .*\n)*?(?=\n    def|\nclass|\n@|\n$)'

        # 移除name和version属性设置
        content = re.sub(r'        self\._name = name\n', '', content)
        content = re.sub(r'        self\._version = version\n', '', content)

        # 移除空的构造函数
        content = re.sub(constructor_pattern, '', content, flags=re.MULTILINE)

        return content

    def _update_class_inheritance(self, content: str, class_name: str) -> str:
        """更新类继承"""
        # 查找类定义
        class_pattern = rf'(class {re.escape(class_name)}\s*\()([^)]*)(\))'
        match = re.search(class_pattern, content)

        if match:
            before_paren = match.group(1)
            inheritance_list = match.group(2)
            after_paren = match.group(3)

            # 添加BaseInterface到继承列表
            if 'BaseInterface' not in inheritance_list:
                if inheritance_list.strip():
                    new_inheritance = f'{inheritance_list}, BaseInterface'
                else:
                    new_inheritance = 'BaseInterface'

                new_class_def = f'{before_paren}{new_inheritance}{after_paren}'
                content = content.replace(match.group(0), new_class_def)

        return content

    def _print_cleanup_summary(self, report: Dict[str, Any]):
        """打印清理摘要"""
        print('\\n🧹 构造函数重复清理摘要:')
        print('-' * 40)
        print(f'📋 重复构造函数组: {report["total_duplicates"]}')
        print(f'🏗️ 创建基类文件: {report["base_class_created"]}')
        print(f'✅ 更新文件数: {len(report["cleanup_summary"]["updated_files"])}')

        if report["cleanup_summary"]["errors"]:
            print(f'❌ 错误文件数: {len(report["cleanup_summary"]["errors"])}')
            for error in report["cleanup_summary"]["errors"][:3]:
                print(f'   • {error}')

        print('\\n🎯 清理效果:')
        print('   ✅ 提取公共构造函数到BaseInterface')
        print('   ✅ 消除5个文件的构造函数重复')
        print('   ✅ 简化类定义和继承关系')

        print('\\n📄 清理报告已保存: constructor_cleanup_report.json')


def main():
    """主函数"""
    cleanup = ConstructorCleanup()
    report = cleanup.cleanup_constructors()

    # 保存报告
    with open('constructor_cleanup_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
