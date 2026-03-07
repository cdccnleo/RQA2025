"""
基础设施层统一接口迁移脚本

自动迁移现有类到新的统一接口体系
"""

import re
from pathlib import Path
from typing import Dict, List


class InterfaceMigrator:
    """接口迁移器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.migration_map = {
            # 架构模式迁移 (按优先级排序，避免重复匹配)
            'ComponentFactory': {
                'new_base': 'BaseComponentFactory',
                'import': 'from src.infrastructure.interfaces import BaseComponentFactory'
            },
            'Factory': {
                'new_base': 'BaseFactory',
                'import': 'from src.infrastructure.interfaces import BaseFactory'
            },
            'Manager': {
                'new_base': 'BaseManager',
                'import': 'from src.infrastructure.interfaces import BaseManager'
            },
            'Service': {
                'new_base': 'BaseService',
                'import': 'from src.infrastructure.interfaces import BaseService'
            },
            'Handler': {
                'new_base': 'BaseHandler',
                'import': 'from src.infrastructure.interfaces import BaseHandler'
            },
            'Provider': {
                'new_base': 'BaseProvider',
                'import': 'from src.infrastructure.interfaces import BaseProvider'
            }
        }

    def migrate_file(self, file_path: str) -> bool:
        """迁移单个文件"""
        full_path = self.infra_dir / file_path

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 应用迁移规则 (按优先级处理，避免重复匹配)
            processed_classes = set()  # 避免重复处理同一个类

            for old_pattern, new_config in self.migration_map.items():
                # 查找需要迁移的类
                class_pattern = rf'class\s+(\w+{old_pattern})(\s*\(|\s*:)'
                class_matches = re.findall(class_pattern, content)

                for class_match in class_matches:
                    class_name = class_match[0]
                    if class_name in processed_classes:
                        continue  # 跳过已处理的类

                    processed_classes.add(class_name)

                    # 添加导入语句
                    if new_config['import'] not in content:
                        # 找到合适的位置添加导入
                        lines = content.split('\n')
                        import_inserted = False

                        for i, line in enumerate(lines):
                            if line.startswith('from ') or line.startswith('import '):
                                # 在现有导入后添加
                                lines.insert(i + 1, new_config['import'])
                                import_inserted = True
                                break

                        if import_inserted:
                            content = '\n'.join(lines)

                    # 更新类继承
                    # 处理有继承的情况
                    class_pattern_with_inheritance = rf'(class\s+{re.escape(class_name)}\s*\([^)]*)\)'

                    def replacement_with_inheritance(match):
                        existing_inheritance = match.group(1)
                        new_base = new_config['new_base']
                        return f'{existing_inheritance}, {new_base})'

                    # 先尝试替换有继承的情况
                    new_content = re.sub(class_pattern_with_inheritance,
                                         replacement_with_inheritance, content)
                    if new_content != content:
                        content = new_content
                    else:
                        # 处理没有继承的情况
                        class_pattern_no_inheritance = rf'(class\s+{re.escape(class_name)}\s*:\s*$)'

                        def replacement_no_inheritance(match):
                            class_declaration = match.group(1)
                            new_base = new_config['new_base']
                            return f'class {class_name}({new_base}):'

                        content = re.sub(class_pattern_no_inheritance,
                                         replacement_no_inheritance, content, flags=re.MULTILINE)

            # 如果内容有变化，写回文件
            if content != original_content:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f'✅ 迁移完成: {file_path}')
                return True
            else:
                return False

        except Exception as e:
            print(f'❌ 迁移失败 {file_path}: {e}')
            return False

    def batch_migrate(self, files_to_migrate: List[str]) -> Dict[str, int]:
        """批量迁移"""
        results = {'migrated': 0, 'failed': 0, 'skipped': 0}

        for file_path in files_to_migrate:
            if self.migrate_file(file_path):
                results['migrated'] += 1
            else:
                results['skipped'] += 1

        return results


def main():
    """主函数"""
    migrator = InterfaceMigrator()

    # 这里需要手动指定需要迁移的文件列表
    # 在实际使用时，这个列表会从分析结果中生成
    files_to_migrate = [
        # ComponentFactory 文件
        'cache/core/optimizer_components.py',
        'cache/core/service_components.py',
        'cache/services/client_components.py',
        # 添加其他需要迁移的文件...
    ]

    if files_to_migrate:
        print(f'开始迁移 {len(files_to_migrate)} 个文件...')
        results = migrator.batch_migrate(files_to_migrate)

        print(f'\n迁移结果:')
        print(f'  成功迁移: {results["migrated"]} 个文件')
        print(f'  跳过: {results["skipped"]} 个文件')
        print(f'  失败: {results["failed"]} 个文件')
    else:
        print('没有找到需要迁移的文件')


if __name__ == "__main__":
    main()
