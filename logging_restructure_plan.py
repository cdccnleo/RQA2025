"""
Logging模块重构计划 - Phase 1

目标: 将19个子目录优化为8-10个功能明确的目录
"""

import os
import shutil
from pathlib import Path
from typing import Dict


class LoggingRestructurePlan:
    """Logging模块重构计划"""

    def __init__(self):
        self.logging_dir = Path('src/infrastructure/logging')
        self.backup_dir = Path('src/infrastructure/logging_backup')

        # 重构映射方案
        self.restructure_map = {
            # 保留的核心目录
            'core': ['core', 'engine'],  # 合并engine到core
            'foundation': ['foundation'],
            'config': ['config'],
            'utils': ['utils', 'formatters'],  # 合并formatters到utils
            'system': ['system', 'integrations', 'storages'],  # 合并集成和存储
            'business': ['business'],
            'services': ['services'],
            'plugins': ['plugins'],
            'distributed': ['distributed'],
            'data': ['data'],
            'handlers': ['handlers'],
            'monitors': ['monitors'],
            'processors': ['processors'],
            'security': ['security']
        }

    def analyze_current_structure(self) -> Dict:
        """分析当前目录结构"""
        structure = {}
        for item in os.listdir(self.logging_dir):
            item_path = self.logging_dir / item
            if item_path.is_dir() and not item.startswith('.'):
                py_files = list(item_path.glob('*.py'))
                structure[item] = {
                    'file_count': len(py_files),
                    'files': [f.name for f in py_files]
                }
        return structure

    def create_backup(self):
        """创建备份"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        shutil.copytree(self.logging_dir, self.backup_dir)
        print(f'✅ 已创建备份: {self.backup_dir}')

    def merge_directories(self):
        """执行目录合并"""
        print('🔄 开始目录合并...')

        # 创建目标目录
        for target_dir in self.restructure_map.keys():
            target_path = self.logging_dir / target_dir
            target_path.mkdir(exist_ok=True)

        # 移动文件
        for target_dir, source_dirs in self.restructure_map.items():
            target_path = self.logging_dir / target_dir

            for source_dir in source_dirs:
                source_path = self.logging_dir / source_dir
                if source_path.exists() and source_dir != target_dir:
                    print(f'  合并 {source_dir}/ → {target_dir}/')

                    # 移动所有文件
                    for file_path in source_path.glob('*'):
                        if file_path.is_file():
                            dest_path = target_path / file_path.name
                            if dest_path.exists():
                                # 处理文件名冲突
                                base = dest_path.stem
                                ext = dest_path.suffix
                                counter = 1
                                while dest_path.exists():
                                    dest_path = target_path / f"{base}_{source_dir}_{counter}{ext}"
                                    counter += 1

                            shutil.move(str(file_path), str(dest_path))
                            print(f'    移动: {file_path.name}')

                    # 删除空目录
                    if not list(source_path.iterdir()):
                        source_path.rmdir()
                        print(f'    删除空目录: {source_dir}/')

    def update_imports(self):
        """更新导入语句"""
        print('🔄 更新导入语句...')

        for root, dirs, files in os.walk(self.logging_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 查找并更新导入语句
                        import_updates = {
                            'from .engine': 'from .core',
                            'from ..engine': 'from ..core',
                            'from .formatters': 'from .utils',
                            'from ..formatters': 'from ..utils',
                            'from .integrations': 'from .system',
                            'from ..integrations': 'from ..system',
                            'from .storages': 'from .system',
                            'from ..storages': 'from ..system'
                        }

                        updated_content = content
                        for old_import, new_import in import_updates.items():
                            updated_content = updated_content.replace(old_import, new_import)

                        if updated_content != content:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(updated_content)
                            print(f'  更新导入: {file_path.relative_to(self.logging_dir)}')

                    except Exception as e:
                        print(f'  跳过文件: {file_path.name} ({e})')

    def validate_restructure(self) -> Dict:
        """验证重构结果"""
        print('🔍 验证重构结果...')

        new_structure = self.analyze_current_structure()
        validation = {
            'old_structure': self.analyze_current_structure(),
            'new_structure': new_structure,
            'directories_before': len([d for d in self.backup_dir.iterdir() if d.is_dir()]),
            'directories_after': len([d for d in self.logging_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]),
            'files_preserved': sum(info['file_count'] for info in new_structure.values())
        }

        print(f'  目录数量: {validation["directories_before"]} → {validation["directories_after"]}')
        print(f'  文件总数: {validation["files_preserved"]}')

        return validation

    def generate_report(self, validation: Dict):
        """生成重构报告"""
        print('\n📊 Logging模块重构报告')
        print('=' * 40)

        print(f'原始目录数: {validation["directories_before"]}')
        print(f'重构后目录数: {validation["directories_after"]}')
        print(f'文件总数: {validation["files_preserved"]}')

        print('\n🏗️ 新目录结构:')
        for dir_name, info in validation['new_structure'].items():
            print(f'  {dir_name}/ ({info["file_count"]} 个文件)')

        improvement = validation["directories_before"] - validation["directories_after"]
        print(f'\\n✅ 优化效果: 减少了 {improvement} 个目录')

        print('\\n💡 重构映射:')
        for target, sources in self.restructure_map.items():
            if len(sources) > 1:
                print(f'  {target}/ ← {" + ".join(sources)}')

    def execute_restructure(self):
        """执行完整重构"""
        print('🚀 开始Logging模块重构')
        print('=' * 40)

        # 1. 创建备份
        self.create_backup()

        # 2. 合并目录
        self.merge_directories()

        # 3. 更新导入
        self.update_imports()

        # 4. 验证结果
        validation = self.validate_restructure()

        # 5. 生成报告
        self.generate_report(validation)

        print('\\n✅ Logging模块重构完成！')


def main():
    """主函数"""
    plan = LoggingRestructurePlan()
    plan.execute_restructure()


if __name__ == "__main__":
    main()
