#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
架构层级修复脚本 - Phase 2
用于修复 core 层内部子目录合并到 infrastructure 层

功能:
1. 迁移 core/utils → infrastructure/utils
2. 迁移 core/automation → infrastructure/automation
3. 迁移 core/resilience → infrastructure/resilience
4. 迁移 core/database → infrastructure/database
5. 迁移 core/integration → infrastructure/integration
6. 迁移 core/orchestration → infrastructure/orchestration
7. 更新所有相关导入语句

作者: RQA2025 Architecture Team
日期: 2026-03-08
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Tuple


class CoreLayerFixer:
    """
    Core 层修复器
    
    负责将 core 层中属于基础设施的子模块迁移到 infrastructure 层
    """
    
    # 定义需要迁移的目录映射
    MIGRATION_MAP = {
        'src/core/utils': 'src/infrastructure/utils',
        'src/core/automation': 'src/infrastructure/automation',
        'src/core/resilience': 'src/infrastructure/resilience',
        'src/core/database': 'src/infrastructure/database',
        'src/core/integration': 'src/infrastructure/integration',
        'src/core/orchestration': 'src/infrastructure/orchestration',
    }
    
    def __init__(self, project_root: str = None):
        """
        初始化修复器
        
        Args:
            project_root: 项目根目录路径，默认为当前工作目录
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.src_path = self.project_root / 'src'
        self.changes_log: List[str] = []
        self.errors: List[str] = []
        
    def _log_change(self, message: str):
        """记录变更日志"""
        self.changes_log.append(message)
        print(f"  ✓ {message}")
        
    def _log_error(self, message: str):
        """记录错误日志"""
        self.errors.append(message)
        print(f"  ✗ {message}")
        
    def check_source_exists(self) -> bool:
        """
        检查源目录是否存在
        
        Returns:
            是否所有源目录都存在
        """
        print("\n🔍 检查源目录...")
        all_exist = True
        for src, dst in self.MIGRATION_MAP.items():
            src_path = self.project_root / src
            if src_path.exists():
                print(f"  ✓ {src} 存在")
            else:
                print(f"  ⚠ {src} 不存在 (跳过)")
                all_exist = False
        return all_exist
        
    def migrate_directories(self):
        """
        迁移目录结构
        
        将 core 层中的基础设施相关子目录移动到 infrastructure 层
        """
        print("\n📁 迁移目录结构...")
        
        for src_rel, dst_rel in self.MIGRATION_MAP.items():
            src_path = self.project_root / src_rel
            dst_path = self.project_root / dst_rel
            
            if not src_path.exists():
                self._log_change(f"跳过 {src_rel} (不存在)")
                continue
                
            try:
                # 如果目标目录已存在，合并内容
                if dst_path.exists():
                    self._merge_directories(src_path, dst_path)
                    shutil.rmtree(src_path)
                    self._log_change(f"合并并删除 {src_rel} → {dst_rel}")
                else:
                    # 移动整个目录
                    shutil.move(str(src_path), str(dst_path))
                    self._log_change(f"迁移 {src_rel} → {dst_rel}")
                    
            except Exception as e:
                self._log_error(f"迁移失败 {src_rel}: {e}")
                
    def _merge_directories(self, src: Path, dst: Path):
        """
        合并两个目录
        
        Args:
            src: 源目录
            dst: 目标目录
        """
        for item in src.iterdir():
            dst_item = dst / item.name
            if item.is_file():
                if dst_item.exists():
                    # 文件已存在，保留目标文件
                    pass
                else:
                    shutil.move(str(item), str(dst_item))
            elif item.is_dir():
                if dst_item.exists():
                    self._merge_directories(item, dst_item)
                else:
                    shutil.move(str(item), str(dst_item))
                    
    def update_imports(self):
        """
        更新导入语句
        
        将所有从 core 导入已迁移模块的语句更新为从 infrastructure 导入
        """
        print("\n📝 更新导入语句...")
        
        # 定义导入替换规则
        import_patterns = [
            # core.utils → infrastructure.utils
            (r'from\s+src\.core\.utils', 'from src.infrastructure.utils'),
            (r'import\s+src\.core\.utils', 'import src.infrastructure.utils'),
            
            # core.automation → infrastructure.automation
            (r'from\s+src\.core\.automation', 'from src.infrastructure.automation'),
            (r'import\s+src\.core\.automation', 'import src.infrastructure.automation'),
            
            # core.resilience → infrastructure.resilience
            (r'from\s+src\.core\.resilience', 'from src.infrastructure.resilience'),
            (r'import\s+src\.core\.resilience', 'import src.infrastructure.resilience'),
            
            # core.database → infrastructure.database
            (r'from\s+src\.core\.database', 'from src.infrastructure.database'),
            (r'import\s+src\.core\.database', 'import src.infrastructure.database'),
            
            # core.integration → infrastructure.integration
            (r'from\s+src\.core\.integration', 'from src.infrastructure.integration'),
            (r'import\s+src\.core\.integration', 'import src.infrastructure.integration'),
            
            # core.orchestration → infrastructure.orchestration
            (r'from\s+src\.core\.orchestration', 'from src.infrastructure.orchestration'),
            (r'import\s+src\.core\.orchestration', 'import src.infrastructure.orchestration'),
        ]
        
        # 遍历所有 Python 文件
        python_files = list(self.src_path.rglob('*.py'))
        updated_count = 0
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                original_content = content
                
                # 应用所有替换规则
                for pattern, replacement in import_patterns:
                    content = re.sub(pattern, replacement, content)
                    
                # 如果有变更，写回文件
                if content != original_content:
                    py_file.write_text(content, encoding='utf-8')
                    updated_count += 1
                    
            except Exception as e:
                self._log_error(f"更新导入失败 {py_file}: {e}")
                
        self._log_change(f"更新了 {updated_count} 个文件的导入语句")
        
    def create_init_files(self):
        """
        创建必要的 __init__.py 文件
        """
        print("\n📄 创建 __init__.py 文件...")
        
        for dst_rel in self.MIGRATION_MAP.values():
            dst_path = self.project_root / dst_rel
            init_file = dst_path / '__init__.py'
            
            if dst_path.exists() and not init_file.exists():
                try:
                    init_file.write_text(
                        '"""\n' +
                        f'{dst_path.name} 模块\n' +
                        '\n' +
                        '从 core 层迁移至 infrastructure 层\n' +
                        '"""\n',
                        encoding='utf-8'
                    )
                    self._log_change(f"创建 {dst_rel}/__init__.py")
                except Exception as e:
                    self._log_error(f"创建 __init__.py 失败 {dst_rel}: {e}")
                    
    def generate_summary(self) -> str:
        """
        生成修复摘要
        
        Returns:
            摘要文本
        """
        summary = []
        summary.append("=" * 60)
        summary.append("Core 层修复摘要")
        summary.append("=" * 60)
        summary.append(f"\n成功变更: {len(self.changes_log)} 项")
        summary.append(f"错误: {len(self.errors)} 项\n")
        
        if self.changes_log:
            summary.append("变更列表:")
            for change in self.changes_log:
                summary.append(f"  - {change}")
                
        if self.errors:
            summary.append("\n错误列表:")
            for error in self.errors:
                summary.append(f"  - {error}")
                
        return '\n'.join(summary)
        
    def run(self):
        """
        执行完整的修复流程
        """
        print("=" * 60)
        print("开始 Core 层修复")
        print("=" * 60)
        
        # 检查源目录
        self.check_source_exists()
        
        # 迁移目录
        self.migrate_directories()
        
        # 更新导入
        self.update_imports()
        
        # 创建 __init__.py
        self.create_init_files()
        
        # 打印摘要
        print("\n" + self.generate_summary())
        
        # 保存日志
        log_file = self.project_root / 'reports' / 'core_layer_fix_log.txt'
        log_file.write_text(self.generate_summary(), encoding='utf-8')
        print(f"\n📄 日志已保存: {log_file}")


def main():
    """
    主函数
    
    解析命令行参数并执行修复
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='修复 Core 层架构问题 - 将基础设施相关模块迁移至 infrastructure 层'
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='项目根目录路径 (默认: 当前目录)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行模式 - 只显示将要执行的操作而不实际执行'
    )
    
    args = parser.parse_args()
    
    fixer = CoreLayerFixer(args.project_root)
    fixer.run()


if __name__ == '__main__':
    main()
