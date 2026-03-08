#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
其他层级修复脚本 - Phase 2 P1
用于迁移其他目录到正确的层级

功能:
1. 迁移 ai_quality → ml/quality
2. 迁移 api → gateway/api
3. 迁移 async_processor → infrastructure/async
4. 迁移 distributed → infrastructure/distributed
5. 迁移 pipeline → data/pipeline
6. 迁移 boundary → core/boundary
7. 迁移 interfaces → core/interfaces
8. 迁移 mobile → gateway/mobile
9. 迁移 rl → ml/rl
10. 迁移 rollback → infrastructure/rollback
11. 迁移 testing → infrastructure/testing
12. 迁移 tools → infrastructure/devtools
13. 迁移 web → gateway/web

作者: RQA2025 Architecture Team
日期: 2026-03-08
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict


class OtherLayerFixer:
    """
    其他层级修复器
    
    负责将分散的目录迁移到正确的架构层级
    """
    
    # 定义需要迁移的目录映射
    MIGRATION_MAP = {
        # P1 优先级
        'src/ai_quality': 'src/ml/quality',
        'src/api': 'src/gateway/api_integration',
        'src/async_processor': 'src/infrastructure/async',
        'src/distributed': 'src/infrastructure/distributed',
        'src/pipeline': 'src/data/pipeline',
        
        # P2 优先级
        'src/boundary': 'src/core/boundary',
        'src/interfaces': 'src/core/interfaces',
        'src/mobile': 'src/gateway/mobile',
        'src/rl': 'src/ml/rl',
        'src/rollback': 'src/infrastructure/rollback',
        'src/testing': 'src/infrastructure/testing',
        'src/tools': 'src/infrastructure/devtools',
        'src/web': 'src/gateway/web',
    }
    
    # 导入替换规则
    IMPORT_PATTERNS: Dict[str, tuple] = {
        'ai_quality': (r'from\s+src\.ai_quality', r'import\s+src\.ai_quality',
                      'from src.ml.quality', 'import src.ml.quality'),
        'api': (r'from\s+src\.api', r'import\s+src\.api',
                'from src.gateway.api_integration', 'import src.gateway.api_integration'),
        'async_processor': (r'from\s+src\.async_processor', r'import\s+src\.async_processor',
                           'from src.infrastructure.async', 'import src.infrastructure.async'),
        'distributed': (r'from\s+src\.distributed', r'import\s+src\.distributed',
                       'from src.infrastructure.distributed', 'import src.infrastructure.distributed'),
        'pipeline': (r'from\s+src\.pipeline', r'import\s+src\.pipeline',
                    'from src.data.pipeline', 'import src.data.pipeline'),
        'boundary': (r'from\s+src\.boundary', r'import\s+src\.boundary',
                    'from src.core.boundary', 'import src.core.boundary'),
        'interfaces': (r'from\s+src\.interfaces', r'import\s+src\.interfaces',
                      'from src.core.interfaces', 'import src.core.interfaces'),
        'mobile': (r'from\s+src\.mobile', r'import\s+src\.mobile',
                  'from src.gateway.mobile', 'import src.gateway.mobile'),
        'rl': (r'from\s+src\.rl', r'import\s+src\.rl',
              'from src.ml.rl', 'import src.ml.rl'),
        'rollback': (r'from\s+src\.rollback', r'import\s+src\.rollback',
                    'from src.infrastructure.rollback', 'import src.infrastructure.rollback'),
        'testing': (r'from\s+src\.testing', r'import\s+src\.testing',
                   'from src.infrastructure.testing', 'import src.infrastructure.testing'),
        'tools': (r'from\s+src\.tools', r'import\s+src\.tools',
                 'from src.infrastructure.devtools', 'import src.infrastructure.devtools'),
        'web': (r'from\s+src\.web', r'import\s+src\.web',
               'from src.gateway.web', 'import src.gateway.web'),
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
        
    def check_source_exists(self) -> Dict[str, bool]:
        """
        检查源目录是否存在
        
        Returns:
            各目录存在状态字典
        """
        print("\n🔍 检查源目录...")
        status = {}
        for src, dst in self.MIGRATION_MAP.items():
            src_path = self.project_root / src
            exists = src_path.exists()
            status[src] = exists
            if exists:
                print(f"  ✓ {src} 存在")
            else:
                print(f"  ⚠ {src} 不存在 (跳过)")
        return status
        
    def migrate_directories(self, status: Dict[str, bool]):
        """
        迁移目录结构
        
        Args:
            status: 目录存在状态字典
        """
        print("\n📁 迁移目录结构...")
        
        for src_rel, dst_rel in self.MIGRATION_MAP.items():
            if not status.get(src_rel, False):
                continue
                
            src_path = self.project_root / src_rel
            dst_path = self.project_root / dst_rel
            
            try:
                # 如果目标目录已存在，合并内容
                if dst_path.exists():
                    self._merge_directories(src_path, dst_path)
                    shutil.rmtree(src_path)
                    self._log_change(f"合并并删除 {src_rel} → {dst_rel}")
                else:
                    # 确保父目录存在
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
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
        """
        print("\n📝 更新导入语句...")
        
        # 遍历所有 Python 文件
        python_files = list(self.src_path.rglob('*.py'))
        updated_count = 0
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                original_content = content
                
                # 应用所有替换规则
                for module_name, patterns in self.IMPORT_PATTERNS.items():
                    from_pattern, import_pattern, from_repl, import_repl = patterns
                    content = re.sub(from_pattern, from_repl, content)
                    content = re.sub(import_pattern, import_repl, content)
                    
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
                    module_name = dst_path.name
                    parent_name = dst_path.parent.name
                    init_file.write_text(
                        f'"""\n'
                        f'{module_name} 模块\n'
                        f'\n'
                        f'从 src 根目录迁移至 {parent_name} 层\n'
                        f'"""\n',
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
        summary.append("其他层级修复摘要")
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
        print("开始其他层级修复")
        print("=" * 60)
        
        # 检查源目录
        status = self.check_source_exists()
        
        # 迁移目录
        self.migrate_directories(status)
        
        # 更新导入
        self.update_imports()
        
        # 创建 __init__.py
        self.create_init_files()
        
        # 打印摘要
        print("\n" + self.generate_summary())
        
        # 保存日志
        log_file = self.project_root / 'reports' / 'other_layer_fix_log.txt'
        log_file.write_text(self.generate_summary(), encoding='utf-8')
        print(f"\n📄 日志已保存: {log_file}")


def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='修复其他层级架构问题 - 将分散目录迁移至正确层级'
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='项目根目录路径 (默认: 当前目录)'
    )
    
    args = parser.parse_args()
    
    fixer = OtherLayerFixer(args.project_root)
    fixer.run()


if __name__ == '__main__':
    main()
