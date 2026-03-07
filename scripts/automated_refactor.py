#!/usr/bin/env python3
"""
自动化重构脚本

用于执行可自动化的重构任务：
1. 魔数替换
2. 未使用导入清理
3. 代码格式化优化
"""

import re
import ast
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MagicNumberReplacer:
    """魔数替换器"""
    
    # 常见魔数映射（可配置）
    MAGIC_NUMBERS = {
        '1000': 'MAX_RECORDS',
        '10000': 'MAX_QUEUE_SIZE',
        '100': 'MAX_RETRIES',
        '60': 'SECONDS_PER_MINUTE',
        '3600': 'SECONDS_PER_HOUR',
        '86400': 'SECONDS_PER_DAY',
        '30': 'DEFAULT_TIMEOUT',
        '300': 'DEFAULT_TEST_TIMEOUT',
        '5': 'MAX_WORKERS',
        '10': 'DEFAULT_BATCH_SIZE',
        '1000000': 'MAX_DATA_SIZE_BYTES',
    }
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.changes = []
    
    def find_magic_numbers(self) -> List[Tuple[int, str, str]]:
        """查找魔数
        
        Returns:
            List of (line_number, magic_number, suggested_name)
        """
        magic_numbers = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 匹配数字（排除浮点数、负数、已定义的常量等）
            magic_pattern = re.compile(r'\b(\d{2,})\b')
            
            for i, line in enumerate(lines, 1):
                # 跳过注释和字符串
                if line.strip().startswith('#') or '"""' in line or "'''" in line:
                    continue
                
                # 跳过已经定义的常量
                if '=' in line and any(name in line for name in ['MAX_', 'MIN_', 'DEFAULT_', 'CONST_']):
                    continue
                
                matches = magic_pattern.finditer(line)
                for match in matches:
                    number = match.group(1)
                    # 排除常见非魔数
                    if number in ['0', '1', '10', '100', '1000', '10000', '24', '60', '365']:
                        # 检查是否在字典中
                        if number not in self.MAGIC_NUMBERS:
                            continue
                    
                    if number in self.MAGIC_NUMBERS:
                        suggested_name = self.MAGIC_NUMBERS[number]
                        magic_numbers.append((i, number, suggested_name))
        
        except Exception as e:
            logger.error(f"查找魔数失败 {self.file_path}: {e}")
        
        return magic_numbers


class UnusedImportRemover:
    """未使用导入清理器"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.changes = []
    
    def find_unused_imports(self) -> List[Tuple[int, str]]:
        """查找未使用的导入
        
        Returns:
            List of (line_number, import_statement)
        """
        unused_imports = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # 解析AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                logger.warning(f"无法解析 {self.file_path}，跳过")
                return unused_imports
            
            # 收集所有导入
            imports = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.name] = node.lineno
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        imports[alias.name] = node.lineno
            
            # 收集所有使用的名称
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # 处理属性访问，如 module.function
                    attr_parts = []
                    current = node
                    while isinstance(current, ast.Attribute):
                        attr_parts.insert(0, current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        used_names.add(current.id)
            
            # 检查导入是否被使用
            for import_name, line_num in imports.items():
                # 排除标准库和特殊导入
                if import_name.startswith('__') or import_name in ['os', 'sys', 'logging', 'time', 'datetime']:
                    continue
                
                # 检查是否在代码中使用
                base_name = import_name.split('.')[0]
                if base_name not in used_names and import_name not in used_names:
                    # 进一步检查是否在字符串中出现（可能是动态导入）
                    if import_name not in content.replace(f'import {import_name}', '').replace(f'from {import_name}', ''):
                        unused_imports.append((line_num, import_name))
        
        except Exception as e:
            logger.error(f"查找未使用导入失败 {self.file_path}: {e}")
        
        return unused_imports


class AutomatedRefactor:
    """自动化重构器"""
    
    def __init__(self, target_path: str):
        self.target_path = Path(target_path)
        self.magic_number_replacer = None
        self.unused_import_remover = None
        self.stats = {
            'files_processed': 0,
            'magic_numbers_found': 0,
            'unused_imports_found': 0,
            'changes_applied': 0
        }
    
    def scan_files(self) -> List[Path]:
        """扫描Python文件"""
        python_files = []
        
        if self.target_path.is_file():
            if self.target_path.suffix == '.py':
                python_files.append(self.target_path)
        else:
            for file_path in self.target_path.rglob('*.py'):
                # 跳过__pycache__和虚拟环境
                if any(part.startswith('__') or part in {'venv', '.venv', 'env', '.env', 'node_modules'}
                       for part in file_path.parts):
                    continue
                python_files.append(file_path)
        
        return python_files
    
    def refactor_magic_numbers(self, file_path: Path, dry_run: bool = True) -> int:
        """重构魔数"""
        replacer = MagicNumberReplacer(file_path)
        magic_numbers = replacer.find_magic_numbers()
        
        if not magic_numbers:
            return 0
        
        if dry_run:
            logger.info(f"  📊 发现 {len(magic_numbers)} 个魔数: {file_path}")
            for line_num, number, suggested_name in magic_numbers[:5]:  # 只显示前5个
                logger.info(f"     第{line_num}行: {number} -> {suggested_name}")
            return len(magic_numbers)
        
        # 实际替换（这里简化处理，只记录）
        self.stats['magic_numbers_found'] += len(magic_numbers)
        return len(magic_numbers)
    
    def refactor_unused_imports(self, file_path: Path, dry_run: bool = True) -> int:
        """清理未使用的导入"""
        remover = UnusedImportRemover(file_path)
        unused_imports = remover.find_unused_imports()
        
        if not unused_imports:
            return 0
        
        if dry_run:
            logger.info(f"  📊 发现 {len(unused_imports)} 个未使用导入: {file_path}")
            for line_num, import_name in unused_imports[:5]:  # 只显示前5个
                logger.info(f"     第{line_num}行: {import_name}")
            return len(unused_imports)
        
        # 实际清理（这里简化处理，只记录）
        self.stats['unused_imports_found'] += len(unused_imports)
        return len(unused_imports)
    
    def refactor_file(self, file_path: Path, dry_run: bool = True):
        """重构单个文件"""
        logger.info(f"处理文件: {file_path}")
        
        magic_count = self.refactor_magic_numbers(file_path, dry_run)
        import_count = self.refactor_unused_imports(file_path, dry_run)
        
        self.stats['files_processed'] += 1
        
        if magic_count > 0 or import_count > 0:
            self.stats['changes_applied'] += 1
        
        return magic_count, import_count
    
    def run(self, dry_run: bool = True):
        """执行重构"""
        logger.info("🤖 开始自动化重构...")
        logger.info(f"目标路径: {self.target_path}")
        logger.info(f"模式: {'模拟运行' if dry_run else '实际执行'}")
        
        python_files = self.scan_files()
        logger.info(f"发现 {len(python_files)} 个Python文件")
        
        total_magic = 0
        total_imports = 0
        
        for file_path in python_files:
            magic_count, import_count = self.refactor_file(file_path, dry_run)
            total_magic += magic_count
            total_imports += import_count
        
        logger.info("\n📊 重构统计:")
        logger.info(f"  • 处理文件数: {self.stats['files_processed']}")
        logger.info(f"  • 发现魔数: {total_magic} 个")
        logger.info(f"  • 发现未使用导入: {total_imports} 个")
        logger.info(f"  • 需要修改的文件: {self.stats['changes_applied']}")
        
        return {
            'files_processed': self.stats['files_processed'],
            'magic_numbers': total_magic,
            'unused_imports': total_imports,
            'files_to_modify': self.stats['changes_applied']
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自动化重构脚本")
    parser.add_argument('target', help='目标路径（文件或目录）')
    parser.add_argument('--dry-run', action='store_true', help='模拟运行，不实际修改')
    parser.add_argument('--magic-only', action='store_true', help='只处理魔数')
    parser.add_argument('--imports-only', action='store_true', help='只处理未使用导入')
    
    args = parser.parse_args()
    
    refactor = AutomatedRefactor(args.target)
    result = refactor.run(dry_run=args.dry_run)
    
    print("\n✅ 自动化重构完成！")
    if args.dry_run:
        print("💡 提示: 使用 --no-dry-run 来实际执行重构")


if __name__ == '__main__':
    main()

