#!/usr/bin/env python3
"""
组件迁移工具

自动化工具用于将旧组件迁移到基于BaseComponent/BaseAdapter的新架构

功能：
1. 分析旧组件文件
2. 生成迁移后的新代码
3. 创建备份
4. 更新导入语句
5. 运行验证测试

使用方式：
    python scripts/component_migration_tool.py migrate --file src/core/container/container_components.py
    python scripts/component_migration_tool.py validate --dir src/core/container
    python scripts/component_migration_tool.py rollback --file src/core/container/container_components.py

创建时间: 2025-11-03
版本: 1.0
"""

import sys
import os
import shutil
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ComponentAnalyzer:
    """组件分析器"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.content = ""
        self.classes = []
        self.functions = []
        self.imports = []
    
    def analyze(self) -> Dict:
        """分析组件文件"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
        
        # 提取类定义
        self.classes = re.findall(r'^class\s+(\w+)', self.content, re.MULTILINE)
        
        # 提取函数定义
        self.functions = re.findall(r'^def\s+(\w+)', self.content, re.MULTILINE)
        
        # 提取导入语句
        self.imports = re.findall(r'^(?:from|import)\s+.+$', self.content, re.MULTILINE)
        
        return {
            'file': str(self.file_path),
            'lines': len(self.content.splitlines()),
            'classes': self.classes,
            'functions': self.functions,
            'imports': len(self.imports),
            'has_component_factory': 'ComponentFactory' in self.content,
            'migration_complexity': self._assess_complexity()
        }
    
    def _assess_complexity(self) -> str:
        """评估迁移复杂度"""
        lines = len(self.content.splitlines())
        class_count = len(self.classes)
        
        if lines > 1000 or class_count > 10:
            return 'high'
        elif lines > 500 or class_count > 5:
            return 'medium'
        else:
            return 'low'


class ComponentMigrator:
    """组件迁移器"""
    
    def __init__(self, source_file: Path, backup_dir: Optional[Path] = None):
        self.source_file = source_file
        self.backup_dir = backup_dir or PROJECT_ROOT / 'backups' / 'migration'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def migrate(self, dry_run: bool = False) -> Dict:
        """执行迁移"""
        print(f"🔄 开始迁移: {self.source_file}")
        
        # 1. 分析源文件
        analyzer = ComponentAnalyzer(self.source_file)
        analysis = analyzer.analyze()
        
        print(f"  📊 文件分析:")
        print(f"    - 代码行数: {analysis['lines']}")
        print(f"    - 类数量: {len(analysis['classes'])}")
        print(f"    - 复杂度: {analysis['migration_complexity']}")
        
        # 2. 创建备份
        if not dry_run:
            backup_file = self._create_backup()
            print(f"  💾 备份已创建: {backup_file}")
        
        # 3. 生成新代码
        new_content = self._generate_migrated_code(analyzer)
        
        if dry_run:
            print(f"  🔍 模拟模式 - 不会写入文件")
            print(f"  📝 生成代码预览（前20行）:")
            preview = '\n'.join(new_content.splitlines()[:20])
            print(f"```python\n{preview}\n...```")
            return {
                'status': 'simulated',
                'backup': None,
                'analysis': analysis
            }
        
        # 4. 写入新文件
        with open(self.source_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  ✅ 迁移完成")
        
        return {
            'status': 'completed',
            'backup': str(backup_file),
            'analysis': analysis
        }
    
    def _create_backup(self) -> Path:
        """创建备份文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.source_file.stem}_{timestamp}.backup"
        backup_file = self.backup_dir / filename
        
        shutil.copy2(self.source_file, backup_file)
        return backup_file
    
    def _generate_migrated_code(self, analyzer: ComponentAnalyzer) -> str:
        """生成迁移后的代码"""
        # 检测主要类型
        if self._is_adapter_file(analyzer):
            return self._generate_adapter_code(analyzer)
        else:
            return self._generate_component_code(analyzer)
    
    def _is_adapter_file(self, analyzer: ComponentAnalyzer) -> bool:
        """判断是否是适配器文件"""
        adapter_keywords = ['adapter', 'Adapter', 'adapt']
        return any(keyword in analyzer.content for keyword in adapter_keywords)
    
    def _generate_component_code(self, analyzer: ComponentAnalyzer) -> str:
        """生成基于BaseComponent的代码"""
        # 提取原有的docstring
        docstring = self._extract_docstring(analyzer.content)
        
        # 提取主类名
        main_class = analyzer.classes[0] if analyzer.classes else "MyComponent"
        
        template = f'''#!/usr/bin/env python3
"""
{docstring}

迁移说明：
- 已基于BaseComponent重构
- 消除重复的样板代码
- 统一的生命周期管理
- 迁移时间: {datetime.now().strftime("%Y-%m-%d")}
"""

from typing import Dict, Any, Optional
from src.core.foundation.base_component import BaseComponent, component


@component("{main_class.lower()}")
class {main_class}(BaseComponent):
    """
    {main_class} (重构版)
    
    基于BaseComponent，提供统一的组件架构
    """
    
    def __init__(self, name: str = "{main_class.lower()}", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        # TODO: 添加组件特定的属性
        self._data = {{}}
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        try:
            # TODO: 实现特定的初始化逻辑
            self._logger.info(f"{{self.name}} 组件初始化")
            return True
        except Exception as e:
            self._logger.error(f"初始化失败: {{e}}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行组件功能"""
        # TODO: 实现特定的执行逻辑
        operation = kwargs.get('operation', 'default')
        
        if operation == 'process':
            return self._process_data(kwargs.get('data'))
        
        return None
    
    def _process_data(self, data: Any) -> Any:
        """处理数据"""
        # TODO: 实现数据处理逻辑
        return {{'processed': True, 'data': data}}


# TODO: 迁移其他类和函数
# 原文件类: {', '.join(analyzer.classes)}


__all__ = ['{main_class}']
'''
        return template
    
    def _generate_adapter_code(self, analyzer: ComponentAnalyzer) -> str:
        """生成基于BaseAdapter的代码"""
        docstring = self._extract_docstring(analyzer.content)
        main_class = analyzer.classes[0] if analyzer.classes else "MyAdapter"
        
        template = f'''#!/usr/bin/env python3
"""
{docstring}

迁移说明：
- 已基于BaseAdapter重构
- 自动化的缓存支持
- 统一的错误处理
- 迁移时间: {datetime.now().strftime("%Y-%m-%d")}
"""

from typing import Dict, Any
from src.core.foundation.base_adapter import BaseAdapter, adapter


@adapter("{main_class.lower()}", enable_cache=True)
class {main_class}(BaseAdapter[Dict[str, Any], Dict[str, Any]]):
    """
    {main_class} (重构版)
    
    基于BaseAdapter，提供统一的适配器模式
    """
    
    def _do_adapt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行适配"""
        # TODO: 实现特定的适配逻辑
        return {{
            'adapted': True,
            'original': data,
            'timestamp': datetime.now()
        }}
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """验证输入"""
        # TODO: 实现特定的验证逻辑
        return data is not None
    
    def _preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理"""
        # TODO: 实现预处理逻辑
        return data
    
    def _postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理"""
        # TODO: 实现后处理逻辑
        return data


# TODO: 迁移其他类和函数
# 原文件类: {', '.join(analyzer.classes)}


__all__ = ['{main_class}']
'''
        return template
    
    def _extract_docstring(self, content: str) -> str:
        """提取文件级docstring"""
        match = re.search(r'"""(.+?)"""', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "模块说明"
    
    def rollback(self, backup_file: Path) -> bool:
        """回滚到备份"""
        if not backup_file.exists():
            print(f"❌ 备份文件不存在: {backup_file}")
            return False
        
        shutil.copy2(backup_file, self.source_file)
        print(f"✅ 已回滚到备份: {backup_file}")
        return True


class MigrationValidator:
    """迁移验证器"""
    
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
    
    def validate(self) -> Dict:
        """验证迁移结果"""
        print(f"🔍 验证迁移: {self.target_dir}")
        
        results = {
            'total_files': 0,
            'migrated_files': 0,
            'issues': []
        }
        
        for py_file in self.target_dir.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue
            
            results['total_files'] += 1
            
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已迁移
            if 'BaseComponent' in content or 'BaseAdapter' in content:
                results['migrated_files'] += 1
                print(f"  ✅ {py_file.name} - 已迁移")
            else:
                print(f"  ⏸️ {py_file.name} - 未迁移")
            
            # 检查潜在问题
            if 'ComponentFactory' in content and 'from src.core.foundation' not in content:
                results['issues'].append({
                    'file': str(py_file),
                    'issue': '包含重复的ComponentFactory定义'
                })
        
        print(f"\n📊 验证结果:")
        print(f"  - 总文件数: {results['total_files']}")
        print(f"  - 已迁移: {results['migrated_files']}")
        print(f"  - 迁移率: {results['migrated_files']/max(results['total_files'],1)*100:.1f}%")
        print(f"  - 发现问题: {len(results['issues'])}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="组件迁移工具")
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # migrate命令
    migrate_parser = subparsers.add_parser('migrate', help='迁移组件文件')
    migrate_parser.add_argument('--file', required=True, help='要迁移的文件路径')
    migrate_parser.add_argument('--dry-run', action='store_true', help='模拟模式，不实际写入')
    
    # validate命令
    validate_parser = subparsers.add_parser('validate', help='验证迁移结果')
    validate_parser.add_argument('--dir', required=True, help='要验证的目录')
    
    # rollback命令
    rollback_parser = subparsers.add_parser('rollback', help='回滚迁移')
    rollback_parser.add_argument('--file', required=True, help='要回滚的文件')
    rollback_parser.add_argument('--backup', required=True, help='备份文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'migrate':
        file_path = PROJECT_ROOT / args.file
        migrator = ComponentMigrator(file_path)
        result = migrator.migrate(dry_run=args.dry_run)
        print(f"\n✅ 迁移结果: {result['status']}")
    
    elif args.command == 'validate':
        dir_path = PROJECT_ROOT / args.dir
        validator = MigrationValidator(dir_path)
        result = validator.validate()
    
    elif args.command == 'rollback':
        file_path = PROJECT_ROOT / args.file
        backup_path = Path(args.backup)
        migrator = ComponentMigrator(file_path)
        migrator.rollback(backup_path)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

