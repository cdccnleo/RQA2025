#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动实现测试钩子脚本
为未实现测试钩子的基础设施模块自动添加钩子
"""

import os
import re
import ast
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class ModuleAnalysis:
    """模块分析结果"""
    name: str
    file_path: str
    has_test_hook: bool
    dependencies: List[str]
    constructor_params: List[str]
    needs_hook: bool

class AutoHookImplementer:
    """自动实现测试钩子"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"
        
        # 需要添加钩子的模块
        self.target_modules = {
            'monitoring/system_monitor.py': {
                'dependencies': ['psutil', 'os', 'threading'],
                'hook_params': ['psutil_mock', 'os_mock']
            },
            'monitoring/application_monitor.py': {
                'dependencies': ['influxdb_client'],
                'hook_params': ['influx_client_mock']
            },
            'database/database_manager.py': {
                'dependencies': ['json', 'pathlib'],
                'hook_params': ['file_reader_mock']
            }
        }

    def analyze_module(self, module_path: Path) -> ModuleAnalysis:
        """分析模块"""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 检查是否已有测试钩子
            has_test_hook = self._check_existing_hooks(tree)
            
            # 提取依赖
            dependencies = self._extract_dependencies(content)
            
            # 提取构造函数参数
            constructor_params = self._extract_constructor_params(tree)
            
            # 判断是否需要钩子
            needs_hook = not has_test_hook and any(
                dep in dependencies for dep in ['ConfigManager', 'psutil', 'influxdb_client']
            )
            
            return ModuleAnalysis(
                name=module_path.stem,
                file_path=str(module_path),
                has_test_hook=has_test_hook,
                dependencies=dependencies,
                constructor_params=constructor_params,
                needs_hook=needs_hook
            )
            
        except Exception as e:
            print(f"分析模块 {module_path} 时出错: {e}")
            return ModuleAnalysis(
                name=module_path.stem,
                file_path=str(module_path),
                has_test_hook=False,
                dependencies=[],
                constructor_params=[],
                needs_hook=False
            )

    def _check_existing_hooks(self, tree: ast.AST) -> bool:
        """检查是否已有测试钩子"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        # 检查是否有可选参数
                        for arg in item.args.args:
                            if arg.arg != 'self' and arg.arg != 'config':
                                return True
        return False

    def _extract_dependencies(self, content: str) -> List[str]:
        """提取依赖"""
        dependencies = []
        
        # 查找import语句
        import_patterns = [
            r'from\s+([\w.]+)\s+import',
            r'import\s+([\w.]+)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
        
        return dependencies

    def _extract_constructor_params(self, tree: ast.AST) -> List[str]:
        """提取构造函数参数"""
        params = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        for arg in item.args.args:
                            if arg.arg != 'self':
                                params.append(arg.arg)
                        break
        
        return params

    def generate_hook_implementation(self, module_path: Path, analysis: ModuleAnalysis) -> str:
        """生成测试钩子实现"""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分析当前构造函数
            tree = ast.parse(content)
            
            # 找到类定义
            class_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_def = node
                    break
            
            if not class_def:
                return content
            
            # 找到构造函数
            init_func = None
            for item in class_def.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    init_func = item
                    break
            
            if not init_func:
                return content
            
            # 生成新的构造函数
            new_init = self._generate_new_constructor(init_func, analysis)
            
            # 替换构造函数
            new_content = self._replace_constructor(content, init_func, new_init)
            
            return new_content
            
        except Exception as e:
            print(f"生成钩子实现时出错: {e}")
            return content

    def _generate_new_constructor(self, init_func: ast.FunctionDef, analysis: ModuleAnalysis) -> str:
        """生成新的构造函数"""
        # 基础参数
        params = ['self', 'config: Dict[str, Any]']
        
        # 添加测试钩子参数
        if 'ConfigManager' in analysis.dependencies:
            params.append('config_manager: Optional[ConfigManager] = None')
        
        if 'psutil' in analysis.dependencies:
            params.append('psutil_mock: Optional[Any] = None')
        
        if 'influxdb_client' in analysis.dependencies:
            params.append('influx_client_mock: Optional[Any] = None')
        
        # 生成构造函数体
        body_lines = [
            '"""',
            f'初始化{analysis.name}',
            ':param config: 系统配置',
        ]
        
        if 'ConfigManager' in analysis.dependencies:
            body_lines.append(':param config_manager: 可选的配置管理器实例，用于测试时注入mock对象')
        
        if 'psutil' in analysis.dependencies:
            body_lines.append(':param psutil_mock: 可选的psutil mock，用于测试时注入mock对象')
        
        if 'influxdb_client' in analysis.dependencies:
            body_lines.append(':param influx_client_mock: 可选的InfluxDB客户端mock，用于测试时注入mock对象')
        
        body_lines.extend([
            '"""',
            'self.config = config',
            '',
            '# 测试钩子：允许注入mock的依赖'
        ])
        
        if 'ConfigManager' in analysis.dependencies:
            body_lines.extend([
                'if config_manager is not None:',
                '    self.config_manager = config_manager',
                'else:',
                '    self.config_manager = ConfigManager(config)',
                ''
            ])
        
        if 'psutil' in analysis.dependencies:
            body_lines.extend([
                'if psutil_mock is not None:',
                '    self.psutil = psutil_mock',
                'else:',
                '    import psutil',
                '    self.psutil = psutil',
                ''
            ])
        
        if 'influxdb_client' in analysis.dependencies:
            body_lines.extend([
                'if influx_client_mock is not None:',
                '    self.influx_client = influx_client_mock',
                'else:',
                '    self.influx_client = None  # 延迟初始化',
                ''
            ])
        
        # 添加原始构造函数体
        original_body = self._extract_constructor_body(init_func)
        body_lines.extend(original_body)
        
        # 生成完整的构造函数
        param_str = ', '.join(params)
        body_str = '\n        '.join(body_lines)
        
        return f"""    def __init__({param_str}):
        {body_str}"""

    def _extract_constructor_body(self, init_func: ast.FunctionDef) -> List[str]:
        """提取构造函数体"""
        body_lines = []
        
        for node in init_func.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr != 'config':
                        body_lines.append(f'self.{target.attr} = {ast.unparse(node.value)}')
            elif isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    # 跳过ConfigManager的初始化
                    node_str = ast.unparse(node.value)
                    if 'ConfigManager' not in node_str:
                        body_lines.append(ast.unparse(node))
        
        return body_lines

    def _replace_constructor(self, content: str, old_init: ast.FunctionDef, new_init: str) -> str:
        """替换构造函数"""
        lines = content.split('\n')
        
        # 找到构造函数的位置
        start_line = old_init.lineno - 1
        end_line = old_init.end_lineno
        
        # 替换构造函数
        new_lines = lines[:start_line]
        new_lines.append(new_init)
        new_lines.extend(lines[end_line:])
        
        return '\n'.join(new_lines)

    def create_test_file(self, module_path: Path, analysis: ModuleAnalysis) -> str:
        """创建测试文件"""
        module_name = module_path.stem
        test_content = f'''import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.{module_path.parent.name}.{module_name} import {analysis.name.title().replace('_', '')}

@pytest.fixture
def mock_config_manager():
    """创建mock的ConfigManager"""
    mock_cm = MagicMock()
    mock_cm.get_config.return_value = {{
        'test_config': 'test_value'
    }}
    return mock_cm

@pytest.fixture
def {module_name}(mock_config_manager):
    """创建{analysis.name.title().replace('_', '')}实例"""
    config = {{'test': 'config'}}
    return {analysis.name.title().replace('_', '')}(config, config_manager=mock_config_manager)

class Test{analysis.name.title().replace('_', '')}:
    """{analysis.name.title().replace('_', '')}测试类"""

    def test_init_with_test_hook(self, mock_config_manager):
        """测试使用测试钩子初始化"""
        config = {{'test': 'config'}}
        manager = {analysis.name.title().replace('_', '')}(config, config_manager=mock_config_manager)
        
        assert manager.config_manager == mock_config_manager

    def test_init_without_test_hook(self):
        """测试不使用测试钩子初始化"""
        config = {{'test': 'config'}}
        with patch('src.infrastructure.{module_path.parent.name}.{module_name}.ConfigManager') as mock_cm_class:
            mock_cm_instance = MagicMock()
            mock_cm_class.return_value = mock_cm_instance
            
            manager = {analysis.name.title().replace('_', '')}(config)
            
            assert manager.config_manager == mock_cm_instance
            mock_cm_class.assert_called_once_with(config)

    def test_load_config_from_manager(self, {module_name}, mock_config_manager):
        """测试从配置管理器加载配置"""
        # 这里添加具体的测试逻辑
        assert {module_name}.config_manager == mock_config_manager
'''
        return test_content

    def auto_implement_hooks(self, dry_run: bool = False) -> Dict[str, bool]:
        """自动实现测试钩子"""
        results = {}
        
        for module_path_str, config in self.target_modules.items():
            module_path = self.infrastructure_dir / module_path_str
            
            if not module_path.exists():
                print(f"模块不存在: {module_path}")
                continue
            
            print(f"处理模块: {module_path}")
            
            # 分析模块
            analysis = self.analyze_module(module_path)
            
            if not analysis.needs_hook:
                print(f"  跳过: 模块 {analysis.name} 不需要钩子")
                results[str(module_path)] = False
                continue
            
            if analysis.has_test_hook:
                print(f"  跳过: 模块 {analysis.name} 已有测试钩子")
                results[str(module_path)] = False
                continue
            
            print(f"  需要实现测试钩子: {analysis.name}")
            
            if not dry_run:
                # 生成新的实现
                new_content = self.generate_hook_implementation(module_path, analysis)
                
                # 备份原文件
                backup_path = module_path.with_suffix('.py.bak')
                with open(module_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # 写入新内容
                with open(module_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  已更新: {module_path}")
                print(f"  备份: {backup_path}")
                
                # 创建测试文件
                test_dir = self.project_root / "tests" / "unit" / "infrastructure" / module_path.parent.name
                test_dir.mkdir(parents=True, exist_ok=True)
                
                test_file = test_dir / f"test_{module_path.stem}.py"
                if not test_file.exists():
                    test_content = self.create_test_file(module_path, analysis)
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(test_content)
                    print(f"  已创建测试文件: {test_file}")
                
                results[str(module_path)] = True
            else:
                print(f"  模拟: 将为 {analysis.name} 添加测试钩子")
                results[str(module_path)] = True
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自动实现测试钩子")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际修改文件")
    parser.add_argument("--backup", action="store_true", help="创建备份文件")
    
    args = parser.parse_args()
    
    implementer = AutoHookImplementer(args.project_root)
    
    print("开始自动实现测试钩子...")
    if args.dry_run:
        print("模拟运行模式")
    
    results = implementer.auto_implement_hooks(dry_run=args.dry_run)
    
    print("\n实现结果:")
    for module_path, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {module_path}")
    
    if args.dry_run:
        print("\n这是模拟运行，没有实际修改文件。使用 --dry-run=False 来实际实现钩子。")

if __name__ == "__main__":
    main() 