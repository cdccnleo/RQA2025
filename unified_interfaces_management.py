"""
统一接口管理 - Phase 1.4

创建全局接口管理方案，统一各模块接口定义
"""

import re
from pathlib import Path
from typing import Dict, List, Set


class UnifiedInterfaceManager:
    """统一接口管理器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.global_interfaces_dir = self.infra_dir / 'interfaces'

    def analyze_interface_distribution(self) -> Dict:
        """分析接口分布情况"""
        interface_distribution = {
            'global_interfaces': {},
            'module_interfaces': {},
            'duplicate_interfaces': set()
        }

        # 分析全局接口
        if self.global_interfaces_dir.exists():
            for file_path in self.global_interfaces_dir.glob('*.py'):
                if file_path.name != '__init__.py':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        interfaces = self._extract_interfaces(content)
                        interface_distribution['global_interfaces'][file_path.stem] = interfaces

                    except Exception as e:
                        print(f"读取全局接口文件 {file_path.name} 失败: {e}")

        # 分析各模块接口
        for module_dir in self.infra_dir.iterdir():
            if (module_dir.is_dir() and
                not module_dir.name.startswith('.') and
                    module_dir.name != 'interfaces'):

                interfaces_file = module_dir / 'interfaces.py'
                if interfaces_file.exists():
                    try:
                        with open(interfaces_file, 'r', encoding='utf-8') as f:
                            content = f.read()

                        interfaces = self._extract_interfaces(content)
                        interface_distribution['module_interfaces'][module_dir.name] = interfaces

                    except Exception as e:
                        print(f"读取模块接口文件 {module_dir.name}/interfaces.py 失败: {e}")

        # 找出重复接口
        all_interfaces = set()
        for interfaces in interface_distribution['global_interfaces'].values():
            all_interfaces.update(interfaces)

        for module, interfaces in interface_distribution['module_interfaces'].items():
            duplicates = all_interfaces.intersection(interfaces)
            if duplicates:
                interface_distribution['duplicate_interfaces'].update(duplicates)

        return interface_distribution

    def _extract_interfaces(self, content: str) -> Set[str]:
        """提取接口名称"""
        # 匹配以I开头的类定义
        interface_pattern = r'class\s+(I\w+)\s*[:\(]'
        interfaces = set(re.findall(interface_pattern, content, re.MULTILINE))

        return interfaces

    def create_unified_interface_system(self, distribution: Dict):
        """创建统一接口系统"""
        print('🏗️ 创建统一接口系统')
        print('=' * 30)

        # 1. 分析需要迁移的接口
        interfaces_to_migrate = self._identify_interfaces_to_migrate(distribution)

        # 2. 创建模块接口引用文件
        self._create_module_interface_references(interfaces_to_migrate)

        # 3. 更新全局__init__.py
        self._update_global_init(distribution)

        # 4. 生成接口使用报告
        self._generate_interface_usage_report(distribution, interfaces_to_migrate)

    def _identify_interfaces_to_migrate(self, distribution: Dict) -> Dict[str, List[str]]:
        """识别需要迁移的接口"""
        interfaces_to_migrate = {}

        # 检查重复的接口
        duplicate_interfaces = distribution['duplicate_interfaces']

        for interface in duplicate_interfaces:
            # 找到哪些模块定义了这个接口
            modules_with_interface = []
            for module, interfaces in distribution['module_interfaces'].items():
                if interface in interfaces:
                    modules_with_interface.append(module)

            if modules_with_interface:
                interfaces_to_migrate[interface] = modules_with_interface

        # 检查常用的接口（在多个模块中使用）
        common_interfaces = self._find_common_interfaces(distribution)
        for interface, modules in common_interfaces.items():
            if len(modules) > 1 and interface not in interfaces_to_migrate:
                interfaces_to_migrate[interface] = modules

        return interfaces_to_migrate

    def _find_common_interfaces(self, distribution: Dict) -> Dict[str, List[str]]:
        """查找常用接口"""
        interface_usage = {}

        # 统计每个接口在哪些模块中使用
        for module, interfaces in distribution['module_interfaces'].items():
            for interface in interfaces:
                if interface not in interface_usage:
                    interface_usage[interface] = []
                interface_usage[interface].append(module)

        # 返回使用频率高的接口
        common_interfaces = {name: modules for name, modules in interface_usage.items()
                             if len(modules) >= 2}

        return common_interfaces

    def _create_module_interface_references(self, interfaces_to_migrate: Dict[str, List[str]]):
        """创建模块接口引用文件"""
        print('\\n📝 创建模块接口引用文件...')

        for interface_name, modules in interfaces_to_migrate.items():
            for module in modules:
                self._create_interface_reference_file(module, interface_name)

    def _create_interface_reference_file(self, module: str, interface_name: str):
        """为模块创建接口引用文件"""
        module_dir = self.infra_dir / module
        reference_file = module_dir / f'interface_{interface_name.lower()}.py'

        # 检查接口是否已经在全局定义
        global_interface_file = self._find_global_interface_file(interface_name)

        if global_interface_file:
            # 创建引用文件
            content = f'''"""
{interface_name} 接口引用

此文件引用全局定义的 {interface_name} 接口
"""

from src.infrastructure.interfaces.{global_interface_file} import {interface_name}

# 重新导出以保持兼容性
__all__ = ['{interface_name}']
'''

            try:
                with open(reference_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f'  ✅ 创建引用文件: {module}/interface_{interface_name.lower()}.py')

            except Exception as e:
                print(f'  ❌ 创建引用文件失败 {module}/interface_{interface_name.lower()}.py: {e}')

    def _find_global_interface_file(self, interface_name: str) -> str:
        """查找全局接口文件"""
        global_interfaces = {
            'ICacheComponent': 'cache',
            'ICacheManager': 'cache',
            'ICacheFactory': 'cache',
            'IConfigComponent': 'config',
            'IConfigManager': 'config',
            'IConfigLoader': 'config',
            'IMonitorComponent': 'monitor',
            'IMonitor': 'monitor',
            'ISystemMonitor': 'monitor'
        }

        return global_interfaces.get(interface_name, 'standard_interfaces')

    def _update_global_init(self, distribution: Dict):
        """更新全局__init__.py"""
        global_init = self.global_interfaces_dir / '__init__.py'

        # 收集所有全局接口
        all_global_interfaces = set()
        for interfaces in distribution['global_interfaces'].values():
            all_global_interfaces.update(interfaces)

        # 生成__init__.py内容
        imports = []
        all_exports = []

        for interface_file, interfaces in distribution['global_interfaces'].items():
            if interfaces:
                import_line = f"from .{interface_file} import {', '.join(sorted(interfaces))}"
                imports.append(import_line)
                all_exports.extend(interfaces)

        content = f'''"""
基础设施层统一接口定义

此模块包含所有核心组件的接口定义
"""

# 导入所有接口
{chr(10).join(imports)}

# 导出所有接口
__all__ = {sorted(all_exports)}
'''

        try:
            with open(global_init, 'w', encoding='utf-8') as f:
                f.write(content)

            print('  ✅ 更新全局 __init__.py')

        except Exception as e:
            print(f'  ❌ 更新全局 __init__.py 失败: {e}')

    def _generate_interface_usage_report(self, distribution: Dict, interfaces_to_migrate: Dict):
        """生成接口使用报告"""
        print('\\n📊 接口使用报告')
        print('=' * 20)

        print(f'全局接口文件数: {len(distribution["global_interfaces"])}')
        print(f'模块接口文件数: {len(distribution["module_interfaces"])}')
        print(f'重复接口数: {len(distribution["duplicate_interfaces"])}')
        print(f'需要迁移的接口数: {len(interfaces_to_migrate)}')

        if interfaces_to_migrate:
            print('\\n需要迁移的接口:')
            for interface, modules in interfaces_to_migrate.items():
                print(f'  • {interface}: {len(modules)} 个模块 ({", ".join(modules)})')

    def implement_interface_unification(self):
        """实施接口统一化"""
        print('🚀 开始接口统一化实施')
        print('=' * 30)

        # 1. 分析接口分布
        distribution = self.analyze_interface_distribution()

        # 2. 创建统一接口系统
        self.create_unified_interface_system(distribution)

        print('\\n✅ 接口统一化完成！')


def main():
    """主函数"""
    manager = UnifiedInterfaceManager()
    manager.implement_interface_unification()


if __name__ == "__main__":
    main()
