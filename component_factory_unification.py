"""
ComponentFactory统一化工具 - Phase 2.2

解决36个ComponentFactory重复类名问题
"""

import os
import re
from pathlib import Path
from typing import Dict, List


class ComponentFactoryUnification:
    """ComponentFactory统一化工具"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.global_interfaces_dir = self.infra_dir / 'interfaces'

    def create_global_component_factory_interface(self):
        """创建全局ComponentFactory接口"""
        interface_content = '''"""
全局ComponentFactory接口定义

统一所有组件工厂的接口规范
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class IComponentFactory(ABC):
    """组件工厂接口

    定义所有组件工厂必须实现的统一接口
    """

    @abstractmethod
    def create_component(self, component_type: str, config: Dict[str, Any]) -> Any:
        """创建组件

        Args:
            component_type: 组件类型
            config: 组件配置

        Returns:
            创建的组件实例
        """
        pass

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """获取支持的组件类型

        Returns:
            支持的组件类型列表
        """
        pass

    @abstractmethod
    def validate_config(self, component_type: str, config: Dict[str, Any]) -> bool:
        """验证配置

        Args:
            component_type: 组件类型
            config: 配置字典

        Returns:
            配置是否有效
        """
        pass


class BaseComponentFactory(IComponentFactory):
    """基础组件工厂

    提供ComponentFactory的通用实现
    """

    def __init__(self):
        self._components = {}
        self._supported_types = []

    def get_supported_types(self) -> List[str]:
        """获取支持的组件类型"""
        return self._supported_types.copy()

    def validate_config(self, component_type: str, config: Dict[str, Any]) -> bool:
        """验证配置 - 基础实现"""
        if not isinstance(config, dict):
            return False

        if component_type not in self._supported_types:
            return False

        return True

    def register_component_type(self, component_type: str):
        """注册组件类型"""
        if component_type not in self._supported_types:
            self._supported_types.append(component_type)

    def unregister_component_type(self, component_type: str):
        """注销组件类型"""
        if component_type in self._supported_types:
            self._supported_types.remove(component_type)
'''

        # 保存到全局interfaces目录
        interface_file = self.global_interfaces_dir / 'component_factory.py'
        with open(interface_file, 'w', encoding='utf-8') as f:
            f.write(interface_content)

        print(f'✅ 创建全局ComponentFactory接口: {interface_file}')

        # 更新全局__init__.py
        self._update_global_init('IComponentFactory', 'BaseComponentFactory')

        return interface_file

    def _update_global_init(self, *classes):
        """更新全局interfaces/__init__.py"""
        init_file = self.global_interfaces_dir / '__init__.py'

        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 添加新的导入
            import_lines = []
            for cls in classes:
                import_lines.append(f"from .component_factory import {cls}")

            # 找到__all__列表并更新
            if '__all__' in content:
                # 在__all__中添加新类
                all_pattern = r"__all__\s*=\s*\[(.*?)\]"
                match = re.search(all_pattern, content, re.DOTALL)
                if match:
                    existing_all = match.group(1)
                    quoted_classes = [f'"{cls}"' for cls in classes]
                    new_all = existing_all.rstrip() + f",\n    {', '.join(quoted_classes)}"
                    content = content.replace(match.group(0), f"__all__ = [{new_all}]")

            # 添加导入语句
            if import_lines:
                # 找到最后一个导入语句后添加
                lines = content.split('\n')
                last_import_idx = -1
                for i, line in enumerate(lines):
                    if line.startswith('from .') or line.startswith('import '):
                        last_import_idx = i

                if last_import_idx >= 0:
                    lines.insert(last_import_idx + 1, '')
                    for import_line in import_lines:
                        lines.insert(last_import_idx + 2, import_line)
                        last_import_idx += 1

                    content = '\n'.join(lines)

            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print('✅ 更新全局interfaces/__init__.py')

        except Exception as e:
            print(f'❌ 更新全局__init__.py失败: {e}')

    def analyze_component_factory_implementations(self) -> Dict:
        """分析所有ComponentFactory的实现"""
        implementations = {}

        component_factory_files = self._find_component_factory_files()

        for file_path in component_factory_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取类实现
                class_match = re.search(
                    r'class ComponentFactory:(.*?)(?=\nclass|\n@|\n\n\n|\Z)', content, re.DOTALL)
                if class_match:
                    implementation = class_match.group(1).strip()
                    rel_path = str(file_path.relative_to(self.infra_dir))

                    # 分析实现特点
                    features = self._analyze_implementation_features(implementation)

                    implementations[rel_path] = {
                        'implementation': implementation,
                        'features': features,
                        'line_count': len(implementation.split('\n'))
                    }

            except Exception as e:
                print(f'❌ 分析文件失败 {file_path}: {e}')

        return implementations

    def _find_component_factory_files(self) -> List[Path]:
        """查找所有包含ComponentFactory的文件"""
        files = []

        for root, dirs, files_in_dir in os.walk(self.infra_dir):
            for file in files_in_dir:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if 'class ComponentFactory' in content:
                            files.append(file_path)

                    except Exception:
                        continue

        return files

    def _analyze_implementation_features(self, implementation: str) -> Dict:
        """分析实现的特征"""
        features = {
            'has_logger': 'logger' in implementation,
            'has_init': '__init__' in implementation,
            'has_create_component': 'create_component' in implementation,
            'has_caching': '_components' in implementation or 'cache' in implementation.lower(),
            'has_validation': 'validate' in implementation.lower(),
            'has_registration': 'register' in implementation.lower(),
            'complexity': len(re.findall(r'def ', implementation))  # 方法数量
        }

        return features

    def create_unified_implementations(self, implementations: Dict):
        """创建统一的实现"""
        print('\\n🔄 创建统一实现...')

        # 按功能分组实现
        grouped_implementations = self._group_implementations_by_functionality(implementations)

        unified_implementations = {}

        for group_name, files in grouped_implementations.items():
            unified_class = self._create_unified_component_factory_class(group_name, files)
            unified_implementations[group_name] = unified_class

        return unified_implementations

    def _group_implementations_by_functionality(self, implementations: Dict) -> Dict[str, List]:
        """按功能分组实现"""
        groups = {
            'cache_components': [],
            'error_components': [],
            'health_components': [],
            'logging_components': [],
            'resource_components': [],
            'utils_components': []
        }

        for file_path, data in implementations.items():
            if 'cache' in file_path:
                groups['cache_components'].append(file_path)
            elif 'error' in file_path:
                groups['error_components'].append(file_path)
            elif 'health' in file_path:
                groups['health_components'].append(file_path)
            elif 'logging' in file_path:
                groups['logging_components'].append(file_path)
            elif 'resource' in file_path:
                groups['resource_components'].append(file_path)
            elif 'utils' in file_path:
                groups['utils_components'].append(file_path)

        # 过滤空组
        return {k: v for k, v in groups.items() if v}

    def _create_unified_component_factory_class(self, group_name: str, files: List[str]) -> str:
        """创建统一的组件工厂类"""
        class_name = f"{group_name.replace('_', ' ').title().replace(' ', '')}Factory"

        class_content = f'''"""
{group_name} 统一组件工厂

统一以下模块的ComponentFactory实现:
{chr(10).join(f"- {f}" for f in files)}
"""

from src.infrastructure.interfaces import BaseComponentFactory
from typing import Dict, Any, Optional
import logging


class {class_name}(BaseComponentFactory):
    """{group_name} 组件工厂"""

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._initialize_supported_types()

    def _initialize_supported_types(self):
        """初始化支持的组件类型"""
        # 这里可以根据具体需求初始化支持的类型
        pass

    def create_component(self, component_type: str, config: Dict[str, Any]) -> Any:
        """创建组件

        Args:
            component_type: 组件类型
            config: 组件配置

        Returns:
            创建的组件实例

        Raises:
            ValueError: 当组件类型不支持或配置无效时
        """
        if not self.validate_config(component_type, config):
            raise ValueError("Invalid config for component type: " + component_type)

        # 统一的组件创建逻辑
        # 这里可以根据component_type调用相应的创建方法

        # 缓存已创建的组件
        cache_key = component_type + ":" + str(hash(str(config)))
        if cache_key in self._components:
            return self._components[cache_key]

        # 创建新组件
        component = self._create_component_instance(component_type, config)

        # 缓存组件
        self._components[cache_key] = component

        self._logger.info("Created " + component_type + " component")
        return component

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]) -> Any:
        """创建组件实例 - 子类必须实现"""
        raise NotImplementedError("Subclasses must implement _create_component_instance")
'''

        return class_content

    def generate_migration_plan(self, implementations: Dict) -> str:
        """生成迁移计划"""
        plan = f"""# ComponentFactory迁移计划

## 概述

需要迁移 {len(implementations)} 个ComponentFactory实现到统一的接口架构。

## 迁移步骤

### 1. 创建全局接口
- ✅ 已创建 `IComponentFactory` 接口
- ✅ 已创建 `BaseComponentFactory` 基类

### 2. 创建统一实现类
需要创建以下统一实现类:

"""

        grouped = self._group_implementations_by_functionality(implementations)

        for group_name, files in grouped.items():
            class_name = f"{group_name.replace('_', ' ').title().replace(' ', '')}Factory"
            plan += f"#### {class_name}\n"
            plan += f"- **负责模块**: {len(files)} 个文件\n"
            plan += f"- **文件列表**:\n"
            for file in files:
                plan += f"  - {file}\n"
            plan += "\n"

        plan += """### 3. 更新现有代码
对于每个现有的ComponentFactory类：

1. 继承相应的统一实现类
2. 实现 `_create_component_instance` 方法
3. 移除重复的通用代码
4. 更新导入语句

### 4. 向后兼容性
- 保留原有的类名作为别名
- 提供过渡期支持
- 逐步迁移使用方

### 5. 测试验证
- 验证所有组件创建功能正常
- 检查配置验证逻辑正确
- 确认日志记录正常工作

## 风险评估

### 高风险
- 缓存逻辑可能不一致
- 组件创建参数可能不同

### 中风险
- 日志记录格式不统一
- 异常处理方式不同

### 低风险
- 接口方法签名相同
- 基本功能逻辑相似

## 实施时间表

- **Week 1**: 创建统一接口和基类 ✅
- **Week 2**: 实现分组的统一工厂类
- **Week 3**: 迁移现有实现
- **Week 4**: 测试和验证
"""

        return plan

    def implement_unification(self):
        """实施统一化"""
        print('🚀 开始ComponentFactory统一化')
        print('=' * 50)

        # 1. 创建全局接口
        interface_file = self.create_global_component_factory_interface()

        # 2. 分析现有实现
        implementations = self.analyze_component_factory_implementations()
        print(f'✅ 分析了 {len(implementations)} 个ComponentFactory实现')

        # 3. 生成迁移计划
        migration_plan = self.generate_migration_plan(implementations)
        plan_file = Path('component_factory_migration_plan.md')
        with open(plan_file, 'w', encoding='utf-8') as f:
            f.write(migration_plan)
        print(f'✅ 生成迁移计划: {plan_file}')

        # 4. 创建统一实现类
        unified_implementations = self.create_unified_implementations(implementations)

        # 保存统一实现类
        for class_name, implementation in unified_implementations.items():
            impl_file = Path(f'{class_name}.py')
            with open(impl_file, 'w', encoding='utf-8') as f:
                f.write(implementation)
            print(f'✅ 创建统一实现类: {impl_file}')

        print(f'\\n📊 统一化总结:')
        print(f'  ComponentFactory实例数: {len(implementations)}')
        print(f'  功能分组数: {len(unified_implementations)}')
        print(f'  全局接口: IComponentFactory, BaseComponentFactory')

        return {
            'interface_file': interface_file,
            'implementations': implementations,
            'unified_classes': unified_implementations,
            'migration_plan': migration_plan
        }


def main():
    """主函数"""
    unifier = ComponentFactoryUnification()
    result = unifier.implement_unification()

    print('\\n✅ ComponentFactory统一化完成！')
    print('生成的文件:')
    print('  - src/infrastructure/interfaces/component_factory.py')
    print('  - component_factory_migration_plan.md')
    print('  - 多个统一实现类文件')


if __name__ == "__main__":
    main()
