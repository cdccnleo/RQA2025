"""
架构模式统一化工具 - Phase 2.3

统一Factory、Manager、Service等架构模式
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


class ArchitecturePatternsUnification:
    """架构模式统一化工具"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.global_interfaces_dir = self.infra_dir / 'interfaces'

        self.patterns = {
            'Factory': {
                'interface': 'IFactory',
                'base_class': 'BaseFactory',
                'description': '工厂模式：负责创建对象'
            },
            'Manager': {
                'interface': 'IManager',
                'base_class': 'BaseManager',
                'description': '管理器模式：统一管理某一类资源'
            },
            'Service': {
                'interface': 'IService',
                'base_class': 'BaseService',
                'description': '服务模式：提供业务逻辑服务'
            },
            'Handler': {
                'interface': 'IHandler',
                'base_class': 'BaseHandler',
                'description': '处理器模式：处理特定类型的请求'
            },
            'Provider': {
                'interface': 'IProvider',
                'base_class': 'BaseProvider',
                'description': '提供者模式：提供特定类型的服务'
            },
            'Repository': {
                'interface': 'IRepository',
                'base_class': 'BaseRepository',
                'description': '仓库模式：封装数据访问逻辑'
            }
        }

    def analyze_architecture_patterns(self) -> Dict:
        """分析架构模式使用情况"""
        print('🔍 分析架构模式使用情况...')
        pattern_usage = defaultdict(lambda: {'classes': [], 'locations': []})

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 查找各种模式的类
                        for pattern_name in self.patterns.keys():
                            # 查找以模式结尾的类名
                            pattern = rf'class\s+(\w+{pattern_name})\s*[:\(]'
                            matches = re.findall(pattern, content)

                            for class_name in matches:
                                pattern_usage[pattern_name]['classes'].append(class_name)
                                pattern_usage[pattern_name]['locations'].append(rel_path)

                    except Exception as e:
                        continue

        return dict(pattern_usage)

    def create_unified_architecture_interfaces(self, pattern_usage: Dict):
        """创建统一的架构接口"""
        print('\\n🏗️ 创建统一架构接口...')

        created_interfaces = {}

        for pattern_name, usage in pattern_usage.items():
            if len(usage['classes']) >= 3:  # 只有当有3个或更多类时才创建统一接口
                interface_name = self.patterns[pattern_name]['interface']
                base_class_name = self.patterns[pattern_name]['base_class']
                description = self.patterns[pattern_name]['description']

                # 创建接口文件
                interface_content = self._generate_architecture_interface(
                    pattern_name, interface_name, base_class_name, description, usage['classes']
                )

                interface_file = self.global_interfaces_dir / f'{pattern_name.lower()}_pattern.py'
                with open(interface_file, 'w', encoding='utf-8') as f:
                    f.write(interface_content)

                created_interfaces[pattern_name] = {
                    'interface_file': interface_file,
                    'interface_name': interface_name,
                    'base_class_name': base_class_name,
                    'implementing_classes': usage['classes']
                }

                print(f'✅ 创建 {pattern_name} 模式接口: {interface_file}')

        return created_interfaces

    def _generate_architecture_interface(self, pattern_name: str, interface_name: str,
                                         base_class_name: str, description: str,
                                         implementing_classes: List[str]) -> str:
        """生成架构模式接口"""
        content = f'''"""
{pattern_name} 模式统一接口

{description}

当前实现类:
{chr(10).join(f"- {cls}" for cls in implementing_classes[:10])}
{"- ... 还有更多实现类" if len(implementing_classes) > 10 else ""}
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging


class {interface_name}(ABC):
    """{pattern_name} 接口

    定义所有{pattern_name}必须实现的统一接口
    """

    @abstractmethod
    def get_name(self) -> str:
        """获取{pattern_name}名称"""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """获取{pattern_name}版本"""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """检查{pattern_name}健康状态"""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """获取{pattern_name}统计信息"""
        pass


class {base_class_name}({interface_name}):
    """{pattern_name} 基类

    提供{pattern_name}的通用实现
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self._logger = logging.getLogger(__name__)
        self._start_time = None
        self._request_count = 0
        self._error_count = 0

    def get_name(self) -> str:
        """获取名称"""
        return self._name

    def get_version(self) -> str:
        """获取版本"""
        return self._version

    def is_healthy(self) -> bool:
        """检查健康状态 - 基础实现"""
        # 这里可以实现基本的健康检查逻辑
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {{
            "name": self._name,
            "version": self._version,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "uptime": 0,  # 可以计算实际运行时间
            "health_status": self.is_healthy()
        }}

    def _record_request(self):
        """记录请求"""
        self._request_count += 1

    def _record_error(self):
        """记录错误"""
        self._error_count += 1

    def _log_operation(self, operation: str, details: Optional[Dict[str, Any]] = None):
        """记录操作日志"""
        message = operation + " in " + self._name
        if details:
            message += " - " + str(details)
        self._logger.info(message)
'''

        return content

    def generate_pattern_migration_guide(self, created_interfaces: Dict, pattern_usage: Dict) -> str:
        """生成模式迁移指南"""
        guide = """# 架构模式迁移指南

## 概述

本文档指导如何将现有的架构模式类迁移到统一的接口体系。

## 支持的架构模式

"""

        for pattern_name, config in self.patterns.items():
            if pattern_name in created_interfaces:
                interface_info = created_interfaces[pattern_name]
                usage = pattern_usage[pattern_name]

                guide += f"### {pattern_name} 模式\n\n"
                guide += f"**接口**: `{config['interface']}`\n"
                guide += f"**基类**: `{config['base_class']}`\n"
                guide += f"**描述**: {config['description']}\n"
                guide += f"**当前实现类**: {len(usage['classes'])} 个\n\n"

                guide += "**实现类列表**:\n"
                for i, cls in enumerate(usage['classes'][:10]):
                    guide += f"- {cls}\n"
                if len(usage['classes']) > 10:
                    guide += f"- ... 还有 {len(usage['classes']) - 10} 个\n"

                guide += f"\n**迁移步骤**:\n"
                guide += f"1. 让 `{cls}` 继承 `{config['base_class']}`\n"
                guide += "2. 实现必要的抽象方法\n"
                guide += "3. 更新构造函数参数\n"
                guide += "4. 移除重复的通用代码\n\n"

        guide += """## 迁移策略

### 渐进式迁移
1. **第一阶段**: 核心类迁移 (Manager, Service, Factory)
2. **第二阶段**: 扩展类迁移 (Handler, Provider, Repository)
3. **第三阶段**: 清理和优化

### 向后兼容性
- 保留原有的类名作为别名
- 提供过渡期支持
- 逐步更新调用方

### 测试策略
1. 验证接口实现正确性
2. 检查功能完整性
3. 性能基准测试
4. 向后兼容性测试

## 质量保证

### 代码审查要点
- 接口实现完整性
- 方法签名一致性
- 异常处理规范性
- 日志记录统一性

### 自动化检查
- 接口实现检查
- 命名规范验证
- 依赖关系分析

## 实施时间表

- **Week 1-2**: 核心模式迁移 (Factory, Manager, Service)
- **Week 3**: 扩展模式迁移 (Handler, Provider)
- **Week 4**: 测试和优化
"""

        return guide

    def update_global_interfaces_init(self, created_interfaces: Dict):
        """更新全局interfaces的__init__.py"""
        init_file = self.global_interfaces_dir / '__init__.py'

        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 添加新的导入
            import_lines = []
            all_exports = []

            for pattern_name, interface_info in created_interfaces.items():
                import_line = f"from .{pattern_name.lower()}_pattern import {interface_info['interface']}, {interface_info['base_class_name']}"
                import_lines.append(import_line)
                all_exports.extend([interface_info['interface'], interface_info['base_class_name']])

            # 添加到现有内容
            lines = content.split('\n')

            # 找到最后一个导入语句
            last_import_idx = -1
            for i, line in enumerate(lines):
                if line.startswith('from .') or line.startswith('import '):
                    last_import_idx = i

            if last_import_idx >= 0:
                # 添加新导入
                for import_line in import_lines:
                    lines.insert(last_import_idx + 1, import_line)
                    last_import_idx += 1

                # 更新__all__
                all_pattern = r"__all__\s*=\s*\[(.*?)\]"
                match = re.search(all_pattern, content, re.DOTALL)
                if match:
                    existing_all = match.group(1)
                    quoted_exports = [f'"{exp}"' for exp in all_exports]
                    new_all = existing_all.rstrip() + ",\n    " + ", ".join(quoted_exports)
                    content = content.replace(match.group(0), f"__all__ = [{new_all}]")

                # 写回文件
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

                print('✅ 更新全局interfaces/__init__.py')

        except Exception as e:
            print(f'❌ 更新全局__init__.py失败: {e}')

    def implement_architecture_unification(self):
        """实施架构模式统一化"""
        print('🚀 开始架构模式统一化')
        print('=' * 50)

        # 1. 分析现有架构模式
        pattern_usage = self.analyze_architecture_patterns()
        print(f'✅ 分析了 {len(pattern_usage)} 种架构模式')

        # 2. 创建统一架构接口
        created_interfaces = self.create_unified_architecture_interfaces(pattern_usage)

        # 3. 更新全局接口导出
        self.update_global_interfaces_init(created_interfaces)

        # 4. 生成迁移指南
        migration_guide = self.generate_pattern_migration_guide(created_interfaces, pattern_usage)
        guide_file = Path('architecture_patterns_migration_guide.md')
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(migration_guide)
        print(f'✅ 生成迁移指南: {guide_file}')

        # 5. 输出统计信息
        total_patterns = len(created_interfaces)
        total_classes = sum(len(info['implementing_classes'])
                            for info in created_interfaces.values())

        print(f'\\n📊 架构模式统一化统计:')
        print(f'  创建的模式接口: {total_patterns}')
        print(f'  涉及的实现类: {total_classes}')

        for pattern_name, interface_info in created_interfaces.items():
            print(f'  {pattern_name}: {len(interface_info["implementing_classes"])} 个实现类')

        return {
            'pattern_usage': pattern_usage,
            'created_interfaces': created_interfaces,
            'migration_guide': migration_guide
        }


def main():
    """主函数"""
    unifier = ArchitecturePatternsUnification()
    result = unifier.implement_architecture_unification()

    print('\\n✅ 架构模式统一化完成！')
    print('生成的文件:')
    print('  - 多个架构模式接口文件')
    print('  - architecture_patterns_migration_guide.md')


if __name__ == "__main__":
    main()
