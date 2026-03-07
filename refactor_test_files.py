#!/usr/bin/env python3
"""
重构大型测试文件
"""

import os


def extract_class_definitions(content: str):
    """提取类定义及其范围"""
    lines = content.split('\n')
    classes = []
    in_class = False
    class_start = 0
    current_class = ""
    indent_level = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 查找类定义开始
        if stripped.startswith('class '):
            if in_class:
                # 结束之前的类
                classes.append((current_class, class_start, i-1))

            current_class = stripped.split('class ')[1].split('(')[0].split(':')[0].strip()
            class_start = i
            in_class = True
            indent_level = len(line) - len(line.lstrip())

        elif in_class:
            # 检查是否到达类结束
            if stripped and not line.startswith(' ') and not line.startswith('\t') and not stripped.startswith('#'):
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level:
                    classes.append((current_class, class_start, i-1))
                    in_class = False
                    current_class = ""
                    class_start = 0

    # 处理最后一个类
    if in_class:
        classes.append((current_class, class_start, len(lines)-1))

    return classes


def refactor_cloud_native_test_platform():
    """重构云原生测试平台"""

    print('🔄 重构 cloud_native_test_platform.py...')

    source_file = 'src/infrastructure/config/tests/cloud_native_test_platform.py'
    test_models_dir = 'src/infrastructure/config/tests/models'

    try:
        # 读取源文件
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        classes = extract_class_definitions(content)

        # 选择要拆分的配置类（数据模型类）
        config_classes = ['PlatformType', 'TestServiceStatus', 'TestEnvironment',
                          'ContainerConfig', 'KubernetesConfig', 'ServiceInfo', 'TestResult']

        created_files = []

        for class_name in config_classes:
            # 查找对应的类定义
            class_info = next((c for c in classes if c[0] == class_name), None)
            if not class_info:
                continue

            # 创建模型文件
            model_file = os.path.join(test_models_dir, f"{class_name.lower()}.py")
            lines = content.split('\n')

            # 提取类内容
            class_lines = [
                f'"""测试模型: {class_name}"""',
                '',
                'from infrastructure.config.core.imports import (',
                '    dataclass, field, Enum, Dict, Any, Optional, List',
                ')',
                ''
            ]

            start_line, end_line = class_info[1], class_info[2]
            class_content = lines[start_line:end_line+1]
            class_lines.extend(class_content)

            with open(model_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(class_lines))

            created_files.append(model_file)
            print(f'   ✅ 创建模型: {os.path.basename(model_file)}')

        # 重构主测试文件
        main_lines = [
            '"""云原生测试平台"""',
            '',
            'from infrastructure.config.core.imports import (',
            '    Dict, Any, Optional, List, Union, time, threading',
            ')',
            '',
            '# 导入测试模型',
            'from infrastructure.config.tests.models.platformtype import PlatformType',
            'from infrastructure.config.tests.models.testservicestatus import TestServiceStatus',
            'from infrastructure.config.tests.models.testenvironment import TestEnvironment',
            'from infrastructure.config.tests.models.containerconfig import ContainerConfig',
            'from infrastructure.config.tests.models.kubernetesconfig import KubernetesConfig',
            'from infrastructure.config.tests.models.serviceinfo import ServiceInfo',
            'from infrastructure.config.tests.models.testresult import TestResult',
            '',
            '# 导入核心组件',
            'from infrastructure.config.core.common_mixins import ConfigComponentMixin',
            ''
        ]

        # 提取剩余的类（管理器类）
        manager_classes = ['ContainerManager', 'KubernetesManager',
                           'MicroserviceTestRunner', 'CloudNativeTestPlatform']

        lines = content.split('\n')
        for class_name in manager_classes:
            class_info = next((c for c in classes if c[0] == class_name), None)
            if class_info:
                start_line, end_line = class_info[1], class_info[2]
                main_lines.extend(lines[start_line:end_line+1])
                main_lines.append('')

        # 写回主文件
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(main_lines))

        print(f'   ✅ 重构主文件: {os.path.basename(source_file)}')

        # 创建models/__init__.py
        init_content = '''"""测试模型模块"""

from .platformtype import PlatformType
from .testservicestatus import TestServiceStatus
from .testenvironment import TestEnvironment
from .containerconfig import ContainerConfig
from .kubernetesconfig import KubernetesConfig
from .serviceinfo import ServiceInfo
from .testresult import TestResult

__all__ = [
    "PlatformType",
    "TestServiceStatus",
    "TestEnvironment",
    "ContainerConfig",
    "KubernetesConfig",
    "ServiceInfo",
    "TestResult"
]
'''

        init_file = os.path.join(test_models_dir, '__init__.py')
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)

        print(f'   ✅ 创建: models/__init__.py')

        return len(created_files)

    except Exception as e:
        print(f'   ❌ 重构失败: {e}')
        return 0


def refactor_edge_computing_test_platform():
    """重构边缘计算测试平台"""

    print('🔄 重构 edge_computing_test_platform.py...')

    source_file = 'src/infrastructure/config/tests/edge_computing_test_platform.py'
    edge_models_dir = 'src/infrastructure/config/tests/edge_models'

    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        classes = extract_class_definitions(content)

        # 选择配置类
        config_classes = ['EdgeNodeType', 'NodeStatus',
                          'TestType', 'EdgeNodeConfig', 'EdgeNodeInfo']

        created_files = []

        for class_name in config_classes:
            class_info = next((c for c in classes if c[0] == class_name), None)
            if not class_info:
                continue

            model_file = os.path.join(edge_models_dir, f"{class_name.lower()}.py")
            lines = content.split('\n')

            class_lines = [
                f'"""边缘计算测试模型: {class_name}"""',
                '',
                'from infrastructure.config.core.imports import (',
                '    dataclass, field, Enum, Dict, Any, Optional, List',
                ')',
                ''
            ]

            start_line, end_line = class_info[1], class_info[2]
            class_content = lines[start_line:end_line+1]
            class_lines.extend(class_content)

            with open(model_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(class_lines))

            created_files.append(model_file)
            print(f'   ✅ 创建模型: {os.path.basename(model_file)}')

        # 重构主测试文件
        main_lines = [
            '"""边缘计算测试平台"""',
            '',
            'from infrastructure.config.core.imports import (',
            '    Dict, Any, Optional, List, Union, time, threading',
            ')',
            '',
            '# 导入测试模型',
            'from infrastructure.config.tests.edge_models.edgenodetype import EdgeNodeType',
            'from infrastructure.config.tests.edge_models.nodestatus import NodeStatus',
            'from infrastructure.config.tests.edge_models.testtype import TestType',
            'from infrastructure.config.tests.edge_models.edgenodeconfig import EdgeNodeConfig',
            'from infrastructure.config.tests.edge_models.edgenodeinfo import EdgeNodeInfo',
            '',
            '# 导入核心组件',
            'from infrastructure.config.core.common_mixins import ConfigComponentMixin',
            ''
        ]

        # 提取剩余的类
        manager_classes = ['EdgeNodeManager', 'EdgeTestRunner', 'EdgeComputingTestPlatform']

        lines = content.split('\n')
        for class_name in manager_classes:
            class_info = next((c for c in classes if c[0] == class_name), None)
            if class_info:
                start_line, end_line = class_info[1], class_info[2]
                main_lines.extend(lines[start_line:end_line+1])
                main_lines.append('')

        with open(source_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(main_lines))

        print(f'   ✅ 重构主文件: {os.path.basename(source_file)}')

        # 创建edge_models/__init__.py
        init_content = '''"""边缘计算测试模型模块"""

from .edgenodetype import EdgeNodeType
from .nodestatus import NodeStatus
from .testtype import TestType
from .edgenodeconfig import EdgeNodeConfig
from .edgenodeinfo import EdgeNodeInfo

__all__ = [
    "EdgeNodeType",
    "NodeStatus",
    "TestType",
    "EdgeNodeConfig",
    "EdgeNodeInfo"
]
'''

        init_file = os.path.join(edge_models_dir, '__init__.py')
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)

        print(f'   ✅ 创建: edge_models/__init__.py')

        return len(created_files)

    except Exception as e:
        print(f'   ❌ 重构失败: {e}')
        return 0


def main():
    """主函数"""

    print('=== 📦 Phase 2.1: 重构测试文件 ===')
    print()

    # 创建必要的目录
    os.makedirs('src/infrastructure/config/tests/models', exist_ok=True)
    os.makedirs('src/infrastructure/config/tests/edge_models', exist_ok=True)

    total_created = 0

    # 重构测试文件
    created1 = refactor_cloud_native_test_platform()
    total_created += created1

    created2 = refactor_edge_computing_test_platform()
    total_created += created2

    print()
    print('🎯 测试文件重构完成！')
    print(f'   📦 创建了 {total_created} 个模型文件')
    print('   📁 新的测试结构:')
    print('      tests/')
    print('      ├── models/              # 云原生测试模型')
    print('      ├── edge_models/         # 边缘计算测试模型')
    print('      ├── cloud_native_test_platform.py   # 云原生测试逻辑')
    print('      └── edge_computing_test_platform.py # 边缘计算测试逻辑')

    # 验证文件大小
    print()
    print('📊 文件大小验证:')
    test_files = [
        'src/infrastructure/config/tests/cloud_native_test_platform.py',
        'src/infrastructure/config/tests/edge_computing_test_platform.py'
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            rel_path = os.path.relpath(file_path, 'src/infrastructure/config')
            status = '✅' if size_kb < 25 else '⚠️'
            print(f'   {status} {rel_path}: {size_kb:.1f} KB')


if __name__ == '__main__':
    main()
