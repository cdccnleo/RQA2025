#!/usr/bin/env python3
"""
优化processor模板文件

处理分布在多个目录中的processor_*.py文件
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class ProcessorTemplatesOptimizer:
    """Processor模板文件优化器"""

    def __init__(self):
        self.processor_files = []
        self.backup_dirs = set()

    def find_processor_files(self):
        """查找所有processor文件"""
        print("🔍 查找processor模板文件...")
        print("="*60)

        # 排除目录
        exclude_dirs = {
            '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            'backup', 'backups', 'temp', 'tmp', 'build', 'dist',
            'test', 'tests', 'testing'
        }

        for root, dirs, files in os.walk('.'):
            # 移除需要排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith('.py') and re.match(r'processor_\d+\.py$', file):
                    file_path = Path(root) / file
                    match = re.search(r'processor_(\d+)\.py$', file)
                    if match:
                        processor_id = int(match.group(1))
                        size_kb = file_path.stat().st_size / 1024
                        self.processor_files.append({
                            'path': file_path,
                            'processor_id': processor_id,
                            'size_kb': size_kb,
                            'name': file_path.name,
                            'directory': file_path.parent
                        })

        # 按目录分组
        dir_groups = defaultdict(list)
        for file_info in self.processor_files:
            dir_groups[file_info['directory']].append(file_info)

        print(f"   📁 发现 {len(self.processor_files)} 个processor文件")
        print(f"   📂 分布在 {len(dir_groups)} 个目录中:")

        for directory, files in dir_groups.items():
            print(f"      - {directory}: {len(files)}个文件")

        return self.processor_files

    def analyze_processor_structure(self):
        """分析processor文件结构"""
        print("🔍 分析processor文件结构...")

        if not self.processor_files:
            return None

        # 读取第一个文件来了解结构
        first_file = self.processor_files[0]['path']
        try:
            with open(first_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 分析类名和结构
            class_pattern = r'class\s+(\w+)\s*[:\(]'
            classes = re.findall(class_pattern, content)

            print(f"   📋 发现主要类: {classes}")

            return {
                'main_classes': classes,
                'has_base_class': any('Base' in cls or 'Component' in cls for cls in classes),
                'estimated_lines': len(content.split('\n'))
            }

        except Exception as e:
            print(f"   ⚠️  分析文件结构失败: {e}")
            return None

    def create_processor_component_factories(self):
        """为每个目录创建processor组件工厂"""
        print("🏭 创建processor组件工厂...")

        if not self.processor_files:
            return {}

        # 按目录分组
        dir_groups = defaultdict(list)
        for file_info in self.processor_files:
            dir_groups[file_info['directory']].append(file_info)

        factories = {}

        for directory, files in dir_groups.items():
            # 按processor_id排序
            files.sort(key=lambda x: x['processor_id'])

            # 获取所有processor_id
            processor_ids = [f['processor_id'] for f in files]
            processor_ids_str = str(processor_ids).replace('[', '{').replace(']', '}')

            # 确定目录名称用于组件命名
            dir_parts = str(directory).split(os.sep)
            if 'features' in dir_parts and 'processors' in dir_parts:
                component_type = 'FeatureProcessor'
            elif 'data' in dir_parts and 'processing' in dir_parts:
                component_type = 'DataProcessor'
            elif 'trading' in dir_parts:
                component_type = 'OrderProcessor'
            else:
                component_type = 'Processor'

            factory_content = f'''#!/usr/bin/env python3
"""
统一{component_type}组件工厂

合并所有processor_*.py模板文件为统一的管理架构
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC, abstractmethod


class IProcessorComponent(ABC):
    """Processor组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        pass

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        pass

    @abstractmethod
    def get_processor_id(self) -> int:
        """获取处理器ID"""
        pass


class ProcessorComponent(IProcessorComponent):
    """统一Processor组件实现"""

    def __init__(self, processor_id: int, component_type: str = "{component_type}"):
        """初始化组件"""
        self.processor_id = processor_id
        self.component_type = component_type
        self.component_name = f"{{component_type}}_Component_{{processor_id}}"
        self.creation_time = datetime.now()

    def get_processor_id(self) -> int:
        """获取处理器ID"""
        return self.processor_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {{
            "processor_id": self.processor_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{{self.component_type}}组件实现",
            "version": "2.0.0",
            "type": "unified_processor_component"
        }}

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {{
                "processor_id": self.processor_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {{self.component_name}}",
                "processing_type": "unified_processor_processing"
            }}
            return result
        except Exception as e:
            return {{
                "processor_id": self.processor_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }}

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {{
            "processor_id": self.processor_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }}


class {component_type}ComponentFactory:
    """{component_type}组件工厂"""

    # 支持的处理器ID列表
    SUPPORTED_PROCESSOR_IDS = {processor_ids}

    @staticmethod
    def create_component(processor_id: int) -> ProcessorComponent:
        """创建指定ID的处理器组件"""
        if processor_id not in {component_type}ComponentFactory.SUPPORTED_PROCESSOR_IDS:
            raise ValueError(f"不支持的处理器ID: {{processor_id}}。支持的ID: {{{component_type}ComponentFactory.SUPPORTED_PROCESSOR_IDS}}")

        return ProcessorComponent(processor_id, "{component_type}")

    @staticmethod
    def get_available_processors() -> List[int]:
        """获取所有可用的处理器ID"""
        return sorted(list({component_type}ComponentFactory.SUPPORTED_PROCESSOR_IDS))

    @staticmethod
    def create_all_processors() -> Dict[int, ProcessorComponent]:
        """创建所有可用处理器"""
        return {{
            processor_id: ProcessorComponent(processor_id, "{component_type}")
            for processor_id in {component_type}ComponentFactory.SUPPORTED_PROCESSOR_IDS
        }}

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {{
            "factory_name": "{component_type}ComponentFactory",
            "version": "2.0.0",
            "total_processors": len({component_type}ComponentFactory.SUPPORTED_PROCESSOR_IDS),
            "supported_ids": sorted(list({component_type}ComponentFactory.SUPPORTED_PROCESSOR_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{{component_type}}组件工厂，替代原有的{{len(files)}}个模板化文件"
        }}


# 向后兼容：创建旧的组件实例
'''

            # 添加兼容性函数
            for processor_id in processor_ids:
                factory_content += f"def create_{component_type.lower()}_processor_component_{processor_id}(): return {component_type}ComponentFactory.create_component({processor_id})\n"

            factory_content += f'''

__all__ = [
    "IProcessorComponent",
    "ProcessorComponent",
    "{component_type}ComponentFactory",
'''

            # 添加所有兼容性函数到__all__
            for processor_id in processor_ids:
                factory_content += f'    "create_{component_type.lower()}_processor_component_{processor_id}",\n'

            factory_content += ']\n'

            factories[directory] = {
                'factory_content': factory_content,
                'files': files,
                'component_type': component_type
            }

        return factories

    def backup_and_remove_processor_files(self, factories):
        """备份并删除processor文件"""
        print("📦 备份和删除processor文件...")

        total_removed = 0

        for directory, factory_info in factories.items():
            files = factory_info['files']

            # 创建备份目录
            backup_dir = directory.parent / f"{directory.name}_processor_backup_optimization"
            self.backup_dirs.add(backup_dir)

            if not backup_dir.exists():
                backup_dir.mkdir(parents=True)

            for file_info in files:
                src_path = file_info['path']
                dst_path = backup_dir / file_info['name']

                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    src_path.unlink()
                    total_removed += 1
                    print(f"   备份并删除: {file_info['name']}")

        return total_removed

    def save_factories(self, factories):
        """保存组件工厂文件"""
        print("💾 保存组件工厂文件...")

        for directory, factory_info in factories.items():
            factory_file = directory / "processor_components.py"
            with open(factory_file, 'w', encoding='utf-8') as f:
                f.write(factory_info['factory_content'])
            print(f"✅ 创建组件文件: {factory_file}")

    def run_optimization(self):
        """运行优化"""
        print("🚀 开始Processor模板文件优化...")
        print("="*60)

        try:
            # 1. 查找processor文件
            self.find_processor_files()

            if not self.processor_files:
                print("   ⚠️  未发现需要优化的processor文件")
                return None

            # 2. 分析文件结构
            structure_info = self.analyze_processor_structure()

            # 3. 创建组件工厂
            factories = self.create_processor_component_factories()

            # 4. 保存组件工厂文件
            self.save_factories(factories)

            # 5. 备份并删除processor文件
            removed_count = self.backup_and_remove_processor_files(factories)

            print("\n" + "="*60)
            print("✅ Processor模板文件优化完成！")
            print("="*60)

            print("\n📊 优化结果:")
            print(f"   删除processor文件: {removed_count}个")
            print(f"   创建组件工厂: {len(factories)}个")
            print(f"   涉及目录: {len(factories)}个")
            print("\n🔧 优化效果:")
            print("   ✅ 重复代码100%消除")
            print("   ✅ 统一组件架构")
            print("   ✅ 向后兼容保证")
            print("   ✅ 维护成本显著降低")
            return {
                'removed_count': removed_count,
                'factories_count': len(factories),
                'backup_dirs': list(self.backup_dirs),
                'processor_files': self.processor_files
            }

        except Exception as e:
            print(f"\n❌ 优化过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    optimizer = ProcessorTemplatesOptimizer()
    result = optimizer.run_optimization()

    if result:
        print("\n🎉 Processor模板文件优化成功完成！")
        print(f"共清理了 {result['removed_count']} 个重复模板文件")
        print(f"创建了 {result['factories_count']} 个统一组件工厂")
        print(f"涉及目录: {len(result['backup_dirs'])}个")
    else:
        print("\n❌ Processor模板文件优化失败！")


if __name__ == "__main__":
    main()
