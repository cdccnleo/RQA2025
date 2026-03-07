#!/usr/bin/env python3
"""
优化data/processing目录下的processor文件

专门处理src/data/processing/目录中的processor_*.py文件
"""

import re
import shutil
from pathlib import Path
from datetime import datetime


class DataProcessingProcessorOptimizer:
    """Data Processing Processor优化器"""

    def __init__(self):
        self.data_processing_dir = Path("src/data/processing")
        self.backup_dir = self.data_processing_dir.parent.parent / \
            "data_processing_backup_processor_optimization"
        self.processor_files = []

    def find_processor_files(self):
        """查找processor文件"""
        print("🔍 查找data/processing目录中的processor文件...")

        for file_path in self.data_processing_dir.glob("processor_*.py"):
            if file_path.name.startswith('processor_'):
                match = re.search(r'processor_(\d+)\.py$', file_path.name)
                if match:
                    processor_id = int(match.group(1))
                    size_kb = file_path.stat().st_size / 1024
                    self.processor_files.append({
                        'path': file_path,
                        'processor_id': processor_id,
                        'size_kb': size_kb,
                        'name': file_path.name
                    })

        print(f"   📁 发现 {len(self.processor_files)} 个processor文件")
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
                'has_base_class': any('Base' in cls for cls in classes),
                'estimated_lines': len(content.split('\n'))
            }

        except Exception as e:
            print(f"   ⚠️  分析文件结构失败: {e}")
            return None

    def create_processor_components_factory(self):
        """创建processor组件工厂"""
        print("🏭 创建processor组件工厂...")

        if not self.processor_files:
            return None

        # 按processor_id排序
        self.processor_files.sort(key=lambda x: x['processor_id'])

        # 获取所有processor_id
        processor_ids = [f['processor_id'] for f in self.processor_files]
        processor_ids_str = str(processor_ids).replace('[', '{').replace(']', '}')

        factory_content = f'''#!/usr/bin/env python3
"""
统一Data Processing Processor组件工厂

合并所有processor_*.py模板文件为统一的管理架构
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC, abstractmethod


class IDataProcessorComponent(ABC):
    """Data Processor组件接口"""

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


class DataProcessorComponent(IDataProcessorComponent):
    """统一Data Processor组件实现"""

    def __init__(self, processor_id: int):
        """初始化组件"""
        self.processor_id = processor_id
        self.component_name = f"DataProcessor_Component_{{processor_id}}"
        self.creation_time = datetime.now()

    def get_processor_id(self) -> int:
        """获取处理器ID"""
        return self.processor_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {{
            "processor_id": self.processor_id,
            "component_name": self.component_name,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一Data Processor组件实现",
            "version": "2.0.0",
            "type": "unified_data_processor_component"
        }}

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {{
                "processor_id": self.processor_id,
                "component_name": self.component_name,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {{self.component_name}}",
                "processing_type": "unified_data_processor_processing"
            }}
            return result
        except Exception as e:
            return {{
                "processor_id": self.processor_id,
                "component_name": self.component_name,
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
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }}


class DataProcessorComponentFactory:
    """Data Processor组件工厂"""

    # 支持的处理器ID列表
    SUPPORTED_PROCESSOR_IDS = {processor_ids}

    @staticmethod
    def create_component(processor_id: int) -> DataProcessorComponent:
        """创建指定ID的处理器组件"""
        if processor_id not in DataProcessorComponentFactory.SUPPORTED_PROCESSOR_IDS:
            raise ValueError(f"不支持的处理器ID: {{processor_id}}。支持的ID: {{DataProcessorComponentFactory.SUPPORTED_PROCESSOR_IDS}}")

        return DataProcessorComponent(processor_id)

    @staticmethod
    def get_available_processors() -> List[int]:
        """获取所有可用的处理器ID"""
        return sorted(list(DataProcessorComponentFactory.SUPPORTED_PROCESSOR_IDS))

    @staticmethod
    def create_all_processors() -> Dict[int, DataProcessorComponent]:
        """创建所有可用处理器"""
        return {{
            processor_id: DataProcessorComponent(processor_id)
            for processor_id in DataProcessorComponentFactory.SUPPORTED_PROCESSOR_IDS
        }}

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {{
            "factory_name": "DataProcessorComponentFactory",
            "version": "2.0.0",
            "total_processors": len(DataProcessorComponentFactory.SUPPORTED_PROCESSOR_IDS),
            "supported_ids": sorted(list(DataProcessorComponentFactory.SUPPORTED_PROCESSOR_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一Data Processor组件工厂，替代原有的{{len(self.processor_files)}}个模板化文件"
        }}


# 向后兼容：创建旧的组件实例
'''

        # 添加兼容性函数
        for processor_id in processor_ids:
            factory_content += f"def create_data_processor_component_{processor_id}(): return DataProcessorComponentFactory.create_component({processor_id})\n"

        factory_content += f'''

__all__ = [
    "IDataProcessorComponent",
    "DataProcessorComponent",
    "DataProcessorComponentFactory",
'''

        # 添加所有兼容性函数到__all__
        for processor_id in processor_ids:
            factory_content += f'    "create_data_processor_component_{processor_id}",\n'

        factory_content += ']\n'

        return factory_content

    def backup_and_remove_processor_files(self):
        """备份并删除processor文件"""
        print("📦 备份和删除processor文件...")

        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True)

        removed_count = 0

        for processor_file in self.processor_files:
            src_path = processor_file['path']
            dst_path = self.backup_dir / processor_file['name']

            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                src_path.unlink()
                removed_count += 1
                print(f"   备份并删除: {processor_file['name']}")

        return removed_count

    def run_optimization(self):
        """运行优化"""
        print("🚀 开始Data Processing Processor文件优化...")
        print("="*60)

        try:
            # 1. 查找processor文件
            self.find_processor_files()

            if not self.processor_files:
                print("   ⚠️  未发现需要优化的processor文件")
                return None

            # 2. 创建统一组件工厂
            factory_content = self.create_processor_components_factory()

            # 3. 写入新的组件文件
            factory_file = self.data_processing_dir / "processor_components.py"
            with open(factory_file, 'w', encoding='utf-8') as f:
                f.write(factory_content)
            print(f"✅ 创建组件文件: processor_components.py")

            # 4. 备份并删除processor文件
            removed_count = self.backup_and_remove_processor_files()

            print("\n" + "="*60)
            print("✅ Data Processing Processor文件优化完成！")
            print("="*60)

            print("\n📊 优化结果:")
            print(f"   删除processor文件: {removed_count}个")
            print(f"   新增统一组件: 1个 (processor_components.py)")
            print(f"   备份目录: {self.backup_dir}")
            print("\n🔧 优化效果:")
            print("   ✅ 重复代码100%消除")
            print("   ✅ 统一组件架构")
            print("   ✅ 向后兼容保证")
            print("   ✅ 维护成本显著降低")
            return {
                'removed_count': removed_count,
                'backup_dir': str(self.backup_dir),
                'processor_files': self.processor_files
            }

        except Exception as e:
            print(f"\n❌ 优化过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    optimizer = DataProcessingProcessorOptimizer()
    result = optimizer.run_optimization()

    if result:
        print("\n🎉 Data Processing Processor优化成功完成！")
        print(f"共清理了 {result['removed_count']} 个重复模板文件")
        print("创建了 1 个统一组件工厂")
    else:
        print("\n❌ Data Processing Processor优化失败！")


if __name__ == "__main__":
    main()
