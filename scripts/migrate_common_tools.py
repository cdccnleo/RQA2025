#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用工具迁移脚本
Migrate Common Tools to Appropriate Directories

将 src/backtest/ 中保留的通用工具迁移到更合适的位置
"""

import sys
import shutil
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CommonToolsMigrator:
    """通用工具迁移器"""

    def __init__(self):
        """初始化迁移器"""
        self.project_root = project_root
        self.backup_dir = project_root / "backups" / \
            f"common_tools_migration_{self.get_timestamp()}"

        # 迁移计划
        self.migration_plan = {
            # 可视化工具迁移到 core 目录
            "src/backtest/visualization.py": "src/core/visualization.py",
            "src/backtest/visualizer.py": "src/core/visualizer.py",

            # 通用工具函数迁移到 utils 目录
            "src/strategy/backtest/utils/backtest_utils.py": "src/utils/backtest_utils.py",
        }

        # 需要保留在原地的专用工具
        self.keep_in_place = {
            "src/backtest/config_manager.py",  # 回测专用配置管理器
            "src/backtest/data_loader.py",     # 回测专用数据加载器
        }

    def get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_backup(self, source_path: Path, backup_path: Path):
        """创建备份"""
        try:
            if source_path.exists():
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, backup_path)
                print(f"✅ 已备份: {source_path} -> {backup_path}")
        except Exception as e:
            print(f"❌ 备份失败: {source_path} - {e}")

    def migrate_visualization_tools(self) -> Dict[str, Any]:
        """迁移可视化工具"""
        print("🎨 迁移可视化工具到 src/core/...")

        results = {
            "migrated_files": 0,
            "skipped_files": 0,
            "errors": []
        }

        # 确保目标目录存在
        core_dir = self.project_root / "src" / "core"
        core_dir.mkdir(exist_ok=True)

        for source_str, target_str in self.migration_plan.items():
            if "visualization" in source_str:
                source_path = self.project_root / source_str
                target_path = self.project_root / target_str

                if source_path.exists():
                    try:
                        # 备份源文件
                        backup_path = self.backup_dir / source_str
                        self.create_backup(source_path, backup_path)

                        # 复制到新位置
                        shutil.copy2(source_path, target_path)
                        print(f"✅ 迁移可视化工具: {source_str} -> {target_str}")

                        # 删除原文件
                        source_path.unlink()
                        print(f"🗑️ 删除原文件: {source_str}")

                        results["migrated_files"] += 1

                    except Exception as e:
                        results["errors"].append(f"{source_str}: {e}")
                        print(f"❌ 迁移失败: {source_str} - {e}")
                else:
                    results["skipped_files"] += 1
                    print(f"⚠️ 源文件不存在: {source_str}")

        return results

    def migrate_utils(self) -> Dict[str, Any]:
        """迁移工具函数"""
        print("🔧 迁移工具函数到 src/utils/...")

        results = {
            "migrated_files": 0,
            "skipped_files": 0,
            "errors": []
        }

        # 确保目标目录存在
        utils_dir = self.project_root / "src" / "utils"
        utils_dir.mkdir(exist_ok=True)

        for source_str, target_str in self.migration_plan.items():
            if "utils" in source_str and "backtest_utils" in source_str:
                source_path = self.project_root / source_str
                target_path = self.project_root / target_str

                if source_path.exists():
                    try:
                        # 备份源文件
                        backup_path = self.backup_dir / source_str
                        self.create_backup(source_path, backup_path)

                        # 复制到新位置
                        shutil.copy2(source_path, target_path)
                        print(f"✅ 迁移工具函数: {source_str} -> {target_str}")

                        # 删除原文件
                        source_path.unlink()
                        print(f"🗑️ 删除原文件: {source_str}")

                        # 清理空的目录结构
                        self.cleanup_empty_dirs(source_path.parent)

                        results["migrated_files"] += 1

                    except Exception as e:
                        results["errors"].append(f"{source_str}: {e}")
                        print(f"❌ 迁移失败: {source_str} - {e}")
                else:
                    results["skipped_files"] += 1
                    print(f"⚠️ 源文件不存在: {source_str}")

        return results

    def cleanup_empty_dirs(self, dir_path: Path):
        """清理空的目录"""
        try:
            # 向上清理空的父目录
            current = dir_path
            while current != self.project_root / "src":
                if current.exists() and not any(current.iterdir()):
                    current.rmdir()
                    print(f"🗑️ 删除空目录: {current.relative_to(self.project_root)}")
                else:
                    break
                current = current.parent
        except Exception as e:
            print(f"⚠️ 清理目录时出错: {e}")

    def update_imports(self) -> Dict[str, Any]:
        """更新导入语句"""
        print("🔄 更新导入语句...")

        results = {
            "updated_files": 0,
            "errors": []
        }

        # 需要更新的导入语句
        import_updates = {
            "from src.strategy.backtest.utils.backtest_utils import": "from src.utils.backtest_utils import",
        }

        # 搜索需要更新的文件
        search_dirs = [
            self.project_root / "src" / "strategy",
            self.project_root / "src" / "backtest"
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for py_file in search_dir.rglob("*.py"):
                    if py_file.is_file():
                        try:
                            with open(py_file, 'r', encoding='utf-8') as f:
                                content = f.read()

                            updated = False
                            for old_import, new_import in import_updates.items():
                                if old_import in content:
                                    content = content.replace(old_import, new_import)
                                    updated = True
                                    print(f"🔧 更新导入: {py_file.relative_to(self.project_root)}")

                            if updated:
                                with open(py_file, 'w', encoding='utf-8') as f:
                                    f.write(content)
                                results["updated_files"] += 1

                        except Exception as e:
                            results["errors"].append(f"{py_file}: {e}")
                            print(f"❌ 更新导入失败: {py_file} - {e}")

        return results

    def create_visualization_module(self) -> Dict[str, Any]:
        """创建可视化模块的 __init__.py"""
        print("📦 创建可视化模块...")

        results = {"created": False, "error": None}

        try:
            # 检查是否已有 __init__.py
            init_file = self.project_root / "src" / "core" / "__init__.py"

            if not init_file.exists():
                # 创建基本的 __init__.py
                init_content = '''"""
RQA2025 Core Services
"""

from .visualization import *
from .visualizer import *

__all__ = [
    # 可视化相关
    "BacktestVisualizer",
]
'''
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(init_content)

                results["created"] = True
                print("✅ 创建可视化模块 __init__.py")

        except Exception as e:
            results["error"] = str(e)
            print(f"❌ 创建可视化模块失败: {e}")

        return results

    def create_utils_module(self) -> Dict[str, Any]:
        """更新工具模块的 __init__.py"""
        print("📦 更新工具模块...")

        results = {"updated": False, "error": None}

        try:
            init_file = self.project_root / "src" / "utils" / "__init__.py"

            # 读取现有内容
            if init_file.exists():
                with open(init_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = '''"""
RQA2025 Utilities
"""

'''

            # 添加 backtest_utils 导入
            if "backtest_utils" not in content:
                content += '''
from .backtest_utils import *

__all__ = [
    # 回测工具
    "BacktestUtils",
    "StrategyValidationResult",
]
'''
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                results["updated"] = True
                print("✅ 更新工具模块 __init__.py")

        except Exception as e:
            results["error"] = str(e)
            print(f"❌ 更新工具模块失败: {e}")

        return results

    def execute_migration(self) -> Dict[str, Any]:
        """
        执行完整的通用工具迁移

        Returns:
            Dict[str, Any]: 迁移结果
        """
        print("🚀 开始执行通用工具迁移工作...")

        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"📦 备份目录: {self.backup_dir}")

        # 执行迁移步骤
        results = {
            "visualization_migration": self.migrate_visualization_tools(),
            "utils_migration": self.migrate_utils(),
            "import_updates": self.update_imports(),
            "module_creation": {
                "visualization": self.create_visualization_module(),
                "utils": self.create_utils_module()
            }
        }

        # 生成迁移报告
        self.generate_migration_report(results)

        return results

    def generate_migration_report(self, results: Dict[str, Any]):
        """生成迁移报告"""
        report_path = self.project_root / "docs" / "strategy" / "COMMON_TOOLS_MIGRATION_REPORT.md"

        report_content = f"""# 通用工具迁移报告

## 📊 迁移概况

- **迁移时间**: {self.get_timestamp()}
- **备份位置**: `{self.backup_dir}`

## 🎨 可视化工具迁移

- **迁移文件数**: {results['visualization_migration']['migrated_files']}
- **跳过文件数**: {results['visualization_migration']['skipped_files']}
- **错误数**: {len(results['visualization_migration']['errors'])}

### 迁移详情
- `src/backtest/visualization.py` → `src/core/visualization.py`
- `src/backtest/visualizer.py` → `src/core/visualizer.py`

## 🔧 工具函数迁移

- **迁移文件数**: {results['utils_migration']['migrated_files']}
- **跳过文件数**: {results['utils_migration']['skipped_files']}
- **错误数**: {len(results['utils_migration']['errors'])}

### 迁移详情
- `src/strategy/backtest/utils/backtest_utils.py` → `src/utils/backtest_utils.py`

## 🔄 导入更新

- **更新文件数**: {results['import_updates']['updated_files']}
- **错误数**: {len(results['import_updates']['errors'])}

## 📦 模块创建

- **可视化模块**: {"✅ 创建成功" if results['module_creation']['visualization']['created'] else "❌ 创建失败"}
- **工具模块**: {"✅ 更新成功" if results['module_creation']['utils']['updated'] else "❌ 更新失败"}

## 🗂️ 保留文件

以下文件因其专用性而保留在原位置：

### src/backtest/ 保留文件
- `config_manager.py` - 回测专用配置管理器
- `data_loader.py` - 回测专用数据加载器
- `__init__.py` - 包结构文件
- 各子目录的 `__init__.py`

## ✅ 迁移完成

通用工具已成功迁移到适当的位置：
- **可视化工具** → `src/core/`
- **工具函数** → `src/utils/`
- **专用工具** → 保留在 `src/backtest/`

---
**迁移报告生成时间**: {self.get_timestamp()}
"""

        # 确保目录存在
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📋 迁移报告已生成: {report_path}")


def main():
    """主函数"""
    try:
        migrator = CommonToolsMigrator()
        results = migrator.execute_migration()

        print("\n" + "="*60)
        print("🎉 通用工具迁移完成！")
        print("="*60)

        viz = results["visualization_migration"]
        utils = results["utils_migration"]
        imports = results["import_updates"]

        print("\n📊 迁移统计:")
        print(f"  • 可视化工具迁移: {viz['migrated_files']} 个文件")
        print(f"  • 工具函数迁移: {utils['migrated_files']} 个文件")
        print(f"  • 导入更新: {imports['updated_files']} 个文件")

        print(f"\n📦 备份位置: {migrator.backup_dir}")
        print("\n✅ 通用工具已迁移到最合适的位置！")

    except Exception as e:
        print(f"❌ 迁移过程异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
