#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解决Health模块代码重叠问题
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class HealthOverlapResolver:
    """Health模块重叠解决器"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "backup" / \
            f"health_overlap_resolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.health_dirs = [
            "src/infrastructure/core/health",
            "src/infrastructure/health"
        ]

    def create_backup(self):
        """创建备份"""
        print("🔧 创建备份...")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        for health_dir in self.health_dirs:
            if Path(health_dir).exists():
                backup_path = self.backup_dir / health_dir.replace("src/", "")
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(health_dir, backup_path, dirs_exist_ok=True)
                print(f"   ✓ 备份 {health_dir} -> {backup_path}")

    def analyze_overlap(self) -> Dict[str, Any]:
        """分析重叠情况"""
        print("🔍 分析代码重叠...")

        overlap_analysis = {
            "health_checkers": [],
            "duplicate_classes": [],
            "similar_functions": [],
            "recommendations": []
        }

        # 分析health_checker.py文件
        health_checker_files = [
            "src/infrastructure/core/health/unified_health_checker.py",
            "src/infrastructure/health/health_checker.py",
            "src/infrastructure/health/enhanced_health_checker.py",
            "src/infrastructure/health/core/checker.py"
        ]

        for file_path in health_checker_files:
            if Path(file_path).exists():
                overlap_analysis["health_checkers"].append({
                    "file": file_path,
                    "size": Path(file_path).stat().st_size,
                    "lines": len(Path(file_path).read_text(encoding='utf-8').splitlines())
                })

        # 检测重复的类名
        class_names = set()
        for file_path in health_checker_files:
            if Path(file_path).exists():
                content = Path(file_path).read_text(encoding='utf-8')
                for line in content.splitlines():
                    if line.strip().startswith("class ") and ":" in line:
                        class_name = line.strip().split("class ")[1].split(
                            "(")[0].split(":")[0].strip()
                        if class_name in class_names:
                            overlap_analysis["duplicate_classes"].append(class_name)
                        class_names.add(class_name)

        # 生成建议
        if len(overlap_analysis["health_checkers"]) > 1:
            overlap_analysis["recommendations"].append({
                "type": "merge",
                "description": "合并多个健康检查器实现",
                "action": "保留最完整的实现，删除重复文件"
            })

        if overlap_analysis["duplicate_classes"]:
            overlap_analysis["recommendations"].append({
                "type": "rename",
                "description": "重命名重复的类名",
                "action": "为每个实现添加前缀以区分"
            })

        return overlap_analysis

    def merge_health_checkers(self):
        """合并健康检查器"""
        print("🔗 合并健康检查器...")

        # 选择主要实现（enhanced_health_checker.py作为主要实现）
        primary_file = "src/infrastructure/health/enhanced_health_checker.py"
        target_file = "src/infrastructure/core/health/unified_health_checker.py"

        if Path(primary_file).exists():
            # 备份当前的主要实现
            if Path(target_file).exists():
                backup_target = self.backup_dir / "unified_health_checker_backup.py"
                shutil.copy2(target_file, backup_target)
                print(f"   ✓ 备份 {target_file} -> {backup_target}")

            # 复制增强版到统一位置
            shutil.copy2(primary_file, target_file)
            print(f"   ✓ 复制 {primary_file} -> {target_file}")

            # 更新导入路径
            self._update_imports(target_file)

    def _update_imports(self, file_path: str):
        """更新导入路径"""
        if not Path(file_path).exists():
            return

        content = Path(file_path).read_text(encoding='utf-8')

        # 更新相对导入
        updated_content = content.replace(
            "from ..config.unified_manager import UnifiedConfigManager",
            "from ...config.unified_manager import UnifiedConfigManager"
        )

        # 更新其他导入
        updated_content = updated_content.replace(
            "from src.infrastructure.interfaces.base import IHealthChecker",
            "from ...interfaces.base import IHealthChecker"
        )

        Path(file_path).write_text(updated_content, encoding='utf-8')
        print(f"   ✓ 更新导入路径 {file_path}")

    def remove_duplicate_files(self):
        """删除重复文件"""
        print("🗑️  删除重复文件...")

        files_to_remove = [
            "src/infrastructure/health/health_checker.py",
            "src/infrastructure/health/enhanced_health_checker.py",
            "src/infrastructure/health/core/checker.py"
        ]

        for file_path in files_to_remove:
            if Path(file_path).exists():
                # 移动到备份目录而不是直接删除
                backup_path = self.backup_dir / file_path.replace("src/", "")
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(file_path, backup_path)
                print(f"   ✓ 移动 {file_path} -> {backup_path}")

        # 清理空目录
        empty_dirs = [
            "src/infrastructure/health/core",
            "src/infrastructure/health"
        ]

        for dir_path in empty_dirs:
            if Path(dir_path).exists() and not any(Path(dir_path).iterdir()):
                Path(dir_path).rmdir()
                print(f"   ✓ 删除空目录 {dir_path}")

    def update_init_files(self):
        """更新__init__.py文件"""
        print("📝 更新__init__.py文件...")

        # 更新core/health/__init__.py
        core_health_init = "src/infrastructure/core/health/__init__.py"
        if not Path(core_health_init).exists():
            init_content = '''"""
统一健康检查模块
"""

from .unified_health_checker import UnifiedHealthChecker, HealthStatus, HealthCheck

__all__ = [
    'UnifiedHealthChecker',
    'HealthStatus', 
    'HealthCheck'
]
'''
            Path(core_health_init).write_text(init_content, encoding='utf-8')
            print(f"   ✓ 创建 {core_health_init}")

    def create_migration_guide(self):
        """创建迁移指南"""
        print("📋 创建迁移指南...")

        guide_content = f"""# Health模块重叠解决报告

## 问题描述
发现多个健康检查器实现存在功能重叠：
- src/infrastructure/core/health/unified_health_checker.py
- src/infrastructure/health/health_checker.py  
- src/infrastructure/health/enhanced_health_checker.py
- src/infrastructure/health/core/checker.py

## 解决方案
1. 保留最完整的实现：enhanced_health_checker.py
2. 将其移动到统一位置：core/health/unified_health_checker.py
3. 删除其他重复实现
4. 更新导入路径和__init__.py文件

## 迁移后的结构
```
src/infrastructure/core/health/
├── __init__.py
└── unified_health_checker.py  # 统一的健康检查器实现
```

## 使用方式
```python
from src.infrastructure.health import EnhancedHealthChecker, HealthStatus

# 创建健康检查器
checker = UnifiedHealthChecker()

# 注册检查函数
checker.register_check("database", lambda: True)

# 执行检查
status = checker.check_health()
```

## 注意事项
- 所有重复文件已备份到 {self.backup_dir}
- 如需恢复，可从备份目录复制文件
- 请更新所有引用旧路径的代码
"""

        guide_path = self.backup_dir / "MIGRATION_GUIDE.md"
        guide_path.write_text(guide_content, encoding='utf-8')
        print(f"   ✓ 创建迁移指南 {guide_path}")

    def run(self):
        """执行重叠解决流程"""
        print("🚀 开始解决Health模块重叠问题...")

        try:
            # 1. 创建备份
            self.create_backup()

            # 2. 分析重叠
            analysis = self.analyze_overlap()
            print(f"   📊 发现 {len(analysis['health_checkers'])} 个健康检查器文件")
            print(f"   📊 发现 {len(analysis['duplicate_classes'])} 个重复类名")

            # 3. 合并实现
            self.merge_health_checkers()

            # 4. 删除重复文件
            self.remove_duplicate_files()

            # 5. 更新初始化文件
            self.update_init_files()

            # 6. 创建迁移指南
            self.create_migration_guide()

            print("✅ Health模块重叠问题解决完成！")
            print(f"📁 备份位置: {self.backup_dir}")

        except Exception as e:
            print(f"❌ 解决过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    resolver = HealthOverlapResolver()
    resolver.run()


if __name__ == "__main__":
    main()
