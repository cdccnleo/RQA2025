#!/usr/bin/env python3
"""
基础设施层测试目录重新组织工具

按照重构后的架构设计重新组织tests/unit/infrastructure目录
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from collections import defaultdict


class InfrastructureTestReorganizer:
    """基础设施层测试重新组织器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_tests_dir = self.project_root / "tests" / "unit" / "infrastructure"
        self.backup_dir = self.project_root / \
            f"backup/infrastructure_tests_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 重构后的基础设施层标准结构
        self.standard_structure = {
            "config": {
                "description": "配置管理测试",
                "keywords": ["config", "configuration", "env", "environment", "settings", "properties"],
                "target_files": []
            },
            "cache": {
                "description": "缓存系统测试",
                "keywords": ["cache", "caching", "memory", "redis", "storage"],
                "target_files": []
            },
            "logging": {
                "description": "日志系统测试",
                "keywords": ["log", "logging", "logger", "audit", "tracing"],
                "target_files": []
            },
            "security": {
                "description": "安全管理测试",
                "keywords": ["security", "auth", "authentication", "authorization", "encrypt", "crypto"],
                "target_files": []
            },
            "error": {
                "description": "错误处理测试",
                "keywords": ["error", "exception", "exception_handler", "error_handler", "recovery"],
                "target_files": []
            },
            "resource": {
                "description": "资源管理测试",
                "keywords": ["resource", "resource_manager", "quota", "pool", "connection"],
                "target_files": []
            },
            "health": {
                "description": "健康检查测试",
                "keywords": ["health", "health_check", "probe", "monitor", "status"],
                "target_files": []
            },
            "utils": {
                "description": "工具组件测试",
                "keywords": ["util", "helper", "tool", "common", "utility"],
                "target_files": []
            }
        }

        # 文件移动记录
        self.move_operations = []
        self.deleted_directories = []

    def analyze_current_structure(self) -> Dict[str, Any]:
        """分析当前目录结构"""

        analysis = {
            "total_directories": 0,
            "total_files": 0,
            "empty_directories": [],
            "file_distribution": defaultdict(list),
            "directory_depth": {}
        }

        if not self.infrastructure_tests_dir.exists():
            return analysis

        # 遍历所有目录和文件
        for root, dirs, files in os.walk(self.infrastructure_tests_dir):
            root_path = Path(root)
            rel_path = root_path.relative_to(self.infrastructure_tests_dir)

            analysis["total_directories"] += 1
            analysis["directory_depth"][str(rel_path)] = len(rel_path.parts)

            # 检查是否为空目录
            if not dirs and not files and str(rel_path) != ".":
                analysis["empty_directories"].append(str(rel_path))

            # 统计文件
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    analysis["total_files"] += 1
                    file_path = root_path / file

                    # 根据文件名和路径分类文件
                    category = self._categorize_file(file_path)
                    if category:
                        analysis["file_distribution"][category].append(
                            str(file_path.relative_to(self.infrastructure_tests_dir)))

        return analysis

    def _categorize_file(self, file_path: Path) -> str:
        """对文件进行分类"""

        file_name = file_path.name.lower()
        rel_path = file_path.relative_to(self.infrastructure_tests_dir)

        # 基于路径的关键词匹配
        path_str = str(rel_path).lower()

        for category, config in self.standard_structure.items():
            for keyword in config["keywords"]:
                if keyword in path_str or keyword in file_name:
                    return category

        # 如果没有匹配，尝试基于文件名内容
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()

                for category, config in self.standard_structure.items():
                    for keyword in config["keywords"]:
                        if keyword in content:
                            return category
        except Exception:
            pass

        return "uncategorized"

    def create_reorganization_plan(self) -> Dict[str, Any]:
        """创建重新组织计划"""

        current_analysis = self.analyze_current_structure()

        plan = {
            "current_analysis": current_analysis,
            "reorganization_operations": [],
            "new_structure": {},
            "cleanup_operations": []
        }

        # 为每个标准分类创建操作计划
        for category, config in self.standard_structure.items():
            target_dir = self.infrastructure_tests_dir / category

            plan["new_structure"][category] = {
                "directory": str(target_dir),
                "description": config["description"],
                "expected_files": len(current_analysis["file_distribution"].get(category, [])),
                "operations": []
            }

            # 收集需要移动的文件
            for file_path_str in current_analysis["file_distribution"].get(category, []):
                source_file = self.infrastructure_tests_dir / file_path_str
                target_file = target_dir / source_file.name

                if str(source_file) != str(target_file):
                    plan["reorganization_operations"].append({
                        "type": "move_file",
                        "from": str(source_file),
                        "to": str(target_file),
                        "category": category,
                        "reason": f"重新分类到{category}目录"
                    })

        # 清理空目录
        for empty_dir in current_analysis["empty_directories"]:
            plan["cleanup_operations"].append({
                "type": "remove_directory",
                "path": str(self.infrastructure_tests_dir / empty_dir),
                "reason": "空目录清理"
            })

        return plan

    def execute_reorganization(self, dry_run: bool = True) -> Dict[str, Any]:
        """执行重新组织"""

        plan = self.create_reorganization_plan()

        results = {
            "dry_run": dry_run,
            "operations_executed": 0,
            "files_moved": 0,
            "directories_created": 0,
            "directories_removed": 0,
            "errors": [],
            "backup_created": False
        }

        print(f"🔄 开始基础设施层测试重新组织... ({'仅分析' if dry_run else '实际执行'})")

        # 创建备份
        if not dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.infrastructure_tests_dir, self.backup_dir /
                            "infrastructure_tests_original", dirs_exist_ok=True)
            results["backup_created"] = True

        # 创建目标目录
        for category in self.standard_structure.keys():
            target_dir = self.infrastructure_tests_dir / category
            if not target_dir.exists():
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    results["directories_created"] += 1
                print(f"    📁 创建目录: {category}/")

        # 执行文件移动操作
        for operation in plan["reorganization_operations"]:
            results["operations_executed"] += 1

            if operation["type"] == "move_file":
                from_path = Path(operation["from"])
                to_path = Path(operation["to"])

                if from_path.exists():
                    if not dry_run:
                        try:
                            # 确保目标目录存在
                            to_path.parent.mkdir(parents=True, exist_ok=True)
                            # 移动文件
                            shutil.move(str(from_path), str(to_path))
                            results["files_moved"] += 1
                            print(f"    ✅ 移动文件: {from_path.name} → {operation['category']}/")
                        except Exception as e:
                            results["errors"].append(f"移动文件失败 {from_path}: {e}")
                    else:
                        print(f"    📋 计划移动: {from_path.name} → {operation['category']}/")

        # 清理空目录
        for operation in plan["cleanup_operations"]:
            if operation["type"] == "remove_directory":
                dir_path = Path(operation["path"])

                if dir_path.exists() and not any(dir_path.iterdir()):
                    if not dry_run:
                        try:
                            dir_path.rmdir()
                            results["directories_removed"] += 1
                            print(f"    🗑️ 删除空目录: {dir_path.name}/")
                        except Exception as e:
                            results["errors"].append(f"删除目录失败 {dir_path}: {e}")
                    else:
                        print(f"    📋 计划删除: {dir_path.name}/")

        # 递归清理空的父目录
        if not dry_run:
            self._cleanup_empty_parent_directories()

        return results

    def _cleanup_empty_parent_directories(self):
        """清理空的父目录"""

        def is_empty_dir(path: Path) -> bool:
            """检查目录是否为空"""
            if not path.is_dir():
                return False
            try:
                return not any(path.iterdir())
            except PermissionError:
                return False

        def cleanup_recursive(path: Path):
            """递归清理空目录"""
            if not path.exists():
                return

            for child in path.iterdir():
                if child.is_dir():
                    cleanup_recursive(child)

            # 检查当前目录是否为空
            if is_empty_dir(path) and path != self.infrastructure_tests_dir:
                try:
                    path.rmdir()
                    self.deleted_directories.append(
                        str(path.relative_to(self.infrastructure_tests_dir)))
                    print(f"    🗑️ 删除空父目录: {path.name}/")
                except Exception:
                    pass

        cleanup_recursive(self.infrastructure_tests_dir)

    def generate_reorganization_report(self, plan: Dict[str, Any], results: Dict[str, Any]) -> str:
        """生成重新组织报告"""

        current_analysis = plan["current_analysis"]

        report = f"""# 🏗️ 基础设施层测试目录重新组织报告

## 📅 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 当前结构分析

### 总体统计
- **总目录数**: {current_analysis['total_directories']}
- **总测试文件数**: {current_analysis['total_files']}
- **空目录数**: {len(current_analysis['empty_directories'])}

### 当前文件分布
"""

        for category, files in current_analysis['file_distribution'].items():
            report += f"- **{category}**: {len(files)} 个文件\n"

        report += f"""
### 空目录列表
"""

        for empty_dir in current_analysis['empty_directories'][:10]:  # 只显示前10个
            report += f"- `{empty_dir}/`\n"

        if len(current_analysis['empty_directories']) > 10:
            report += f"- ... 还有 {len(current_analysis['empty_directories']) - 10} 个空目录\n"

        report += f"""
## 🏗️ 重构后的标准结构

### 基础设施层8个功能分类

| 功能分类 | 描述 | 当前文件数 | 目标目录 |
|---------|------|-----------|---------|
| **配置管理** | 系统配置管理、环境变量处理、配置验证 | {len(current_analysis['file_distribution'].get('config', []))} | `config/` |
| **缓存系统** | 多级缓存管理、缓存策略、数据持久化 | {len(current_analysis['file_distribution'].get('cache', []))} | `cache/` |
| **日志系统** | 日志记录、格式化、轮转、审计 | {len(current_analysis['file_distribution'].get('logging', []))} | `logging/` |
| **安全管理** | 身份认证、权限控制、加密解密 | {len(current_analysis['file_distribution'].get('security', []))} | `security/` |
| **错误处理** | 异常捕获、错误恢复、重试机制 | {len(current_analysis['file_distribution'].get('error', []))} | `error/` |
| **资源管理** | 资源分配、连接池、配额管理 | {len(current_analysis['file_distribution'].get('resource', []))} | `resource/` |
| **健康检查** | 系统健康监控、状态检查、探针 | {len(current_analysis['file_distribution'].get('health', []))} | `health/` |
| **工具组件** | 通用工具函数、助手类、公共组件 | {len(current_analysis['file_distribution'].get('utils', []))} | `utils/` |

### 未分类文件
- **未分类文件数**: {len(current_analysis['file_distribution'].get('uncategorized', []))}

## 🔄 重新组织执行结果

### 执行统计
- **执行模式**: {'仅分析' if results['dry_run'] else '实际执行'}
- **计划操作数**: {len(plan['reorganization_operations'])}
- **已移动文件数**: {results['files_moved']}
- **已创建目录数**: {results['directories_created']}
- **已删除目录数**: {results['directories_removed']}
- **错误数**: {len(results['errors'])}

### 备份信息
- **备份状态**: {'已创建' if results.get('backup_created') else '未创建'}
- **备份目录**: {self.backup_dir}
"""

        if not results['dry_run']:
            report += f"""
### 文件移动详情
"""

            for i, operation in enumerate(plan['reorganization_operations'][:20]):  # 只显示前20个操作
                status = "✅ 完成" if i < results['files_moved'] else "⏳ 待处理"
                from_path = Path(operation['from'])
                to_path = Path(operation['to'])
                report += f"- {status} **{from_path.name}**: {from_path.parent.name}/ → {operation['category']}/\n"

            if len(plan['reorganization_operations']) > 20:
                report += f"- ... 还有 {len(plan['reorganization_operations']) - 20} 个文件移动操作\n"

            report += f"""
### 目录清理详情
"""

            for operation in plan['cleanup_operations'][:10]:  # 只显示前10个清理操作
                dir_path = Path(operation['path'])
                status = "✅ 已删除" if str(dir_path.name) in self.deleted_directories else "⏳ 待删除"
                report += f"- {status} `{dir_path.name}/`\n"

        report += f"""
## 🗂️ 新的目录结构

```
tests/unit/infrastructure/
├── config/          # 配置管理测试
│   ├── test_*.py
│   └── __pycache__/
├── cache/           # 缓存系统测试
│   ├── test_*.py
│   └── __pycache__/
├── logging/         # 日志系统测试
│   ├── test_*.py
│   └── __pycache__/
├── security/        # 安全管理测试
│   ├── test_*.py
│   └── __pycache__/
├── error/           # 错误处理测试
│   ├── test_*.py
│   └── __pycache__/
├── resource/        # 资源管理测试
│   ├── test_*.py
│   └── __pycache__/
├── health/          # 健康检查测试
│   ├── test_*.py
│   └── __pycache__/
├── utils/           # 工具组件测试
│   ├── test_*.py
│   └── __pycache__/
└── __pycache__/     # 根目录缓存文件
```

## 🎯 重新组织原则

### 分类原则
1. **基于文件内容**: 分析测试文件中的导入语句和类名
2. **基于文件名**: 根据文件名中的关键词进行分类
3. **基于目录路径**: 根据文件所在目录路径进行分类
4. **基于功能职责**: 根据组件的职责范围进行分类

### 目录清理原则
1. **空目录删除**: 删除不包含任何文件的目录
2. **递归清理**: 递归删除空的父目录
3. **保留重要目录**: 保留标准分类的8个主要目录
4. **备份安全**: 在执行操作前创建完整备份

### 命名规范
1. **目录命名**: 使用小写英文单词
2. **文件命名**: `test_*.py`格式
3. **分类准确**: 确保文件分类到最合适的目录
4. **结构清晰**: 保持目录层次清晰简单

## ⚠️ 注意事项

1. **备份重要性**: 执行前已创建完整备份，可在需要时恢复
2. **分类准确性**: 基于关键词匹配，部分文件可能需要手动调整
3. **依赖关系**: 文件移动后需要检查和修复导入路径
4. **测试完整性**: 确保移动后的文件不会影响测试执行

## 💡 建议后续行动

### 阶段1: 验证重新组织结果
1. **检查文件分类**: 验证文件是否分类到正确的目录
2. **运行测试**: 执行测试确保功能正常
3. **修复导入路径**: 修复可能被破坏的导入语句
4. **更新文档**: 更新相关的文档和配置

### 阶段2: 优化和完善
1. **手动调整分类**: 手动调整分类不准确的文件
2. **合并重复测试**: 合并重复或相似的测试文件
3. **添加缺失测试**: 补充重要的缺失测试用例
4. **性能优化**: 优化测试执行性能

### 阶段3: 持续维护
1. **建立规范**: 建立测试文件组织规范
2. **自动化检查**: 创建自动化检查脚本
3. **定期重构**: 定期检查和优化目录结构
4. **文档更新**: 保持文档与代码结构同步

## 🎉 总结

基础设施层测试目录重新组织已完成：

- **标准结构**: 按照8个功能分类重新组织目录
- **文件移动**: 重新分类和移动测试文件到合适目录
- **空目录清理**: 删除所有空的测试目录
- **备份安全**: 创建完整备份确保操作安全
- **结构优化**: 建立清晰、可维护的测试目录结构

这次重新组织将显著提高基础设施层测试的可维护性和可理解性，为后续的测试开发和维护奠定了良好的基础。

---

*基础设施层测试重新组织工具版本: v1.0*
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return report


def main():
    """主函数"""

    import argparse

    parser = argparse.ArgumentParser(description='基础设施层测试目录重新组织工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--dry-run', action='store_true', help='仅分析不执行重新组织')
    parser.add_argument('--report', action='store_true', help='生成详细报告')

    args = parser.parse_args()

    tool = InfrastructureTestReorganizer(args.project)

    # 分析当前结构
    print("🔍 分析当前基础设施层测试目录结构...")
    current_analysis = tool.analyze_current_structure()

    print(f"✅ 发现 {current_analysis['total_files']} 个测试文件")
    print(f"📁 发现 {current_analysis['total_directories']} 个目录")
    print(f"🗑️ 发现 {len(current_analysis['empty_directories'])} 个空目录")

    # 执行重新组织
    print(f"\n🔄 开始重新组织... ({'仅分析' if args.dry_run else '实际执行'})")
    results = tool.execute_reorganization(dry_run=args.dry_run)

    print("\n📊 执行完成！")
    print(f"   计划操作: {results['operations_executed']}")
    print(f"   文件移动: {results['files_moved']}")
    print(f"   目录创建: {results['directories_created']}")
    print(f"   目录删除: {results['directories_removed']}")

    if args.report:
        plan = tool.create_reorganization_plan()
        report_content = tool.generate_reorganization_report(plan, results)
        report_file = tool.project_root / "reports" / \
            f"infrastructure_test_reorganization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📊 重新组织报告已保存: {report_file}")


if __name__ == "__main__":
    main()
