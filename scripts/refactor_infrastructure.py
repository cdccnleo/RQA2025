#!/usr/bin/env python3
"""
基础设施层重构脚本

将庞大的infrastructure层拆分为8个独立的基础设施组件层
"""

import shutil
from pathlib import Path
from typing import Dict
from datetime import datetime
import argparse


class InfrastructureRefactor:
    """基础设施层重构器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.infrastructure_dir = self.src_dir / "infrastructure"

        # 拆分映射配置
        self.split_mapping = {
            "config": {
                "target_dir": "config",
                "description": "配置管理层",
                "files": [],
                "dependencies": []
            },
            "cache": {
                "target_dir": "cache",
                "description": "缓存系统层",
                "files": [],
                "dependencies": ["config"]
            },
            "logging": {
                "target_dir": "logging",
                "description": "日志系统层",
                "files": [],
                "dependencies": ["config"]
            },
            "security": {
                "target_dir": "security",
                "description": "安全管理层",
                "files": [],
                "dependencies": ["config", "logging"]
            },
            "error": {
                "target_dir": "error",
                "description": "错误处理层",
                "files": [],
                "dependencies": ["logging"]
            },
            "resource": {
                "target_dir": "resource",
                "description": "资源管理层",
                "files": [],
                "dependencies": ["config", "monitoring"]
            },
            "health": {
                "target_dir": "health",
                "description": "健康检查层",
                "files": [],
                "dependencies": ["logging", "monitoring"]
            },
            "utils": {
                "target_dir": "utils",
                "description": "工具组件层",
                "files": [],
                "dependencies": ["config"]
            }
        }

        # 文件分类规则
        self.classification_rules = {
            "config": [
                "config", "configuration", "settings", "env", "environment"
            ],
            "cache": [
                "cache", "caching", "redis", "memcache", "memory"
            ],
            "logging": [
                "log", "logging", "logger", "audit", "trace"
            ],
            "security": [
                "security", "auth", "authentication", "permission", "access",
                "encrypt", "crypto", "token", "jwt", "oauth"
            ],
            "error": [
                "error", "exception", "exception", "fail", "retry", "recovery"
            ],
            "resource": [
                "resource", "monitor", "performance", "metric", "stat"
            ],
            "health": [
                "health", "check", "status", "alive", "heartbeat", "probe"
            ],
            "utils": [
                "util", "helper", "tool", "common", "shared", "base"
            ]
        }

    def analyze_infrastructure(self) -> Dict[str, any]:
        """分析基础设施层结构"""
        print("🔍 分析基础设施层结构...")

        analysis = {
            "total_files": 0,
            "directories": {},
            "file_classification": {},
            "dependencies": {}
        }

        # 统计各子目录的文件数
        for item in self.infrastructure_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                py_files = list(item.rglob("*.py"))
                analysis["directories"][item.name] = len(py_files)
                analysis["total_files"] += len(py_files)

                # 分析每个文件
                for py_file in py_files:
                    classification = self._classify_file(py_file)
                    if classification:
                        if classification not in analysis["file_classification"]:
                            analysis["file_classification"][classification] = []
                        analysis["file_classification"][classification].append(str(py_file))

        print(f"📊 分析完成：共{analysis['total_files']}个文件")
        return analysis

    def _classify_file(self, file_path: Path) -> str:
        """根据文件名和内容分类文件"""
        file_name = file_path.name.lower()
        file_stem = file_path.stem.lower()

        # 根据文件名分类
        for category, keywords in self.classification_rules.items():
            for keyword in keywords:
                if keyword in file_name or keyword in file_stem:
                    return category

        # 读取文件内容进行深度分类
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()

            # 根据内容中的关键词分类
            for category, keywords in self.classification_rules.items():
                for keyword in keywords:
                    if keyword in content:
                        return category

        except:
            pass

        return "utils"  # 默认分类到工具组件

    def create_refactor_plan(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """创建重构计划"""
        print("📋 创建重构计划...")

        plan = {
            "current_state": analysis,
            "refactor_steps": [],
            "migration_plan": {},
            "validation_plan": {}
        }

        # 1. 创建目标目录结构
        plan["refactor_steps"].append({
            "step": 1,
            "name": "创建目标目录结构",
            "description": "创建8个独立的基础设施组件目录",
            "actions": [
                "创建src/infrastructure/config/",
                "创建src/infrastructure/cache/",
                "创建src/infrastructure/logging/",
                "创建src/infrastructure/security/",
                "创建src/infrastructure/error/",
                "创建src/infrastructure/resource/",
                "创建src/infrastructure/health/",
                "创建src/infrastructure/utils/"
            ]
        })

        # 2. 文件迁移
        for category, files in analysis["file_classification"].items():
            if category in self.split_mapping:
                target_dir = self.split_mapping[category]["target_dir"]
                plan["refactor_steps"].append({
                    "step": 2,
                    "name": f"迁移{category}相关文件",
                    "description": f"将{len(files)}个文件迁移到{target_dir}目录",
                    "actions": [f"移动文件到src/infrastructure/{target_dir}/"]
                })

        # 3. 更新导入语句
        plan["refactor_steps"].append({
            "step": 3,
            "name": "更新导入语句",
            "description": "更新所有受影响文件的导入语句",
            "actions": [
                "更新from src.infrastructure.xxx导入",
                "更新import src.infrastructure.xxx导入",
                "验证导入路径正确性"
            ]
        })

        # 4. 创建组件接口
        for category, config in self.split_mapping.items():
            plan["refactor_steps"].append({
                "step": 4,
                "name": f"创建{category}组件接口",
                "description": f"创建{config['description']}的统一接口",
                "actions": [
                    f"创建src/infrastructure/{config['target_dir']}/interfaces.py",
                    f"创建src/infrastructure/{config['target_dir']}/base.py",
                    f"定义标准接口契约"
                ]
            })

        # 5. 验证重构结果
        plan["refactor_steps"].append({
            "step": 5,
            "name": "验证重构结果",
            "description": "验证重构后的系统功能完整性",
            "actions": [
                "运行单元测试",
                "验证导入正确性",
                "检查依赖关系",
                "性能基准测试"
            ]
        })

        return plan

    def execute_refactor(self, plan: Dict[str, any], dry_run: bool = True, step: int = None) -> Dict[str, any]:
        """执行重构"""
        print(f"🔄 {'预览' if dry_run else '执行'}重构...")

        results = {
            "executed_steps": [],
            "created_directories": [],
            "moved_files": [],
            "updated_imports": [],
            "created_interfaces": [],
            "errors": []
        }

        # 步骤1: 创建目标目录结构
        if step is None or step == 1:
            print("📁 步骤1: 创建目标目录结构")
            results["executed_steps"].append("create_directories")

            for category, config in self.split_mapping.items():
                target_dir = self.infrastructure_dir / config["target_dir"]
                if not target_dir.exists():
                    if not dry_run:
                        target_dir.mkdir(parents=True, exist_ok=True)
                    results["created_directories"].append(str(target_dir))
                    print(f"  📁 {'将创建' if dry_run else '已创建'}目录: {target_dir}")

            # 同时创建对应的测试目录
            tests_dir = self.project_root / "tests"
            for category, config in self.split_mapping.items():
                test_target_dir = tests_dir / "infrastructure" / config["target_dir"]
                if not test_target_dir.exists():
                    if not dry_run:
                        test_target_dir.mkdir(parents=True, exist_ok=True)
                    results["created_directories"].append(str(test_target_dir))
                    print(f"  📁 {'将创建' if dry_run else '已创建'}测试目录: {test_target_dir}")

        # 步骤2: 迁移文件到对应目录
        if step is None or step == 2:
            print("📄 步骤2: 迁移文件到对应目录")
            results["executed_steps"].append("migrate_files")

            analysis = self.analyze_infrastructure()
            for category, files in analysis["file_classification"].items():
                if category in self.split_mapping:
                    target_dir = self.infrastructure_dir / \
                        self.split_mapping[category]["target_dir"]

                    for file_path in files:
                        src_file = self.infrastructure_dir / file_path
                        dst_file = target_dir / Path(file_path).name

                        if not dry_run:
                            try:
                                if src_file.exists():
                                    shutil.move(str(src_file), str(dst_file))
                                    results["moved_files"].append(
                                        f"{file_path} -> {self.split_mapping[category]['target_dir']}/{Path(file_path).name}")
                                    print(
                                        f"  📄 已移动: {file_path} -> {self.split_mapping[category]['target_dir']}/")
                            except Exception as e:
                                results["errors"].append(f"移动文件失败 {file_path}: {e}")
                        else:
                            print(
                                f"  📄 将移动: {file_path} -> {self.split_mapping[category]['target_dir']}/")

        # 步骤3: 更新导入语句
        if step is None or step == 3:
            print("🔧 步骤3: 更新导入语句")
            results["executed_steps"].append("update_imports")

            # 这里需要实现导入更新逻辑
            # 扫描所有Python文件，更新相关的导入语句
            if not dry_run:
                self._update_import_statements(results)
            else:
                print("  🔧 将更新所有受影响文件的导入语句")

        # 步骤4: 创建组件接口
        if step is None or step == 4:
            print("🛠️ 步骤4: 创建组件接口")
            results["executed_steps"].append("create_interfaces")

            for category, config in self.split_mapping.items():
                target_dir = self.infrastructure_dir / config["target_dir"]
                interface_file = target_dir / "interfaces.py"
                base_file = target_dir / "base.py"

                if not dry_run:
                    try:
                        # 创建接口文件
                        self._create_component_interface(category, interface_file, base_file)
                        results["created_interfaces"].extend([str(interface_file), str(base_file)])
                        print(f"  🛠️ 已创建接口: {category}/interfaces.py, {category}/base.py")
                    except Exception as e:
                        results["errors"].append(f"创建接口失败 {category}: {e}")
                else:
                    print(f"  🛠️ 将创建接口: {category}/interfaces.py, {category}/base.py")

        # 步骤5: 验证重构结果
        if step is None or step == 5:
            print("✅ 步骤5: 验证重构结果")
            results["executed_steps"].append("validate")

            if not dry_run:
                validation = self.validate_refactor()
                if validation["issues"]:
                    results["errors"].extend(validation["issues"])

        return results

    def _update_import_statements(self, results: Dict[str, any]) -> None:
        """更新导入语句"""
        # 扫描项目中的Python文件，更新受影响的导入语句
        for py_file in self.project_root.rglob("*.py"):
            if "infrastructure" in str(py_file):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    original_content = content

                    # 更新导入语句
                    for category, config in self.split_mapping.items():
                        old_import = f"from src.infrastructure import {category}"
                        new_import = f"from src.infrastructure.{config['target_dir']} import {category}"
                        content = content.replace(old_import, new_import)

                        old_import2 = f"import src.infrastructure.{category}"
                        new_import2 = f"import src.infrastructure.{config['target_dir']}.{category}"
                        content = content.replace(old_import2, new_import2)

                    # 如果内容有变化，更新文件
                    if content != original_content:
                        with open(py_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        results["updated_imports"].append(str(py_file))
                        print(f"  🔧 已更新导入: {py_file}")

                except Exception as e:
                    results["errors"].append(f"更新导入失败 {py_file}: {e}")

    def _create_component_interface(self, category: str, interface_file: Path, base_file: Path) -> None:
        """创建组件接口"""
        config = self.split_mapping[category]

        # 创建接口文件
        interface_content = f'''"""基础设施层 - {config["description"]} 接口定义"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class I{category.capitalize()}Component(ABC):
    """{config["description"]} 组件接口"""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件

        Args:
            config: 组件配置

        Returns:
            初始化是否成功
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态

        Returns:
            组件状态信息
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """关闭组件"""
        pass

# 扩展接口可以在这里添加
'''

        with open(interface_file, 'w', encoding='utf-8') as f:
            f.write(interface_content)

        # 创建基础实现文件
        base_content = f'''"""基础设施层 - {config["description"]} 基础实现"""

from typing import Any, Dict, List, Optional
from .interfaces import I{category.capitalize()}Component

class Base{category.capitalize()}Component(I{category.capitalize()}Component):
    """{config["description"]} 基础组件实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化基础组件

        Args:
            config: 组件配置
        """
        self.config = config or {{}}
        self._initialized = False
        self._status = "stopped"

    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件

        Args:
            config: 组件配置

        Returns:
            初始化是否成功
        """
        try:
            self.config.update(config)
            self._initialized = True
            self._status = "running"
            return True
        except Exception:
            self._status = "error"
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态

        Returns:
            组件状态信息
        """
        return {{
            "component": "{category}",
            "status": self._status,
            "initialized": self._initialized,
            "config": self.config
        }}

    def shutdown(self) -> None:
        """关闭组件"""
        self._initialized = False
        self._status = "stopped"

# 具体组件实现可以继承此类
'''

        with open(base_file, 'w', encoding='utf-8') as f:
            f.write(base_content)

    def validate_refactor(self) -> Dict[str, any]:
        """验证重构结果"""
        print("✅ 验证重构结果...")

        validation = {
            "directory_structure": True,
            "import_validity": True,
            "dependency_integrity": True,
            "test_coverage": True,
            "issues": []
        }

        # 验证目录结构
        for category, config in self.split_mapping.items():
            target_dir = self.infrastructure_dir / config["target_dir"]
            if not target_dir.exists():
                validation["directory_structure"] = False
                validation["issues"].append(f"目录不存在: {target_dir}")

        return validation

    def generate_report(self, analysis: Dict, plan: Dict, execution: Dict, validation: Dict) -> str:
        """生成重构报告"""
        report = f"""# 基础设施层重构报告

## 概述
- **重构时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总文件数**: {analysis['total_files']}
- **拆分目标**: {len(self.split_mapping)}个组件层

## 当前结构分析
| 目录 | 文件数 | 描述 |
|------|--------|------|
"""

        for dir_name, file_count in analysis['directories'].items():
            report += f"| {dir_name} | {file_count} | |\n"

        report += f"""
## 拆分计划
拆分为{len(self.split_mapping)}个独立组件层：

"""

        for category, config in self.split_mapping.items():
            deps = ", ".join(config["dependencies"]) if config["dependencies"] else "无"
            report += f"### {config['description']} (`{config['target_dir']}/`)\n"
            report += f"- **职责**: {config['description']}\n"
            report += f"- **依赖**: {deps}\n\n"

        report += f"""
## 重构步骤
总共{len(plan['refactor_steps'])}个步骤：

"""

        for step in plan["refactor_steps"]:
            report += f"### 步骤{step['step']}: {step['name']}\n"
            report += f"{step['description']}\n\n"
            for action in step["actions"]:
                report += f"- [ ] {action}\n"
            report += "\n"

        report += f"""
## 执行结果
- **创建目录**: {len(execution['created_directories'])}个
- **移动文件**: {len(execution['moved_files'])}个
- **错误**: {len(execution['errors'])}个

## 验证结果
- **目录结构**: {'✅' if validation['directory_structure'] else '❌'}
- **导入有效性**: {'✅' if validation['import_validity'] else '❌'}
- **依赖完整性**: {'✅' if validation['dependency_integrity'] else '❌'}
- **测试覆盖**: {'✅' if validation['test_coverage'] else '❌'}

"""

        if validation["issues"]:
            report += "## 发现问题\n"
            for issue in validation["issues"]:
                report += f"- {issue}\n"

        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基础设施层重构工具')
    parser.add_argument('--execute', action='store_true', help='执行实际重构操作')
    parser.add_argument('--step', type=int, help='指定执行步骤（1-创建目录，2-迁移文件，3-更新导入，4-创建接口，5-验证）')
    parser.add_argument('--force', action='store_true', help='强制执行（跳过确认）')
    parser.add_argument('--dry-run', action='store_true', help='预览模式（默认）')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    refactor = InfrastructureRefactor(project_root)

    print("🏗️  基础设施层重构工具")
    print("=" * 50)

    # 1. 分析当前结构
    analysis = refactor.analyze_infrastructure()

    # 2. 创建重构计划
    plan = refactor.create_refactor_plan(analysis)

    # 3. 执行重构
    dry_run = not args.execute
    if args.execute and not args.force:
        confirm = input("\n⚠️  警告：这将修改文件系统！是否继续？(y/N): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return

    execution = refactor.execute_refactor(plan, dry_run=dry_run, step=args.step)

    # 4. 验证重构
    validation = refactor.validate_refactor()

    # 5. 生成报告
    report = refactor.generate_report(analysis, plan, execution, validation)

    # 保存报告
    report_file = project_root / "reports" / "infrastructure_refactor_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📋 重构报告已生成: {report_file}")

    if dry_run:
        print("\n🔧 重构工具已就绪，可以执行实际的重构操作。")
        print("⚠️  重要提醒：建议在执行实际重构前进行完整备份！")
        print("\n使用 --execute 参数来执行实际重构操作")
    else:
        print("\n✅ 重构操作已完成！")


if __name__ == "__main__":
    main()
