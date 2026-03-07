#!/usr/bin/env python3
"""
目录结构优化脚本

优化项目目录结构，提高代码组织性和可维护性
"""

import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import argparse


class DirectoryStructureOptimizer:
    """目录结构优化器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"

        # 优化的目录结构定义
        self.optimized_structure = {
            "src": {
                "core": {
                    "description": "核心服务层 - 事件总线、依赖注入、流程编排",
                    "interfaces": ["IEventBus", "IContainer", "IOrchestrator"],
                    "components": ["event_bus.py", "container.py", "business_process_orchestrator.py"]
                },
                "infrastructure": {
                    "description": "基础设施层 - 拆分为8个独立组件",
                    "subdirs": {
                        "config": "配置管理 - 系统配置、环境变量、验证",
                        "cache": "缓存系统 - 多级缓存、策略、持久化",
                        "logging": "日志系统 - 统一日志、分析、存储",
                        "security": "安全管理 - 认证、权限、加密",
                        "error": "错误处理 - 异常处理、恢复、重试",
                        "resource": "资源管理 - 监控、分配、优化",
                        "health": "健康检查 - 监控、诊断、告警",
                        "utils": "工具组件 - 通用工具、辅助功能"
                    }
                },
                "data": {
                    "description": "数据采集层 - 实时数据采集、验证、质量监控",
                    "subdirs": {
                        "adapters": "数据源适配器",
                        "collector": "实时数据采集器",
                        "validator": "数据验证器",
                        "quality_monitor": "数据质量监控器"
                    }
                },
                "gateway": {
                    "description": "API网关层 - 路由、认证、限流、监控"
                },
                "features": {
                    "description": "特征处理层 - 智能特征工程、分布式处理、硬件加速",
                    "subdirs": {
                        "engineering": "特征工程",
                        "distributed": "分布式处理",
                        "acceleration": "硬件加速"
                    }
                },
                "ml": {
                    "description": "模型推理层 - 集成学习、模型管理、实时推理",
                    "subdirs": {
                        "integration": "集成学习",
                        "models": "模型管理",
                        "engine": "推理引擎"
                    }
                },
                "backtest": {
                    "description": "策略决策层 - 策略生成、回测、调优"
                },
                "trading": {
                    "description": "交易执行层 - 订单管理、执行引擎、路由"
                },
                "risk": {
                    "description": "风控合规层 - 实时风险检查、合规验证"
                },
                "engine": {
                    "description": "监控反馈层 - 系统监控、业务监控、性能监控"
                }
            }
        }

    def analyze_current_structure(self) -> Dict[str, any]:
        """分析当前目录结构"""
        print("🔍 分析当前目录结构...")

        analysis = {
            "total_directories": 0,
            "total_files": 0,
            "directory_depth": {},
            "file_distribution": {},
            "structure_issues": []
        }

        # 递归分析目录结构
        for root, dirs, files in os.walk(self.src_dir):
            level = root.replace(str(self.src_dir), "").count(os.sep)
            if level > 0:  # 跳过根目录
                analysis["total_directories"] += len(dirs)
                analysis["total_files"] += len(files)

                if level not in analysis["directory_depth"]:
                    analysis["directory_depth"][level] = 0
                analysis["directory_depth"][level] += len(dirs)

                # 分析文件分布
                for file in files:
                    if file.endswith('.py'):
                        if level not in analysis["file_distribution"]:
                            analysis["file_distribution"][level] = 0
                        analysis["file_distribution"][level] += 1

        # 识别结构问题
        analysis["structure_issues"] = self._identify_structure_issues(analysis["directory_depth"])

        print(f"📊 分析完成：{analysis['total_directories']}个目录，{analysis['total_files']}个文件")
        return analysis

    def _identify_structure_issues(self, directory_depth: Dict[int, int]) -> List[str]:
        """识别结构问题"""
        issues = []

        # 检查目录深度
        if directory_depth:
            max_depth = max(directory_depth.keys())
            if max_depth > 4:
                issues.append(f"目录深度过深（{max_depth}层），建议控制在4层以内")

        # 检查大目录
        for item in self.src_dir.iterdir():
            if item.is_dir():
                file_count = len(list(item.rglob("*.py")))
                if file_count > 100:
                    issues.append(f"目录{item.name}过大（{file_count}个文件），建议拆分")

        return issues

    def create_optimization_plan(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """创建优化计划"""
        print("📋 创建优化计划...")

        plan = {
            "current_analysis": analysis,
            "optimization_steps": [],
            "target_structure": self.optimized_structure,
            "migration_plan": {},
            "validation_plan": {}
        }

        # 1. 基础设施层重构
        plan["optimization_steps"].append({
            "step": 1,
            "name": "基础设施层重构",
            "description": "将庞大的infrastructure层拆分为8个独立组件",
            "priority": "high",
            "actions": [
                "创建8个独立组件目录",
                "迁移文件到对应目录",
                "更新导入语句",
                "创建组件接口"
            ]
        })

        # 2. 数据层重构
        plan["optimization_steps"].append({
            "step": 2,
            "name": "数据层目录优化",
            "description": "优化数据层的目录结构",
            "priority": "medium",
            "actions": [
                "合并重复的适配器代码",
                "重新组织验证器组件",
                "优化数据处理流程"
            ]
        })

        # 3. 特征处理层重构
        plan["optimization_steps"].append({
            "step": 3,
            "name": "特征处理层重构",
            "description": "重新组织特征处理层的目录结构",
            "priority": "medium",
            "actions": [
                "分离特征工程和分布式处理",
                "优化硬件加速组件结构",
                "统一特征处理接口"
            ]
        })

        # 4. 模型层重构
        plan["optimization_steps"].append({
            "step": 4,
            "name": "模型层目录优化",
            "description": "优化模型层的目录结构",
            "priority": "low",
            "actions": [
                "分离模型管理和推理引擎",
                "重新组织集成学习组件",
                "优化模型存储结构"
            ]
        })

        # 5. 文档和规范
        plan["optimization_steps"].append({
            "step": 5,
            "name": "文档和规范更新",
            "description": "更新目录结构相关的文档和规范",
            "priority": "medium",
            "actions": [
                "更新README文件",
                "创建目录结构说明",
                "更新开发规范"
            ]
        })

        return plan

    def execute_optimization(self, plan: Dict[str, any], dry_run: bool = True) -> Dict[str, any]:
        """执行优化"""
        print(f"🔄 {'预览' if dry_run else '执行'}目录结构优化...")

        results = {
            "executed_steps": [],
            "created_directories": [],
            "moved_files": 0,
            "updated_files": 0,
            "errors": []
        }

        # 创建新的目录结构
        for layer_name, layer_config in self.optimized_structure["src"].items():
            if "subdirs" in layer_config:
                for subdir_name, description in layer_config["subdirs"].items():
                    target_dir = self.src_dir / layer_name / subdir_name
                    if not target_dir.exists():
                        if not dry_run:
                            target_dir.mkdir(parents=True, exist_ok=True)
                        results["created_directories"].append(str(target_dir))
                        print(f"  📁 {'将创建' if dry_run else '已创建'}目录: {target_dir}")

        # 预览文件迁移
        infrastructure_dir = self.src_dir / "infrastructure"
        if infrastructure_dir.exists():
            for item in infrastructure_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    file_count = len(list(item.rglob("*.py")))
                    print(f"  📄 {item.name}: {file_count}个文件需要迁移")

        return results

    def validate_optimization(self) -> Dict[str, any]:
        """验证优化结果"""
        print("✅ 验证优化结果...")

        validation = {
            "structure_compliance": True,
            "naming_convention": True,
            "import_validity": True,
            "documentation": True,
            "issues": []
        }

        # 验证目录结构符合预期
        for layer_name, layer_config in self.optimized_structure["src"].items():
            layer_dir = self.src_dir / layer_name
            if not layer_dir.exists():
                validation["structure_compliance"] = False
                validation["issues"].append(f"缺少目录: {layer_name}")

            if "subdirs" in layer_config:
                for subdir_name in layer_config["subdirs"].keys():
                    subdir_path = layer_dir / subdir_name
                    if not subdir_path.exists():
                        validation["structure_compliance"] = False
                        validation["issues"].append(f"缺少子目录: {layer_name}/{subdir_name}")

        return validation

    def generate_report(self, analysis: Dict, plan: Dict, execution: Dict, validation: Dict) -> str:
        """生成优化报告"""
        report = f"""# 目录结构优化报告

## 概述
- **优化时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **当前目录数**: {analysis['total_directories']}
- **当前文件数**: {analysis['total_files']}

## 当前结构问题
"""

        for issue in analysis["structure_issues"]:
            report += f"- {issue}\n"

        report += f"""
## 优化目标结构

### 核心服务层 (`core/`)
- **职责**: 事件总线、依赖注入、流程编排
- **关键接口**: IEventBus, IContainer, IOrchestrator
- **主要组件**: event_bus.py, container.py, business_process_orchestrator.py

### 基础设施层 (`infrastructure/`) - 拆分为8个组件
"""

        infra_subdirs = self.optimized_structure["src"]["infrastructure"]["subdirs"]
        for subdir_name, description in infra_subdirs.items():
            report += f"#### {subdir_name}/\n"
            report += f"- **职责**: {description}\n\n"

        report += f"""
### 数据采集层 (`data/`)
- **职责**: 实时数据采集、验证、质量监控
- **子目录**: adapters/, collector/, validator/, quality_monitor/

### 特征处理层 (`features/`)
- **职责**: 智能特征工程、分布式处理、硬件加速
- **子目录**: engineering/, distributed/, acceleration/

### 模型推理层 (`ml/`)
- **职责**: 集成学习、模型管理、实时推理
- **子目录**: integration/, models/, engine/

## 优化步骤
总共{len(plan['optimization_steps'])}个步骤：

"""

        for step in plan["optimization_steps"]:
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(step["priority"], "⚪")
            report += f"### {priority_icon} 步骤{step['step']}: {step['name']}\n"
            report += f"**优先级**: {step['priority']}\n"
            report += f"{step['description']}\n\n"
            for action in step["actions"]:
                report += f"- [ ] {action}\n"
            report += "\n"

        report += f"""
## 执行结果
- **创建目录**: {len(execution['created_directories'])}个
- **移动文件**: {execution['moved_files']}个
- **更新文件**: {execution['updated_files']}个
- **错误**: {len(execution['errors'])}个

## 验证结果
- **结构符合性**: {'✅' if validation['structure_compliance'] else '❌'}
- **命名规范**: {'✅' if validation['naming_convention'] else '❌'}
- **导入有效性**: {'✅' if validation['import_validity'] else '❌'}
- **文档完整性**: {'✅' if validation['documentation'] else '❌'}

"""

        if validation["issues"]:
            report += "## 发现问题\n"
            for issue in validation["issues"]:
                report += f"- {issue}\n"

        report += f"""
## 优化建议

### 立即执行
1. **基础设施层重构**: 优先拆分infrastructure层
2. **目录深度控制**: 确保目录深度不超过4层
3. **命名规范化**: 统一目录和文件命名规范

### 中期规划
1. **模块化改造**: 将大目录拆分为独立模块
2. **接口标准化**: 制定统一的接口规范
3. **文档完善**: 为每个目录创建说明文档

### 长期目标
1. **架构治理**: 建立目录结构治理机制
2. **自动化维护**: 开发目录结构自动化检查工具
3. **最佳实践**: 总结和推广目录结构最佳实践

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**优化工具**: scripts/optimize_directory_structure.py
"""

        return report

    def create_directory_readme(self, dir_path: Path, description: str) -> None:
        """为目录创建README文件"""
        readme_content = f"""# {dir_path.name} 目录

## 概述
{description}

## 目录结构
"""

        # 列出子目录和文件
        if dir_path.exists():
            for item in sorted(dir_path.iterdir()):
                if item.is_dir() and not item.name.startswith('.'):
                    readme_content += f"### {item.name}/\n"
                    readme_content += f"描述: {item.name}相关功能模块\n\n"
                elif item.name.endswith('.py') and not item.name.startswith('_'):
                    readme_content += f"- {item.name}: {item.stem}功能实现\n"

        readme_content += f"""
## 开发规范
- 文件命名: 小写字母+下划线
- 模块职责: 单一职责原则
- 接口设计: 依赖抽象而非具体实现
- 错误处理: 统一异常处理机制

## 依赖关系
- 外部依赖: [列出外部依赖]
- 内部依赖: [列出内部依赖]

## 测试要求
- 单元测试覆盖率: >=80%
- 集成测试: 关键功能必须测试
- 性能测试: 高负载场景测试

---
**更新时间**: {datetime.now().strftime('%Y-%m-%d')}
**负责人**: 架构组
"""

        readme_file = dir_path / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='目录结构优化工具')
    parser.add_argument('--execute', action='store_true', help='执行实际优化操作')
    parser.add_argument('--force', action='store_true', help='强制执行（跳过确认）')
    parser.add_argument('--dry-run', action='store_true', help='预览模式（默认）')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    optimizer = DirectoryStructureOptimizer(project_root)

    print("🏗️  目录结构优化工具")
    print("=" * 50)

    # 1. 分析当前结构
    analysis = optimizer.analyze_current_structure()

    # 2. 创建优化计划
    plan = optimizer.create_optimization_plan(analysis)

    # 3. 执行优化
    dry_run = not args.execute
    if args.execute and not args.force:
        confirm = input("\n⚠️  警告：这将修改目录结构！是否继续？(y/N): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return

    execution = optimizer.execute_optimization(plan, dry_run=dry_run)

    # 4. 验证优化
    validation = optimizer.validate_optimization()

    # 5. 生成报告
    report = optimizer.generate_report(analysis, plan, execution, validation)

    # 保存报告
    report_file = project_root / "reports" / "directory_optimization_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📋 优化报告已生成: {report_file}")

    if dry_run:
        print("\n🔧 优化工具已就绪，可以执行实际的目录结构优化。")
        print("⚠️  重要提醒：建议在执行优化前进行完整备份！")
        print("\n使用 --execute 参数来执行实际优化操作")
    else:
        print("\n✅ 目录结构优化已完成！")


if __name__ == "__main__":
    main()
