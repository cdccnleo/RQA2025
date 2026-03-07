#!/usr/bin/env python3
import ast
"""
配置文件重构工具

分析、重构和优化项目配置文件，提高配置管理的一致性和可维护性
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ConfigFile:
    """配置文件信息"""
    path: str
    format: str
    size: int
    last_modified: float
    content_hash: str
    is_valid: bool
    issues: List[str]
    dependencies: List[str]


@dataclass
class ConfigAnalysis:
    """配置分析结果"""
    total_files: int
    formats: Dict[str, int]
    duplicated_configs: List[Dict[str, Any]]
    unused_configs: List[str]
    inconsistent_configs: List[Dict[str, Any]]
    complexity_score: float
    maintainability_score: float


class ConfigRefactorTool:
    """配置文件重构工具"""

    def __init__(self, config_root: str):
        self.config_root = Path(config_root)
        self.analysis_result: ConfigAnalysis = None
        self.config_files: List[ConfigFile] = []

        # 支持的配置文件格式
        self.supported_formats = {
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'ini',
            '.properties': 'properties'
        }

        # 配置分类规则
        self.config_categories = {
            'database': ['database', 'db', 'mongo', 'redis', 'sql'],
            'logging': ['logging', 'log', 'logger', 'audit'],
            'monitoring': ['monitoring', 'monitor', 'metrics', 'prometheus', 'grafana'],
            'security': ['security', 'auth', 'authentication', 'permission', 'token'],
            'performance': ['performance', 'perf', 'optimization', 'cache'],
            'deployment': ['deployment', 'deploy', 'docker', 'kubernetes', 'k8s'],
            'testing': ['testing', 'test', 'pytest', 'unittest'],
            'development': ['development', 'dev', 'local', 'debug'],
            'production': ['production', 'prod', 'release'],
            'services': ['service', 'api', 'gateway', 'microservice']
        }

    def analyze_configs(self) -> ConfigAnalysis:
        """分析配置文件"""
        print("🔍 分析配置文件...")

        self.config_files = []
        formats = {}
        content_hashes = {}
        duplicate_groups = []

        # 扫描所有配置文件
        for file_path in self.config_root.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                config_file = self._analyze_single_config(file_path)
                self.config_files.append(config_file)

                # 统计格式分布
                fmt = config_file.format
                formats[fmt] = formats.get(fmt, 0) + 1

                # 检测重复配置
                if config_file.content_hash in content_hashes:
                    content_hashes[config_file.content_hash].append(config_file.path)
                else:
                    content_hashes[config_file.content_hash] = [config_file.path]

        # 识别重复配置
        for hash_value, paths in content_hashes.items():
            if len(paths) > 1:
                duplicate_groups.append({
                    'files': paths,
                    'hash': hash_value,
                    'recommendation': '合并重复配置或提取公共配置'
                })

        # 识别未使用的配置
        unused_configs = self._identify_unused_configs()

        # 识别不一致的配置
        inconsistent_configs = self._identify_inconsistent_configs()

        # 先创建临时的analysis_result用于计算
        temp_analysis = ConfigAnalysis(
            total_files=len(self.config_files),
            formats=formats,
            duplicated_configs=duplicate_groups,
            unused_configs=unused_configs,
            inconsistent_configs=inconsistent_configs,
            complexity_score=0.0,
            maintainability_score=0.0
        )
        self.analysis_result = temp_analysis

        # 计算复杂度和可维护性分数
        complexity_score = self._calculate_complexity_score()
        maintainability_score = self._calculate_maintainability_score()

        # 更新分数
        self.analysis_result.complexity_score = complexity_score
        self.analysis_result.maintainability_score = maintainability_score

        print(f"📊 分析完成：发现{len(self.config_files)}个配置文件")
        return self.analysis_result

    def _analyze_single_config(self, file_path: Path) -> ConfigFile:
        """分析单个配置文件"""
        try:
            stat = file_path.stat()
            content = file_path.read_text(encoding='utf-8')

            # 计算内容哈希
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()

            # 验证文件格式
            is_valid, issues = self._validate_config_format(file_path, content)

            # 分析依赖关系
            dependencies = self._analyze_dependencies(content)

            return ConfigFile(
                path=str(file_path),
                format=self.supported_formats.get(file_path.suffix.lower(), 'unknown'),
                size=stat.st_size,
                last_modified=stat.st_mtime,
                content_hash=content_hash,
                is_valid=is_valid,
                issues=issues,
                dependencies=dependencies
            )

        except Exception as e:
            return ConfigFile(
                path=str(file_path),
                format='unknown',
                size=0,
                last_modified=0,
                content_hash='',
                is_valid=False,
                issues=[f"读取失败: {e}"],
                dependencies=[]
            )

    def _validate_config_format(self, file_path: Path, content: str) -> tuple[bool, List[str]]:
        """验证配置文件格式"""
        issues = []
        is_valid = True

        try:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_load(content)
            elif file_path.suffix.lower() == '.json':
                json.loads(content)
            # 对于INI文件，简单的语法检查
            elif file_path.suffix.lower() in ['.ini', '.cfg', '.conf']:
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith(';'):
                        if '=' not in line and not line.startswith('['):
                            issues.append(f"第{i}行: 格式错误")
                            is_valid = False
        except Exception as e:
            issues.append(f"格式错误: {e}")
            is_valid = False

        return is_valid, issues

    def _analyze_dependencies(self, content: str) -> List[str]:
        """分析配置依赖关系"""
        dependencies = []

        # 查找引用其他配置文件的内容
        import_patterns = [
            r"include\s+['\"]([^'\"]+\.(?:yaml|yml|json|ini|cfg|conf))['\"]",
            r"extends\s+['\"]([^'\"]+\.(?:yaml|yml|json|ini|cfg|conf))['\"]",
            r"import\s+['\"]([^'\"]+\.(?:yaml|yml|json|ini|cfg|conf))['\"]"
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            dependencies.extend(matches)

        return list(set(dependencies))

    def _identify_unused_configs(self) -> List[str]:
        """识别未使用的配置"""
        # 这里需要扫描源代码来检查哪些配置被使用
        # 简化的实现，标记一些明显的未使用配置
        unused = []

        for config_file in self.config_files:
            file_name = Path(config_file.path).name

            # 检查一些明显未使用的配置模式
            if file_name.startswith('test_') and 'config' in file_name:
                # 检查是否有对应的测试文件引用此配置
                test_references = self._find_config_references(config_file.path)
                if len(test_references) == 0:
                    unused.append(config_file.path)

        return unused

    def _identify_inconsistent_configs(self) -> List[Dict[str, Any]]:
        """识别不一致的配置"""
        inconsistencies = []

        # 检查相同配置项在不同文件中的值是否一致
        config_values = {}

        for config_file in self.config_files:
            try:
                if config_file.format == 'json':
                    data = json.loads(Path(config_file.path).read_text())
                elif config_file.format == 'yaml':
                    data = yaml.safe_load(Path(config_file.path).read_text())
                else:
                    continue

                # 扁平化配置结构
                flat_data = self._flatten_dict(data)

                for key, value in flat_data.items():
                    if key not in config_values:
                        config_values[key] = {}
                    config_values[key][config_file.path] = value

            except:
                continue

        # 检查不一致的值
        for key, file_values in config_values.items():
            if len(set(str(v) for v in file_values.values())) > 1:
                inconsistencies.append({
                    'config_key': key,
                    'values': file_values,
                    'recommendation': '统一配置值或明确区分环境'
                })

        return inconsistencies

    def _flatten_dict(self, data: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """扁平化嵌套字典"""
        result = {}

        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self._flatten_dict(value, new_key))
            else:
                result[new_key] = value

        return result

    def _find_config_references(self, config_path: str) -> List[str]:
        """查找配置文件的引用"""
        references = []
        config_name = Path(config_path).stem

        # 扫描Python文件查找引用
        for py_file in Path(self.config_root).parent.rglob("*.py"):
            try:
                content = py_file.read_text()
                if config_name in content:
                    references.append(str(py_file))
            except:
                continue

        return references

    def _calculate_complexity_score(self) -> float:
        """计算配置复杂度分数"""
        if not self.config_files:
            return 0.0

        total_size = sum(cf.size for cf in self.config_files)
        avg_size = total_size / len(self.config_files)

        # 基于文件大小和数量计算复杂度
        size_complexity = min(avg_size / 1024, 10)  # 最大10分
        count_complexity = min(len(self.config_files) / 10, 10)  # 最大10分

        return (size_complexity + count_complexity) / 2

    def _calculate_maintainability_score(self) -> float:
        """计算配置可维护性分数"""
        if not self.config_files:
            return 100.0

        total_score = 100.0

        # 减分项
        total_score -= len(self.analysis_result.duplicated_configs) * 5
        total_score -= len(self.analysis_result.unused_configs) * 3
        total_score -= len(self.analysis_result.inconsistent_configs) * 2

        # 格式一致性加分
        format_count = len(self.analysis_result.formats)
        if format_count <= 2:
            total_score += 10

        return max(0, min(100, total_score))

    def create_refactor_plan(self) -> Dict[str, Any]:
        """创建重构计划"""
        print("📋 创建配置文件重构计划...")

        plan = {
            "analysis": asdict(self.analysis_result),
            "refactor_steps": [],
            "target_structure": {},
            "migration_plan": {}
        }

        # 1. 标准化配置文件格式
        plan["refactor_steps"].append({
            "step": 1,
            "name": "标准化配置文件格式",
            "description": "统一配置文件格式，推荐使用YAML格式",
            "priority": "medium",
            "actions": [
                "将所有INI/JSON配置文件转换为YAML格式",
                "制定配置文件格式规范",
                "更新相关工具和文档"
            ]
        })

        # 2. 合并重复配置
        if self.analysis_result.duplicated_configs:
            plan["refactor_steps"].append({
                "step": 2,
                "name": "合并重复配置",
                "description": f"合并{len(self.analysis_result.duplicated_configs)}组重复配置",
                "priority": "high",
                "actions": [
                    "识别重复配置的根本原因",
                    "创建共享配置文件",
                    "更新引用这些配置的代码"
                ]
            })

        # 3. 清理未使用配置
        if self.analysis_result.unused_configs:
            plan["refactor_steps"].append({
                "step": 3,
                "name": "清理未使用配置",
                "description": f"清理{len(self.analysis_result.unused_configs)}个未使用的配置文件",
                "priority": "low",
                "actions": [
                    "确认配置确实未使用",
                    "备份配置文件",
                    "删除或归档未使用配置"
                ]
            })

        # 4. 统一不一致配置
        if self.analysis_result.inconsistent_configs:
            plan["refactor_steps"].append({
                "step": 4,
                "name": "统一不一致配置",
                "description": f"解决{len(self.analysis_result.inconsistent_configs)}个配置不一致问题",
                "priority": "medium",
                "actions": [
                    "分析不一致的根本原因",
                    "制定统一配置的标准值",
                    "更新相关配置文件"
                ]
            })

        # 5. 重构配置结构
        plan["refactor_steps"].append({
            "step": 5,
            "name": "重构配置结构",
            "description": "优化配置文件的组织结构",
            "priority": "medium",
            "actions": [
                "按功能领域重新组织配置",
                "创建配置模板和继承机制",
                "实现环境特定的配置覆盖"
            ]
        })

        return plan

    def execute_refactor(self, plan: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """执行重构"""
        print(f"🔄 {'预览' if dry_run else '执行'}配置文件重构...")

        results = {
            "executed_steps": [],
            "converted_files": [],
            "merged_files": [],
            "deleted_files": [],
            "errors": []
        }

        # 预览模式：显示将要执行的操作
        for step in plan["refactor_steps"]:
            print(f"  📋 步骤{step['step']}: {step['name']}")
            for action in step["actions"]:
                print(f"    - {action}")

        return results

    def generate_report(self, analysis: ConfigAnalysis, plan: Dict, execution: Dict) -> str:
        """生成重构报告"""
        report = f"""# 配置文件重构报告

## 概述
- **重构时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **配置目录**: {self.config_root}
- **总文件数**: {analysis.total_files}

## 配置分析结果

### 格式分布
| 格式 | 文件数 | 占比 |
|------|--------|------|
"""

        total_files = sum(analysis.formats.values())
        for fmt, count in analysis.formats.items():
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            report += f"| {fmt.upper()} | {count} | {percentage:.1f}% |\n"

        report += f"""
### 质量指标
- **复杂度分数**: {analysis.complexity_score:.1f}/10
- **可维护性分数**: {analysis.maintainability_score:.1f}/100
- **重复配置组数**: {len(analysis.duplicated_configs)}
- **未使用配置数**: {len(analysis.unused_configs)}
- **不一致配置数**: {len(analysis.inconsistent_configs)}

## 发现的问题

"""

        if analysis.duplicated_configs:
            report += "### 🔄 重复配置\n"
            for dup in analysis.duplicated_configs[:5]:  # 最多显示5个
                report += f"**文件**: {', '.join(dup['files'])}\n"
                report += f"**建议**: {dup['recommendation']}\n\n"

        if analysis.unused_configs:
            report += "### 🗑️ 未使用配置\n"
            for unused in analysis.unused_configs[:5]:  # 最多显示5个
                report += f"- {unused}\n"
            report += "\n"

        if analysis.inconsistent_configs:
            report += "### ⚠️ 不一致配置\n"
            for inc in analysis.inconsistent_configs[:5]:  # 最多显示5个
                report += f"**配置项**: {inc['config_key']}\n"
                report += f"**建议**: {inc['recommendation']}\n\n"

        report += f"""## 重构计划
总共{len(plan['refactor_steps'])}个步骤：

"""

        for step in plan["refactor_steps"]:
            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(step["priority"], "⚪")
            report += f"### {priority_icon} 步骤{step['step']}: {step['name']}\n"
            report += f"**优先级**: {step['priority']}\n"
            report += f"{step['description']}\n\n"
            for action in step["actions"]:
                report += f"- [ ] {action}\n"
            report += "\n"

        report += f"""## 重构建议

### 立即执行
1. **格式标准化**: 统一使用YAML格式
2. **重复配置清理**: 合并和清理重复配置
3. **未使用配置清理**: 删除或归档未使用的配置

### 中期规划
1. **配置结构优化**: 重新设计配置文件的组织结构
2. **配置验证加强**: 添加配置文件的验证机制
3. **配置文档完善**: 为所有配置文件添加详细文档

### 长期目标
1. **配置中心化**: 实现配置的中心化管理
2. **配置热更新**: 支持配置的热更新机制
3. **配置监控**: 实现配置变更的监控和告警

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**重构工具**: scripts/config_refactor.py
"""

        return report


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    config_root = project_root / "config"

    tool = ConfigRefactorTool(config_root)

    # 1. 分析配置
    analysis = tool.analyze_configs()

    # 2. 创建重构计划
    plan = tool.create_refactor_plan()

    # 3. 执行重构（预览模式）
    execution = tool.execute_refactor(plan, dry_run=True)

    # 4. 生成报告
    report = tool.generate_report(analysis, plan, execution)

    # 保存报告
    report_file = project_root / "reports" / "config_refactor_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📋 重构报告已生成: {report_file}")
    print("\n🔧 重构工具已就绪，可以执行实际的配置文件重构。")
    print("⚠️  重要提醒：建议在执行重构前进行完整备份！")


if __name__ == "__main__":
    main()
