#!/usr/bin/env python3
"""
测试文件架构合规性分析脚本

分析当前测试文件是否符合新的架构设计，生成废弃和删除建议。
"""

import re
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ArchitectureCompliance:
    """架构合规性分析结果"""
    file_path: str
    layer: str
    compliance_status: str  # "COMPLIANT", "DEPRECATED", "TO_DELETE", "NEEDS_UPDATE"
    issues: List[str]
    recommendations: List[str]


class TestArchitectureAnalyzer:
    """测试架构合规性分析器"""

    def __init__(self):
        self.base_path = Path("tests/unit")
        self.src_path = Path("src")

        # 定义各层的架构要求
        self.layer_requirements = {
            "features": {
                "core_components": ["FeatureEngineer", "FeatureProcessor", "FeatureSelector", "FeatureStandardizer", "FeatureSaver"],
                "type_definitions": ["FeatureType", "FeatureConfig"],
                "processors": ["BaseFeatureProcessor", "TechnicalProcessor"],
                "analyzers": ["SentimentAnalyzer"],
                "deprecated_modules": ["HighFreqOptimizer", "OrderBookAnalyzer"],  # 暂未启用
                "file_patterns": ["test_*", "auto_test_*"]
            },
            "infrastructure": {
                "core_components": ["CacheManager", "DatabaseManager", "MonitorManager", "ConfigManager"],
                "distributed_components": ["DistributedCache", "DistributedMonitor", "ClusterConfig"],
                "ha_components": ["LoadBalancer", "CircuitBreaker", "HealthChecker"],
                "deprecated_modules": ["LegacyMonitor", "OldCacheManager"],
                "file_patterns": ["test_*", "auto_test_*"]
            },
            "integration": {
                "core_components": ["SystemIntegrationManager", "LayerInterface", "UnifiedConfigManager"],
                "data_integration": ["DataIntegration", "DataValidator"],
                "deprecated_modules": ["OldIntegration", "LegacyInterface"],
                "file_patterns": ["test_*", "auto_test_*"]
            }
        }

        # 定义废弃的测试模式
        self.deprecated_patterns = [
            r"auto_test_.*",  # 自动生成的测试文件
            r"test_.*_isolated",  # 隔离测试文件
            r"test_.*_comprehensive",  # 过于复杂的测试文件
            r"test_.*_coverage",  # 覆盖率测试文件
            r"test_.*_simple",  # 简单测试文件
            r"test_.*_standalone",  # 独立测试文件
            r"test_.*_offline",  # 离线测试文件
            r"test_.*_manager_offline",  # 离线管理器测试
            r"test_.*_engineer_isolated",  # 隔离工程师测试
            r"test_.*_metadata_isolated",  # 隔离元数据测试
            r"test_.*_importance_isolated",  # 隔离重要性测试
        ]

        # 定义需要删除的测试模式
        self.delete_patterns = [
            r"isolated_impl",  # 隔离实现文件
            r"features_config\.ini",  # 配置文件
            r"test_features_config\.ini",  # 测试配置文件
            r"conftest\.py",  # 配置文件（保留核心的）
        ]

    def analyze_layer_compliance(self, layer: str) -> List[ArchitectureCompliance]:
        """分析指定层的架构合规性"""
        layer_path = self.base_path / layer
        if not layer_path.exists():
            return []

        results = []
        requirements = self.layer_requirements.get(layer, {})

        for test_file in layer_path.rglob("*.py"):
            if test_file.name == "__init__.py":
                continue

            compliance = self._analyze_file_compliance(test_file, layer, requirements)
            results.append(compliance)

        return results

    def _analyze_file_compliance(self, file_path: Path, layer: str, requirements: Dict) -> ArchitectureCompliance:
        """分析单个文件的架构合规性"""
        issues = []
        recommendations = []

        # 检查文件名模式
        file_name = file_path.name

        # 检查是否需要删除
        if any(re.match(pattern, file_name) for pattern in self.delete_patterns):
            return ArchitectureCompliance(
                file_path=str(file_path),
                layer=layer,
                compliance_status="TO_DELETE",
                issues=["文件符合删除模式"],
                recommendations=["建议删除此文件"]
            )

        # 检查是否已废弃
        if any(re.match(pattern, file_name) for pattern in self.deprecated_patterns):
            return ArchitectureCompliance(
                file_path=str(file_path),
                layer=layer,
                compliance_status="DEPRECATED",
                issues=["文件符合废弃模式"],
                recommendations=["建议废弃此文件，保留核心功能测试"]
            )

        # 检查文件内容合规性
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否测试了核心组件
            core_components = requirements.get("core_components", [])
            tested_components = self._extract_tested_components(content)

            missing_components = [comp for comp in core_components if comp not in tested_components]
            if missing_components:
                issues.append(f"缺少对核心组件的测试: {missing_components}")
                recommendations.append("建议添加对核心组件的测试")

            # 检查是否测试了废弃模块
            deprecated_modules = requirements.get("deprecated_modules", [])
            tested_deprecated = [comp for comp in deprecated_modules if comp in tested_components]
            if tested_deprecated:
                issues.append(f"测试了废弃模块: {tested_deprecated}")
                recommendations.append("建议移除对废弃模块的测试")

            # 检查测试质量
            if len(content.split('\n')) < 50:
                issues.append("测试文件过小，可能测试覆盖不足")
                recommendations.append("建议增加测试用例")

            if len(content.split('\n')) > 500:
                issues.append("测试文件过大，可能过于复杂")
                recommendations.append("建议拆分为多个测试文件")

        except Exception as e:
            issues.append(f"无法读取文件: {e}")
            recommendations.append("检查文件编码和格式")

        # 确定合规状态
        if issues:
            if "TO_DELETE" in [issue for issue in issues if "删除" in issue]:
                status = "TO_DELETE"
            elif "DEPRECATED" in [issue for issue in issues if "废弃" in issue]:
                status = "DEPRECATED"
            else:
                status = "NEEDS_UPDATE"
        else:
            status = "COMPLIANT"

        return ArchitectureCompliance(
            file_path=str(file_path),
            layer=layer,
            compliance_status=status,
            issues=issues,
            recommendations=recommendations
        )

    def _extract_tested_components(self, content: str) -> List[str]:
        """从测试内容中提取被测试的组件"""
        components = []

        # 常见的测试模式
        patterns = [
            r"class Test(\w+)",  # 测试类
            r"def test_(\w+)",   # 测试方法
            r"from.*import.*(\w+)",  # 导入语句
            r"(\w+)\(\)",        # 函数调用
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            components.extend(matches)

        return list(set(components))

    def generate_report(self) -> str:
        """生成架构合规性分析报告"""
        report = []
        report.append("# 测试文件架构合规性分析报告")
        report.append("")

        total_files = 0
        compliant_files = 0
        deprecated_files = 0
        to_delete_files = 0
        needs_update_files = 0

        for layer in ["features", "infrastructure", "integration"]:
            report.append(f"## {layer}层分析")
            report.append("")

            compliance_results = self.analyze_layer_compliance(layer)

            # 按状态分组
            status_groups = {}
            for result in compliance_results:
                status = result.compliance_status
                if status not in status_groups:
                    status_groups[status] = []
                status_groups[status].append(result)

            # 统计
            total_files += len(compliance_results)
            compliant_files += len(status_groups.get("COMPLIANT", []))
            deprecated_files += len(status_groups.get("DEPRECATED", []))
            to_delete_files += len(status_groups.get("TO_DELETE", []))
            needs_update_files += len(status_groups.get("NEEDS_UPDATE", []))

            # 输出结果
            for status, results in status_groups.items():
                status_name = {
                    "COMPLIANT": "✅ 合规",
                    "DEPRECATED": "⚠️ 废弃",
                    "TO_DELETE": "🗑️ 删除",
                    "NEEDS_UPDATE": "🔧 需更新"
                }.get(status, status)

                report.append(f"### {status_name} ({len(results)}个文件)")
                report.append("")

                for result in results:
                    report.append(f"**{result.file_path}**")
                    if result.issues:
                        report.append("问题:")
                        for issue in result.issues:
                            report.append(f"- {issue}")
                    if result.recommendations:
                        report.append("建议:")
                        for rec in result.recommendations:
                            report.append(f"- {rec}")
                    report.append("")

        # 总体统计
        report.append("## 总体统计")
        report.append("")
        report.append(f"- 总文件数: {total_files}")
        report.append(f"- 合规文件: {compliant_files} ({compliant_files/total_files*100:.1f}%)")
        report.append(f"- 废弃文件: {deprecated_files} ({deprecated_files/total_files*100:.1f}%)")
        report.append(f"- 需删除文件: {to_delete_files} ({to_delete_files/total_files*100:.1f}%)")
        report.append(f"- 需更新文件: {needs_update_files} ({needs_update_files/total_files*100:.1f}%)")
        report.append("")

        return "\n".join(report)

    def generate_cleanup_script(self) -> str:
        """生成清理脚本"""
        script_lines = [
            "#!/usr/bin/env python3",
            '"""测试文件清理脚本"""',
            "",
            "import os",
            "import shutil",
            "from pathlib import Path",
            "",
            "# 需要删除的文件列表",
            "files_to_delete = ["
        ]

        for layer in ["features", "infrastructure", "integration"]:
            compliance_results = self.analyze_layer_compliance(layer)
            for result in compliance_results:
                if result.compliance_status == "TO_DELETE":
                    script_lines.append(f'    "{result.file_path}",')

        script_lines.extend([
            "]",
            "",
            "# 需要废弃的文件列表（移动到废弃目录）",
            "files_to_deprecate = ["
        ])

        for layer in ["features", "infrastructure", "integration"]:
            compliance_results = self.analyze_layer_compliance(layer)
            for result in compliance_results:
                if result.compliance_status == "DEPRECATED":
                    script_lines.append(f'    "{result.file_path}",')

        script_lines.extend([
            "]",
            "",
            "def cleanup_test_files():",
            '    """清理测试文件"""',
            '    print("开始清理测试文件...")',
            "",
            "    # 创建废弃目录",
            '    deprecated_dir = Path("tests/deprecated")',
            '    deprecated_dir.mkdir(exist_ok=True)',
            "",
            "    # 删除文件",
            '    for file_path in files_to_delete:',
            '        path = Path(file_path)',
            '        if path.exists():',
            '            print(f"删除文件: {file_path}")',
            '            path.unlink()',
            '        else:',
            '            print(f"文件不存在: {file_path}")',
            "",
            "    # 移动废弃文件",
            '    for file_path in files_to_deprecate:',
            '        path = Path(file_path)',
            '        if path.exists():',
            '            new_path = deprecated_dir / path.name',
            '            print(f"移动文件: {file_path} -> {new_path}")',
            '            shutil.move(str(path), str(new_path))',
            '        else:',
            '            print(f"文件不存在: {file_path}")',
            "",
            '    print("清理完成！")',
            "",
            'if __name__ == "__main__":',
            '    cleanup_test_files()',
        ])

        return "\n".join(script_lines)


def main():
    """主函数"""
    analyzer = TestArchitectureAnalyzer()

    # 生成分析报告
    report = analyzer.generate_report()

    # 保存报告
    with open("reports/testing/architecture_compliance_analysis.md", "w", encoding="utf-8") as f:
        f.write(report)

    # 生成清理脚本
    cleanup_script = analyzer.generate_cleanup_script()

    # 保存清理脚本
    with open("scripts/project/cleanup_test_files.py", "w", encoding="utf-8") as f:
        f.write(cleanup_script)

    print("✅ 架构合规性分析完成！")
    print("📊 报告已保存到: reports/testing/architecture_compliance_analysis.md")
    print("🧹 清理脚本已保存到: scripts/project/cleanup_test_files.py")


if __name__ == "__main__":
    main()
