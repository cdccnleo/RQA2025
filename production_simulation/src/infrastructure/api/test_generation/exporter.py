"""
测试套件导出器

职责: 将测试套件导出为不同格式(JSON, YAML, HTML等)
原位置: APITestCaseGenerator.export_test_cases, _export_json, _export_yaml
"""

import json
from pathlib import Path
from typing import Dict
from .models import TestSuite, TestScenario, TestCase


class TestSuiteExporter:
    """测试套件导出器 - 负责导出测试套件到不同格式"""
    
    def __init__(self):
        """初始化导出器"""
        pass
    
    def export(
        self,
        test_suites=None,
        format_type: str = "json",
        output_dir: str = "docs/api/tests",
        # 兼容性参数
        test_suite=None,
        output_path=None,
        format=None,
        include_metadata=None,
        include_statistics=None,
        pretty_print=None
    ):
        """
        导出测试套件

        Args:
            test_suites: 测试套件字典（新API）
            format_type: 导出格式 (json, yaml, html)
            output_dir: 输出目录
            # 兼容性参数（旧API）
            test_suite: 单个测试套件
            output_path: 输出文件路径
            format: 格式类型（兼容旧API）
            include_metadata: 是否包含元数据
            include_statistics: 是否包含统计信息
            pretty_print: 是否美化输出
        """
        # 处理兼容性：如果使用旧API参数，转换为新API
        use_legacy_api = test_suite is not None

        if use_legacy_api:
            test_suites = {'combined': test_suite}
            if format:
                format_type = format

        if not test_suites:
            raise ValueError("必须提供 test_suites 或 test_suite 参数")

        # 处理输出路径
        if use_legacy_api and output_path:
            # 旧API：直接输出到指定文件
            output_file_path = Path(output_path)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type == "json":
                self._export_json(test_suites, output_file_path)
            elif format_type == "yaml":
                self._export_yaml(test_suites, output_file_path)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")
        else:
            # 新API：输出到目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if format_type == "json":
                self._export_json(test_suites, output_path / "test_suites.json")
            elif format_type == "yaml":
                self._export_yaml(test_suites, output_path / "test_suites.yaml")
            elif format_type == "html":
                self._export_html(test_suites, output_path / "test_suites.html")
            elif format_type == "markdown":
                self._export_markdown(test_suites, output_path / "test_suites.md")
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")

        return True  # 成功导出
    
    def _export_json(self, test_suites: Dict[str, TestSuite], output_file: Path):
        """导出为JSON格式"""
        data = {}
        
        for suite_name, suite in test_suites.items():
            data[suite_name] = {
                "id": suite.id,
                "name": suite.name,
                "description": suite.description,
                "created_at": suite.created_at,
                "updated_at": suite.updated_at,
                "scenarios": [
                    {
                        "id": scenario.id,
                        "name": scenario.name,
                        "description": scenario.description,
                        "endpoint": scenario.endpoint,
                        "method": scenario.method,
                        "setup_steps": scenario.setup_steps,
                        "teardown_steps": scenario.teardown_steps,
                        "test_cases": [
                            {
                                "id": test_case.id,
                                "title": test_case.title,
                                "description": test_case.description,
                                "priority": test_case.priority,
                                "category": test_case.category,
                                "preconditions": test_case.preconditions,
                                "test_steps": test_case.test_steps,
                                "expected_results": test_case.expected_results,
                                "status": test_case.status,
                                "tags": test_case.tags
                            }
                            for test_case in scenario.test_cases
                        ]
                    }
                    for scenario in suite.scenarios
                ]
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON导出完成: {output_file}")
    
    def _export_yaml(self, test_suites: Dict[str, TestSuite], output_file: Path):
        """导出为YAML格式"""
        try:
            import yaml
            
            # 准备数据（与JSON相同）
            data = {}
            for suite_name, suite in test_suites.items():
                data[suite_name] = self._suite_to_dict(suite)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            
            print(f"✅ YAML导出完成: {output_file}")
            
        except ImportError:
            print("⚠️  YAML库未安装，跳过YAML导出")
    
    def _export_html(self, test_suites: Dict[str, TestSuite], output_file: Path):
        """导出为HTML格式"""
        html_content = self._generate_html_report(test_suites)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML导出完成: {output_file}")
    
    def _export_markdown(self, test_suites: Dict[str, TestSuite], output_file: Path):
        """导出为Markdown格式"""
        md_content = self._generate_markdown_report(test_suites)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"✅ Markdown导出完成: {output_file}")
    
    def _suite_to_dict(self, suite: TestSuite) -> Dict:
        """将测试套件转换为字典"""
        return {
            "id": suite.id,
            "name": suite.name,
            "description": suite.description,
            "scenarios": [
                {
                    "id": s.id,
                    "name": s.name,
                    "endpoint": s.endpoint,
                    "method": s.method,
                    "test_cases": [
                        {
                            "id": tc.id,
                            "title": tc.title,
                            "priority": tc.priority
                        }
                        for tc in s.test_cases
                    ]
                }
                for s in suite.scenarios
            ]
        }
    
    def _generate_html_report(self, test_suites: Dict[str, TestSuite]) -> str:
        """生成HTML报告"""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>API测试套件</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .suite { margin: 20px 0; border: 1px solid #ddd; padding: 15px; }
        .scenario { margin: 10px 0; background: #f9f9f9; padding: 10px; }
        .test-case { margin: 5px 0; padding: 5px; border-left: 3px solid #4CAF50; }
    </style>
</head>
<body>
    <h1>API测试套件文档</h1>
"""
        
        for suite_name, suite in test_suites.items():
            html += f"""
    <div class="suite">
        <h2>{suite.name}</h2>
        <p>{suite.description}</p>
        <p>场景数量: {len(suite.scenarios)}</p>
"""
            
            for scenario in suite.scenarios:
                html += f"""
        <div class="scenario">
            <h3>{scenario.name}</h3>
            <p>{scenario.endpoint} [{scenario.method}]</p>
            <p>测试用例: {len(scenario.test_cases)}个</p>
        </div>
"""
            
            html += "    </div>\n"
        
        html += """
</body>
</html>
"""
        return html
    
    def _generate_markdown_report(self, test_suites: Dict[str, TestSuite]) -> str:
        """生成Markdown报告"""
        md_lines = ["# API测试套件文档\n"]
        
        for suite_name, suite in test_suites.items():
            md_lines.append(f"## {suite.name}\n")
            md_lines.append(f"{suite.description}\n")
            md_lines.append(f"**场景数量**: {len(suite.scenarios)}\n")
            
            for scenario in suite.scenarios:
                md_lines.append(f"### {scenario.name}\n")
                md_lines.append(f"- **端点**: `{scenario.endpoint}`\n")
                md_lines.append(f"- **方法**: `{scenario.method}`\n")
                md_lines.append(f"- **测试用例**: {len(scenario.test_cases)}个\n")
        
        return "\n".join(md_lines)


# 避免pytest将导出器类误判为测试类
TestSuiteExporter.__test__ = False  # type: ignore[attr-defined]
