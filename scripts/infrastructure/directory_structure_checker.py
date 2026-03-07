#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动化目录结构检查脚本
用于检测重复目录、空壳文件、架构不一致等问题
"""

import os
import re
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime


class DirectoryStructureChecker:
    """目录结构检查器"""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.issues = []
        self.stats = {
            "total_directories": 0,
            "total_files": 0,
            "empty_files": 0,
            "stub_files": 0,
            "duplicate_directories": 0,
            "naming_issues": 0
        }

    def check_directory_structure(self) -> Dict:
        """检查目录结构"""
        print("🔍 开始检查目录结构...")

        # 检查src目录
        src_issues = self._check_src_directory()

        # 检查tests目录
        tests_issues = self._check_tests_directory()

        # 检查重复目录
        duplicate_issues = self._check_duplicate_directories()

        # 检查空壳文件
        stub_issues = self._check_stub_files()

        # 检查命名规范
        naming_issues = self._check_naming_conventions()

        # 生成报告
        report = self._generate_report()

        return report

    def _check_src_directory(self) -> List[Dict]:
        """检查src目录结构"""
        issues = []
        src_path = self.root_path / "src"

        if not src_path.exists():
            issues.append({
                "type": "error",
                "message": "src目录不存在",
                "path": "src"
            })
            return issues

        # 检查主要模块目录
        expected_modules = {
            "acceleration": "硬件加速模块",
            "trading": "交易核心模块",
            "data": "数据处理模块",
            "models": "模型管理模块",
            "features": "特征工程模块",
            "infrastructure": "基础设施模块",
            "backtest": "回测模块",
            "engine": "实时引擎模块"
        }

        for module, description in expected_modules.items():
            module_path = src_path / module
            if not module_path.exists():
                issues.append({
                    "type": "warning",
                    "message": f"缺少{description}: {module}",
                    "path": f"src/{module}"
                })

        return issues

    def _check_tests_directory(self) -> List[Dict]:
        """检查tests目录结构"""
        issues = []
        tests_path = self.root_path / "tests"

        if not tests_path.exists():
            issues.append({
                "type": "error",
                "message": "tests目录不存在",
                "path": "tests"
            })
            return issues

        # 检查测试目录结构
        expected_test_dirs = {
            "unit": "单元测试",
            "integration": "集成测试",
            "e2e": "端到端测试",
            "performance": "性能测试"
        }

        for test_dir, description in expected_test_dirs.items():
            test_path = tests_path / test_dir
            if not test_path.exists():
                issues.append({
                    "type": "warning",
                    "message": f"缺少{description}: {test_dir}",
                    "path": f"tests/{test_dir}"
                })

        return issues

    def _check_duplicate_directories(self) -> List[Dict]:
        """检查重复目录"""
        issues = []

        # 检查src目录下的重复
        src_path = self.root_path / "src"
        if src_path.exists():
            src_dirs = [d.name for d in src_path.iterdir() if d.is_dir()]
            duplicates = self._find_duplicates(src_dirs)

            for duplicate in duplicates:
                issues.append({
                    "type": "error",
                    "message": f"发现重复目录: {duplicate}",
                    "path": f"src/{duplicate}"
                })

        # 检查tests目录下的重复
        tests_path = self.root_path / "tests"
        if tests_path.exists():
            unit_path = tests_path / "unit"
            if unit_path.exists():
                unit_dirs = [d.name for d in unit_path.iterdir() if d.is_dir()]
                duplicates = self._find_duplicates(unit_dirs)

                for duplicate in duplicates:
                    issues.append({
                        "type": "error",
                        "message": f"发现重复测试目录: {duplicate}",
                        "path": f"tests/unit/{duplicate}"
                    })

        return issues

    def _find_duplicates(self, items: List[str]) -> List[str]:
        """查找重复项"""
        seen = set()
        duplicates = []

        for item in items:
            if item in seen:
                duplicates.append(item)
            else:
                seen.add(item)

        return duplicates

    def _check_stub_files(self) -> List[Dict]:
        """检查空壳文件"""
        issues = []

        # 检查src目录下的空壳文件
        src_path = self.root_path / "src"
        if src_path.exists():
            for file_path in src_path.rglob("*.py"):
                if self._is_stub_file(file_path):
                    issues.append({
                        "type": "warning",
                        "message": f"发现空壳文件: {file_path.name}",
                        "path": str(file_path.relative_to(self.root_path)),
                        "content": self._get_file_preview(file_path)
                    })

        return issues

    def _is_stub_file(self, file_path: Path) -> bool:
        """判断是否为空壳文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # 空文件
            if not content:
                return True

            # 只有pass语句的文件
            if re.match(r'^class\s+\w+:\s*pass\s*$', content, re.MULTILINE):
                return True

            # 只有一行导入的文件
            lines = content.split('\n')
            if len(lines) <= 2 and all(line.strip().startswith(('import', 'from', '#', '')) for line in lines):
                return True

            return False
        except Exception:
            return False

    def _get_file_preview(self, file_path: Path) -> str:
        """获取文件预览"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content[:200] + "..." if len(content) > 200 else content
        except Exception:
            return "无法读取文件内容"

    def _check_naming_conventions(self) -> List[Dict]:
        """检查命名规范"""
        issues = []

        # 检查目录命名规范
        for root, dirs, files in os.walk(self.root_path):
            for dir_name in dirs:
                if not re.match(r'^[a-z_]+$', dir_name):
                    issues.append({
                        "type": "warning",
                        "message": f"目录命名不符合规范: {dir_name}",
                        "path": os.path.join(root, dir_name)
                    })

        return issues

    def _generate_report(self) -> Dict:
        """生成检查报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_issues": len(self.issues),
                "errors": len([i for i in self.issues if i["type"] == "error"]),
                "warnings": len([i for i in self.issues if i["type"] == "warning"])
            },
            "issues": self.issues,
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []

        if any(i["type"] == "error" for i in self.issues):
            recommendations.append("发现严重问题，建议立即修复")

        if any("重复" in i["message"] for i in self.issues):
            recommendations.append("发现重复目录，建议合并或删除重复项")

        if any("空壳" in i["message"] for i in self.issues):
            recommendations.append("发现空壳文件，建议实现或删除")

        if any("命名" in i["message"] for i in self.issues):
            recommendations.append("发现命名规范问题，建议统一命名规范")

        if not recommendations:
            recommendations.append("目录结构良好，继续保持")

        return recommendations

    def save_report(self, report: Dict, output_path: str = "reports/directory_check_report.json"):
        """保存检查报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"📄 检查报告已保存到: {output_file}")

    def print_report(self, report: Dict):
        """打印检查报告"""
        print("\n" + "="*60)
        print("📊 目录结构检查报告")
        print("="*60)

        summary = report["summary"]
        print(f"总问题数: {summary['total_issues']}")
        print(f"错误: {summary['errors']}")
        print(f"警告: {summary['warnings']}")

        if report["issues"]:
            print("\n🔍 发现的问题:")
            for i, issue in enumerate(report["issues"], 1):
                print(f"{i}. [{issue['type'].upper()}] {issue['message']}")
                print(f"   路径: {issue['path']}")
                if 'content' in issue:
                    print(f"   内容: {issue['content'][:100]}...")
                print()
        else:
            print("\n✅ 未发现任何问题")

        if report["recommendations"]:
            print("💡 建议:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec}")


def main():
    """主函数"""
    checker = DirectoryStructureChecker()
    report = checker.check_directory_structure()

    # 打印报告
    checker.print_report(report)

    # 保存报告
    checker.save_report(report)

    # 返回退出码
    if report["summary"]["errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
