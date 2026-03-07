#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
定期维护检查脚本
用于定期检查项目健康状况，包括代码质量、文档完整性、测试覆盖等
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict


class MaintenanceChecker:
    """维护检查器"""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "summary": {},
            "recommendations": []
        }

    def run_all_checks(self) -> Dict:
        """运行所有检查"""
        print("🔍 开始定期维护检查...")

        # 运行各项检查
        self._check_directory_structure()
        self._check_code_quality()
        self._check_test_coverage()
        self._check_documentation()
        self._check_dependencies()
        self._check_security()

        # 生成总结
        self._generate_summary()

        return self.report

    def _check_directory_structure(self):
        """检查目录结构"""
        print("📁 检查目录结构...")

        # 运行目录结构检查脚本
        try:
            result = subprocess.run(
                ["python", "scripts/directory_structure_checker.py"],
                capture_output=True,
                text=True,
                cwd=self.root_path
            )

            if result.returncode == 0:
                self.report["checks"]["directory_structure"] = {
                    "status": "pass",
                    "message": "目录结构良好"
                }
            else:
                self.report["checks"]["directory_structure"] = {
                    "status": "fail",
                    "message": "目录结构存在问题",
                    "details": result.stdout
                }
        except Exception as e:
            self.report["checks"]["directory_structure"] = {
                "status": "error",
                "message": f"检查失败: {e}"
            }

    def _check_code_quality(self):
        """检查代码质量"""
        print("📝 检查代码质量...")

        # 检查Python文件数量
        python_files = list(self.root_path.rglob("*.py"))
        total_lines = 0

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
            except Exception:
                continue

        self.report["checks"]["code_quality"] = {
            "status": "pass",
            "python_files": len(python_files),
            "total_lines": total_lines,
            "message": f"发现 {len(python_files)} 个Python文件，共 {total_lines} 行代码"
        }

    def _check_test_coverage(self):
        """检查测试覆盖率"""
        print("🧪 检查测试覆盖率...")

        # 检查测试文件数量
        test_files = list(self.root_path.rglob("test_*.py"))
        src_files = list((self.root_path / "src").rglob("*.py"))

        test_ratio = len(test_files) / len(src_files) if src_files else 0

        if test_ratio >= 0.5:
            status = "pass"
            message = f"测试覆盖率良好 ({test_ratio:.1%})"
        elif test_ratio >= 0.3:
            status = "warning"
            message = f"测试覆盖率一般 ({test_ratio:.1%})"
        else:
            status = "fail"
            message = f"测试覆盖率较低 ({test_ratio:.1%})"

        self.report["checks"]["test_coverage"] = {
            "status": status,
            "test_files": len(test_files),
            "src_files": len(src_files),
            "test_ratio": test_ratio,
            "message": message
        }

    def _check_documentation(self):
        """检查文档完整性"""
        print("📚 检查文档完整性...")

        # 检查文档文件
        doc_files = list(self.root_path.rglob("*.md"))
        doc_files.extend(list(self.root_path.rglob("*.rst")))

        # 检查关键文档
        key_docs = [
            "README.md",
            "docs/architecture_design.md",
            "docs/development_guide.md"
        ]

        missing_docs = []
        for doc in key_docs:
            if not (self.root_path / doc).exists():
                missing_docs.append(doc)

        if not missing_docs:
            status = "pass"
            message = "关键文档完整"
        else:
            status = "warning"
            message = f"缺少关键文档: {', '.join(missing_docs)}"

        self.report["checks"]["documentation"] = {
            "status": status,
            "total_docs": len(doc_files),
            "missing_docs": missing_docs,
            "message": message
        }

    def _check_dependencies(self):
        """检查依赖项"""
        print("📦 检查依赖项...")

        # 检查requirements.txt
        requirements_file = self.root_path / "requirements.txt"

        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    requirements = f.readlines()

                # 统计依赖项数量
                package_count = len(
                    [line for line in requirements if line.strip() and not line.startswith('#')])

                self.report["checks"]["dependencies"] = {
                    "status": "pass",
                    "package_count": package_count,
                    "message": f"发现 {package_count} 个依赖包"
                }
            except Exception as e:
                self.report["checks"]["dependencies"] = {
                    "status": "error",
                    "message": f"读取依赖文件失败: {e}"
                }
        else:
            self.report["checks"]["dependencies"] = {
                "status": "warning",
                "message": "未找到requirements.txt文件"
            }

    def _check_security(self):
        """检查安全问题"""
        print("🔒 检查安全问题...")

        # 检查是否有硬编码的密钥
        hardcoded_keys = []

        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # 检查常见的硬编码密钥模式
                    patterns = [
                        r'api_key\s*=\s*["\'][^"\']+["\']',
                        r'password\s*=\s*["\'][^"\']+["\']',
                        r'secret\s*=\s*["\'][^"\']+["\']',
                        r'token\s*=\s*["\'][^"\']+["\']'
                    ]

                    for pattern in patterns:
                        import re
                        if re.search(pattern, content, re.IGNORECASE):
                            hardcoded_keys.append(str(file_path))
                            break
            except Exception:
                continue

        if not hardcoded_keys:
            status = "pass"
            message = "未发现硬编码密钥"
        else:
            status = "fail"
            message = f"发现 {len(hardcoded_keys)} 个文件包含硬编码密钥"

        self.report["checks"]["security"] = {
            "status": status,
            "hardcoded_keys": hardcoded_keys,
            "message": message
        }

    def _generate_summary(self):
        """生成检查总结"""
        checks = self.report["checks"]

        # 统计各状态数量
        status_counts = {"pass": 0, "warning": 0, "fail": 0, "error": 0}
        for check in checks.values():
            status_counts[check["status"]] += 1

        # 生成总结
        total_checks = len(checks)
        pass_rate = status_counts["pass"] / total_checks if total_checks > 0 else 0

        self.report["summary"] = {
            "total_checks": total_checks,
            "pass_count": status_counts["pass"],
            "warning_count": status_counts["warning"],
            "fail_count": status_counts["fail"],
            "error_count": status_counts["error"],
            "pass_rate": pass_rate
        }

        # 生成建议
        recommendations = []

        if status_counts["fail"] > 0:
            recommendations.append("发现严重问题，建议立即修复")

        if status_counts["warning"] > 0:
            recommendations.append("发现警告问题，建议尽快处理")

        if pass_rate < 0.8:
            recommendations.append("整体健康状况一般，建议改进")
        else:
            recommendations.append("项目健康状况良好，继续保持")

        self.report["recommendations"] = recommendations

    def print_report(self):
        """打印检查报告"""
        print("\n" + "="*60)
        print("📊 定期维护检查报告")
        print("="*60)

        summary = self.report["summary"]
        print(f"总检查项: {summary['total_checks']}")
        print(f"通过: {summary['pass_count']}")
        print(f"警告: {summary['warning_count']}")
        print(f"失败: {summary['fail_count']}")
        print(f"错误: {summary['error_count']}")
        print(f"通过率: {summary['pass_rate']:.1%}")

        print("\n🔍 详细检查结果:")
        for check_name, check_result in self.report["checks"].items():
            status_icon = {
                "pass": "✅",
                "warning": "⚠️",
                "fail": "❌",
                "error": "💥"
            }.get(check_result["status"], "❓")

            print(f"{status_icon} {check_name}: {check_result['message']}")

        if self.report["recommendations"]:
            print("\n💡 建议:")
            for i, rec in enumerate(self.report["recommendations"], 1):
                print(f"{i}. {rec}")

    def save_report(self, output_path: str = "reports/maintenance_report.json"):
        """保存检查报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)

        print(f"\n📄 维护报告已保存到: {output_file}")


def main():
    """主函数"""
    checker = MaintenanceChecker()
    report = checker.run_all_checks()

    # 打印报告
    checker.print_report()

    # 保存报告
    checker.save_report()

    # 返回退出码
    if report["summary"]["fail_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
