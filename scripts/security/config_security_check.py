#!/usr/bin/env python3
"""
配置文件安全检查脚本

检查项目配置文件中的安全风险，包括敏感信息、硬编码密码等
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


class ConfigSecurityChecker:
    """配置文件安全检查器"""

    def __init__(self):
        self.security_issues = []
        self.sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'private_key\s*=\s*["\'][^"\']+["\']',
            r'access_token\s*=\s*["\'][^"\']+["\']',
            r'credential\s*=\s*["\'][^"\']+["\']',
            r'password\s*:\s*["\'][^"\']+["\']',
            r'secret\s*:\s*["\'][^"\']+["\']',
            r'api_key\s*:\s*["\'][^"\']+["\']',
            r'token\s*:\s*["\'][^"\']+["\']',
            r'private_key\s*:\s*["\'][^"\']+["\']',
            r'access_token\s*:\s*["\'][^"\']+["\']',
            r'credential\s*:\s*["\'][^"\']+["\']'
        ]

        self.config_files = [
            'config.ini',
            'config.yml',
            'config.yaml',
            'settings.py',
            'config.py',
            '.env',
            'environment.yml',
            'requirements.txt',
            'docker-compose.yml',
            'deploy/*.yml',
            'deploy/*.yaml'
        ]

    def check_config_files(self) -> Dict[str, Any]:
        """检查配置文件安全"""
        print("🔒 开始配置文件安全检查...")

        results = {
            "total_files_checked": 0,
            "files_with_issues": 0,
            "security_issues": [],
            "recommendations": []
        }

        # 检查配置文件
        for pattern in self.config_files:
            files = list(Path('.').glob(pattern))
            for file_path in files:
                if file_path.is_file():
                    results["total_files_checked"] += 1
                    file_issues = self._check_single_file(file_path)
                    if file_issues:
                        results["files_with_issues"] += 1
                        results["security_issues"].extend(file_issues)

        # 生成建议
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _check_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """检查单个文件的安全问题"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

                for line_num, line in enumerate(lines, 1):
                    for pattern in self.sensitive_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            issues.append({
                                "file": str(file_path),
                                "line": line_num,
                                "issue_type": "hardcoded_credential",
                                "description": f"发现硬编码凭据: {line.strip()}",
                                "severity": "high",
                                "recommendation": "使用环境变量或密钥管理系统"
                            })

                # 检查其他安全问题
                if self._check_insecure_config(content, file_path):
                    issues.append({
                        "file": str(file_path),
                        "line": 0,
                        "issue_type": "insecure_configuration",
                        "description": "发现不安全的配置",
                        "severity": "medium",
                        "recommendation": "检查配置安全性"
                    })

        except Exception as e:
            issues.append({
                "file": str(file_path),
                "line": 0,
                "issue_type": "file_read_error",
                "description": f"无法读取文件: {e}",
                "severity": "low",
                "recommendation": "检查文件权限和格式"
            })

        return issues

    def _check_insecure_config(self, content: str, file_path: Path) -> bool:
        """检查不安全配置"""
        insecure_patterns = [
            r'debug\s*=\s*true',
            r'DEBUG\s*=\s*True',
            r'allow_all\s*=\s*true',
            r'permissive\s*=\s*true',
            r'unsafe\s*=\s*true'
        ]

        for pattern in insecure_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成安全建议"""
        recommendations = [
            "使用环境变量存储敏感信息",
            "实施密钥管理系统",
            "定期轮换密码和密钥",
            "启用配置文件加密",
            "实施访问控制",
            "定期进行安全审计",
            "使用安全的配置模板",
            "实施配置验证"
        ]

        if results["files_with_issues"] > 0:
            recommendations.extend([
                "立即修复发现的硬编码凭据",
                "审查所有配置文件",
                "实施配置管理最佳实践"
            ])

        return recommendations

    def print_results(self, results: Dict[str, Any]):
        """打印检查结果"""
        print("\n" + "="*60)
        print("🔒 配置文件安全检查结果")
        print("="*60)
        print(f"📁 检查文件数: {results['total_files_checked']}")
        print(f"⚠️ 发现问题文件: {results['files_with_issues']}")
        print(f"🔍 安全问题数: {len(results['security_issues'])}")
        print()

        if results['security_issues']:
            print("🚨 发现的安全问题:")
            for issue in results['security_issues']:
                print(f"  📄 {issue['file']}:{issue['line']}")
                print(f"     🔴 {issue['issue_type']}: {issue['description']}")
                print(f"     💡 建议: {issue['recommendation']}")
                print()

        print("📋 安全建议:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")

        print("="*60)

    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存检查结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"✅ 安全检查结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="配置文件安全检查")
    parser.add_argument("--output", default="security_config_report.json", help="输出文件")

    args = parser.parse_args()

    try:
        # 创建检查器
        checker = ConfigSecurityChecker()

        # 执行检查
        results = checker.check_config_files()

        # 打印结果
        checker.print_results(results)

        # 保存结果
        checker.save_results(results, args.output)

        # 返回状态码
        if results['files_with_issues'] > 0:
            print("❌ 发现安全问题，请及时修复")
            exit(1)
        else:
            print("✅ 配置文件安全检查通过")
            exit(0)

    except Exception as e:
        print(f"❌ 安全检查时出错: {e}")
        exit(1)


if __name__ == "__main__":
    main()
