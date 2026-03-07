#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文档结构验证脚本
Documentation Structure Verification Script

验证文档结构的完整性和正确性，确保文件移动和重构后所有引用都已正确更新。
"""

import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DocsStructureVerifier:
    """文档结构验证器"""

    def __init__(self):
        """初始化验证器"""
        self.project_root = project_root
        self.docs_root = project_root / "docs"
        self.issues = []
        self.warnings = []

    def verify_structure(self) -> Dict[str, Any]:
        """
        验证文档结构

        Returns:
            Dict[str, Any]: 验证结果
        """
        print("🔍 开始验证文档结构...")

        results = {
            "structure_check": self._check_directory_structure(),
            "file_integrity_check": self._check_file_integrity(),
            "reference_check": self._check_references(),
            "index_consistency_check": self._check_index_consistency(),
            "issues": self.issues,
            "warnings": self.warnings
        }

        # 输出结果摘要
        self._print_summary(results)

        return results

    def _check_directory_structure(self) -> Dict[str, Any]:
        """检查目录结构"""
        print("📁 检查目录结构...")

        # 检查实际存在的关键文件和目录
        expected_structure = {
            "docs/architecture/": "架构设计文档目录",
            "docs/architecture/strategy_layer_architecture_design.md": "策略服务层架构设计",
            "docs/strategy/": "策略相关文档目录",
            "docs/DEVELOPMENT_INDEX.md": "文档索引"
        }

        structure_results = {}

        for path_str, description in expected_structure.items():
            # 移除docs/前缀，因为我们已经在docs_root中
            if path_str.startswith('docs/'):
                path_str = path_str[5:]
            path = self.docs_root / path_str
            if path_str.endswith('/'):
                # 检查目录
                if path.exists() and path.is_dir():
                    structure_results[path_str] = {"status": "present", "type": "directory"}
                else:
                    structure_results[path_str] = {"status": "missing", "type": "directory"}
                    self.issues.append(f"缺少目录: {path_str}")
            else:
                # 检查文件
                if path.exists() and path.is_file():
                    structure_results[path_str] = {"status": "present", "type": "file"}
                else:
                    structure_results[path_str] = {"status": "missing", "type": "file"}
                    self.issues.append(f"缺少文件: {path_str}")

        return structure_results

    def _check_file_integrity(self) -> Dict[str, Any]:
        """检查文件完整性"""
        print("📄 检查文件完整性...")

        integrity_results = {}

        # 检查关键文件的大小和内容
        key_files = {
            "docs/architecture/strategy_layer_architecture_design.md": 10000,  # 预期最小大小
            "docs/DEVELOPMENT_INDEX.md": 5000
        }

        for file_path, min_size in key_files.items():
            # 移除docs/前缀，因为我们已经在docs_root中
            if file_path.startswith('docs/'):
                file_path = file_path[5:]
            full_path = self.docs_root / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                integrity_results[file_path] = {
                    "size": size,
                    "status": "ok" if size >= min_size else "too_small"
                }

                if size < min_size:
                    self.warnings.append(f"文件过小: {file_path} ({size} bytes)")

                # 检查文件是否可读且包含预期内容
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            integrity_results[file_path]["status"] = "empty"
                            self.issues.append(f"文件为空: {file_path}")
                        elif len(content) < 100:
                            integrity_results[file_path]["status"] = "too_short"
                            self.warnings.append(f"文件内容过短: {file_path}")
                except Exception as e:
                    integrity_results[file_path]["status"] = "unreadable"
                    self.issues.append(f"文件不可读: {file_path} - {e}")
            else:
                integrity_results[file_path] = {"status": "missing"}

        return integrity_results

    def _check_references(self) -> Dict[str, Any]:
        """检查引用一致性"""
        print("🔗 检查引用一致性...")

        reference_results = {}

        # 检查文档索引中的引用
        index_file = self.docs_root / "DEVELOPMENT_INDEX.md"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查关键引用
                key_references = [
                    "architecture/strategy_layer_architecture_design.md",
                    "architecture/BUSINESS_PROCESS_DRIVEN_ARCHITECTURE.md",
                    "README.md",
                    "USER_GUIDE.md"
                ]

                for ref in key_references:
                    if ref in content:
                        reference_results[ref] = {"status": "found"}
                    else:
                        reference_results[ref] = {"status": "missing"}
                        self.warnings.append(f"文档索引中缺少引用: {ref}")

            except Exception as e:
                reference_results["index_read_error"] = {"status": "error", "error": str(e)}
                self.issues.append(f"读取文档索引失败: {e}")

        return reference_results

    def _check_index_consistency(self) -> Dict[str, Any]:
        """检查索引一致性"""
        print("📋 检查索引一致性...")

        index_results = {}

        # 检查文档索引版本和更新时间
        index_file = self.docs_root / "DEVELOPMENT_INDEX.md"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查版本信息
                if "**文档版本**:" in content:
                    index_results["version_info"] = {"status": "present"}
                else:
                    index_results["version_info"] = {"status": "missing"}
                    self.warnings.append("文档索引缺少版本信息")

                # 检查更新日志
                if "## 文档更新日志" in content:
                    index_results["update_log"] = {"status": "present"}
                else:
                    index_results["update_log"] = {"status": "missing"}
                    self.warnings.append("文档索引缺少更新日志")

            except Exception as e:
                index_results["index_check_error"] = {"status": "error", "error": str(e)}
                self.issues.append(f"检查文档索引失败: {e}")

        return index_results

    def _print_summary(self, results: Dict[str, Any]):
        """打印验证摘要"""
        print("\n" + "="*60)
        print("📊 文档结构验证结果")
        print("="*60)

        total_checks = 0
        passed_checks = 0

        # 计算通过的检查
        for category, checks in results.items():
            if category in ["issues", "warnings"]:
                continue
            if isinstance(checks, dict):
                total_checks += len(checks)
                for check_name, check_result in checks.items():
                    if isinstance(check_result, dict):
                        status = check_result.get("status", "unknown")
                        if status in ["present", "ok", "found"]:
                            passed_checks += 1

        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0

        print(
            f"📈 总体状态: {'✅ 通过' if success_rate >= 90 else '⚠️ 需要注意' if success_rate >= 70 else '❌ 需要修复'}")
        print(f"📋 总检查数: {total_checks}")
        print(f"✅ 通过检查: {passed_checks}")
        print(f"❌ 失败检查: {total_checks - passed_checks}")
        print(f"📊 成功率: {success_rate:.1f}%")

        # 显示问题
        if results["issues"]:
            print(f"\n❌ 发现 {len(results['issues'])} 个问题:")
            for issue in results["issues"][:5]:  # 只显示前5个
                print(f"  • {issue}")
            if len(results["issues"]) > 5:
                print(f"  ... 还有 {len(results['issues']) - 5} 个问题")

        if results["warnings"]:
            print(f"\n⚠️ 发现 {len(results['warnings'])} 个警告:")
            for warning in results["warnings"][:3]:  # 只显示前3个
                print(f"  • {warning}")
            if len(results["warnings"]) > 3:
                print(f"  ... 还有 {len(results['warnings']) - 3} 个警告")

        print("="*60)


def main():
    """主函数"""
    try:
        verifier = DocsStructureVerifier()
        results = verifier.verify_structure()

        # 根据验证结果设置退出码
        if results.get("issues"):
            print(f"\n❌ 发现 {len(results['issues'])} 个问题需要修复")
            sys.exit(1)
        elif results.get("warnings"):
            print(f"\n⚠️ 发现 {len(results['warnings'])} 个警告建议处理")
            sys.exit(0)  # 警告不影响退出码
        else:
            print("\n✅ 文档结构验证全部通过")
            sys.exit(0)

    except Exception as e:
        print(f"❌ 文档结构验证失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
