#!/usr/bin/env python3
"""
RQA2025 文档同步器

自动同步代码实现与架构文档，确保文档与代码的一致性。

功能：
- 自动检测文档与代码的差异
- 生成更新的文档内容
- 支持架构文档、API文档等同步
- 提供文档质量评估

使用方法：
python scripts/documentation_synchronizer.py --check src/infrastructure/resource/
python scripts/documentation_synchronizer.py --sync docs/architecture/ARCHITECTURE_OVERVIEW.md
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from scripts.generate_architecture_docs import ArchitectureAnalyzer


@dataclass
class DocumentationIssue:
    """文档问题"""
    issue_type: str  # 'missing_component', 'outdated_stats', 'incorrect_reference'
    severity: str  # 'high', 'medium', 'low'
    file_path: str
    description: str
    suggested_fix: str
    line_number: Optional[int] = None


@dataclass
class SyncResult:
    """同步结果"""
    issues_found: List[DocumentationIssue] = field(default_factory=list)
    changes_made: List[str] = field(default_factory=list)
    success: bool = True
    error_message: str = ""


class DocumentationSynchronizer:
    """文档同步器"""

    def __init__(self):
        self.architecture_analyzer = ArchitectureAnalyzer()

    def check_consistency(self, code_path: str, docs_path: str) -> SyncResult:
        """
        检查文档与代码的一致性

        Args:
            code_path: 代码路径
            docs_path: 文档路径

        Returns:
            SyncResult: 检查结果
        """
        result = SyncResult()

        try:
            print("🔍 检查文档一致性...")

            # 分析代码结构
            code_analysis = self.architecture_analyzer.analyze_directory(Path(code_path))

            # 检查架构文档
            architecture_issues = self._check_architecture_docs(code_analysis, docs_path)
            result.issues_found.extend(architecture_issues)

            print(f"📊 发现 {len(result.issues_found)} 个文档问题")

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def sync_documentation(self, code_path: str, docs_path: str) -> SyncResult:
        """
        同步文档与代码

        Args:
            code_path: 代码路径
            docs_path: 文档路径

        Returns:
            SyncResult: 同步结果
        """
        result = SyncResult()

        try:
            print("🔄 开始文档同步...")

            # 检查一致性
            check_result = self.check_consistency(code_path, docs_path)
            result.issues_found = check_result.issues_found

            # 应用修复
            print(f"🔧 开始应用修复，共 {len(result.issues_found)} 个问题")
            for issue in result.issues_found:
                print(f"📋 处理问题: {issue.issue_type} (严重程度: {issue.severity})")
                if issue.severity == 'high':
                    print(f"🔧 应用高优先级修复: {issue.description}")
                    success = self._apply_fix(issue, docs_path)
                    if success:
                        result.changes_made.append(f"修复: {issue.description}")
                        print(f"✅ 修复成功")
                    else:
                        result.success = False
                        result.error_message = f"修复失败: {issue.description}"
                        print(f"❌ 修复失败")
                else:
                    print(f"⏭️ 跳过低优先级问题: {issue.description}")

            print(f"✅ 完成 {len(result.changes_made)} 个文档更新")

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def _check_architecture_docs(self, code_analysis: Dict[str, Any], docs_path: str) -> List[DocumentationIssue]:
        """检查架构文档"""
        issues = []

        # 检查架构总览文档
        overview_path = Path(docs_path) / "ARCHITECTURE_OVERVIEW.md"
        if overview_path.exists():
            issues.extend(self._check_overview_doc(code_analysis, overview_path))

        # 检查基础设施层架构设计文档
        infra_design_path = Path(docs_path) / "infrastructure_architecture_design.md"
        if infra_design_path.exists():
            issues.extend(self._check_infrastructure_design_doc(code_analysis, infra_design_path))

        return issues

    def _check_overview_doc(self, code_analysis: Dict[str, Any], doc_path: Path) -> List[DocumentationIssue]:
        """检查架构总览文档"""
        issues = []

        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查文件数量统计
            file_count_pattern = r'infrastructure层 \(Infrastructure Layer\) - (\d+)个文件'
            match = re.search(file_count_pattern, content)
            if not match:
                # 尝试更宽松的匹配 - 匹配基础设施层标题中的数字
                file_count_pattern = r'基础设施层 \(Infrastructure Layer\) - (\d+)个文件'
                match = re.search(file_count_pattern, content)
            if match:
                documented_count = int(match.group(1))
                actual_count = code_analysis['metrics']['total_modules']
                if actual_count and documented_count != actual_count:
                    issues.append(DocumentationIssue(
                        issue_type='outdated_stats',
                        severity='high',
                        file_path=str(doc_path),
                        description=f"文件数量统计不准确: 文档显示{documented_count}个，实际{actual_count}个",
                        suggested_fix=f"更新为{actual_count}个文件"
                    ))

            # 检查表格中的文件数量统计
            table_pattern = r'\|\s*\*\*基础设施层\*\*\s*\|\s*(\d+)文件\s*\|'
            match = re.search(table_pattern, content)
            if match:
                documented_count = int(match.group(1))
                actual_count = code_analysis['metrics']['total_modules']
                if documented_count != actual_count:
                    issues.append(DocumentationIssue(
                        issue_type='outdated_stats',
                        severity='high',
                        file_path=str(doc_path),
                        description=f"表格中文件数量统计不准确: 文档显示{documented_count}文件，实际{actual_count}文件",
                        suggested_fix=f"更新表格中的文件数量为{actual_count}文件"
                    ))

            # 检查章节标题中的文件数量统计
            section_pattern = r'#### \d+️⃣ 基础设施层 \((\d+)文件\)'
            match = re.search(section_pattern, content)
            if match:
                documented_count = int(match.group(1))
                actual_count = code_analysis['metrics']['total_modules']
                if documented_count != actual_count:
                    issues.append(DocumentationIssue(
                        issue_type='outdated_stats',
                        severity='high',
                        file_path=str(doc_path),
                        description=f"章节标题中文件数量统计不准确: 文档显示{documented_count}文件，实际{actual_count}文件",
                        suggested_fix=f"更新章节标题中的文件数量为{actual_count}文件"
                    ))

            # 检查组件数量描述
            component_pattern = r'\*\*组件数量\*\*:\s*(\d+)个文件'
            match = re.search(component_pattern, content)
            if match:
                documented_count = int(match.group(1))
                actual_count = code_analysis['metrics']['total_modules']
                if documented_count != actual_count:
                    issues.append(DocumentationIssue(
                        issue_type='outdated_stats',
                        severity='high',
                        file_path=str(doc_path),
                        description=f"组件数量描述不准确: 文档显示{documented_count}个文件，实际{actual_count}个文件",
                        suggested_fix=f"更新组件数量为{actual_count}个文件"
                    ))

            # 检查列表项中的文件数量统计
            list_item_pattern = r'-\s*✅\s*\*\*基础设施层\*\*:\s*(\d+)个文件'
            matches = re.findall(list_item_pattern, content)
            for match in matches:
                documented_count = int(match)
                actual_count = code_analysis['metrics']['total_modules']
                if documented_count != actual_count:
                    issues.append(DocumentationIssue(
                        issue_type='outdated_stats',
                        severity='high',
                        file_path=str(doc_path),
                        description=f"列表项中文件数量统计不准确: 文档显示{documented_count}个文件，实际{actual_count}个文件",
                        suggested_fix=f"更新列表项中的文件数量为{actual_count}个文件"
                    ))
                    break  # 只报告第一个匹配项

            # 检查组件存在性
            components_to_check = [
                ('ResourceManager', 'CoreResourceManager'),
                ('ConnectionPool', 'PoolComponent'),
                ('QuotaManager', None),  # 预期不存在
                ('UnifiedResourceManager', None),  # 文档可能缺失
            ]

            for doc_component, actual_component in components_to_check:
                if actual_component:
                    # 检查实际存在的组件是否在文档中提及
                    if actual_component not in content:
                        issues.append(DocumentationIssue(
                            issue_type='missing_component',
                            severity='medium',
                            file_path=str(doc_path),
                            description=f"文档未提及实际存在的组件: {actual_component}",
                            suggested_fix=f"添加{actual_component}组件说明"
                        ))
                else:
                    # 检查文档中提到的组件是否实际存在
                    if doc_component in content:
                        # 检查实际代码中是否存在这个组件
                        found = False
                        for module in code_analysis['modules'].values():
                            if doc_component in module.get('classes', []):
                                found = True
                                break
                        if not found:
                            issues.append(DocumentationIssue(
                                issue_type='incorrect_reference',
                                severity='medium',
                                file_path=str(doc_path),
                                description=f"文档提及但代码中不存在的组件: {doc_component}",
                                suggested_fix=f"移除或更正{doc_component}组件引用"
                            ))

        except Exception as e:
            issues.append(DocumentationIssue(
                issue_type='parse_error',
                severity='high',
                file_path=str(doc_path),
                description=f"文档解析失败: {e}",
                suggested_fix="检查文档格式是否正确"
            ))

        return issues

    def _check_infrastructure_design_doc(self, code_analysis: Dict[str, Any], doc_path: Path) -> List[DocumentationIssue]:
        """检查基础设施层架构设计文档"""
        issues = []

        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查文件数量统计
            # 匹配类似 "160+个模块化文件" 的模式
            file_count_patterns = [
                r'(\d+)\+个模块化文件',
                r'文件数量.*(\d+)\+.*文件',
                r'(\d+)\+.*文件'
            ]

            actual_count = code_analysis['metrics']['total_modules']

            for pattern in file_count_patterns:
                match = re.search(pattern, content)
                if match:
                    documented_count = int(match.group(1))
                    if documented_count != actual_count:
                        issues.append(DocumentationIssue(
                            issue_type='outdated_stats',
                            severity='high',
                            file_path=str(doc_path),
                            description=f"基础设施层文件数量统计不准确: 文档显示{documented_count}+个，实际{actual_count}个",
                            suggested_fix=f"更新为{actual_count}个文件"
                        ))
                    break

            # 检查组件一致性
            # 这里可以添加更多具体的检查逻辑

        except Exception as e:
            issues.append(DocumentationIssue(
                issue_type='parse_error',
                severity='high',
                file_path=str(doc_path),
                description=f"基础设施层架构设计文档解析失败: {e}",
                suggested_fix="检查文档格式是否正确"
            ))

        return issues

    def _apply_fix(self, issue: DocumentationIssue, docs_path: str) -> bool:
        """应用修复"""
        try:
            print(f"🔧 尝试修复问题: {issue.issue_type} - {issue.description}")
            if issue.issue_type == 'outdated_stats':
                # 判断是哪个文档和问题类型
                if 'infrastructure_architecture_design.md' in str(issue.file_path):
                    result = self._fix_infrastructure_design_file_count(issue, docs_path)
                elif '表格中文件数量统计' in issue.description:
                    result = self._fix_table_file_count(issue, docs_path)
                elif '章节标题中文件数量统计' in issue.description:
                    result = self._fix_section_file_count(issue, docs_path)
                elif '组件数量描述' in issue.description:
                    result = self._fix_component_count(issue, docs_path)
                elif '列表项中文件数量统计' in issue.description:
                    result = self._fix_list_item_count(issue, docs_path)
                else:
                    result = self._fix_file_count(issue, docs_path)
                if result:
                    print(f"✅ 文件数量统计已更新")
                return result
            elif issue.issue_type == 'missing_component':
                result = self._add_component_doc(issue, docs_path)
                if result:
                    print(f"✅ 组件文档已添加")
                return result

            print(f"⚠️ 未知问题类型: {issue.issue_type}")
            return False
        except Exception as e:
            print(f"❌ 应用修复失败: {e}")
            return False

    def _fix_file_count(self, issue: DocumentationIssue, docs_path: str) -> bool:
        """修复文件数量统计"""
        try:
            # 从问题描述中提取实际数量
            match = re.search(r'实际(\d+)个', issue.description)
            if not match:
                return False

            actual_count = match.group(1)
            doc_path = Path(issue.file_path)

            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换文件数量 - 主标题
            pattern = r'基础设施层 \(Infrastructure Layer\) - \d+个文件'
            replacement = f'基础设施层 (Infrastructure Layer) - {actual_count}个文件'
            new_content = re.sub(pattern, replacement, content)

            # 替换Mermaid图表中的文件数量
            mermaid_pattern = r'CS3\[基础设施层<br/>\d+文件\]'
            mermaid_replacement = f'CS3[基础设施层<br/>{actual_count}文件]'
            new_content = re.sub(mermaid_pattern, mermaid_replacement, new_content)

            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return True
        except Exception:
            return False

    def _add_component_doc(self, issue: DocumentationIssue, docs_path: str) -> bool:
        """添加组件文档"""
        try:
            # 从问题描述中提取组件名
            match = re.search(r'实际存在的组件: (\w+)', issue.description)
            if not match:
                return False

            component_name = match.group(1)
            doc_path = Path(issue.file_path)

            # 在文档中添加组件说明（简化版）
            addition = f"""- ✅ **{component_name}**: 新增组件，负责相关功能

"""

            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 在合适的位置添加（这里简化处理，实际可能需要更智能的位置识别）
            if "## 2.1.1 核心组件" in content:
                content = content.replace(
                    "## 2.1.1 核心组件",
                    "## 2.1.1 核心组件\n" + addition
                )

                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                return True

            return False
        except Exception:
            return False

    def _fix_table_file_count(self, issue: DocumentationIssue, docs_path: str) -> bool:
        """修复表格中的文件数量统计"""
        try:
            # 从问题描述中提取实际数量
            match = re.search(r'实际(\d+)文件', issue.description)
            if not match:
                print(f"❌ 无法从描述中提取数量: {issue.description}")
                return False

            actual_count = match.group(1)
            doc_path = Path(issue.file_path)

            print(f"📄 读取架构总览文档: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换表格中的文件数量
            table_pattern = r'(\|\s*\*\*基础设施层\*\*\s*\|\s*)\d+(文件\s*\|)'
            replacement = f'\\g<1>{actual_count}\\g<2>'
            print(f"🔄 替换表格模式: {table_pattern}")
            print(f"🔄 替换内容: {replacement}")

            new_content = re.sub(table_pattern, replacement, content)

            if new_content != content:
                print(f"✅ 表格内容已更改，保存文件...")
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            else:
                print(f"⚠️ 表格内容未更改，可能模式不匹配")
                return False
        except Exception as e:
            print(f"❌ 修复表格文件数量统计失败: {e}")
            return False

    def _fix_section_file_count(self, issue: DocumentationIssue, docs_path: str) -> bool:
        """修复章节标题中的文件数量统计"""
        try:
            # 从问题描述中提取实际数量
            match = re.search(r'实际(\d+)文件', issue.description)
            if not match:
                print(f"❌ 无法从描述中提取数量: {issue.description}")
                return False

            actual_count = match.group(1)
            doc_path = Path(issue.file_path)

            print(f"📄 读取架构总览文档: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换章节标题中的文件数量
            section_pattern = r'(#### \d+️⃣ 基础设施层 \()\d+(文件\))'
            replacement = f'\\g<1>{actual_count}\\g<2>'
            print(f"🔄 替换章节模式: {section_pattern}")
            print(f"🔄 替换内容: {replacement}")

            new_content = re.sub(section_pattern, replacement, content)

            if new_content != content:
                print(f"✅ 章节标题已更改，保存文件...")
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            else:
                print(f"⚠️ 章节标题未更改，可能模式不匹配")
                return False
        except Exception as e:
            print(f"❌ 修复章节标题文件数量统计失败: {e}")
            return False

    def _fix_component_count(self, issue: DocumentationIssue, docs_path: str) -> bool:
        """修复组件数量描述"""
        try:
            # 从问题描述中提取实际数量
            match = re.search(r'实际(\d+)个文件', issue.description)
            if not match:
                print(f"❌ 无法从描述中提取数量: {issue.description}")
                return False

            actual_count = match.group(1)
            doc_path = Path(issue.file_path)

            print(f"📄 读取架构总览文档: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换组件数量描述
            component_pattern = r'(\*\*组件数量\*\*:\s*)\d+(个文件)'
            replacement = f'\\g<1>{actual_count}\\g<2>'
            print(f"🔄 替换组件模式: {component_pattern}")
            print(f"🔄 替换内容: {replacement}")

            new_content = re.sub(component_pattern, replacement, content)

            if new_content != content:
                print(f"✅ 组件数量已更改，保存文件...")
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            else:
                print(f"⚠️ 组件数量未更改，可能模式不匹配")
                return False
        except Exception as e:
            print(f"❌ 修复组件数量描述失败: {e}")
            return False

    def _fix_list_item_count(self, issue: DocumentationIssue, docs_path: str) -> bool:
        """修复列表项中的文件数量统计"""
        try:
            # 从问题描述中提取实际数量
            match = re.search(r'实际(\d+)个文件', issue.description)
            if not match:
                print(f"❌ 无法从描述中提取数量: {issue.description}")
                return False

            actual_count = match.group(1)
            doc_path = Path(issue.file_path)

            print(f"📄 读取架构总览文档: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换列表项中的文件数量
            list_item_pattern = r'(-\s*✅\s*\*\*基础设施层\*\*:\s*)\d+(个文件)'
            replacement = f'\\g<1>{actual_count}\\g<2>'
            print(f"🔄 替换列表项模式: {list_item_pattern}")
            print(f"🔄 替换内容: {replacement}")

            new_content = re.sub(list_item_pattern, replacement, content)

            if new_content != content:
                print(f"✅ 列表项已更改，保存文件...")
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            else:
                print(f"⚠️ 列表项未更改，可能模式不匹配")
                return False
        except Exception as e:
            print(f"❌ 修复列表项文件数量统计失败: {e}")
            return False

    def _fix_infrastructure_design_file_count(self, issue: DocumentationIssue, docs_path: str) -> bool:
        """修复基础设施层架构设计文档的文件数量统计"""
        try:
            # 从问题描述中提取实际数量
            match = re.search(r'实际(\d+)个', issue.description)
            if not match:
                print(f"❌ 无法从描述中提取数量: {issue.description}")
                return False

            actual_count = match.group(1)
            doc_path = Path(issue.file_path)

            print(f"📄 读取基础设施层架构设计文档: {doc_path}")
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换文件数量 - 匹配 "160+个模块化文件" 这样的模式
            pattern = r'(\d+)\+个模块化文件'
            replacement = f'{actual_count}+个模块化文件'
            print(f"🔄 替换模式: {pattern}")
            print(f"🔄 替换内容: {replacement}")

            new_content = re.sub(pattern, replacement, content)

            if new_content != content:
                print(f"✅ 内容已更改，保存文件...")
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            else:
                print(f"⚠️ 内容未更改，可能模式不匹配")
                return False
        except Exception as e:
            print(f"❌ 修复基础设施层架构设计文档文件数量统计失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025 文档同步器")
    parser.add_argument("code_path", help="代码路径")
    parser.add_argument("--docs-path", "-d", default="docs/architecture",
                        help="文档路径")
    parser.add_argument("--check", action="store_true",
                        help="仅检查，不同步")
    parser.add_argument("--sync", action="store_true",
                        help="执行同步")

    args = parser.parse_args()

    synchronizer = DocumentationSynchronizer()

    if args.check:
        result = synchronizer.check_consistency(args.code_path, args.docs_path)

        if result.issues_found:
            print("\n📋 发现的文档问题:")
            for issue in result.issues_found:
                severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[issue.severity]
                print(f"{severity_icon} {issue.description}")
                print(f"   💡 {issue.suggested_fix}")
        else:
            print("✅ 文档与代码一致")

    elif args.sync:
        result = synchronizer.sync_documentation(args.code_path, args.docs_path)

        if result.success:
            print(f"✅ 同步完成，共应用 {len(result.changes_made)} 个修复")
            for change in result.changes_made:
                print(f"  • {change}")
        else:
            print(f"❌ 同步失败: {result.error_message}")

    else:
        print("请指定 --check 或 --sync 参数")


if __name__ == "__main__":
    main()
