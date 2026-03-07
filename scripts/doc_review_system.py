#!/usr/bin/env python3
"""
RQA2025 文档审查系统
用于自动检查和验证文档质量，确保文档规范性和一致性
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class DocumentationReviewer:
    """文档审查器"""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed_checks = []

        # 文档规范配置
        self.doc_standards = {
            'architecture': {
                'required_sections': [
                    '文档概述', '核心业务目标', '架构概述', '架构设计理念',
                    '分层架构', '核心组件设计', '性能优化设计', '总结'
                ],
                'required_patterns': [
                    r'## 📋 文档概述',
                    r'## 🎯 核心业务目标',
                    r'## 🏛️.*架构',
                    r'## 📊.*价值实现',
                    r'## 📋 总结'
                ]
            },
            'readme': {
                'required_sections': [
                    '模块定位', '架构概述', '主要组件', '实现状态',
                    '典型用法', '版本信息'
                ],
                'required_patterns': [
                    r'## \d+\. ',
                    r'### \d+\.\d+ ',
                    r'```python',
                    r'```mermaid'
                ]
            },
            'api': {
                'required_sections': [
                    '概述', '接口定义', '使用示例', '错误处理'
                ],
                'required_patterns': [
                    r'### \w+',
                    r'```python',
                    r'@abstractmethod'
                ]
            }
        }

        # 质量检查规则
        self.quality_rules = {
            'structure': [
                ('missing_headers', r'^#{1,6} ', '文档缺少标题结构'),
                ('broken_links', r'\[.*\]\(.*\)', '可能存在损坏的链接'),
                ('inconsistent_formatting', r'^\s*#+\s*$', '标题格式不一致')
            ],
            'content': [
                ('empty_sections', r'^#{1,6}\s*$', '存在空章节'),
                ('missing_examples', r'```python.*```', '缺少代码示例'),
                ('inconsistent_naming', r'[a-z][A-Z]', '命名不一致')
            ],
            'metadata': [
                ('missing_version', r'版本.*v\d+', '缺少版本信息'),
                ('missing_date', r'\d{4}年\d{1,2}月\d{1,2}日', '缺少更新日期'),
                ('missing_author', r'作者.*[:：]', '缺少作者信息')
            ]
        }

    def review_document(self, file_path: str) -> Dict[str, Any]:
        """审查单个文档"""
        if not os.path.exists(file_path):
            return {'error': f'文件不存在: {file_path}'}

        print(f"正在审查文档: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        doc_type = self._identify_doc_type(file_path)
        results = {
            'file_path': file_path,
            'doc_type': doc_type,
            'issues': [],
            'warnings': [],
            'score': 0,
            'recommendations': []
        }

        # 结构检查
        structure_issues = self._check_document_structure(content, doc_type)
        results['issues'].extend(structure_issues)

        # 内容质量检查
        quality_issues = self._check_content_quality(content)
        results['issues'].extend(quality_issues)

        # 规范符合性检查
        compliance_issues = self._check_standards_compliance(content, doc_type)
        results['issues'].extend(compliance_issues)

        # 计算质量分数
        results['score'] = self._calculate_quality_score(results['issues'])

        # 生成改进建议
        results['recommendations'] = self._generate_recommendations(results['issues'])

        return results

    def _identify_doc_type(self, file_path: str) -> str:
        """识别文档类型"""
        path_parts = Path(file_path).parts

        if 'architecture' in path_parts:
            if 'README.md' in file_path:
                return 'architecture_readme'
            else:
                return 'architecture'
        elif 'api' in path_parts:
            return 'api'
        elif 'README.md' in str(file_path).upper():
            return 'readme'
        else:
            return 'general'

    def _check_document_structure(self, content: str, doc_type: str) -> List[Dict]:
        """检查文档结构"""
        issues = []
        lines = content.split('\n')

        # 检查必需章节
        if doc_type in self.doc_standards:
            standards = self.doc_standards[doc_type]
            for section in standards.get('required_sections', []):
                if section not in content:
                    issues.append({
                        'type': 'missing_section',
                        'severity': 'high',
                        'message': f'缺少必需章节: {section}',
                        'line': 0
                    })

        # 检查标题层级
        header_levels = []
        for i, line in enumerate(lines):
            if line.startswith('#'):
                level = len(line.split()[0]) if line.split() else 0
                header_levels.append(level)

                # 检查标题格式
                if not re.match(r'^#{1,6}\s+\S', line):
                    issues.append({
                        'type': 'header_format',
                        'severity': 'medium',
                        'message': f'标题格式不正确: {line.strip()}',
                        'line': i + 1
                    })

        # 检查标题层级跳跃
        for i in range(1, len(header_levels)):
            if header_levels[i] > header_levels[i-1] + 1:
                issues.append({
                    'type': 'header_hierarchy',
                    'severity': 'low',
                    'message': f'标题层级跳跃过大 (第{i+1}个标题)',
                    'line': 0
                })

        return issues

    def _check_content_quality(self, content: str) -> List[Dict]:
        """检查内容质量"""
        issues = []
        lines = content.split('\n')

        # 检查代码块完整性
        in_code_block = False
        code_block_start = 0

        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_block_start = i
                else:
                    in_code_block = False

        if in_code_block:
            issues.append({
                'type': 'unclosed_code_block',
                'severity': 'high',
                'message': '存在未闭合的代码块',
                'line': code_block_start + 1
            })

        # 检查链接有效性
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        for link_text, link_url in links:
            if not link_url or link_url == '#':
                issues.append({
                    'type': 'broken_link',
                    'severity': 'medium',
                    'message': f'可能损坏的链接: [{link_text}]({link_url})',
                    'line': 0
                })

        # 检查中英文混用
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        english_pattern = re.compile(r'[a-zA-Z]')

        for i, line in enumerate(lines):
            chinese_chars = chinese_pattern.findall(line)
            english_chars = english_pattern.findall(line)

            if chinese_chars and english_chars and len(chinese_chars) > len(english_chars):
                # 中文内容中混杂英文
                issues.append({
                    'type': 'language_mixed',
                    'severity': 'low',
                    'message': '中英文混用可能影响可读性',
                    'line': i + 1
                })

        return issues

    def _check_standards_compliance(self, content: str, doc_type: str) -> List[Dict]:
        """检查规范符合性"""
        issues = []

        if doc_type in self.doc_standards:
            standards = self.doc_standards[doc_type]
            for pattern in standards.get('required_patterns', []):
                if not re.search(pattern, content, re.MULTILINE):
                    issues.append({
                        'type': 'missing_pattern',
                        'severity': 'medium',
                        'message': f'缺少必需的格式模式: {pattern}',
                        'line': 0
                    })

        return issues

    def _calculate_quality_score(self, issues: List[Dict]) -> float:
        """计算质量分数"""
        if not issues:
            return 100.0

        # 根据问题严重程度扣分
        score = 100.0
        severity_weights = {
            'high': 10,
            'medium': 5,
            'low': 2
        }

        for issue in issues:
            severity = issue.get('severity', 'medium')
            score -= severity_weights.get(severity, 5)

        return max(0.0, score)

    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        issue_types = {}
        for issue in issues:
            issue_type = issue['type']
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1

        # 根据问题类型生成建议
        if 'missing_section' in issue_types:
            recommendations.append("补充缺失的文档章节，确保内容完整性")

        if 'header_format' in issue_types:
            recommendations.append("统一标题格式，使用标准 Markdown 格式")

        if 'unclosed_code_block' in issue_types:
            recommendations.append("检查并修复未闭合的代码块")

        if 'broken_link' in issue_types:
            recommendations.append("检查并修复损坏的链接")

        if 'missing_pattern' in issue_types:
            recommendations.append("按照文档规范补充必需的格式元素")

        if not recommendations:
            recommendations.append("文档质量良好，继续保持")

        return recommendations

    def review_directory(self, directory: str) -> Dict[str, Any]:
        """审查整个目录"""
        results = {
            'total_files': 0,
            'reviewed_files': 0,
            'average_score': 0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
            'file_results': []
        }

        if not os.path.exists(directory):
            return {'error': f'目录不存在: {directory}'}

        total_score = 0

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    results['total_files'] += 1

                    try:
                        file_result = self.review_document(file_path)
                        results['file_results'].append(file_result)
                        results['reviewed_files'] += 1

                        if 'score' in file_result:
                            total_score += file_result['score']

                            # 质量分布统计
                            score = file_result['score']
                            if score >= 90:
                                results['quality_distribution']['excellent'] += 1
                            elif score >= 75:
                                results['quality_distribution']['good'] += 1
                            elif score >= 60:
                                results['quality_distribution']['fair'] += 1
                            else:
                                results['quality_distribution']['poor'] += 1

                    except Exception as e:
                        results['file_results'].append({
                            'file_path': file_path,
                            'error': str(e)
                        })

        if results['reviewed_files'] > 0:
            results['average_score'] = total_score / results['reviewed_files']

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成审查报告"""
        report = f"""
# RQA2025 文档质量审查报告

**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
**审查范围**: docs/ 目录
**文档总数**: {results.get('total_files', 0)}
**审查文档数**: {results.get('reviewed_files', 0)}
**平均质量分数**: {results.get('average_score', 0):.1f}

## 📊 质量分布

- **优秀 (90-100)**: {results.get('quality_distribution', {}).get('excellent', 0)} 个文档
- **良好 (75-89)**: {results.get('quality_distribution', {}).get('good', 0)} 个文档
- **一般 (60-74)**: {results.get('quality_distribution', {}).get('fair', 0)} 个文档
- **需改进 (0-59)**: {results.get('quality_distribution', {}).get('poor', 0)} 个文档

## 📋 详细结果

"""

        for file_result in results.get('file_results', []):
            if 'error' in file_result:
                report += f"### ❌ {file_result['file_path']}\n"
                report += f"**错误**: {file_result['error']}\n\n"
                continue

            score = file_result.get('score', 0)
            status = "✅" if score >= 75 else "⚠️" if score >= 60 else "❌"

            report += f"### {status} {file_result['file_path']}\n"
            report += f"**文档类型**: {file_result.get('doc_type', 'unknown')}\n"
            report += f"**质量分数**: {score:.1f}\n"

            if file_result.get('issues'):
                report += "**发现问题**:\n"
                for issue in file_result['issues']:
                    severity_icon = {"high": "🔴", "medium": "🟡",
                                     "low": "🟢"}.get(issue.get('severity'), "🟢")
                    report += f"- {severity_icon} {issue['message']}\n"
                report += "\n"

            if file_result.get('recommendations'):
                report += "**改进建议**:\n"
                for rec in file_result['recommendations']:
                    report += f"- {rec}\n"
                report += "\n"

        report += """
## 🎯 质量改进建议

### 总体改进方向
1. **完善文档结构**: 确保所有文档包含必需的章节
2. **统一格式规范**: 遵循统一的 Markdown 格式标准
3. **加强内容质量**: 补充代码示例和详细说明
4. **定期审查更新**: 建立文档定期审查机制

### 具体改进措施
1. 为所有API文档添加使用示例
2. 统一各文档的版本信息格式
3. 完善架构文档的图表说明
4. 加强文档间的交叉引用

---
*此报告由文档审查系统自动生成*
"""

        return report


def main():
    """主函数"""
    reviewer = DocumentationReviewer()

    # 审查整个文档目录
    print("开始文档质量审查...")
    results = reviewer.review_directory('docs/')

    # 生成报告
    report = reviewer.generate_report(results)
    print(report)

    # 保存报告
    with open('DOCUMENTATION_REVIEW_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n审查完成！报告已保存到: DOCUMENTATION_REVIEW_REPORT.md")


if __name__ == "__main__":
    main()
