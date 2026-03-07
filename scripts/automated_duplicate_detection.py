#!/usr/bin/env python3
"""
自动化代码重复检测系统

用于持续监控和阻止代码重复的产生
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class AutomatedDuplicateDetector:
    """自动化代码重复检测器"""

    def __init__(self, infra_dir: str = 'src/infrastructure'):
        self.infra_dir = Path(infra_dir)
        self.baseline_file = Path('duplicate_detection_baseline.json')
        self.report_file = Path('duplicate_detection_report.json')
        self.thresholds = {
            'critical': 10,  # 10+次重复为严重
            'high': 5,       # 5-9次重复为高
            'medium': 3,     # 3-4次重复为中
            'low': 2         # 2次重复为低
        }

    def perform_detection(self) -> Dict[str, Any]:
        """执行重复检测"""
        print('🔍 开始自动化代码重复检测')
        print('=' * 50)

        # 扫描代码文件
        code_files = self._scan_code_files()
        print(f'📁 扫描到 {len(code_files)} 个Python文件')

        # 提取代码块
        code_blocks = self._extract_code_blocks(code_files)
        print(f'🔍 提取到 {len(code_blocks)} 个代码块')

        # 检测重复
        duplicates = self._detect_duplicates(code_blocks)
        print(f'⚠️ 发现 {len(duplicates["functions"])} 组重复函数')
        print(f'⚠️ 发现 {len(duplicates["classes"])} 组重复类')

        # 评估严重程度
        severity_assessment = self._assess_severity(duplicates)

        # 检查基线变化
        baseline_changes = self._check_baseline_changes(duplicates)

        # 生成报告
        report = {
            'timestamp': self._get_timestamp(),
            'summary': {
                'total_files': len(code_files),
                'total_blocks': len(code_blocks),
                'duplicate_functions': len(duplicates['functions']),
                'duplicate_classes': len(duplicates['classes']),
                'total_occurrences': self._calculate_total_occurrences(duplicates)
            },
            'severity_assessment': severity_assessment,
            'baseline_changes': baseline_changes,
            'duplicates': duplicates,
            'recommendations': self._generate_recommendations(duplicates, baseline_changes)
        }

        # 保存报告
        with open(self.report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 更新基线（如果没有重大问题）
        if not baseline_changes['has_critical_increase']:
            self._update_baseline(duplicates)

        print('\\n✅ 代码重复检测完成')
        self._print_detection_summary(report)

        return report

    def _scan_code_files(self) -> List[Path]:
        """扫描Python代码文件"""
        code_files = []
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    code_files.append(Path(root) / file)
        return code_files

    def _extract_code_blocks(self, files: List[Path]) -> List[Dict[str, Any]]:
        """提取代码块"""
        code_blocks = []

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取函数
                functions = self._extract_functions(content, str(file_path))
                code_blocks.extend(functions)

                # 提取类
                classes = self._extract_classes(content, str(file_path))
                code_blocks.extend(classes)

            except Exception as e:
                print(f'⚠️ 读取文件失败 {file_path}: {e}')

        return code_blocks

    def _extract_functions(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """提取函数定义"""
        functions = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                # 提取函数签名
                func_start = i
                func_name = line.strip().split('(')[0].replace('def ', '')

                # 查找函数结束（简单的缩进检测）
                func_end = func_start
                indent_level = len(line) - len(line.lstrip())

                for j in range(func_start + 1, len(lines)):
                    if lines[j].strip() == '':
                        continue
                    if len(lines[j]) - len(lines[j].lstrip()) <= indent_level:
                        break
                    func_end = j

                # 提取函数内容
                func_content = '\n'.join(lines[func_start:func_end + 1])
                if len(func_content.strip()) > 50:  # 只处理有意义的函数
                    functions.append({
                        'type': 'function',
                        'file': file_path,
                        'signature': line.strip(),
                        'content': func_content,
                        'hash': hashlib.md5(func_content.encode()).hexdigest()[:16]
                    })

        return functions

    def _extract_classes(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """提取类定义"""
        classes = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if line.strip().startswith('class '):
                # 提取类签名
                class_start = i
                class_name = line.strip().split('(')[0].replace('class ', '')

                # 查找类结束（简单的缩进检测）
                class_end = class_start
                indent_level = len(line) - len(line.lstrip())

                for j in range(class_start + 1, len(lines)):
                    if lines[j].strip() == '':
                        continue
                    if len(lines[j]) - len(lines[j].lstrip()) <= indent_level and lines[j].strip():
                        break
                    class_end = j

                # 提取类内容
                class_content = '\n'.join(lines[class_start:class_end + 1])
                if len(class_content.strip()) > 100:  # 只处理有意义的类
                    classes.append({
                        'type': 'class',
                        'file': file_path,
                        'signature': line.strip(),
                        'content': class_content,
                        'hash': hashlib.md5(class_content.encode()).hexdigest()[:16]
                    })

        return classes

    def _detect_duplicates(self, code_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检测重复代码"""
        hash_groups = defaultdict(list)

        # 按哈希分组
        for block in code_blocks:
            hash_groups[block['hash']].append(block)

        # 过滤出重复的组（出现2次以上）
        duplicates = {
            'functions': [],
            'classes': []
        }

        for hash_val, blocks in hash_groups.items():
            if len(blocks) >= 2:
                group_info = {
                    'hash': hash_val,
                    'occurrences': len(blocks),
                    'blocks': blocks
                }

                # 分类存储
                if blocks[0]['type'] == 'function':
                    duplicates['functions'].append(group_info)
                elif blocks[0]['type'] == 'class':
                    duplicates['classes'].append(group_info)

        return duplicates

    def _assess_severity(self, duplicates: Dict[str, Any]) -> Dict[str, Any]:
        """评估重复严重程度"""
        severity = {
            'overall_severity': 'low',
            'severity_score': 0,
            'breakdown': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'most_severe_duplicate': None,
            'risk_assessment': ''
        }

        max_occurrences = 0
        most_severe = None

        # 计算严重程度
        for dup_type in ['functions', 'classes']:
            for group in duplicates[dup_type]:
                count = group['occurrences']
                max_occurrences = max(max_occurrences, count)

                if count >= self.thresholds['critical']:
                    severity['breakdown']['critical'] += 1
                    severity['severity_score'] += count * 3
                elif count >= self.thresholds['high']:
                    severity['breakdown']['high'] += 1
                    severity['severity_score'] += count * 2
                elif count >= self.thresholds['medium']:
                    severity['breakdown']['medium'] += 1
                    severity['severity_score'] += count * 1.5
                else:
                    severity['breakdown']['low'] += 1
                    severity['severity_score'] += count

                if most_severe is None or count > most_severe['occurrences']:
                    most_severe = {
                        'type': dup_type[:-1],  # 移除's'
                        'occurrences': count,
                        'signature': group['blocks'][0]['signature']
                    }

        severity['most_severe_duplicate'] = most_severe

        # 确定整体严重程度
        if severity['severity_score'] > 2000:
            severity['overall_severity'] = 'critical'
            severity['risk_assessment'] = '极高风险：严重影响代码质量和维护效率'
        elif severity['severity_score'] > 1000:
            severity['overall_severity'] = 'high'
            severity['risk_assessment'] = '高风险：显著影响开发效率'
        elif severity['severity_score'] > 500:
            severity['overall_severity'] = 'medium'
            severity['risk_assessment'] = '中等风险：需要逐步清理'
        else:
            severity['overall_severity'] = 'low'
            severity['risk_assessment'] = '低风险：基本可接受'

        return severity

    def _check_baseline_changes(self, duplicates: Dict[str, Any]) -> Dict[str, Any]:
        """检查基线变化"""
        changes = {
            'has_changes': False,
            'has_increase': False,
            'has_critical_increase': False,
            'changes_detail': []
        }

        if not self.baseline_file.exists():
            changes['has_changes'] = True
            changes['changes_detail'].append('首次建立基线')
            return changes

        try:
            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                baseline = json.load(f)

            # 比较重复数量变化
            current_total = len(duplicates['functions']) + len(duplicates['classes'])
            baseline_total = (baseline.get('summary', {}).get('duplicate_functions', 0) +
                              baseline.get('summary', {}).get('duplicate_classes', 0))

            if current_total != baseline_total:
                changes['has_changes'] = True
                changes['changes_detail'].append(
                    f'重复组数量变化: {baseline_total} -> {current_total} ({current_total - baseline_total:+d})'
                )

                if current_total > baseline_total:
                    changes['has_increase'] = True
                    if current_total > baseline_total * 1.1:  # 增加10%以上
                        changes['has_critical_increase'] = True

        except Exception as e:
            changes['has_changes'] = True
            changes['changes_detail'].append(f'基线文件读取失败: {e}')

        return changes

    def _update_baseline(self, duplicates: Dict[str, Any]):
        """更新基线"""
        baseline = {
            'timestamp': self._get_timestamp(),
            'summary': {
                'duplicate_functions': len(duplicates['functions']),
                'duplicate_classes': len(duplicates['classes'])
            },
            'duplicates': duplicates
        }

        with open(self.baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)

        print(f'📊 基线已更新: {self.baseline_file}')

    def _calculate_total_occurrences(self, duplicates: Dict[str, Any]) -> int:
        """计算总重复出现次数"""
        total = 0
        for dup_type in ['functions', 'classes']:
            for group in duplicates[dup_type]:
                total += group['occurrences']
        return total

    def _generate_recommendations(self, duplicates: Dict[str, Any],
                                  baseline_changes: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于严重程度
        severity = self._assess_severity(duplicates)

        if severity['overall_severity'] == 'critical':
            recommendations.append('🚨 紧急：立即启动大规模代码重复消除项目')
            recommendations.append('📋 建议：暂停新功能开发，优先处理重复代码')
        elif severity['overall_severity'] == 'high':
            recommendations.append('⚠️ 重要：制定代码重复清理计划')
            recommendations.append('👥 建议：成立专项清理小组')

        # 基于基线变化
        if baseline_changes['has_critical_increase']:
            recommendations.append('🚫 禁止：阻止任何增加重复代码的提交')
            recommendations.append('🔍 检查：审查最近的代码提交')

        # 通用建议
        recommendations.extend([
            '🛠️ 建立：代码模板和最佳实践指南',
            '🔄 集成：将重复检测纳入CI/CD流水线',
            '📚 培训：开展代码重用和设计模式培训'
        ])

        return recommendations

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _print_detection_summary(self, report: Dict[str, Any]):
        """打印检测摘要"""
        summary = report['summary']
        severity = report['severity_assessment']

        print('\\n📊 代码重复检测摘要:')
        print('-' * 40)
        print(f'📁 扫描文件: {summary["total_files"]}')
        print(f'🔍 代码块数: {summary["total_blocks"]}')
        print(f'⚠️ 重复函数: {summary["duplicate_functions"]} 组')
        print(f'⚠️ 重复类: {summary["duplicate_classes"]} 组')
        print(f'🔢 总重复次数: {summary["total_occurrences"]}')

        print(f'\\n⚠️ 严重程度: {severity["overall_severity"].upper()}')
        print(f'📊 严重评分: {severity["severity_score"]}')
        print('📈 严重程度分布:')
        for level, count in severity['breakdown'].items():
            if count > 0:
                print(f'   {level}: {count}')

        if severity['most_severe_duplicate']:
            severe = severity['most_severe_duplicate']
            print(f'\\n🎯 最严重重复:')
            print(f'   类型: {severe["type"]}')
            print(f'   重复次数: {severe["occurrences"]}')
            print(f'   签名: {severe["signature"][:60]}...')

        print(f'\\n💰 风险评估: {severity["risk_assessment"]}')

        if report['baseline_changes']['has_changes']:
            print('\\n📊 基线变化:')
            for change in report['baseline_changes']['changes_detail']:
                print(f'   • {change}')

        print('\\n💡 建议:')
        for rec in report['recommendations'][:3]:  # 显示前3条
            print(f'   {rec}')

        print(f'\\n📄 详细报告已保存: {self.report_file}')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动化代码重复检测')
    parser.add_argument('--check', action='store_true',
                        help='只检查，不更新基线')
    parser.add_argument('--strict', action='store_true',
                        help='严格模式：发现重复立即失败')
    parser.add_argument('--dir', default='src/infrastructure',
                        help='要检测的目录')

    args = parser.parse_args()

    detector = AutomatedDuplicateDetector(args.dir)
    report = detector.perform_detection()

    # 检查是否应该失败
    severity = report['severity_assessment']
    if args.strict and severity['overall_severity'] in ['critical', 'high']:
        print('\\n❌ 检测失败：发现严重代码重复')
        sys.exit(1)

    if args.check and report['baseline_changes']['has_critical_increase']:
        print('\\n❌ 基线检查失败：重复代码显著增加')
        sys.exit(1)


if __name__ == "__main__":
    main()
