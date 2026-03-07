#!/usr/bin/env python3
"""
重复代码标记和分类工具

识别、标记和分类最严重的重复代码组
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class DuplicateCodeMarker:
    """重复代码标记器"""

    def __init__(self):
        self.report_file = Path('duplicate_detection_report.json')
        self.marker_file = Path('duplicate_code_markers.json')
        self.priority_thresholds = {
            'critical': 10,  # 10+次重复
            'high': 5,       # 5-9次重复
            'medium': 3,     # 3-4次重复
            'low': 2         # 2次重复
        }

    def mark_critical_duplicates(self) -> Dict[str, Any]:
        """标记关键重复代码"""
        print('🏷️ 开始标记重复代码')
        print('=' * 40)

        # 加载检测报告
        if not self.report_file.exists():
            print('❌ 找不到重复检测报告文件')
            return {}

        with open(self.report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)

        # 分析和标记重复代码
        marked_duplicates = self._analyze_and_mark_duplicates(report['duplicates'])

        # 生成标记报告
        marker_report = {
            'timestamp': report['timestamp'],
            'summary': {
                'total_marked': len(marked_duplicates),
                'by_priority': self._count_by_priority(marked_duplicates),
                'by_type': self._count_by_type(marked_duplicates)
            },
            'marked_duplicates': marked_duplicates,
            'action_plan': self._generate_action_plan(marked_duplicates)
        }

        # 保存标记报告
        with open(self.marker_file, 'w', encoding='utf-8') as f:
            json.dump(marker_report, f, indent=2, ensure_ascii=False)

        print('\\n✅ 重复代码标记完成')
        self._print_marker_summary(marker_report)

        return marker_report

    def _analyze_and_mark_duplicates(self, duplicates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析并标记重复代码"""
        marked_duplicates = []

        for dup_type in ['functions', 'classes']:
            for group in duplicates[dup_type]:
                marked_group = self._mark_duplicate_group(group, dup_type)
                if marked_group:
                    marked_duplicates.append(marked_group)

        # 按优先级排序
        marked_duplicates.sort(key=lambda x: self._get_priority_score(x['priority']), reverse=True)

        return marked_duplicates

    def _mark_duplicate_group(self, group: Dict[str, Any], dup_type: str) -> Dict[str, Any]:
        """标记单个重复组"""
        occurrences = group['occurrences']

        # 确定优先级
        if occurrences >= self.priority_thresholds['critical']:
            priority = 'critical'
        elif occurrences >= self.priority_thresholds['high']:
            priority = 'high'
        elif occurrences >= self.priority_thresholds['medium']:
            priority = 'medium'
        elif occurrences >= self.priority_thresholds['low']:
            priority = 'low'
        else:
            return None  # 不标记少于2次的重复

        # 分析重复模式
        pattern_analysis = self._analyze_duplicate_pattern(group['blocks'])

        # 生成标记信息
        marked_group = {
            'id': f"{dup_type}_{group['hash'][:8]}",
            'type': dup_type[:-1],  # 移除's'
            'priority': priority,
            'occurrences': occurrences,
            'signature': group['blocks'][0]['signature'],
            'pattern_analysis': pattern_analysis,
            'affected_files': [block['file'] for block in group['blocks']],
            'affected_modules': list(set(self._extract_module_from_file(f) for f in [block['file'] for block in group['blocks']])),
            'refactoring_strategy': self._suggest_refactoring_strategy(group, dup_type),
            'estimated_effort': self._estimate_effort(priority, occurrences, pattern_analysis),
            'tags': self._generate_tags(group, dup_type)
        }

        return marked_group

    def _analyze_duplicate_pattern(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析重复模式"""
        pattern = {
            'is_constructor': False,
            'is_getter_setter': False,
            'is_utility_function': False,
            'is_interface_method': False,
            'has_complex_logic': False,
            'avg_length': 0,
            'language_patterns': []
        }

        if not blocks:
            return pattern

        # 检查函数签名模式
        signature = blocks[0]['signature']

        if '__init__' in signature:
            pattern['is_constructor'] = True
        elif signature.startswith('def get_') or signature.startswith('def set_'):
            pattern['is_getter_setter'] = True
        elif any(word in signature.lower() for word in ['util', 'helper', 'common', 'tool']):
            pattern['is_utility_function'] = True
        elif signature.strip().endswith(':') and 'self' not in signature:
            pattern['is_interface_method'] = True

        # 计算平均长度
        total_length = sum(len(block.get('content', '')) for block in blocks)
        pattern['avg_length'] = total_length / len(blocks)

        # 检查是否有复杂逻辑
        complex_indicators = ['if ', 'for ', 'while ', 'try:', 'except:', 'class ']
        for block in blocks:
            content = block.get('content', '')
            if any(indicator in content for indicator in complex_indicators):
                pattern['has_complex_logic'] = True
                break

        # 语言模式分析
        pattern['language_patterns'] = self._extract_language_patterns(blocks)

        return pattern

    def _extract_language_patterns(self, blocks: List[Dict[str, Any]]) -> List[str]:
        """提取语言模式"""
        patterns = []

        if not blocks:
            return patterns

        content = blocks[0].get('content', '')

        if 'from abc import ABC' in content or '@abstractmethod' in content:
            patterns.append('abstract_base_class')
        if 'class ' in content and '(ABC)' in content:
            patterns.append('interface_inheritance')
        if 'def __init__' in content and 'self.' in content:
            patterns.append('standard_constructor')
        if 'logging.getLogger' in content:
            patterns.append('standard_logging')
        if 'try:' in content and 'except' in content:
            patterns.append('standard_error_handling')

        return patterns

    def _suggest_refactoring_strategy(self, group: Dict[str, Any], dup_type: str) -> str:
        """建议重构策略"""
        blocks = group['blocks']
        if not blocks:
            return 'unknown'

        pattern = self._analyze_duplicate_pattern(blocks)

        if pattern['is_constructor']:
            return 'extract_base_class_constructor'
        elif pattern['is_interface_method']:
            return 'move_to_interface_base'
        elif pattern['is_utility_function']:
            return 'create_utility_module'
        elif pattern['is_getter_setter']:
            return 'use_property_decorators'
        elif dup_type == 'classes':
            return 'extract_common_base_class'
        else:
            return 'extract_common_function'

    def _estimate_effort(self, priority: str, occurrences: int,
                         pattern: Dict[str, Any]) -> str:
        """估算工作量"""
        base_effort = {
            'critical': 4,  # 人天
            'high': 2,
            'medium': 1,
            'low': 0.5
        }

        effort = base_effort.get(priority, 1)

        # 根据复杂度调整
        if pattern['has_complex_logic']:
            effort *= 1.5

        # 根据文件数量调整
        if occurrences > 5:
            effort *= 1.2

        # 格式化输出
        if effort >= 4:
            return f"{effort:.0f}人天"
        elif effort >= 1:
            return f"{effort:.1f}人天"
        else:
            return f"{effort:.1f}人时"

    def _generate_tags(self, group: Dict[str, Any], dup_type: str) -> List[str]:
        """生成标签"""
        tags = []

        blocks = group['blocks']
        if not blocks:
            return tags

        pattern = self._analyze_duplicate_pattern(blocks)

        if pattern['is_constructor']:
            tags.append('constructor')
        if pattern['is_getter_setter']:
            tags.append('getter_setter')
        if pattern['is_utility_function']:
            tags.append('utility')
        if pattern['is_interface_method']:
            tags.append('interface')
        if pattern['has_complex_logic']:
            tags.append('complex')
        if dup_type == 'classes':
            tags.append('class_duplication')

        # 添加模块标签
        modules = set()
        for block in blocks:
            module = self._extract_module_from_file(block['file'])
            if module:
                modules.add(module)

        tags.extend([f"module:{m}" for m in modules])

        return tags

    def _extract_module_from_file(self, file_path: str) -> str:
        """从文件路径提取模块名"""
        parts = file_path.replace('\\', '/').split('/')
        if 'infrastructure' in parts:
            infra_index = parts.index('infrastructure')
            if infra_index + 1 < len(parts):
                return parts[infra_index + 1]
        return 'unknown'

    def _count_by_priority(self, marked_duplicates: List[Dict[str, Any]]) -> Dict[str, int]:
        """按优先级统计"""
        counts = defaultdict(int)
        for dup in marked_duplicates:
            counts[dup['priority']] += 1
        return dict(counts)

    def _count_by_type(self, marked_duplicates: List[Dict[str, Any]]) -> Dict[str, int]:
        """按类型统计"""
        counts = defaultdict(int)
        for dup in marked_duplicates:
            counts[dup['type']] += 1
        return dict(counts)

    def _get_priority_score(self, priority: str) -> int:
        """获取优先级分数"""
        scores = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return scores.get(priority, 0)

    def _generate_action_plan(self, marked_duplicates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成行动计划"""
        plan = {
            'immediate_actions': [],
            'phased_approach': {
                'week_1_2': [],
                'month_1_2': [],
                'month_2_3': []
            },
            'total_effort': '0人天',
            'success_criteria': []
        }

        # 分类标记的重复
        critical = [d for d in marked_duplicates if d['priority'] == 'critical']
        high = [d for d in marked_duplicates if d['priority'] == 'high']

        # 立即行动：前3个最高优先级的重复
        immediate_items = marked_duplicates[:3]
        plan['immediate_actions'] = [
            f"修复 {item['id']}: {item['signature'][:50]}... ({item['occurrences']}处重复)"
            for item in immediate_items
        ]

        # 分阶段计划
        plan['phased_approach']['week_1_2'] = [
            f"修复 {item['id']}: {item['signature'][:50]}... ({item['occurrences']}处重复)"
            for item in marked_duplicates[:5]  # 前5个
        ]

        plan['phased_approach']['month_1_2'] = [
            f"修复 {item['id']}: {item['signature'][:50]}... ({item['occurrences']}处重复)"
            for item in marked_duplicates[5:15]  # 5-15个
        ]

        plan['phased_approach']['month_2_3'] = [
            f"修复 {item['id']}: {item['signature'][:50]}... ({item['occurrences']}处重复)"
            for item in marked_duplicates[15:]  # 剩余的
        ]

        # 计算总工作量
        total_effort_days = sum(
            float(item['estimated_effort'].replace('人天', '').replace('人时', '')) / 8
            if '人时' in item['estimated_effort']
            else float(item['estimated_effort'].replace('人天', ''))
            for item in marked_duplicates
        )
        plan['total_effort'] = f"{total_effort_days:.1f}人天"

        # 成功标准
        plan['success_criteria'] = [
            "代码重复率降低50%以上",
            "所有critical和high优先级重复代码被清理",
            "建立自动化重复检测和预防机制",
            "代码可维护性指数提升到50+",
            "团队开发效率显著提升"
        ]

        return plan

    def _print_marker_summary(self, report: Dict[str, Any]):
        """打印标记摘要"""
        summary = report['summary']

        print('\\n🏷️ 重复代码标记摘要:')
        print('-' * 40)
        print(f'📊 标记重复组: {summary["total_marked"]}')
        print('📈 按优先级分布:')
        for priority, count in summary['by_priority'].items():
            print(f'   {priority}: {count}')

        print('🔧 按类型分布:')
        for dup_type, count in summary['by_type'].items():
            print(f'   {dup_type}: {count}')

        print(f'\\n🎯 行动计划 ({report["action_plan"]["total_effort"]}):')

        print('\\n🚀 立即行动:')
        for action in report['action_plan']['immediate_actions']:
            print(f'   • {action}')

        print('\\n📅 第1-2周:')
        for action in report['action_plan']['phased_approach']['week_1_2'][:3]:
            print(f'   • {action}')

        print('\\n✅ 成功标准:')
        for criterion in report['action_plan']['success_criteria'][:3]:
            print(f'   ✓ {criterion}')

        print(f'\\n📄 详细标记报告已保存: {self.marker_file}')


def main():
    """主函数"""
    marker = DuplicateCodeMarker()
    report = marker.mark_critical_duplicates()

    if not report:
        print('❌ 重复代码标记失败')
        exit(1)


if __name__ == "__main__":
    main()
