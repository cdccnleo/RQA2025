#!/usr/bin/env python3
import ast
"""
持续优化引擎

提供持续的架构优化机制：
1. 渐进式重构
2. 债务优先级管理
3. 质量度量体系
4. 自动优化建议
"""

import re
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class ContinuousOptimizationEngine:
    """持续优化引擎"""

    def __init__(self):
        self.optimization_history = []
        self.debt_register = []
        self.quality_metrics = {}
        self.optimization_queue = []
        self.engine_status = 'initialized'

    def start_continuous_optimization(self):
        """启动持续优化"""
        print("🚀 启动持续优化引擎...")
        self.engine_status = 'running'

        try:
            while self.engine_status == 'running':
                self._perform_optimization_cycle()
                time.sleep(3600)  # 每小时执行一次优化周期

        except KeyboardInterrupt:
            print("🛑 收到停止信号，正在关闭优化引擎...")
            self.stop_optimization()

    def stop_optimization(self):
        """停止优化"""
        self.engine_status = 'stopped'
        self._generate_optimization_summary()

    def _perform_optimization_cycle(self):
        """执行优化周期"""
        cycle_start = datetime.now()
        print(f"\n🔄 开始优化周期 - {cycle_start.strftime('%H:%M:%S')}")

        try:
            # 1. 收集当前状态
            current_state = self._collect_current_state()

            # 2. 分析改进机会
            opportunities = self._analyze_improvement_opportunities(current_state)

            # 3. 优先级排序
            prioritized_opportunities = self._prioritize_opportunities(opportunities)

            # 4. 执行自动优化
            executed_optimizations = self._execute_automated_optimizations(
                prioritized_opportunities)

            # 5. 更新债务注册表
            self._update_debt_register()

            # 6. 生成优化报告
            self._generate_cycle_report(cycle_start, executed_optimizations)

        except Exception as e:
            print(f"❌ 优化周期执行失败: {e}")

    def _collect_current_state(self) -> Dict:
        """收集当前状态"""
        print("📊 收集当前状态...")

        current_state = {
            'timestamp': datetime.now().isoformat(),
            'architecture_violations': self._scan_architecture_violations(),
            'code_quality_metrics': self._measure_code_quality(),
            'dependency_health': self._check_dependency_health(),
            'debt_status': self._get_current_debt_status()
        }

        return current_state

    def _scan_architecture_violations(self) -> List[Dict]:
        """扫描架构违规"""
        violations = []

        # 扫描各层级的违规情况
        layer_mapping = {
            'src/core': 'core',
            'src/infrastructure': 'infrastructure',
            'src/data': 'data',
            'src/gateway': 'gateway',
            'src/features': 'features',
            'src/ml': 'ml',
            'src/backtest': 'backtest',
            'src/risk': 'risk',
            'src/trading': 'trading',
            'src/engine': 'engine'
        }

        for layer_path, layer_name in layer_mapping.items():
            layer_dir = Path(layer_path)
            if not layer_dir.exists():
                continue

            for file_path in layer_dir.rglob('*.py'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    file_violations = self._check_file_violations(
                        str(file_path), content, layer_name)
                    violations.extend(file_violations)

                except Exception as e:
                    continue

        return violations

    def _check_file_violations(self, file_path: str, content: str, layer: str) -> List[Dict]:
        """检查文件违规"""
        violations = []

        # 检查业务概念使用
        forbidden_concepts = {
            'data': ['trading', 'strategy', 'execution', 'model', 'risk', 'order'],
            'features': ['trading', 'order', 'execution'],
            'ml': ['trading', 'order', 'execution'],
            'core': ['trading', 'strategy', 'execution', 'model', 'risk', 'order'],
            'infrastructure': ['trading', 'strategy', 'execution']
        }

        forbidden_in_layer = forbidden_concepts.get(layer, [])
        for concept in forbidden_in_layer:
            if re.search(r'\b' + re.escape(concept) + r'\b', content, re.IGNORECASE):
                violations.append({
                    'type': 'business_concept_violation',
                    'file': file_path,
                    'layer': layer,
                    'concept': concept,
                    'severity': 'high',
                    'description': f"禁止在{layer}层使用业务概念: {concept}"
                })

        # 检查长函数
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno
                    end_line = node.body[-1].lineno if node.body else start_line
                    func_lines = end_line - start_line + 1

                    if func_lines > 30:
                        violations.append({
                            'type': 'long_function',
                            'file': file_path,
                            'function': node.name,
                            'lines': func_lines,
                            'severity': 'medium',
                            'description': f"函数过长: {node.name} ({func_lines}行)"
                        })
        except:
            pass

        return violations

    def _measure_code_quality(self) -> Dict:
        """测量代码质量"""
        print("📈 测量代码质量...")

        quality_metrics = {
            'total_files': 0,
            'total_lines': 0,
            'average_complexity': 0,
            'longest_function': 0,
            'largest_class': 0
        }

        # 扫描Python文件
        src_path = Path('src')
        if src_path.exists():
            for file_path in src_path.rglob('*.py'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    quality_metrics['total_files'] += 1
                    quality_metrics['total_lines'] += len(content.split('\n'))

                    # 分析函数复杂度
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                start_line = node.lineno
                                end_line = node.body[-1].lineno if node.body else start_line
                                func_lines = end_line - start_line + 1
                                quality_metrics['longest_function'] = max(
                                    quality_metrics['longest_function'],
                                    func_lines
                                )
                    except:
                        pass

                except Exception as e:
                    continue

        if quality_metrics['total_files'] > 0:
            quality_metrics['average_lines_per_file'] = quality_metrics['total_lines'] / \
                quality_metrics['total_files']

        return quality_metrics

    def _check_dependency_health(self) -> Dict:
        """检查依赖健康状况"""
        dependency_health = {
            'total_dependencies': 0,
            'valid_dependencies': 0,
            'invalid_dependencies': 0,
            'circular_dependencies': 0
        }

        # 这里可以集成依赖分析工具的结果
        # 暂时返回模拟数据
        dependency_health['compliance_rate'] = 85.0  # 假设合规率

        return dependency_health

    def _get_current_debt_status(self) -> Dict:
        """获取当前债务状态"""
        debt_status = {
            'total_debt_items': len(self.debt_register),
            'high_priority_debt': len([d for d in self.debt_register if d.get('priority') == 'high']),
            'medium_priority_debt': len([d for d in self.debt_register if d.get('priority') == 'medium']),
            'low_priority_debt': len([d for d in self.debt_register if d.get('priority') == 'low'])
        }

        return debt_status

    def _analyze_improvement_opportunities(self, current_state: Dict) -> List[Dict]:
        """分析改进机会"""
        opportunities = []

        # 基于架构违规的改进机会
        violations = current_state.get('architecture_violations', [])
        for violation in violations:
            opportunity = {
                'type': 'fix_violation',
                'violation': violation,
                'priority': self._calculate_violation_priority(violation),
                'estimated_effort': self._estimate_fix_effort(violation),
                'description': f"修复架构违规: {violation.get('description', '')}",
                'automated': self._can_be_automated_fix(violation)
            }
            opportunities.append(opportunity)

        # 基于代码质量的改进机会
        quality_metrics = current_state.get('code_quality_metrics', {})
        if quality_metrics.get('longest_function', 0) > 50:
            opportunities.append({
                'type': 'refactor_long_functions',
                'priority': 'high',
                'estimated_effort': 'high',
                'description': f"重构长函数 (最长{quality_metrics['longest_function']}行)",
                'automated': False
            })

        # 基于依赖健康的改进机会
        dependency_health = current_state.get('dependency_health', {})
        compliance_rate = dependency_health.get('compliance_rate', 100)
        if compliance_rate < 90:
            opportunities.append({
                'type': 'improve_dependency_compliance',
                'priority': 'medium',
                'estimated_effort': 'medium',
                'description': f"提升依赖合规率 (当前{compliance_rate}%)",
                'automated': False
            })

        return opportunities

    def _calculate_violation_priority(self, violation: Dict) -> str:
        """计算违规优先级"""
        severity = violation.get('severity', 'low')

        if severity == 'high':
            return 'high'
        elif severity == 'medium':
            return 'medium'
        else:
            return 'low'

    def _estimate_fix_effort(self, violation: Dict) -> str:
        """估计修复工作量"""
        violation_type = violation.get('type', '')

        effort_mapping = {
            'business_concept_violation': 'low',
            'long_function': 'high',
            'large_class': 'high',
            'complex_condition': 'medium',
            'magic_number': 'low',
            'bare_except': 'low',
            'hard_coded_dependency': 'high'
        }

        return effort_mapping.get(violation_type, 'medium')

    def _can_be_automated_fix(self, violation: Dict) -> bool:
        """判断是否可以自动化修复"""
        violation_type = violation.get('type', '')

        automatable_types = [
            'magic_number',
            'bare_except'
        ]

        return violation_type in automatable_types

    def _prioritize_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """优先级排序"""
        # 按优先级排序
        priority_order = {'high': 3, 'medium': 2, 'low': 1}

        sorted_opportunities = sorted(
            opportunities,
            key=lambda x: (
                priority_order.get(x.get('priority', 'low'), 1),
                -x.get('automated', False)  # 优先处理可自动化的
            ),
            reverse=True
        )

        return sorted_opportunities

    def _execute_automated_optimizations(self, opportunities: List[Dict]) -> List[Dict]:
        """执行自动化优化"""
        executed = []

        for opportunity in opportunities[:5]:  # 限制每次执行的数量
            if opportunity.get('automated', False):
                try:
                    result = self._execute_single_optimization(opportunity)
                    if result:
                        executed.append({
                            'opportunity': opportunity,
                            'result': 'success',
                            'timestamp': datetime.now().isoformat()
                        })
                        print(f"✅ 自动优化成功: {opportunity['description']}")
                    else:
                        executed.append({
                            'opportunity': opportunity,
                            'result': 'failed',
                            'timestamp': datetime.now().isoformat()
                        })
                        print(f"❌ 自动优化失败: {opportunity['description']}")

                except Exception as e:
                    executed.append({
                        'opportunity': opportunity,
                        'result': 'error',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"⚠️ 自动优化出错: {e}")

        return executed

    def _execute_single_optimization(self, opportunity: Dict) -> bool:
        """执行单个优化"""
        violation = opportunity.get('violation', {})
        violation_type = violation.get('type', '')

        if violation_type == 'magic_number':
            return self._fix_magic_number(violation)
        elif violation_type == 'bare_except':
            return self._fix_bare_except(violation)

        return False

    def _fix_magic_number(self, violation: Dict) -> bool:
        """修复魔法数字"""
        file_path = violation.get('file')
        line_number = violation.get('line')
        number = violation.get('number', '')

        if not all([file_path, line_number, number]):
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')

            if line_number <= len(lines):
                # 生成常量名
                constant_name = f"CONST_{number}"

                # 在文件开头添加常量定义
                import_lines = []
                code_lines = []

                for i, line in enumerate(lines):
                    if i == 0 and not line.startswith('#'):
                        import_lines.append(f"{constant_name} = {number}")
                        import_lines.append("")
                        code_lines.append(line)
                    elif line.startswith(('import ', 'from ')):
                        import_lines.append(line)
                    else:
                        if not import_lines and not line.startswith('#'):
                            import_lines.extend(['', f"{constant_name} = {number}", ''])
                        code_lines.append(line)

                # 替换数字
                target_line = lines[line_number - 1]
                target_line = re.sub(r'\b' + number + r'\b', constant_name, target_line)
                lines[line_number - 1] = target_line

                # 重新组合文件
                new_content = '\n'.join(import_lines + code_lines)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                return True

        except Exception as e:
            print(f"修复魔法数字失败: {e}")

        return False

    def _fix_bare_except(self, violation: Dict) -> bool:
        """修复裸except"""
        file_path = violation.get('file')

        if not file_path:
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换裸except为具体的异常类型
            new_content = re.sub(r'except\s*:', 'except Exception as e:', content)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return True

        except Exception as e:
            print(f"修复裸except失败: {e}")

        return False

    def _update_debt_register(self):
        """更新债务注册表"""
        # 扫描新的债务项
        current_state = self._collect_current_state()
        violations = current_state.get('architecture_violations', [])

        for violation in violations:
            debt_item = {
                'id': f"DEBT_{len(self.debt_register) + 1}",
                'type': violation.get('type', 'unknown'),
                'description': violation.get('description', ''),
                'file': violation.get('file', ''),
                'severity': violation.get('severity', 'low'),
                'priority': self._calculate_debt_priority(violation),
                'status': 'open',
                'created_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }

            # 检查是否已存在
            existing_debt = next(
                (d for d in self.debt_register if d['file'] ==
                 debt_item['file'] and d['type'] == debt_item['type']),
                None
            )

            if not existing_debt:
                self.debt_register.append(debt_item)

    def _calculate_debt_priority(self, violation: Dict) -> str:
        """计算债务优先级"""
        severity = violation.get('severity', 'low')
        violation_type = violation.get('type', '')

        # 高严重度违规
        if severity == 'high':
            return 'high'

        # 特定类型的违规
        high_priority_types = [
            'hard_coded_dependency',
            'circular_dependency',
            'security_violation'
        ]

        if violation_type in high_priority_types:
            return 'high'

        # 中等优先级
        if severity == 'medium' or violation_type in ['long_function', 'large_class']:
            return 'medium'

        return 'low'

    def _generate_cycle_report(self, cycle_start: datetime, executed_optimizations: List[Dict]):
        """生成周期报告"""
        cycle_end = datetime.now()
        duration = cycle_end - cycle_start

        report = {
            'cycle_start': cycle_start.isoformat(),
            'cycle_end': cycle_end.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'executed_optimizations': len(executed_optimizations),
            'successful_optimizations': len([o for o in executed_optimizations if o['result'] == 'success']),
            'failed_optimizations': len([o for o in executed_optimizations if o['result'] == 'failed']),
            'engine_status': self.engine_status
        }

        # 保存到历史记录
        self.optimization_history.append(report)

        # 保持历史记录在合理范围内
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]

        print(f"📊 周期报告: 执行{len(executed_optimizations)}个优化，耗时{duration.total_seconds():.1f}秒")

    def _generate_optimization_summary(self):
        """生成优化总结报告"""
        print("📋 生成持续优化总结报告...")

        summary = {
            'total_cycles': len(self.optimization_history),
            'total_optimizations': sum(h['executed_optimizations'] for h in self.optimization_history),
            'successful_optimizations': sum(h['successful_optimizations'] for h in self.optimization_history),
            'failed_optimizations': sum(h['failed_optimizations'] for h in self.optimization_history),
            'active_debt_items': len([d for d in self.debt_register if d['status'] == 'open']),
            'resolved_debt_items': len([d for d in self.debt_register if d['status'] == 'resolved']),
            'engine_uptime': self._calculate_uptime()
        }

        # 生成详细报告
        report_content = f"""# 持续优化引擎总结报告

## 📊 运行统计
- **总优化周期**: {summary['total_cycles']}
- **总优化次数**: {summary['total_optimizations']}
- **成功优化**: {summary['successful_optimizations']}
- **失败优化**: {summary['failed_optimizations']}
- **引擎运行时间**: {summary['engine_uptime']}

## 📈 债务状态
- **活跃债务项**: {summary['active_debt_items']}
- **已解决债务**: {summary['resolved_debt_items']}
- **债务解决率**: {summary['resolved_debt_items'] / max(1, summary['active_debt_items'] + summary['resolved_debt_items']) * 100:.1f}%

## 🎯 优化效果
- **自动化程度**: {summary['successful_optimizations'] / max(1, summary['total_optimizations']) * 100:.1f}%
- **平均周期耗时**: {sum(h['duration_seconds'] for h in self.optimization_history) / max(1, len(self.optimization_history)):.1f}秒

## 📋 建议
1. **持续监控**: 保持优化引擎运行以维持架构健康
2. **债务管理**: 定期处理高优先级债务项
3. **效果评估**: 定期评估优化效果并调整策略
4. **团队反馈**: 收集团队反馈以改进优化策略

---
*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open('reports/CONTINUOUS_OPTIMIZATION_SUMMARY.md', 'w', encoding='utf-8') as f:
            f.write(report_content)

        print("✅ 持续优化总结报告已生成")

    def _calculate_uptime(self) -> str:
        """计算运行时间"""
        if not self.optimization_history:
            return "0秒"

        first_cycle = datetime.fromisoformat(self.optimization_history[0]['cycle_start'])
        last_cycle = datetime.fromisoformat(self.optimization_history[-1]['cycle_end'])
        uptime = last_cycle - first_cycle

        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}小时{minutes}分钟"
        elif minutes > 0:
            return f"{minutes}分钟{seconds}秒"
        else:
            return f"{seconds}秒"

    def get_status(self) -> Dict:
        """获取引擎状态"""
        return {
            'status': self.engine_status,
            'cycles_completed': len(self.optimization_history),
            'active_debt_items': len([d for d in self.debt_register if d['status'] == 'open']),
            'total_optimizations': sum(h['executed_optimizations'] for h in self.optimization_history),
            'uptime': self._calculate_uptime()
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='持续优化引擎')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'cycle'],
                        help='引擎操作')
    parser.add_argument('--background', action='store_true',
                        help='后台运行')

    args = parser.parse_args()

    engine = ContinuousOptimizationEngine()

    if args.action == 'start':
        if args.background:
            import daemon
            with daemon.DaemonContext():
                engine.start_continuous_optimization()
        else:
            engine.start_continuous_optimization()

    elif args.action == 'stop':
        engine.stop_optimization()

    elif args.action == 'status':
        status = engine.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))

    elif args.action == 'cycle':
        # 执行单个优化周期
        engine._perform_optimization_cycle()
        print("✅ 单次优化周期执行完成")


if __name__ == "__main__":
    main()
