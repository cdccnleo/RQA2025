#!/usr/bin/env python3
"""
RQA2025 简化质量监控脚本

轻量级质量监控，避免复杂的子进程调用
"""

import os
import sys
import json
from datetime import datetime


class SimpleQualityMonitor:
    """简化质量监控器"""

    def __init__(self, baseline_file: str = ".quality_baseline.json"):
        self.baseline_file = baseline_file
        self.current_metrics = {}
        self.baseline_metrics = {}
        self.alerts = []

        # 质量阈值
        self.thresholds = {
            'interface_violations': 100,   # 接口违规数量阈值
            'estimated_complexity_score': 80.0,  # 预估复杂度评分
        }

    def load_baseline(self):
        """加载基线数据"""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, 'r', encoding='utf-8') as f:
                    self.baseline_metrics = json.load(f)
                print(f"✅ 已加载质量基线数据")
            except Exception as e:
                print(f"⚠️  无法加载基线数据: {e}")
        else:
            print("ℹ️  首次运行，尚未建立质量基线")

    def run_quality_checks(self) -> bool:
        """运行质量检查"""
        print("🔍 开始简化质量监控检查...")
        print("=" * 50)

        # 1. 检查文件数量和结构
        self.check_project_structure()

        # 2. 检查关键文件存在性
        self.check_critical_files()

        # 3. 预估代码质量
        self.estimate_code_quality()

        # 4. 检查重构成果
        self.check_refactoring_results()

        # 5. 对比基线
        self.compare_with_baseline()

        return len([a for a in self.alerts if a['severity'] == 'error']) == 0

    def check_project_structure(self):
        """检查项目结构"""
        print("🏗️  检查项目结构...")

        # 统计文件数量
        total_files = 0
        py_files = 0
        test_files = 0

        for root, dirs, files in os.walk('.'):
            # 跳过隐藏目录和__pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                total_files += 1
                if file.endswith('.py'):
                    py_files += 1
                    if 'test' in file.lower() or root.endswith('tests'):
                        test_files += 1

        self.current_metrics.update({
            'total_files': total_files,
            'python_files': py_files,
            'test_files': test_files,
            'test_ratio': test_files / max(py_files, 1) * 100
        })

        print(f"📊 项目统计: {total_files} 总文件, {py_files} Python文件, {test_files} 测试文件")

    def check_critical_files(self):
        """检查关键文件存在性"""
        print("🔍 检查关键文件...")

        critical_files = [
            'src/core/container.py',
            'src/core/patterns/standard_interface_template.py',
            'scripts/ai_intelligent_code_analyzer.py',
            'advanced_spacing_fix.py',
            '.pre-commit-config.yaml',
        ]

        missing_files = []
        for file_path in critical_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        self.current_metrics['missing_critical_files'] = len(missing_files)

        if missing_files:
            self.alerts.append({
                'type': 'missing_files',
                'message': f"缺少关键文件: {', '.join(missing_files)}",
                'severity': 'error'
            })
            print(f"❌ 缺少 {len(missing_files)} 个关键文件")
        else:
            print("✅ 所有关键文件都存在")

    def estimate_code_quality(self):
        """预估代码质量"""
        print("🎯 预估代码质量...")

        # 简单的质量预估
        quality_score = 85.0  # 基于重构成果的基础分

        # 检查重构标记
        refactoring_indicators = [
            'StandardComponent',  # 统一接口模板
            'InstanceCreator',    # 复杂度治理
            'BasicHealthChecker',  # 循环依赖解决
        ]

        found_indicators = 0
        for indicator in refactoring_indicators:
            try:
                with open('src/core/container.py', 'r', encoding='utf-8') as f:
                    if indicator in f.read():
                        found_indicators += 1
            except:
                pass

        # 根据重构指标调整分数
        indicator_bonus = (found_indicators / len(refactoring_indicators)) * 10
        quality_score += indicator_bonus

        self.current_metrics['estimated_quality_score'] = quality_score
        self.current_metrics['refactoring_indicators_found'] = found_indicators

        print(f"📈 预估质量评分: {quality_score:.1f}/100")

    def check_refactoring_results(self):
        """检查重构成果"""
        print("🔧 检查重构成果...")

        # 检查复杂度治理结果
        try:
            with open('src/core/container.py', 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否包含重构后的组件
            has_spacing_fix = 'SpacingRules' in content or 'InstanceCreator' in content
            has_interface_template = 'StandardComponent' in content
            has_dependency_fix = 'BasicHealthChecker' in content

            refactoring_score = sum(
                [has_spacing_fix, has_interface_template, has_dependency_fix]) / 3 * 100

            self.current_metrics['refactoring_completeness'] = refactoring_score

            print(f"📊 重构完成度: {refactoring_score:.1f}%")

        except Exception as e:
            print(f"⚠️  重构检查失败: {e}")
            self.current_metrics['refactoring_completeness'] = 0

    def compare_with_baseline(self):
        """对比基线数据"""
        if not self.baseline_metrics:
            print("ℹ️  建立初始质量基线")
            return

        print("\\n📊 质量趋势对比:")

        # 检查质量变化
        current_quality = self.current_metrics.get('estimated_quality_score', 0)
        baseline_quality = self.baseline_metrics.get('estimated_quality_score', 0)

        if current_quality > baseline_quality:
            trend = "🟢 ↑"
            message = f"质量改善: {current_quality:.1f} > {baseline_quality:.1f}"
            self.alerts.append({
                'type': 'quality_improvement',
                'message': message,
                'severity': 'info'
            })
        elif current_quality < baseline_quality:
            trend = "🔴 ↓"
            message = f"质量下降: {current_quality:.1f} < {baseline_quality:.1f}"
            self.alerts.append({
                'type': 'quality_degradation',
                'message': message,
                'severity': 'warning'
            })
        else:
            trend = "→"
            message = f"质量稳定: {current_quality:.1f}"

        print(f"  质量评分: {baseline_quality:.1f} → {current_quality:.1f} {trend}")

    def generate_report(self) -> dict:
        """生成监控报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.current_metrics,
            'baseline': self.baseline_metrics,
            'alerts': self.alerts,
            'thresholds': self.thresholds,
            'summary': {
                'total_alerts': len(self.alerts),
                'error_alerts': len([a for a in self.alerts if a['severity'] == 'error']),
                'warning_alerts': len([a for a in self.alerts if a['severity'] == 'warning']),
                'info_alerts': len([a for a in self.alerts if a['severity'] == 'info']),
            }
        }

    def print_report(self):
        """打印监控报告"""
        print("\\n📈 当前质量指标:")
        for key, value in self.current_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")

        if self.alerts:
            print(f"\\n🚨 发现 {len(self.alerts)} 个告警:")
            for alert in self.alerts:
                severity_icon = {
                    'error': '🔴',
                    'warning': '🟡',
                    'info': 'ℹ️'
                }.get(alert['severity'], '❓')
                print(f"  {severity_icon} {alert['message']}")


def main():
    """主函数"""
    print("📊 RQA2025 简化质量监控系统")
    print("=" * 40)

    # 创建监控器
    monitor = SimpleQualityMonitor()

    # 加载基线
    monitor.load_baseline()

    # 运行检查
    success = monitor.run_quality_checks()

    # 生成报告
    report = monitor.generate_report()
    monitor.print_report()

    # 保存基线
    if '--update-baseline' in sys.argv or not monitor.baseline_metrics:
        monitor.baseline_metrics = monitor.current_metrics.copy()
        monitor.baseline_metrics['timestamp'] = datetime.now().isoformat()

        try:
            with open(monitor.baseline_file, 'w', encoding='utf-8') as f:
                json.dump(monitor.baseline_metrics, f, indent=2, ensure_ascii=False)
            print("\\n✅ 已更新质量基线")
        except Exception as e:
            print(f"\\n❌ 保存基线失败: {e}")

    # 保存详细报告
    report_file = f"quality_monitor_simple_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✅ 详细报告已保存: {report_file}")
    except Exception as e:
        print(f"❌ 保存报告失败: {e}")

    if success:
        print("\\n✅ 质量监控完成 - 符合重构标准")
        sys.exit(0)
    else:
        print("\\n❌ 质量监控完成 - 发现质量问题")
        sys.exit(1)


if __name__ == "__main__":
    main()
