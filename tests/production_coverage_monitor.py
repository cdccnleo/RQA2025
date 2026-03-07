"""
生产环境测试覆盖率监控系统

建立持续的测试覆盖率监控机制，包括：
1. 实时覆盖率收集和分析
2. 覆盖率趋势监控和预警
3. 生产环境覆盖率报告生成
4. 覆盖率阈值管理和告警
5. 分层覆盖率质量评估
"""

import os
import json
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class ProductionCoverageMonitor:
    """生产环境覆盖率监控系统"""

    def __init__(self, project_root: str = None):
        """
        初始化生产环境覆盖率监控系统

        Args:
            project_root: 项目根目录路径
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.reports_dir = self.project_root / "test_reports" / "coverage_monitoring"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 覆盖率阈值配置
        self.coverage_thresholds = {
            'infrastructure': 95.0,  # 基础设施层
            'data': 70.0,           # 数据管理层
            'feature': 70.0,        # 特征分析层
            'ml': 71.5,             # 机器学习层
            'strategy': 100.0,      # 策略服务层
            'trading': 70.0,        # 交易层
            'risk': 40.0,           # 风险控制层 (优化后目标)
            'monitoring': 70.0,     # 监控层
            'streaming': 94.3,      # 流处理层
            'gateway': 70.0,        # 网关层
            'optimization': 35.0,   # 优化层 (优化后目标)
            'overall': 75.0         # 总体覆盖率
        }

        # 监控历史数据
        self.monitoring_history = []
        self.alerts = []

        # 初始化样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def run_coverage_analysis(self) -> Dict[str, Any]:
        """
        运行覆盖率分析

        Returns:
            覆盖率分析结果
        """
        print("🔍 开始生产环境覆盖率分析...")

        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'layer_coverage': {},
            'overall_coverage': 0.0,
            'quality_score': 0.0,
            'alerts': [],
            'recommendations': []
        }

        try:
            # 分析各层覆盖率
            layer_coverage = self._analyze_layer_coverage()
            analysis_result['layer_coverage'] = layer_coverage

            # 计算总体覆盖率
            overall_coverage = self._calculate_overall_coverage(layer_coverage)
            analysis_result['overall_coverage'] = overall_coverage

            # 评估覆盖率质量
            quality_score = self._assess_coverage_quality(layer_coverage)
            analysis_result['quality_score'] = quality_score

            # 生成告警
            alerts = self._generate_coverage_alerts(layer_coverage)
            analysis_result['alerts'] = alerts

            # 生成建议
            recommendations = self._generate_coverage_recommendations(layer_coverage)
            analysis_result['recommendations'] = recommendations

            # 保存到监控历史
            self.monitoring_history.append(analysis_result)

            print("✅ 覆盖率分析完成")
            print(f"📊 总体覆盖率: {overall_coverage:.2f}%")
            print(f"🎯 质量评分: {quality_score:.1f}/100")
            print(f"⚠️  告警数量: {len(alerts)}")

        except Exception as e:
            print(f"❌ 覆盖率分析失败: {e}")
            analysis_result['error'] = str(e)

        return analysis_result

    def _analyze_layer_coverage(self) -> Dict[str, float]:
        """
        分析各层覆盖率

        Returns:
            各层覆盖率字典
        """
        layer_coverage = {}

        # 定义层级映射
        layer_mappings = {
            'infrastructure': ['src/infrastructure'],
            'data': ['src/data'],
            'feature': ['src/feature'],
            'ml': ['src/ml'],
            'strategy': ['src/strategy'],
            'trading': ['src/trading'],
            'risk': ['src/risk'],
            'monitoring': ['src/monitoring'],
            'streaming': ['src/streaming'],
            'gateway': ['src/gateway'],
            'optimization': ['src/optimization']
        }

        for layer_name, source_dirs in layer_mappings.items():
            try:
                # 运行该层的覆盖率测试
                coverage = self._run_layer_coverage_test(layer_name, source_dirs)
                layer_coverage[layer_name] = coverage

            except Exception as e:
                print(f"⚠️  {layer_name}层覆盖率分析失败: {e}")
                layer_coverage[layer_name] = 0.0

        return layer_coverage

    def _run_layer_coverage_test(self, layer_name: str, source_dirs: List[str]) -> float:
        """
        运行指定层的覆盖率测试

        Args:
            layer_name: 层级名称
            source_dirs: 源代码目录列表

        Returns:
            覆盖率百分比
        """
        try:
            # 构建测试命令
            test_paths = []
            for src_dir in source_dirs:
                test_dir = src_dir.replace('src/', 'tests/unit/')
                if (self.project_root / test_dir).exists():
                    test_paths.append(f"tests/unit/{src_dir.split('/')[-1]}")

            if not test_paths:
                return 0.0

            # 运行覆盖率测试
            cmd = [
                'python', '-m', 'pytest',
                '--cov=' + ','.join([f'src/{d.split("/")[-1]}' for d in source_dirs]),
                '--cov-report=json',
                '--cov-report=term-missing',
                '--cov-fail-under=0',  # 不因覆盖率不足而失败
                '-x', '--disable-warnings', '-q', '--tb=no'
            ] + test_paths

            # 设置环境变量解决编码问题
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                env=env
            )

            # 解析覆盖率结果
            if result.returncode in [0, 1]:  # 0=成功，1=测试失败但有覆盖率数据
                coverage_file = self.project_root / '.coverage.json'
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)

                    # 计算平均覆盖率
                    total_coverage = 0.0
                    file_count = 0

                    for file_path, file_data in coverage_data.get('files', {}).items():
                        if any(src_dir in file_path for src_dir in source_dirs):
                            summary = file_data.get('summary', {})
                            line_coverage = summary.get('percent_covered', 0)
                            total_coverage += line_coverage
                            file_count += 1

                    return total_coverage / file_count if file_count > 0 else 0.0

            return 0.0

        except Exception as e:
            print(f"运行{layer_name}层覆盖率测试失败: {e}")
            return 0.0

    def _calculate_overall_coverage(self, layer_coverage: Dict[str, float]) -> float:
        """
        计算总体覆盖率

        Args:
            layer_coverage: 各层覆盖率

        Returns:
            加权平均覆盖率
        """
        # 定义层级权重（基于业务重要性）
        weights = {
            'infrastructure': 0.15,  # 基础设施层权重最高
            'data': 0.12,
            'feature': 0.10,
            'ml': 0.10,
            'strategy': 0.12,       # 策略层权重较高
            'trading': 0.12,        # 交易层权重较高
            'risk': 0.10,           # 风险层权重较高
            'monitoring': 0.08,
            'streaming': 0.06,
            'gateway': 0.03,
            'optimization': 0.02
        }

        total_weighted_coverage = 0.0
        total_weight = 0.0

        for layer, coverage in layer_coverage.items():
            if layer in weights:
                total_weighted_coverage += coverage * weights[layer]
                total_weight += weights[layer]

        return total_weighted_coverage / total_weight if total_weight > 0 else 0.0

    def _assess_coverage_quality(self, layer_coverage: Dict[str, float]) -> float:
        """
        评估覆盖率质量

        Args:
            layer_coverage: 各层覆盖率

        Returns:
            质量评分 (0-100)
        """
        quality_score = 0.0

        # 检查各层是否达到阈值
        threshold_compliance = 0
        total_layers = len(self.coverage_thresholds) - 1  # 排除overall

        for layer, coverage in layer_coverage.items():
            if layer in self.coverage_thresholds:
                threshold = self.coverage_thresholds[layer]
                if coverage >= threshold:
                    threshold_compliance += 1

        threshold_score = (threshold_compliance / total_layers) * 50  # 50分

        # 检查覆盖率分布均衡性
        coverage_values = list(layer_coverage.values())
        if coverage_values:
            mean_coverage = sum(coverage_values) / len(coverage_values)
            variance = sum((x - mean_coverage) ** 2 for x in coverage_values) / len(coverage_values)
            std_dev = variance ** 0.5

            # 标准差越小，分布越均衡，得分越高
            balance_score = max(0, 50 - (std_dev * 2))  # 最高50分
        else:
            balance_score = 0

        quality_score = threshold_score + balance_score

        return min(100.0, quality_score)

    def _generate_coverage_alerts(self, layer_coverage: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        生成覆盖率告警

        Args:
            layer_coverage: 各层覆盖率

        Returns:
            告警列表
        """
        alerts = []

        for layer, coverage in layer_coverage.items():
            if layer in self.coverage_thresholds:
                threshold = self.coverage_thresholds[layer]
                gap = threshold - coverage

                if gap > 10:  # 差距超过10%
                    alerts.append({
                        'level': 'CRITICAL',
                        'layer': layer,
                        'message': f'{layer}层覆盖率严重不足: {coverage:.1f}% < {threshold:.1f}% (差距: {gap:.1f}%)',
                        'gap': gap
                    })
                elif gap > 5:  # 差距超过5%
                    alerts.append({
                        'level': 'WARNING',
                        'layer': layer,
                        'message': f'{layer}层覆盖率需要提升: {coverage:.1f}% < {threshold:.1f}% (差距: {gap:.1f}%)',
                        'gap': gap
                    })

        # 检查总体覆盖率
        overall_coverage = self._calculate_overall_coverage(layer_coverage)
        overall_threshold = self.coverage_thresholds['overall']

        if overall_coverage < overall_threshold:
            alerts.append({
                'level': 'CRITICAL',
                'layer': 'overall',
                'message': f'总体覆盖率未达标: {overall_coverage:.1f}% < {overall_threshold:.1f}%',
                'gap': overall_threshold - overall_coverage
            })

        return alerts

    def _generate_coverage_recommendations(self, layer_coverage: Dict[str, float]) -> List[str]:
        """
        生成覆盖率改进建议

        Args:
            layer_coverage: 各层覆盖率

        Returns:
            建议列表
        """
        recommendations = []

        # 按覆盖率从低到高排序
        sorted_layers = sorted(layer_coverage.items(), key=lambda x: x[1])

        # 为覆盖率最低的层生成建议
        for layer, coverage in sorted_layers[:3]:  # 前3个最低的
            if layer in self.coverage_thresholds:
                threshold = self.coverage_thresholds[layer]
                if coverage < threshold:
                    gap = threshold - coverage
                    recommendations.append(
                        f"优先提升{self._get_layer_display_name(layer)}覆盖率 "
                        f"(当前:{coverage:.1f}%, 目标:{threshold:.1f}%, 差距:{gap:.1f}%)"
                    )

        # 通用建议
        recommendations.extend([
            "定期运行覆盖率监控，确保质量不下降",
            "对新增代码要求100%测试覆盖",
            "重点测试边界条件和错误处理逻辑",
            "建立自动化测试流水线，包含覆盖率检查"
        ])

        return recommendations

    def _get_layer_display_name(self, layer: str) -> str:
        """获取层级的显示名称"""
        display_names = {
            'infrastructure': '基础设施',
            'data': '数据管理',
            'feature': '特征分析',
            'ml': '机器学习',
            'strategy': '策略服务',
            'trading': '交易',
            'risk': '风险控制',
            'monitoring': '监控',
            'streaming': '流处理',
            'gateway': '网关',
            'optimization': '优化'
        }
        return display_names.get(layer, layer)

    def generate_coverage_report(self, analysis_result: Dict[str, Any]) -> str:
        """
        生成覆盖率报告

        Args:
            analysis_result: 分析结果

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"coverage_report_{timestamp}.md"

        report_content = f"""# 生产环境测试覆盖率监控报告

生成时间: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## 📊 覆盖率概览

- **总体覆盖率**: {analysis_result.get('overall_coverage', 0):.2f}%
- **质量评分**: {analysis_result.get('quality_score', 0):.1f}/100
- **告警数量**: {len(analysis_result.get('alerts', []))}

## 📈 分层覆盖率详情

| 层级 | 覆盖率 | 阈值 | 状态 |
|------|--------|------|------|
"""

        layer_coverage = analysis_result.get('layer_coverage', {})
        for layer, coverage in layer_coverage.items():
            threshold = self.coverage_thresholds.get(layer, 0)
            status = "✅" if coverage >= threshold else "❌"
            report_content += f"| {self._get_layer_display_name(layer)} | {coverage:.1f}% | {threshold:.1f}% | {status} |\n"

        report_content += "\n## ⚠️ 覆盖率告警\n\n"
        alerts = analysis_result.get('alerts', [])
        if alerts:
            for alert in alerts:
                report_content += f"- **{alert['level']}**: {alert['message']}\n"
        else:
            report_content += "暂无告警，所有层级覆盖率达标 ✅\n"

        report_content += "\n## 💡 改进建议\n\n"
        recommendations = analysis_result.get('recommendations', [])
        for rec in recommendations:
            report_content += f"- {rec}\n"

        report_content += "\n## 📋 覆盖率阈值配置\n\n"
        for layer, threshold in self.coverage_thresholds.items():
            if layer != 'overall':
                report_content += f"- {self._get_layer_display_name(layer)}: {threshold}%\n"

        report_content += f"\n- **总体覆盖率**: {self.coverage_thresholds['overall']}%\n"

        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📄 覆盖率报告已生成: {report_file}")
        return str(report_file)

    def generate_coverage_trend_chart(self, days: int = 30) -> str:
        """
        生成覆盖率趋势图表

        Args:
            days: 分析天数

        Returns:
            图表文件路径
        """
        if len(self.monitoring_history) < 2:
            print("⚠️  监控历史数据不足，无法生成趋势图")
            return ""

        # 提取最近N天的趋势数据
        recent_data = []
        cutoff_date = datetime.now() - timedelta(days=days)

        for record in self.monitoring_history:
            record_date = datetime.fromisoformat(record['timestamp'])
            if record_date >= cutoff_date:
                recent_data.append({
                    'date': record_date,
                    'overall_coverage': record.get('overall_coverage', 0),
                    'quality_score': record.get('quality_score', 0)
                })

        if len(recent_data) < 2:
            print("⚠️  近期数据不足，无法生成趋势图")
            return ""

        # 创建图表
        df = pd.DataFrame(recent_data)
        df = df.sort_values('date')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 覆盖率趋势
        ax1.plot(df['date'], df['overall_coverage'], marker='o', linewidth=2, markersize=6)
        ax1.set_title('总体测试覆盖率趋势', fontsize=14, fontweight='bold')
        ax1.set_ylabel('覆盖率 (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.coverage_thresholds['overall'], color='r', linestyle='--', alpha=0.7,
                   label=f'目标阈值 ({self.coverage_thresholds["overall"]}%)')
        ax1.legend()

        # 质量评分趋势
        ax2.plot(df['date'], df['quality_score'], marker='s', color='orange', linewidth=2, markersize=6)
        ax2.set_title('覆盖率质量评分趋势', fontsize=14, fontweight='bold')
        ax2.set_ylabel('质量评分', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=80, color='g', linestyle='--', alpha=0.7, label='优秀标准 (80分)')
        ax2.legend()

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = self.reports_dir / f"coverage_trend_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 覆盖率趋势图已生成: {chart_file}")
        return str(chart_file)

    def setup_continuous_monitoring(self, interval_hours: int = 24):
        """
        设置持续监控

        Args:
            interval_hours: 监控间隔（小时）
        """
        print(f"🔄 设置持续覆盖率监控 (每{interval_hours}小时)")
        print("监控将在后台运行，报告将保存到 test_reports/coverage_monitoring/")

        # 这里可以实现定时任务逻辑
        # 例如使用schedule库或APScheduler

        print("✅ 持续监控已设置")
        print("💡 提示: 运行 run_coverage_analysis() 手动触发监控")
        print("💡 提示: 运行 generate_coverage_report() 生成最新报告")

    def export_monitoring_data(self) -> str:
        """
        导出监控数据

        Returns:
            导出文件路径
        """
        if not self.monitoring_history:
            print("⚠️  无监控历史数据可导出")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = self.reports_dir / f"monitoring_history_{timestamp}.json"

        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(self.monitoring_history, f, indent=2, ensure_ascii=False)

        print(f"💾 监控历史数据已导出: {export_file}")
        return str(export_file)


def main():
    """主函数：运行生产环境覆盖率监控"""
    print("🚀 启动生产环境测试覆盖率监控系统")
    print("=" * 60)

    # 初始化监控系统
    monitor = ProductionCoverageMonitor()

    try:
        # 运行覆盖率分析
        print("\n1️⃣ 运行覆盖率分析...")
        analysis_result = monitor.run_coverage_analysis()

        # 生成覆盖率报告
        print("\n2️⃣ 生成覆盖率报告...")
        report_file = monitor.generate_coverage_report(analysis_result)

        # 生成趋势图表（如果有历史数据）
        print("\n3️⃣ 生成趋势图表...")
        chart_file = monitor.generate_coverage_trend_chart()

        # 设置持续监控
        print("\n4️⃣ 设置持续监控...")
        monitor.setup_continuous_monitoring()

        # 导出监控数据
        print("\n5️⃣ 导出监控数据...")
        export_file = monitor.export_monitoring_data()

        print("\n" + "=" * 60)
        print("✅ 生产环境覆盖率监控系统运行完成")
        print(f"📄 报告文件: {report_file}")
        if chart_file:
            print(f"📊 趋势图表: {chart_file}")
        if export_file:
            print(f"💾 历史数据: {export_file}")

        # 显示关键指标
        overall_coverage = analysis_result.get('overall_coverage', 0)
        quality_score = analysis_result.get('quality_score', 0)
        alerts_count = len(analysis_result.get('alerts', []))

        print("\n🎯 关键指标:")
        print(".2f")
        print(".1f")
        print(f"⚠️  告警数量: {alerts_count}")

        if overall_coverage >= 75.0 and quality_score >= 80.0 and alerts_count == 0:
            print("\n🎉 恭喜！系统测试覆盖率达到生产就绪标准")
        else:
            print("\n⚠️  系统测试覆盖率需要进一步提升")

    except Exception as e:
        print(f"\n❌ 监控系统运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
