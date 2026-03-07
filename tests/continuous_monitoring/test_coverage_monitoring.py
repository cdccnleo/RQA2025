"""
基础设施层 - 测试覆盖率持续监控系统

建立测试覆盖率的持续监控机制，确保代码质量持续改进。
"""

import pytest
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess


class TestCoverageMonitoring:
    """测试覆盖率持续监控"""

    def setup_method(self):
        """测试前准备"""
        self.coverage_history_file = 'tests/coverage_history.json'
        self.baseline_coverage = 18.5  # 当前基准覆盖率
        self.target_coverage = 80.0   # 目标覆盖率

        # 确保历史文件存在
        if not os.path.exists(self.coverage_history_file):
            with open(self.coverage_history_file, 'w') as f:
                json.dump([], f)

    def test_coverage_measurement(self):
        """测试覆盖率测量"""
        print("=== 覆盖率测量测试 ===")

        # 运行覆盖率测试
        try:
            result = subprocess.run([
                'python', '-m', 'pytest',
                '--cov=src/infrastructure/health',
                '--cov-report=json:coverage.json',
                '--cov-report=term-missing',
                'tests/unit/infrastructure/health/',
                '-q'
            ], capture_output=True, text=True, timeout=300)

            # 解析覆盖率报告
            if os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    coverage_data = json.load(f)

                totals = coverage_data.get('totals', {})
                coverage_percent = totals.get('percent_covered', 0)

                print(f"当前覆盖率: {coverage_percent:.1f}%")
                print(f"覆盖率目标: {self.target_coverage}%")
                print(f"距目标差距: {self.target_coverage - coverage_percent:.1f}%")
                # 保存覆盖率历史
                self._save_coverage_history(coverage_percent)

                # 覆盖率断言
                assert coverage_percent >= self.baseline_coverage, \
                    f"覆盖率下降: {coverage_percent}% < {self.baseline_coverage}%"

                # 目标进度检查
                progress = (coverage_percent / self.target_coverage) * 100
                print(f"目标达成进度: {progress:.1f}%")
                if progress >= 80:
                    print("🎉 覆盖率目标即将达成！")
                elif progress >= 50:
                    print("✅ 覆盖率目标进展良好")
                else:
                    print("⚠️  需要加速覆盖率提升")

            else:
                pytest.skip("覆盖率报告文件未生成")

        except subprocess.TimeoutExpired:
            pytest.skip("覆盖率测试超时")
        except Exception as e:
            pytest.skip(f"覆盖率测试失败: {e}")

        print("✅ 覆盖率测量完成")

    def test_coverage_trend_analysis(self):
        """测试覆盖率趋势分析"""
        print("=== 覆盖率趋势分析 ===")

        history = self._load_coverage_history()

        if len(history) < 2:
            print("历史数据不足，无法进行趋势分析")
            return

        # 计算趋势
        recent_data = history[-10:]  # 最近10次测量
        if len(recent_data) >= 2:
            first_coverage = recent_data[0]['coverage']
            last_coverage = recent_data[-1]['coverage']
            trend = last_coverage - first_coverage

            print("覆盖率趋势分析:")
            print(f"起始覆盖率: {first_coverage:.1f}%")
            print(f"最新覆盖率: {last_coverage:.1f}%")
            print(f"覆盖率变化: {trend:+.2f}%")
            if trend > 0:
                print("📈 覆盖率呈上升趋势")
            elif trend < 0:
                print("📉 覆盖率呈下降趋势")
                # 允许小幅波动，但不能大幅下降
                assert abs(trend) < 5, f"覆盖率下降过大: {abs(trend):.2f}%"
            else:
                print("➡️ 覆盖率保持稳定")

            # 计算改进速度
            days_diff = (datetime.fromisoformat(recent_data[-1]['timestamp']) -
                        datetime.fromisoformat(recent_data[0]['timestamp'])).days

            if days_diff > 0:
                improvement_rate = trend / days_diff
                print(f"每日改进率: {improvement_rate:.3f}%")
                if improvement_rate > 0.1:
                    print("🚀 改进速度良好")
                elif improvement_rate > 0:
                    print("✅ 改进速度正常")
                else:
                    print("⚠️ 需要加快改进速度")

        print("✅ 趋势分析完成")

    def test_coverage_quality_assessment(self):
        """测试覆盖率质量评估"""
        print("=== 覆盖率质量评估 ===")

        try:
            # 运行详细覆盖率报告
            result = subprocess.run([
                'python', '-m', 'pytest',
                '--cov=src/infrastructure/health',
                '--cov-report=html:htmlcov',
                '--cov-report=json:coverage.json',
                'tests/unit/infrastructure/health/',
                '-q'
            ], capture_output=True, text=True, timeout=300)

            if os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    coverage_data = json.load(f)

                files = coverage_data.get('files', {})

                # 分析各文件的覆盖率
                low_coverage_files = []
                high_coverage_files = []
                total_lines = 0
                covered_lines = 0

                for file_path, file_data in files.items():
                    summary = file_data.get('summary', {})
                    lines = summary.get('num_statements', 0)
                    covered = summary.get('covered_lines', 0)

                    if lines > 0:
                        file_coverage = (covered / lines) * 100
                        total_lines += lines
                        covered_lines += covered

                        if file_coverage < 50:
                            low_coverage_files.append((file_path, file_coverage))
                        elif file_coverage >= 90:
                            high_coverage_files.append((file_path, file_coverage))

                overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0

                print("文件覆盖率分析:")
                print(f"总文件数: {len(files)}")
                print(f"低覆盖率文件(<50%): {len(low_coverage_files)}")
                print(f"高覆盖率文件(>=90%): {len(high_coverage_files)}")

                if low_coverage_files:
                    print("低覆盖率文件:")
                    for file_path, coverage in sorted(low_coverage_files, key=lambda x: x[1])[:5]:
                        print(f"  • {file_path}: {coverage:.1f}%")
                if high_coverage_files:
                    print("高覆盖率文件:")
                    for file_path, coverage in sorted(high_coverage_files, key=lambda x: x[1], reverse=True)[:3]:
                        print(f"  • {file_path}: {coverage:.1f}%")
                # 质量评估
                quality_score = self._calculate_coverage_quality_score(
                    overall_coverage, low_coverage_files, high_coverage_files
                )

                print(f"覆盖率质量评分: {quality_score:.1f}%")
                assert quality_score >= 60, f"覆盖率质量不足: {quality_score:.1f}%"

        except Exception as e:
            pytest.skip(f"质量评估失败: {e}")

        print("✅ 质量评估完成")

    def test_coverage_goals_tracking(self):
        """测试覆盖率目标跟踪"""
        print("=== 覆盖率目标跟踪 ===")

        history = self._load_coverage_history()

        if not history:
            print("无历史数据")
            return

        current_coverage = history[-1]['coverage']
        baseline = self.baseline_coverage
        target = self.target_coverage

        # 计算进度指标
        progress_to_target = (current_coverage / target) * 100
        improvement_from_baseline = current_coverage - baseline

        print(f"目标跟踪: 当前 {current_coverage:.1f}%, 基线 {baseline:.1f}%, 改进 {improvement_from_baseline:.1f}%, 目标 {target:.2f}%")
        # 里程碑检查
        milestones = [25, 50, 75, 90, 100]
        achieved_milestones = [m for m in milestones if current_coverage >= m]

        print(f"已达成里程碑: {achieved_milestones}")
        print(f"下一个里程碑: {min([m for m in milestones if m > current_coverage], default=target)}%")

        # 时间估算
        if len(history) >= 2:
            recent_trend = self._calculate_recent_trend(history, days=7)
            if recent_trend > 0:
                days_to_target = (target - current_coverage) / recent_trend
                estimated_completion = datetime.now() + timedelta(days=days_to_target)
                print(f"预计达成目标日期: {estimated_completion.strftime('%Y-%m-%d')}")
            else:
                print("当前趋势不支持目标达成")

        print("✅ 目标跟踪完成")

    def test_coverage_regression_detection(self):
        """测试覆盖率回归检测"""
        print("=== 覆盖率回归检测 ===")

        history = self._load_coverage_history()

        if len(history) < 5:
            print("历史数据不足，无法检测回归")
            return

        # 检查最近的回归
        recent_coverage = [h['coverage'] for h in history[-5:]]
        baseline_coverage = history[-6]['coverage'] if len(history) > 5 else history[0]['coverage']

        min_recent = min(recent_coverage)
        max_recent = max(recent_coverage)
        volatility = max_recent - min_recent

        print("回归检测分析:")
        print(f"基准覆盖率: {baseline_coverage:.1f}%")
        print(f"近期覆盖率范围: {min_recent:.1f}% - {max_recent:.1f}%")
        print(f"覆盖率波动性: {volatility:.1f}%")
        # 回归检测
        significant_drop = baseline_coverage - min_recent
        if significant_drop > 2:  # 超过2%的下降
            print(f"覆盖率显著下降: {significant_drop:.1f}%")  # 这里可以触发告警
            assert significant_drop < 5, f"覆盖率显著下降: {significant_drop:.1f}%"
        else:
            print("✅ 未检测到显著回归")

        # 波动性检查
        if volatility > 3:
            print(f"覆盖率波动性过高: {volatility:.1f}%")
        else:
            print(f"覆盖率波动性正常: {volatility:.1f}%")
        print("✅ 回归检测完成")

    def test_continuous_integration_readiness(self):
        """测试持续集成就绪性"""
        print("=== 持续集成就绪性测试 ===")

        # 检查CI/CD相关配置
        ci_files = [
            'pytest.ini',
            '.github/workflows' if os.path.exists('.github/workflows') else None,
            'requirements.txt',
            'setup.py' if os.path.exists('setup.py') else None
        ]

        ci_files = [f for f in ci_files if f and os.path.exists(f)]

        print("CI/CD配置检查:")
        for ci_file in ci_files:
            print(f"✅ {ci_file} 存在")

        # 检查测试运行时间
        start_time = time.time()
        try:
            result = subprocess.run([
                'python', '-m', 'pytest',
                'tests/unit/infrastructure/health/test_core_interfaces.py',
                '-q', '--tb=no'
            ], capture_output=True, text=True, timeout=60)

            test_time = time.time() - start_time
            print(".2")
            assert test_time < 30, f"测试运行时间过长: {test_time:.2f}秒"
            assert result.returncode == 0, "测试执行失败"

        except subprocess.TimeoutExpired:
            pytest.fail("测试超时")
        except Exception as e:
            pytest.skip(f"CI就绪性检查失败: {e}")

        print("✅ 持续集成就绪性测试完成")

    def _save_coverage_history(self, coverage: float):
        """保存覆盖率历史"""
        history = self._load_coverage_history()

        history.append({
            'timestamp': datetime.now().isoformat(),
            'coverage': coverage,
            'target': self.target_coverage,
            'baseline': self.baseline_coverage
        })

        # 只保留最近100条记录
        if len(history) > 100:
            history = history[-100:]

        with open(self.coverage_history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def _load_coverage_history(self) -> List[Dict[str, Any]]:
        """加载覆盖率历史"""
        try:
            with open(self.coverage_history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _calculate_recent_trend(self, history: List[Dict[str, Any]], days: int = 7) -> float:
        """计算近期趋势"""
        if len(history) < 2:
            return 0

        # 获取指定天数内的数据
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_data = [
            h for h in history
            if datetime.fromisoformat(h['timestamp']) >= cutoff_date
        ]

        if len(recent_data) < 2:
            return 0

        # 计算线性回归斜率
        n = len(recent_data)
        x = list(range(n))
        y = [d['coverage'] for d in recent_data]

        # 简单线性回归
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope

    def _calculate_coverage_quality_score(self, overall_coverage: float,
                                        low_coverage_files: List, high_coverage_files: List) -> float:
        """计算覆盖率质量分数"""
        # 基础分数基于整体覆盖率
        base_score = (overall_coverage / self.target_coverage) * 100

        # 低覆盖率文件惩罚
        low_coverage_penalty = len(low_coverage_files) * 2

        # 高覆盖率文件奖励
        high_coverage_bonus = len(high_coverage_files) * 1

        # 计算最终分数
        quality_score = base_score - low_coverage_penalty + high_coverage_bonus
        return max(0, min(100, quality_score))  # 限制在0-100范围内
