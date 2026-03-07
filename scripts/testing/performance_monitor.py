#!/usr/bin/env python3
"""
性能监控器
持续监控测试执行时间变化
确保测试不影响模型执行效率
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging
from dataclasses import dataclass
import psutil
import threading
import queue

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class PerformanceSnapshot:
    """性能快照数据类"""
    timestamp: datetime
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    latency: float
    baseline_time: float
    performance_ratio: float
    trend: str  # 'improving', 'degrading', 'stable'


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, output_dir: str = "reports/performance_monitor"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

        # 监控配置
        self.monitoring_config = {
            'sampling_interval': 1.0,  # 秒
            'trend_window': 10,  # 数据点
            'performance_threshold': 0.1,  # 10%变化阈值
            'memory_threshold': 512,  # MB
            'cpu_threshold': 80.0,  # %
            'alert_interval': 60,  # 秒
        }

        # 性能历史数据
        self.performance_history = {}
        self.baseline_data = {}
        self.monitoring_active = False
        self.alert_queue = queue.Queue()

        # 加载基准数据
        self.load_baseline_data()

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_baseline_data(self):
        """加载基准数据"""
        baseline_file = Path("reports/performance_benchmark/simple_performance_baseline.json")
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                    self.baseline_data = baseline_data.get('baseline_metrics', {})
                self.logger.info(f"已加载基准数据: {len(self.baseline_data)}个测试")
            except Exception as e:
                self.logger.warning(f"加载基准数据失败: {e}")
        else:
            self.logger.warning("未找到基准数据文件")

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': os.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / (1024**2)
        }

    def measure_test_performance(self, test_name: str, test_func, *args, **kwargs) -> PerformanceSnapshot:
        """测量测试性能"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB

        # 执行测试
        result = test_func(*args, **kwargs)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB

        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = psutil.cpu_percent(interval=1)

        # 计算性能指标
        throughput = 1000 / execution_time if execution_time > 0 else 0  # 假设1000个数据点
        latency = execution_time / 1000 if execution_time > 0 else 0

        # 获取基准时间
        baseline_time = self.baseline_data.get(test_name, {}).get('execution_time', execution_time)
        performance_ratio = execution_time / baseline_time if baseline_time > 0 else 1.0

        # 分析趋势
        trend = self._analyze_performance_trend(test_name, execution_time)

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            test_name=test_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency=latency,
            baseline_time=baseline_time,
            performance_ratio=performance_ratio,
            trend=trend
        )

        # 记录历史数据
        if test_name not in self.performance_history:
            self.performance_history[test_name] = []

        self.performance_history[test_name].append(snapshot)

        # 保持历史数据在合理范围内
        if len(self.performance_history[test_name]) > 100:
            self.performance_history[test_name] = self.performance_history[test_name][-50:]

        return snapshot

    def _analyze_performance_trend(self, test_name: str, current_time: float) -> str:
        """分析性能趋势"""
        if test_name not in self.performance_history or len(self.performance_history[test_name]) < 3:
            return 'stable'

        recent_times = [
            snapshot.execution_time for snapshot in self.performance_history[test_name][-5:]]

        if len(recent_times) < 3:
            return 'stable'

        # 计算趋势
        trend_slope = np.polyfit(range(len(recent_times)), recent_times, 1)[0]

        if trend_slope < -0.01:  # 执行时间减少
            return 'improving'
        elif trend_slope > 0.01:  # 执行时间增加
            return 'degrading'
        else:
            return 'stable'

    def check_performance_alerts(self, snapshot: PerformanceSnapshot) -> List[str]:
        """检查性能告警"""
        alerts = []

        # 性能退化告警
        if snapshot.performance_ratio > 1.0 + self.monitoring_config['performance_threshold']:
            alerts.append(
                f"性能退化: {snapshot.test_name} 执行时间增加 {((snapshot.performance_ratio - 1) * 100):.1f}%")

        # 内存使用告警
        if snapshot.memory_usage > self.monitoring_config['memory_threshold']:
            alerts.append(f"内存使用过高: {snapshot.test_name} 使用 {snapshot.memory_usage:.1f}MB")

        # CPU使用告警
        if snapshot.cpu_usage > self.monitoring_config['cpu_threshold']:
            alerts.append(f"CPU使用过高: {snapshot.test_name} 使用 {snapshot.cpu_usage:.1f}%")

        # 趋势告警
        if snapshot.trend == 'degrading':
            alerts.append(f"性能趋势恶化: {snapshot.test_name} 显示性能下降趋势")

        return alerts

    def run_continuous_monitoring(self, test_functions: Dict[str, callable], duration: int = 3600):
        """运行持续监控"""
        self.logger.info(f"开始持续性能监控，持续时间: {duration}秒")

        self.monitoring_active = True
        start_time = time.time()

        # 启动告警处理线程
        alert_thread = threading.Thread(target=self._process_alerts)
        alert_thread.daemon = True
        alert_thread.start()

        try:
            while time.time() - start_time < duration and self.monitoring_active:
                system_info = self.get_system_info()

                # 检查系统资源
                if system_info['memory_used_mb'] > self.monitoring_config['memory_threshold']:
                    self.logger.warning(f"系统内存使用过高: {system_info['memory_used_mb']:.1f}MB")

                if system_info['cpu_percent'] > self.monitoring_config['cpu_threshold']:
                    self.logger.warning(f"系统CPU使用过高: {system_info['cpu_percent']:.1f}%")

                # 运行测试并监控性能
                for test_name, test_func in test_functions.items():
                    try:
                        snapshot = self.measure_test_performance(test_name, test_func)

                        # 检查告警
                        alerts = self.check_performance_alerts(snapshot)
                        for alert in alerts:
                            self.alert_queue.put({
                                'timestamp': datetime.now(),
                                'level': 'warning',
                                'message': alert,
                                'test_name': test_name
                            })

                        # 记录性能数据
                        self.logger.info(f"测试: {test_name}, 执行时间: {snapshot.execution_time:.3f}s, "
                                         f"性能比: {snapshot.performance_ratio:.2f}, 趋势: {snapshot.trend}")

                    except Exception as e:
                        self.logger.error(f"测试 {test_name} 执行失败: {e}")

                # 等待下次采样
                time.sleep(self.monitoring_config['sampling_interval'])

        except KeyboardInterrupt:
            self.logger.info("收到中断信号，停止监控")
        finally:
            self.monitoring_active = False
            self._save_monitoring_data()

    def _process_alerts(self):
        """处理告警队列"""
        while self.monitoring_active:
            try:
                alert = self.alert_queue.get(timeout=1)
                self.logger.warning(f"[告警] {alert['message']}")

                # 保存告警到文件
                alert_file = self.output_dir / "performance_alerts.json"
                alerts = []
                if alert_file.exists():
                    with open(alert_file, 'r', encoding='utf-8') as f:
                        alerts = json.load(f)

                alerts.append(alert)

                with open(alert_file, 'w', encoding='utf-8') as f:
                    json.dump(alerts, f, indent=2)

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"处理告警时出错: {e}")

    def _save_monitoring_data(self):
        """保存监控数据"""
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_duration': time.time(),
            'performance_history': {}
        }

        for test_name, snapshots in self.performance_history.items():
            monitoring_data['performance_history'][test_name] = [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'execution_time': snapshot.execution_time,
                    'memory_usage': snapshot.memory_usage,
                    'cpu_usage': snapshot.cpu_usage,
                    'performance_ratio': snapshot.performance_ratio,
                    'trend': snapshot.trend
                }
                for snapshot in snapshots
            ]

        monitoring_file = self.output_dir / "performance_monitoring_data.json"
        with open(monitoring_file, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2)

        self.logger.info(f"监控数据已保存: {monitoring_file}")

    def generate_monitoring_report(self) -> str:
        """生成监控报告"""
        report_content = []
        report_content.append("# 性能监控报告")
        report_content.append("")
        report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"**监控测试数**: {len(self.performance_history)}")
        report_content.append("")

        # 总体统计
        report_content.append("## 📊 总体统计")
        report_content.append("")

        all_snapshots = []
        for snapshots in self.performance_history.values():
            all_snapshots.extend(snapshots)

        if all_snapshots:
            execution_times = [s.execution_time for s in all_snapshots]
            performance_ratios = [s.performance_ratio for s in all_snapshots]
            memory_usages = [s.memory_usage for s in all_snapshots]
            cpu_usages = [s.cpu_usage for s in all_snapshots]

            report_content.append("| 指标 | 平均值 | 最小值 | 最大值 |")
            report_content.append("|------|--------|--------|--------|")
            report_content.append(
                f"| 执行时间(秒) | {np.mean(execution_times):.3f} | {np.min(execution_times):.3f} | {np.max(execution_times):.3f} |")
            report_content.append(
                f"| 性能比 | {np.mean(performance_ratios):.2f} | {np.min(performance_ratios):.2f} | {np.max(performance_ratios):.2f} |")
            report_content.append(
                f"| 内存使用(MB) | {np.mean(memory_usages):.1f} | {np.min(memory_usages):.1f} | {np.max(memory_usages):.1f} |")
            report_content.append(
                f"| CPU使用率(%) | {np.mean(cpu_usages):.1f} | {np.min(cpu_usages):.1f} | {np.max(cpu_usages):.1f} |")
            report_content.append("")

        # 各测试性能分析
        report_content.append("## 📈 各测试性能分析")
        report_content.append("")

        for test_name, snapshots in self.performance_history.items():
            if not snapshots:
                continue

            report_content.append(f"### {test_name}")

            # 计算统计信息
            execution_times = [s.execution_time for s in snapshots]
            performance_ratios = [s.performance_ratio for s in snapshots]
            trends = [s.trend for s in snapshots]

            avg_time = np.mean(execution_times)
            avg_ratio = np.mean(performance_ratios)
            improving_count = trends.count('improving')
            degrading_count = trends.count('degrading')
            stable_count = trends.count('stable')

            report_content.append(f"- **平均执行时间**: {avg_time:.3f}秒")
            report_content.append(f"- **平均性能比**: {avg_ratio:.2f}")
            report_content.append(
                f"- **性能趋势**: 改善 {improving_count}次, 恶化 {degrading_count}次, 稳定 {stable_count}次")

            # 最近趋势
            if len(snapshots) >= 2:
                recent_trend = snapshots[-1].trend
                recent_ratio = snapshots[-1].performance_ratio
                report_content.append(f"- **最近趋势**: {recent_trend}")
                report_content.append(f"- **最近性能比**: {recent_ratio:.2f}")

            report_content.append("")

        # 告警统计
        alert_file = self.output_dir / "performance_alerts.json"
        if alert_file.exists():
            try:
                with open(alert_file, 'r', encoding='utf-8') as f:
                    alerts = json.load(f)

                report_content.append("## ⚠️ 告警统计")
                report_content.append("")
                report_content.append(f"**总告警数**: {len(alerts)}")
                report_content.append("")

                # 按测试分类告警
                alert_by_test = {}
                for alert in alerts:
                    test_name = alert.get('test_name', 'unknown')
                    if test_name not in alert_by_test:
                        alert_by_test[test_name] = []
                    alert_by_test[test_name].append(alert)

                for test_name, test_alerts in alert_by_test.items():
                    report_content.append(f"### {test_name}")
                    report_content.append(f"- **告警数**: {len(test_alerts)}")

                    # 按告警类型统计
                    alert_types = {}
                    for alert in test_alerts:
                        message = alert.get('message', '')
                        if '性能退化' in message:
                            alert_types['性能退化'] = alert_types.get('性能退化', 0) + 1
                        elif '内存使用' in message:
                            alert_types['内存使用'] = alert_types.get('内存使用', 0) + 1
                        elif 'CPU使用' in message:
                            alert_types['CPU使用'] = alert_types.get('CPU使用', 0) + 1
                        elif '性能趋势' in message:
                            alert_types['性能趋势'] = alert_types.get('性能趋势', 0) + 1

                    for alert_type, count in alert_types.items():
                        report_content.append(f"- **{alert_type}**: {count}次")

                    report_content.append("")
            except Exception as e:
                report_content.append(f"读取告警数据失败: {e}")
                report_content.append("")

        # 建议
        report_content.append("## 💡 性能优化建议")
        report_content.append("")

        # 分析性能问题
        performance_issues = []
        for test_name, snapshots in self.performance_history.items():
            if not snapshots:
                continue

            recent_snapshots = snapshots[-5:]  # 最近5次
            avg_ratio = np.mean([s.performance_ratio for s in recent_snapshots])
            avg_memory = np.mean([s.memory_usage for s in recent_snapshots])
            avg_cpu = np.mean([s.cpu_usage for s in recent_snapshots])

            if avg_ratio > 1.2:
                performance_issues.append(f"{test_name}: 性能退化 {((avg_ratio - 1) * 100):.1f}%")

            if avg_memory > 100:
                performance_issues.append(f"{test_name}: 内存使用过高 {avg_memory:.1f}MB")

            if avg_cpu > 70:
                performance_issues.append(f"{test_name}: CPU使用过高 {avg_cpu:.1f}%")

        if performance_issues:
            report_content.append("### 发现的问题:")
            for issue in performance_issues:
                report_content.append(f"- {issue}")
            report_content.append("")

        report_content.append("### 优化建议:")
        report_content.append("- 🔧 **定期基准测试**: 建议每周运行一次基准测试")
        report_content.append("- 📊 **性能监控**: 持续监控关键测试的性能")
        report_content.append("- ⚠️ **告警设置**: 设置合理的性能告警阈值")
        report_content.append("- 📈 **趋势分析**: 定期分析性能变化趋势")
        report_content.append("- 🚀 **优化算法**: 对性能退化的测试进行算法优化")

        return "\n".join(report_content)


def create_test_functions():
    """创建测试函数"""
    def data_processing_test():
        """数据处理测试"""
        # 生成测试数据
        size = 1000
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, size))

        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, size)
        }, index=dates)

        # 数据处理操作
        df['returns'] = df['close'].pct_change()
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['volatility'] = df['returns'].rolling(20).std()

        return df

    def model_training_test():
        """模型训练测试"""
        # 生成测试数据
        size = 1000
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, size))

        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, size)
        }, index=dates)

        # 特征工程
        df['returns'] = df['close'].pct_change()
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()

        # 模拟模型训练
        features = df[['ma_5', 'ma_20']].dropna()
        target = (df['close'].shift(-1) > df['close']).astype(int)
        target = target[features.index]

        # 简单预测
        predictions = features['ma_5'] > features['ma_20']
        accuracy = (predictions == target).mean() if len(target) > 0 else 0

        return {'accuracy': accuracy, 'predictions': len(predictions)}

    def backtest_test():
        """回测测试"""
        # 生成测试数据
        size = 1000
        dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, size))

        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, size)
        }, index=dates)

        # 回测逻辑
        initial_capital = 100000.0
        cash = initial_capital
        portfolio_values = []

        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()

        for i in range(len(df)):
            if i < 20:
                continue

            current_price = df['close'].iloc[i]
            signal = 1 if df['ma_5'].iloc[i] > df['ma_20'].iloc[i] else -1

            # 简单交易逻辑
            if signal == 1 and cash > 0:
                shares = int(cash * 0.1 / current_price)
                cash -= shares * current_price

            portfolio_value = cash
            portfolio_values.append(portfolio_value)

        return {
            'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
            'total_return': (portfolio_values[-1] - initial_capital) / initial_capital if portfolio_values else 0
        }

    return {
        'data_processing_test': data_processing_test,
        'model_training_test': model_training_test,
        'backtest_test': backtest_test
    }


def main():
    """主函数"""
    print("🔍 开始性能监控")
    print("="*60)

    # 创建性能监控器
    monitor = PerformanceMonitor()

    # 创建测试函数
    test_functions = create_test_functions()

    # 运行持续监控（1小时）
    monitor.run_continuous_monitoring(test_functions, duration=3600)

    # 生成监控报告
    report_content = monitor.generate_monitoring_report()
    report_file = monitor.output_dir / "performance_monitoring_report.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print("="*60)
    print("性能监控完成")
    print("="*60)
    print(f"监控报告: {report_file}")
    print("="*60)


if __name__ == "__main__":
    main()
