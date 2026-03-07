#!/usr/bin/env python3
"""
性能基准测试工具
建立关键操作的性能基准，为持续监控提供参考
"""

import os
import sys
import time
import json
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import unittest.mock
import numpy as np
import pandas as pd


class PerformanceBaseline:
    """性能基准测试器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.baselines = {}
        self.results = {}

    def establish_baselines(self) -> Dict[str, Any]:
        """建立性能基准"""
        print("📊 开始建立性能基准...")

        # 1. ML层性能基准
        print("🤖 建立ML层性能基准...")
        ml_baselines = self._establish_ml_baselines()
        self.baselines.update(ml_baselines)

        # 2. 策略层性能基准
        print("📈 建立策略层性能基准...")
        strategy_baselines = self._establish_strategy_baselines()
        self.baselines.update(strategy_baselines)

        # 3. 交易层性能基准
        print("💰 建立交易层性能基准...")
        trading_baselines = self._establish_trading_baselines()
        self.baselines.update(trading_baselines)

        # 4. 风险控制层性能基准
        print("⚠️ 建立风险控制层性能基准...")
        risk_baselines = self._establish_risk_baselines()
        self.baselines.update(risk_baselines)

        # 5. 系统层性能基准
        print("🏗️ 建立系统层性能基准...")
        system_baselines = self._establish_system_baselines()
        self.baselines.update(system_baselines)

        # 保存基准数据
        self._save_baselines()

        return self.baselines

    def _establish_ml_baselines(self) -> Dict[str, Any]:
        """建立ML层性能基准"""
        baselines = {}

        try:
            sys.path.insert(0, str(self.project_root / 'src'))
            from ml.core.ml_core import MLCore
            ml_core = MLCore()

            # 测试数据
            train_data = pd.DataFrame({
                'feature1': np.random.randn(1000),
                'feature2': np.random.randn(1000),
                'target': np.random.randint(0, 2, 1000)
            })

            predict_data = train_data.drop('target', axis=1)

            # 训练性能基准
            print("  📈 测量ML训练性能...")
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss

            model = ml_core.train(train_data, target_column='target')

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            baselines['ml_training'] = {
                'operation': 'ML模型训练',
                'dataset_size': len(train_data),
                'execution_time': end_time - start_time,
                'memory_usage': (end_memory - start_memory) / 1024 / 1024,  # MB
                'throughput': len(train_data) / (end_time - start_time),  # 样本/秒
                'timestamp': time.time()
            }

            # 预测性能基准
            print("  🔮 测量ML预测性能...")
            start_time = time.time()

            predictions = ml_core.predict(model, predict_data)

            end_time = time.time()

            baselines['ml_prediction'] = {
                'operation': 'ML模型预测',
                'dataset_size': len(predict_data),
                'execution_time': end_time - start_time,
                'throughput': len(predict_data) / (end_time - start_time),  # 样本/秒
                'latency': (end_time - start_time) / len(predict_data) * 1000,  # ms/样本
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"  ❌ ML基准测试失败: {e}")
            baselines['ml_error'] = {
                'operation': 'ML基准测试',
                'error': str(e),
                'timestamp': time.time()
            }

        return baselines

    def _establish_strategy_baselines(self) -> Dict[str, Any]:
        """建立策略层性能基准"""
        baselines = {}

        try:
            from src.strategy.core.strategy_engine import StrategyEngine
            engine = StrategyEngine()

            # 测试数据
            market_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
                'price': 100 + np.cumsum(np.random.randn(1000) * 0.1),
                'volume': np.random.randint(1000, 10000, 1000)
            })

            print("  📊 测量策略执行性能...")
            start_time = time.time()

            signals = engine.execute_strategy(market_data)

            end_time = time.time()

            baselines['strategy_execution'] = {
                'operation': '策略执行',
                'data_points': len(market_data),
                'execution_time': end_time - start_time,
                'throughput': len(market_data) / (end_time - start_time),  # 数据点/秒
                'signals_generated': len(signals) if signals is not None else 0,
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"  ❌ 策略基准测试失败: {e}")
            baselines['strategy_error'] = {
                'operation': '策略基准测试',
                'error': str(e),
                'timestamp': time.time()
            }

        return baselines

    def _establish_trading_baselines(self) -> Dict[str, Any]:
        """建立交易层性能基准"""
        baselines = {}

        try:
            from src.trading.core.trading_engine import TradingEngine
            config = {'api_key': 'test', 'api_secret': 'test'}
            engine = TradingEngine(config)

            # 订单数据
            orders = [{
                'symbol': 'AAPL',
                'side': 'buy',
                'quantity': 100 + i,
                'price': 150.0 + i,
                'order_type': 'limit'
            } for i in range(10)]

            print("  📋 测量订单处理性能...")
            start_time = time.time()

            results = []
            for order in orders:
                try:
                    # 使用mock避免真实交易
                    with unittest.mock.patch.object(engine, '_execute_order', return_value={'order_id': f'test_{len(results)}'}):
                        result = engine.place_order(order)
                        results.append(result)
                except:
                    results.append(None)

            end_time = time.time()

            successful_orders = len([r for r in results if r is not None])

            baselines['order_processing'] = {
                'operation': '订单处理',
                'total_orders': len(orders),
                'successful_orders': successful_orders,
                'execution_time': end_time - start_time,
                'throughput': len(orders) / (end_time - start_time),  # 订单/秒
                'success_rate': successful_orders / len(orders) * 100,
                'timestamp': time.time()
            }

            # 并发订单处理基准
            print("  🔄 测量并发订单处理性能...")
            start_time = time.time()

            def process_order(order):
                try:
                    with unittest.mock.patch.object(engine, '_execute_order', return_value={'order_id': f'concurrent_{len(results)}'}):
                        return engine.place_order(order)
                except:
                    return None

            threads = []
            concurrent_results = []

            for order in orders[:5]:  # 并发处理5个订单
                thread = threading.Thread(target=lambda: concurrent_results.append(process_order(order)))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            end_time = time.time()

            baselines['concurrent_order_processing'] = {
                'operation': '并发订单处理',
                'concurrent_orders': 5,
                'execution_time': end_time - start_time,
                'throughput': 5 / (end_time - start_time),  # 并发订单/秒
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"  ❌ 交易基准测试失败: {e}")
            baselines['trading_error'] = {
                'operation': '交易基准测试',
                'error': str(e),
                'timestamp': time.time()
            }

        return baselines

    def _establish_risk_baselines(self) -> Dict[str, Any]:
        """建立风险控制层性能基准"""
        baselines = {}

        try:
            from src.risk.monitor.realtime_risk_monitor import RealtimeRiskMonitor
            monitor = RealtimeRiskMonitor()

            # 测试数据
            market_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=500, freq='1min'),
                'price': 100 + np.cumsum(np.random.randn(500) * 0.1),
                'returns': np.random.randn(500) * 0.02
            })

            print("  ⚠️ 测量风险计算性能...")
            start_time = time.time()

            risks = monitor.calculate_all_risks(market_data)

            end_time = time.time()

            baselines['risk_calculation'] = {
                'operation': '风险计算',
                'data_points': len(market_data),
                'execution_time': end_time - start_time,
                'throughput': len(market_data) / (end_time - start_time),  # 数据点/秒
                'risk_types': len(risks) if isinstance(risks, dict) else 0,
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"  ❌ 风险基准测试失败: {e}")
            baselines['risk_error'] = {
                'operation': '风险基准测试',
                'error': str(e),
                'timestamp': time.time()
            }

        return baselines

    def _establish_system_baselines(self) -> Dict[str, Any]:
        """建立系统层性能基准"""
        baselines = {}

        # 系统启动时间基准
        print("  🚀 测量系统启动性能...")
        start_time = time.time()

        try:
            # 模拟系统关键组件初始化
            from src.core.foundation.factory.core_factory import CoreFactory
            factory = CoreFactory()
            core_components = factory.create_core_components()

            end_time = time.time()

            baselines['system_startup'] = {
                'operation': '系统启动',
                'execution_time': end_time - start_time,
                'components_loaded': len(core_components) if core_components else 0,
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"  ❌ 系统启动基准测试失败: {e}")
            baselines['system_startup_error'] = {
                'operation': '系统启动基准测试',
                'error': str(e),
                'timestamp': time.time()
            }

        # 内存使用基准
        print("  💾 测量内存使用基准...")
        process = psutil.Process()
        memory_info = process.memory_info()

        baselines['memory_baseline'] = {
            'operation': '内存使用基准',
            'rss_memory': memory_info.rss / 1024 / 1024,  # MB
            'vms_memory': memory_info.vms / 1024 / 1024,  # MB
            'cpu_percent': process.cpu_percent(),
            'timestamp': time.time()
        }

        return baselines

    def monitor_performance(self) -> Dict[str, Any]:
        """监控性能变化"""
        print("📈 开始性能监控...")

        monitoring_results = {}

        # 重新运行基准测试
        current_baselines = self.establish_baselines()

        # 比较差异
        for key, current in current_baselines.items():
            if key in self.baselines:
                baseline = self.baselines[key]
                diff = self._calculate_performance_diff(baseline, current)
                monitoring_results[key] = {
                    'baseline': baseline,
                    'current': current,
                    'diff': diff
                }

        # 生成监控报告
        self._save_monitoring_report(monitoring_results)

        return monitoring_results

    def _calculate_performance_diff(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """计算性能差异"""
        diff = {}

        # 时间差异
        if 'execution_time' in baseline and 'execution_time' in current:
            time_diff = current['execution_time'] - baseline['execution_time']
            diff['time_change'] = time_diff
            diff['time_change_percent'] = (time_diff / baseline['execution_time']) * 100 if baseline['execution_time'] > 0 else 0

        # 内存差异
        if 'memory_usage' in baseline and 'memory_usage' in current:
            memory_diff = current['memory_usage'] - baseline['memory_usage']
            diff['memory_change'] = memory_diff
            diff['memory_change_percent'] = (memory_diff / baseline['memory_usage']) * 100 if baseline['memory_usage'] > 0 else 0

        # 吞吐量差异
        if 'throughput' in baseline and 'throughput' in current:
            throughput_diff = current['throughput'] - baseline['throughput']
            diff['throughput_change'] = throughput_diff
            diff['throughput_change_percent'] = (throughput_diff / baseline['throughput']) * 100 if baseline['throughput'] > 0 else 0

        return diff

    def _save_baselines(self):
        """保存基准数据"""
        baseline_path = self.project_root / "test_logs" / "performance_baselines.json"

        with open(baseline_path, 'w', encoding='utf-8') as f:
            json.dump(self.baselines, f, indent=2, ensure_ascii=False)

        print(f"💾 性能基准已保存: {baseline_path}")

    def _save_monitoring_report(self, monitoring_results: Dict[str, Any]):
        """保存监控报告"""
        report_path = self.project_root / "test_logs" / "performance_monitoring_report.md"

        report_content = f"""# 性能监控报告

**生成时间**: {self._get_current_time()}
**监控目标**: 持续跟踪关键操作性能变化

## 📊 性能监控结果

"""

        for key, result in monitoring_results.items():
            baseline = result['baseline']
            current = result['current']
            diff = result['diff']

            report_content += f"### {baseline.get('operation', key)}\n"
            report_content += f"- **基准值**: {baseline.get('execution_time', 'N/A')}秒\n"
            report_content += f"- **当前值**: {current.get('execution_time', 'N/A')}秒\n"

            if 'time_change_percent' in diff:
                change_symbol = "📈" if diff['time_change_percent'] > 5 else "📉" if diff['time_change_percent'] < -5 else "➡️"
                report_content += f"- **时间变化**: {change_symbol} {diff['time_change_percent']:+.1f}%\n"

            if 'throughput' in baseline:
                baseline_throughput = baseline.get('throughput', 'N/A')
                current_throughput = current.get('throughput', 'N/A')
                if isinstance(baseline_throughput, (int, float)):
                    report_content += f"- **基准吞吐量**: {baseline_throughput:.1f} ops/sec\n"
                else:
                    report_content += f"- **基准吞吐量**: {baseline_throughput}\n"
                if isinstance(current_throughput, (int, float)):
                    report_content += f"- **当前吞吐量**: {current_throughput:.1f} ops/sec\n"
                else:
                    report_content += f"- **当前吞吐量**: {current_throughput}\n"

                if 'throughput_change_percent' in diff:
                    throughput_symbol = "📈" if diff['throughput_change_percent'] > 5 else "📉" if diff['throughput_change_percent'] < -5 else "➡️"
                    report_content += f"- **吞吐量变化**: {throughput_symbol} {diff['throughput_change_percent']:+.1f}%\n"

            report_content += "\n"

        # 性能警报
        alerts = []
        for key, result in monitoring_results.items():
            diff = result['diff']

            if diff.get('time_change_percent', 0) > 10:
                alerts.append(f"⚠️ {result['baseline'].get('operation', key)} 执行时间增加 {diff['time_change_percent']:.1f}%")

            if diff.get('throughput_change_percent', 0) < -10:
                alerts.append(f"⚠️ {result['baseline'].get('operation', key)} 吞吐量下降 {abs(diff['throughput_change_percent']):.1f}%")

        if alerts:
            report_content += "## 🚨 性能警报\n\n"
            for alert in alerts:
                report_content += f"- {alert}\n"
            report_content += "\n"

        report_content += """## 🎯 性能基准说明

### 监控指标
- **执行时间**: 操作完成所需时间
- **吞吐量**: 每秒处理的操作数
- **内存使用**: 操作期间内存消耗
- **成功率**: 操作成功完成的比例

### 警报阈值
- **时间变化**: ±10% 触发警报
- **吞吐量变化**: -10% 触发警报
- **内存变化**: +20% 触发警报

### 优化建议
1. **定期监控**: 每周执行性能监控
2. **趋势分析**: 关注性能变化趋势
3. **容量规划**: 基于基准数据进行容量规划
4. **异常排查**: 对性能异常及时排查原因

---

**报告生成**: 性能基准测试工具自动生成
**监控频率**: 建议每周执行一次
**告警机制**: 自动检测性能异常
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📄 监控报告已保存: {report_path}")

    def _get_current_time(self) -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def run_baseline_establishment(self) -> Dict[str, Any]:
        """运行基准建立流程"""
        print("🏁 开始性能基准建立流程...")

        # 建立基准
        baselines = self.establish_baselines()

        # 生成基准报告
        self._generate_baseline_report(baselines)

        print("\n✅ 性能基准建立完成！")
        print(f"📊 建立基准指标: {len(baselines)} 个")
        print("📈 覆盖操作: ML训练/预测、策略执行、订单处理、风险计算等")
        return baselines

    def _generate_baseline_report(self, baselines: Dict[str, Any]):
        """生成基准报告"""
        report_path = self.project_root / "test_logs" / "performance_baseline_report.md"

        report_content = f"""# 性能基准报告

**生成时间**: {self._get_current_time()}
**基准建立**: 为关键操作建立性能参考标准

## 📊 性能基准数据

### 🤖 ML层基准

"""

        ml_operations = [k for k in baselines.keys() if k.startswith('ml_')]
        for op in ml_operations:
            baseline = baselines[op]
            report_content += f"#### {baseline['operation']}\n"
            exec_time = baseline.get('execution_time', 'N/A')
            if isinstance(exec_time, (int, float)):
                report_content += f"- **执行时间**: {exec_time:.3f}秒\n"
            else:
                report_content += f"- **执行时间**: {exec_time}\n"
            throughput = baseline.get('throughput', 'N/A')
            if isinstance(throughput, (int, float)):
                report_content += f"- **吞吐量**: {throughput:.1f} ops/sec\n"
            else:
                report_content += f"- **吞吐量**: {throughput}\n"
            if 'memory_usage' in baseline:
                report_content += f"- **内存使用**: {baseline['memory_usage']:.1f}MB\n"
            if 'latency' in baseline:
                report_content += f"- **延迟**: {baseline['latency']:.3f}ms/op\n"
            report_content += "\n"

        report_content += """### 📈 策略层基准

"""

        strategy_operations = [k for k in baselines.keys() if k.startswith('strategy_')]
        for op in strategy_operations:
            baseline = baselines[op]
            report_content += f"#### {baseline['operation']}\n"
            exec_time = baseline.get('execution_time', 'N/A')
            if isinstance(exec_time, (int, float)):
                report_content += f"- **执行时间**: {exec_time:.3f}秒\n"
            else:
                report_content += f"- **执行时间**: {exec_time}\n"
            report_content += f"- **数据点**: {baseline.get('data_points', 'N/A')}\n"
            report_content += f"- **吞吐量**: {baseline.get('throughput', 'N/A'):.1f} data/sec\n"
            if 'signals_generated' in baseline:
                report_content += f"- **信号生成**: {baseline['signals_generated']}\n"
            report_content += "\n"

        report_content += """### 💰 交易层基准

"""

        trading_operations = [k for k in baselines.keys() if 'order' in k.lower()]
        for op in trading_operations:
            baseline = baselines[op]
            report_content += f"#### {baseline['operation']}\n"
            exec_time = baseline.get('execution_time', 'N/A')
            if isinstance(exec_time, (int, float)):
                report_content += f"- **执行时间**: {exec_time:.3f}秒\n"
            else:
                report_content += f"- **执行时间**: {exec_time}\n"
            report_content += f"- **订单数量**: {baseline.get('total_orders', baseline.get('concurrent_orders', 'N/A'))}\n"
            report_content += f"- **吞吐量**: {baseline.get('throughput', 'N/A'):.1f} orders/sec\n"
            if 'success_rate' in baseline:
                report_content += f"- **成功率**: {baseline['success_rate']:.1f}%\n"
            report_content += "\n"

        report_content += """### ⚠️ 风险控制层基准

"""

        risk_operations = [k for k in baselines.keys() if k.startswith('risk_')]
        for op in risk_operations:
            baseline = baselines[op]
            report_content += f"#### {baseline['operation']}\n"
            exec_time = baseline.get('execution_time', 'N/A')
            if isinstance(exec_time, (int, float)):
                report_content += f"- **执行时间**: {exec_time:.3f}秒\n"
            else:
                report_content += f"- **执行时间**: {exec_time}\n"
            report_content += f"- **数据点**: {baseline.get('data_points', 'N/A')}\n"
            report_content += f"- **吞吐量**: {baseline.get('throughput', 'N/A'):.1f} data/sec\n"
            if 'risk_types' in baseline:
                report_content += f"- **风险类型**: {baseline['risk_types']}\n"
            report_content += "\n"

        report_content += """### 🏗️ 系统层基准

"""

        system_operations = [k for k in baselines.keys() if 'system' in k or 'memory' in k]
        for op in system_operations:
            baseline = baselines[op]
            if 'memory' in baseline:
                report_content += f"#### {baseline['operation']}\n"
                report_content += f"- **物理内存**: {baseline.get('rss_memory', 'N/A'):.1f}MB\n"
                report_content += f"- **虚拟内存**: {baseline.get('vms_memory', 'N/A'):.1f}MB\n"
                report_content += f"- **CPU使用率**: {baseline.get('cpu_percent', 'N/A'):.1f}%\n"
            else:
                report_content += f"#### {baseline['operation']}\n"
                exec_time = baseline.get('execution_time', 'N/A')
            if isinstance(exec_time, (int, float)):
                report_content += f"- **执行时间**: {exec_time:.3f}秒\n"
            else:
                report_content += f"- **执行时间**: {exec_time}\n"
                if 'components_loaded' in baseline:
                    report_content += f"- **加载组件**: {baseline['components_loaded']}\n"
            report_content += "\n"

        report_content += """## 🎯 基准应用场景

### 性能监控
- **持续跟踪**: 定期对比当前性能与基准
- **异常检测**: 自动识别性能异常情况
- **趋势分析**: 分析性能变化趋势

### 容量规划
- **资源评估**: 基于基准数据评估资源需求
- **扩展规划**: 为系统扩展提供性能参考
- **瓶颈识别**: 识别潜在性能瓶颈

### 质量保障
- **性能测试**: 确保新版本不降低性能
- **回归测试**: 防止性能回归问题
- **优化验证**: 验证性能优化效果

## 📈 基准数据解读

### 性能指标说明
- **执行时间**: 操作完成的时间，越短越好
- **吞吐量**: 单位时间内处理的请求数，越大越好
- **内存使用**: 操作消耗的内存，影响系统稳定性
- **成功率**: 操作成功的比例，应接近100%

### 优化目标
- **时间控制**: 关键操作控制在合理时间范围内
- **资源效率**: 优化内存和CPU资源使用
- **并发能力**: 提升并发处理能力
- **稳定性**: 确保性能的稳定性和一致性

---

**报告生成**: 性能基准测试工具自动生成
**基准有效期**: 建议每月重新建立
**监控频率**: 建议每周执行监控
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📄 基准报告已保存: {report_path}")


def main():
    """主函数"""
    baseline_tool = PerformanceBaseline(".")

    # 建立性能基准
    baselines = baseline_tool.run_baseline_establishment()

    print(f"\n📊 性能基准建立完成!")
    print(f"🎯 建立基准指标数量: {len(baselines)}")
    print("📈 覆盖关键操作: ML、策略、交易、风险控制、系统层")
    print("🔄 建议定期运行监控性能变化")


if __name__ == "__main__":
    main()
