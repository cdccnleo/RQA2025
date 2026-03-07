#!/usr/bin/env python3
"""
高级性能监控系统
建立完整的性能基准和监控体系
"""

import os
import sys
import time
import json
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


class AdvancedPerformanceMonitor:
    """高级性能监控器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.baselines: Dict[str, Dict[str, Any]] = {}
        self.current_metrics: Dict[str, Any] = {}
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # 初始化系统监控
        self.process = psutil.Process(os.getpid())
        self.system_baseline = self._capture_system_baseline()

    def establish_comprehensive_baselines(self) -> Dict[str, Any]:
        """建立全面的性能基准"""
        print("🏗️ 建立全面性能基准...")
        print("=" * 60)

        # 1. 系统层基准
        print("🖥️ 建立系统层基准...")
        system_baselines = self._establish_system_baselines()
        self.baselines.update(system_baselines)

        # 2. 应用层基准
        print("🏭 建立应用层基准...")
        app_baselines = self._establish_application_baselines()
        self.baselines.update(app_baselines)

        # 3. 业务层基准
        print("💼 建立业务层基准...")
        business_baselines = self._establish_business_baselines()
        self.baselines.update(business_baselines)

        # 4. 数据层基准
        print("🗄️ 建立数据层基准...")
        data_baselines = self._establish_data_baselines()
        self.baselines.update(data_baselines)

        # 保存基准线
        self._save_baselines()

        print(f"✅ 共建立 {len(self.baselines)} 个性能基准点")

        return self.baselines

    def _capture_system_baseline(self) -> Dict[str, Any]:
        """捕获系统基准"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_total': psutil.disk_usage('/').total,
            'disk_free': psutil.disk_usage('/').free,
            'platform': sys.platform,
            'python_version': sys.version
        }

    def _establish_system_baselines(self) -> Dict[str, Dict[str, Any]]:
        """建立系统层基准"""
        baselines = {}

        # CPU基准
        cpu_usage = self._measure_cpu_baseline()
        baselines['cpu_baseline'] = {
            'operation': '系统CPU使用率',
            'avg_cpu_percent': cpu_usage['avg'],
            'max_cpu_percent': cpu_usage['max'],
            'min_cpu_percent': cpu_usage['min'],
            'measurement_duration': 10,
            'timestamp': time.time()
        }

        # 内存基准
        memory_usage = self._measure_memory_baseline()
        baselines['memory_baseline'] = {
            'operation': '系统内存使用率',
            'avg_memory_percent': memory_usage['avg'],
            'max_memory_mb': memory_usage['max_mb'],
            'min_memory_mb': memory_usage['min_mb'],
            'measurement_duration': 10,
            'timestamp': time.time()
        }

        # 磁盘IO基准
        disk_io = self._measure_disk_io_baseline()
        baselines['disk_io_baseline'] = {
            'operation': '系统磁盘IO',
            'read_bytes_per_sec': disk_io['read_per_sec'],
            'write_bytes_per_sec': disk_io['write_per_sec'],
            'measurement_duration': 10,
            'timestamp': time.time()
        }

        return baselines

    def _establish_application_baselines(self) -> Dict[str, Dict[str, Any]]:
        """建立应用层基准"""
        baselines = {}

        # 应用启动时间
        startup_time = self._measure_startup_time()
        baselines['app_startup'] = {
            'operation': '应用启动时间',
            'execution_time': startup_time,
            'timestamp': time.time()
        }

        # 模块导入时间
        import_times = self._measure_import_times()
        for module_name, import_time in import_times.items():
            baselines[f'import_{module_name}'] = {
                'operation': f'模块{module_name}导入',
                'execution_time': import_time,
                'timestamp': time.time()
            }

        # 基本功能响应时间
        response_times = self._measure_basic_responses()
        for operation, response_time in response_times.items():
            baselines[f'response_{operation}'] = {
                'operation': f'基本功能{operation}',
                'execution_time': response_time,
                'timestamp': time.time()
            }

        return baselines

    def _establish_business_baselines(self) -> Dict[str, Dict[str, Any]]:
        """建立业务层基准"""
        baselines = {}

        try:
            # ML预测基准
            ml_baseline = self._measure_ml_prediction_baseline()
            baselines['ml_prediction'] = {
                'operation': 'ML模型预测',
                'execution_time': ml_baseline['time'],
                'throughput': ml_baseline['throughput'],
                'memory_usage_mb': ml_baseline['memory_mb'],
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"⚠️ ML基准建立失败: {e}")

        try:
            # 策略执行基准
            strategy_baseline = self._measure_strategy_execution_baseline()
            baselines['strategy_execution'] = {
                'operation': '策略执行',
                'execution_time': strategy_baseline['time'],
                'signals_generated': strategy_baseline['signals'],
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"⚠️ 策略基准建立失败: {e}")

        try:
            # 风险计算基准
            risk_baseline = self._measure_risk_calculation_baseline()
            baselines['risk_calculation'] = {
                'operation': '风险计算',
                'execution_time': risk_baseline['time'],
                'calculations_per_sec': risk_baseline['calculations_per_sec'],
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"⚠️ 风险基准建立失败: {e}")

        return baselines

    def _establish_data_baselines(self) -> Dict[str, Dict[str, Any]]:
        """建立数据层基准"""
        baselines = {}

        try:
            # 数据处理基准
            data_processing = self._measure_data_processing_baseline()
            baselines['data_processing'] = {
                'operation': '数据处理',
                'execution_time': data_processing['time'],
                'records_per_sec': data_processing['records_per_sec'],
                'memory_usage_mb': data_processing['memory_mb'],
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"⚠️ 数据处理基准建立失败: {e}")

        try:
            # 数据查询基准
            data_query = self._measure_data_query_baseline()
            baselines['data_query'] = {
                'operation': '数据查询',
                'execution_time': data_query['time'],
                'query_complexity': data_query['complexity'],
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"⚠️ 数据查询基准建立失败: {e}")

        return baselines

    def _measure_cpu_baseline(self) -> Dict[str, float]:
        """测量CPU基准"""
        measurements = []
        for _ in range(10):
            measurements.append(psutil.cpu_percent(interval=1))

        return {
            'avg': np.mean(measurements),
            'max': np.max(measurements),
            'min': np.min(measurements)
        }

    def _measure_memory_baseline(self) -> Dict[str, float]:
        """测量内存基准"""
        measurements = []
        for _ in range(10):
            mem = psutil.virtual_memory()
            measurements.append(mem.used / 1024 / 1024)  # MB
            time.sleep(1)

        return {
            'avg': np.mean(measurements),
            'max_mb': np.max(measurements),
            'min_mb': np.min(measurements)
        }

    def _measure_disk_io_baseline(self) -> Dict[str, float]:
        """测量磁盘IO基准"""
        io_start = psutil.disk_io_counters()
        time.sleep(10)
        io_end = psutil.disk_io_counters()

        time_diff = 10.0
        read_bytes = io_end.read_bytes - io_start.read_bytes
        write_bytes = io_end.write_bytes - io_start.write_bytes

        return {
            'read_per_sec': read_bytes / time_diff,
            'write_per_sec': write_bytes / time_diff
        }

    def _measure_startup_time(self) -> float:
        """测量应用启动时间"""
        start_time = time.time()
        # 模拟应用启动的关键操作
        try:
            from src.core.integration import get_unified_adapter_factory
            factory = get_unified_adapter_factory()
            # 执行一些初始化操作
            adapter = factory.get_adapter('data')
        except Exception:
            pass

        return time.time() - start_time

    def _measure_import_times(self) -> Dict[str, float]:
        """测量关键模块导入时间"""
        modules_to_test = [
            'src.ml.core.ml_core',
            'src.strategy.core.strategy_service',
            'src.trading.core.trading_engine',
            'src.risk.monitor.realtime_risk_monitor'
        ]

        import_times = {}
        for module in modules_to_test:
            try:
                start_time = time.time()
                __import__(module.replace('src.', '').replace('.', '_'))
                import_times[module.split('.')[-1]] = time.time() - start_time
            except ImportError:
                import_times[module.split('.')[-1]] = float('inf')

        return import_times

    def _measure_basic_responses(self) -> Dict[str, float]:
        """测量基本功能响应时间"""
        response_times = {}

        # 测试配置管理响应
        try:
            start_time = time.time()
            from src.infrastructure.config.core.config_manager import ConfigManager
            config = ConfigManager()
            response_times['config_access'] = time.time() - start_time
        except Exception:
            response_times['config_access'] = float('inf')

        # 测试日志记录响应
        try:
            start_time = time.time()
            from src.infrastructure.logging.logger import StructuredLogger
            logger = StructuredLogger("performance_test")
            logger.info("Performance test log")
            response_times['logging'] = time.time() - start_time
        except Exception:
            response_times['logging'] = float('inf')

        return response_times

    def _measure_ml_prediction_baseline(self) -> Dict[str, Any]:
        """测量ML预测基准"""
        try:
            from src.ml.core.ml_core import MLCore
            import pandas as pd
            import numpy as np

            ml_core = MLCore()

            # 创建测试数据
            X = pd.DataFrame({
                'feature1': np.random.normal(0, 1, 1000),
                'feature2': np.random.normal(0, 1, 1000),
                'feature3': np.random.normal(0, 1, 1000)
            })
            y = pd.Series(np.random.randint(0, 2, 1000))

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear')

            # 测量预测性能
            start_time = time.time()
            predictions = ml_core.predict(model_id, X)
            prediction_time = time.time() - start_time

            return {
                'time': prediction_time,
                'throughput': len(X) / prediction_time,
                'memory_mb': self.process.memory_info().rss / 1024 / 1024
            }

        except Exception as e:
            print(f"ML预测基准测试失败: {e}")
            return {
                'time': float('inf'),
                'throughput': 0,
                'memory_mb': 0
            }

    def _measure_strategy_execution_baseline(self) -> Dict[str, Any]:
        """测量策略执行基准"""
        try:
            from src.strategy.core.strategy_service import UnifiedStrategyService
            from src.strategy.core.unified_strategy_interface import StrategyConfig, StrategyType
            import pandas as pd

            service = UnifiedStrategyService()

            # 创建策略配置
            config = StrategyConfig(
                strategy_id="perf_test_strategy",
                name="Performance Test Strategy",
                description="Strategy for performance testing",
                strategy_type=StrategyType.TREND_FOLLOWING,
                parameters={'window_size': 20, 'threshold': 0.02},
                status="initialized",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            service.create_strategy(config)

            # 创建测试数据
            data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
                'price': 100 + np.random.normal(0, 1, 1000).cumsum(),
                'volume': np.random.poisson(1000, 1000)
            })

            # 执行策略
            start_time = time.time()
            result = service.execute_strategy(config.strategy_id, data)
            execution_time = time.time() - start_time

            signals_count = len(result.get('signals', [])) if isinstance(result, dict) else 0

            return {
                'time': execution_time,
                'signals': signals_count
            }

        except Exception as e:
            print(f"策略执行基准测试失败: {e}")
            return {
                'time': float('inf'),
                'signals': 0
            }

    def _measure_risk_calculation_baseline(self) -> Dict[str, Any]:
        """测量风险计算基准"""
        try:
            from src.risk.monitor.realtime_risk_monitor import RealtimeRiskMonitor
            import pandas as pd

            monitor = RealtimeRiskMonitor()

            # 创建测试数据
            data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
                'price': 100 + np.random.normal(0, 1, 1000).cumsum(),
                'returns': np.random.normal(0, 0.01, 1000)
            })

            # 执行风险计算
            start_time = time.time()
            risks = monitor.calculate_all_risks(data)
            calculation_time = time.time() - start_time

            calculations_per_sec = len(data) / calculation_time if calculation_time > 0 else 0

            return {
                'time': calculation_time,
                'calculations_per_sec': calculations_per_sec
            }

        except Exception as e:
            print(f"风险计算基准测试失败: {e}")
            return {
                'time': float('inf'),
                'calculations_per_sec': 0
            }

    def _measure_data_processing_baseline(self) -> Dict[str, Any]:
        """测量数据处理基准"""
        try:
            from src.data.data_processor import DataProcessor
            import pandas as pd

            processor = DataProcessor()

            # 创建大数据集
            data = {
                'timestamp': pd.date_range('2023-01-01', periods=10000, freq='1s'),
                'price': 100 + np.random.normal(0, 1, 10000).cumsum(),
                'volume': np.random.poisson(1000, 10000),
                'symbol': ['AAPL'] * 10000
            }

            # 执行数据处理
            start_time = time.time()
            processed_data = processor.process(data)
            processing_time = time.time() - start_time

            records_per_sec = 10000 / processing_time if processing_time > 0 else 0

            return {
                'time': processing_time,
                'records_per_sec': records_per_sec,
                'memory_mb': self.process.memory_info().rss / 1024 / 1024
            }

        except Exception as e:
            print(f"数据处理基准测试失败: {e}")
            return {
                'time': float('inf'),
                'records_per_sec': 0,
                'memory_mb': 0
            }

    def _measure_data_query_baseline(self) -> Dict[str, Any]:
        """测量数据查询基准"""
        try:
            # 这里可以集成实际的数据查询测试
            # 目前使用模拟测试
            start_time = time.time()
            time.sleep(0.1)  # 模拟查询时间
            query_time = time.time() - start_time

            return {
                'time': query_time,
                'complexity': 'medium'  # 模拟复杂度
            }

        except Exception as e:
            return {
                'time': float('inf'),
                'complexity': 'unknown'
            }

    def start_continuous_monitoring(self):
        """开始持续监控"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("🔄 持续性能监控已启动")

    def stop_continuous_monitoring(self):
        """停止持续监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("⏹️ 持续性能监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._capture_current_metrics()
                self._check_performance_anomalies()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                print(f"⚠️ 性能监控异常: {e}")
                time.sleep(30)

    def _capture_current_metrics(self):
        """捕获当前性能指标"""
        self.current_metrics = {
            'timestamp': time.time(),
            'cpu_percent': self.process.cpu_percent(),
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'memory_percent': self.process.memory_percent(),
            'num_threads': self.process.num_threads(),
            'num_fds': self._get_num_fds()
        }

    def _get_num_fds(self) -> int:
        """获取文件描述符数量"""
        try:
            return len(self.process.open_files())
        except (psutil.AccessDenied, AttributeError):
            return 0

    def _check_performance_anomalies(self):
        """检查性能异常"""
        if not self.baselines:
            return

        anomalies = []

        # 检查CPU使用率异常
        cpu_baseline = self.baselines.get('cpu_baseline', {})
        if cpu_baseline and self.current_metrics['cpu_percent'] > cpu_baseline.get('max_cpu_percent', 100) * 1.5:
            anomalies.append({
                'type': 'cpu_spike',
                'current': self.current_metrics['cpu_percent'],
                'baseline': cpu_baseline.get('max_cpu_percent'),
                'severity': 'high'
            })

        # 检查内存使用异常
        memory_baseline = self.baselines.get('memory_baseline', {})
        if memory_baseline and self.current_metrics['memory_mb'] > memory_baseline.get('max_memory_mb', 0) * 2:
            anomalies.append({
                'type': 'memory_spike',
                'current': self.current_metrics['memory_mb'],
                'baseline': memory_baseline.get('max_memory_mb'),
                'severity': 'high'
            })

        # 记录异常
        if anomalies:
            self._log_anomalies(anomalies)

    def _log_anomalies(self, anomalies: List[Dict[str, Any]]):
        """记录性能异常"""
        anomaly_log = {
            'timestamp': time.time(),
            'anomalies': anomalies,
            'current_metrics': self.current_metrics
        }

        # 保存到日志文件
        log_file = self.project_root / "test_logs" / "performance_anomalies.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(anomaly_log, ensure_ascii=False) + '\n')

        # 输出告警
        for anomaly in anomalies:
            severity_icon = "🚨" if anomaly['severity'] == 'high' else "⚠️"
            print(f"{severity_icon} 性能异常: {anomaly['type']} - 当前:{anomaly['current']:.2f}, 基准:{anomaly.get('baseline', 'N/A')}")

    def _save_baselines(self):
        """保存基准线"""
        baselines_file = self.project_root / "test_logs" / "comprehensive_performance_baselines.json"

        # 添加元数据
        baselines_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'system_info': self.system_baseline,
                'version': '2.0'
            },
            'baselines': self.baselines
        }

        with open(baselines_file, 'w', encoding='utf-8') as f:
            json.dump(baselines_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 性能基准已保存: {baselines_file}")

    def generate_performance_report(self) -> str:
        """生成性能报告"""
        print("📊 生成性能监控报告...")

        report = f"""# 高级性能监控报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 系统基准信息

- **CPU核心数**: {self.system_baseline.get('cpu_count', 'N/A')}
- **总内存**: {self.system_baseline.get('memory_total', 0) / 1024 / 1024 / 1024:.1f} GB
- **可用内存**: {self.system_baseline.get('memory_available', 0) / 1024 / 1024 / 1024:.1f} GB
- **Python版本**: {self.system_baseline.get('python_version', 'N/A').split()[0]}

## 性能基准汇总

"""

        # 按类别分组显示基准
        categories = {
            '系统层': ['cpu_baseline', 'memory_baseline', 'disk_io_baseline'],
            '应用层': [k for k in self.baselines.keys() if k.startswith(('app_startup', 'import_', 'response_'))],
            '业务层': [k for k in self.baselines.keys() if k in ['ml_prediction', 'strategy_execution', 'risk_calculation']],
            '数据层': [k for k in self.baselines.keys() if k in ['data_processing', 'data_query']]
        }

        for category, keys in categories.items():
            if any(k in self.baselines for k in keys):
                report += f"### {category}\\n\\n"
                report += "| 操作 | 执行时间 | 其他指标 |\\n"
                report += "|------|----------|----------|\\n"

                for key in keys:
                    if key in self.baselines:
                        baseline = self.baselines[key]
                        operation = baseline.get('operation', key)
                        exec_time = baseline.get('execution_time', 'N/A')

                        if isinstance(exec_time, (int, float)) and exec_time != float('inf'):
                            time_str = f"{exec_time:.4f}s"
                        else:
                            time_str = "N/A"

                        # 添加其他指标
                        other_metrics = []
                        if 'throughput' in baseline:
                            other_metrics.append(f"吞吐量: {baseline['throughput']:.1f}")
                        if 'memory_usage_mb' in baseline:
                            other_metrics.append(f"内存: {baseline['memory_usage_mb']:.1f}MB")
                        if 'signals_generated' in baseline:
                            other_metrics.append(f"信号: {baseline['signals_generated']}")

                        other_str = ", ".join(other_metrics) if other_metrics else "-"

                        report += f"| {operation} | {time_str} | {other_str} |\\n"

                report += "\\n"

        # 当前监控状态
        if self.current_metrics:
            report += "## 当前监控状态\\n\\n"
            report += f"- **CPU使用率**: {self.current_metrics.get('cpu_percent', 'N/A'):.1f}%\\n"
            report += f"- **内存使用**: {self.current_metrics.get('memory_mb', 'N/A'):.1f} MB\\n"
            report += f"- **线程数**: {self.current_metrics.get('num_threads', 'N/A')}\\n"
            report += f"- **文件描述符**: {self.current_metrics.get('num_fds', 'N/A')}\\n"

        report += """
## 监控建议

1. **基准维护**: 定期重新建立性能基准，适应系统变化
2. **异常监控**: 关注CPU和内存使用异常，及时处理性能问题
3. **容量规划**: 基于基准数据进行系统容量规划
4. **性能优化**: 识别性能瓶颈，优先优化关键路径

---
**高级性能监控系统自动生成**
**监控状态**: 🔄 持续运行
**基准版本**: 2.0
"""

        # 保存报告
        report_file = self.project_root / "test_logs" / "advanced_performance_report.md"
        report_file.write_text(report, encoding='utf-8')
        print(f"✅ 性能报告已生成: {report_file}")

        return report


def main():
    """主函数"""
    print("🚀 RQA2025 高级性能监控系统")
    print("=" * 50)

    monitor = AdvancedPerformanceMonitor(".")

    try:
        # 建立全面基准
        baselines = monitor.establish_comprehensive_baselines()

        # 启动持续监控
        monitor.start_continuous_monitoring()

        # 生成报告
        report = monitor.generate_performance_report()

        print("\\n🎉 性能监控系统建立完成！")
        print(f"📊 共建立 {len(baselines)} 个性能基准点")
        print("🔄 持续监控已启动")

        # 保持运行一段时间进行监控
        print("⏳ 运行监控 30 秒...")
        time.sleep(30)

    except KeyboardInterrupt:
        print("\\n⏹️ 收到停止信号")
    except Exception as e:
        print(f"❌ 性能监控系统启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止监控
        monitor.stop_continuous_monitoring()
        print("✅ 性能监控系统已关闭")


if __name__ == "__main__":
    main()
