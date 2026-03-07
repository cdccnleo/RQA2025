#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 5: 生产环境模拟

创建生产环境模拟脚本，验证系统在生产负载下的表现
包括并发用户模拟、持续运行测试、资源使用监控等
"""

import threading
import time
import logging
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import os
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入核心业务组件
try:
    from phase4_core_business_fix import CoreBusinessEngine, TransactionType, OrderType
    from phase4_risk_control_system_reconstruction import ComprehensiveRiskControlSystem, Account
    from phase4_portfolio_management_reconstruction import PortfolioManager
    logger.info("成功导入核心业务组件")
except ImportError as e:
    logger.error(f"导入核心组件失败: {e}")
    # 创建模拟组件用于测试

    class MockCoreBusinessEngine:
        def __init__(self):
            self.orders = []
            self.positions = {}

        def submit_order(self, *args, **kwargs):
            return True, "模拟订单提交成功", "MOCK_ORDER_001"

        def get_account_summary(self):
            return {"balance": 100000, "total_value": 100000}


@dataclass
class LoadProfile:
    """负载配置"""
    concurrent_users: int = 100  # 并发用户数
    requests_per_second: int = 50  # 每秒请求数
    duration_minutes: int = 30  # 测试持续时间（分钟）
    ramp_up_seconds: int = 60  # 负载爬坡时间
    ramp_down_seconds: int = 30  # 负载下降时间


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_bytes_sent: float
    network_bytes_recv: float
    active_threads: int
    open_files: int
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0


@dataclass
class BusinessMetrics:
    """业务指标"""
    timestamp: datetime
    orders_submitted: int = 0
    orders_filled: int = 0
    portfolio_updates: int = 0
    risk_checks: int = 0
    total_transactions: int = 0
    active_users: int = 0


class ProductionEnvironmentSimulator:
    """生产环境模拟器"""

    def __init__(self, load_profile: LoadProfile):
        self.load_profile = load_profile
        self.is_running = False
        self.start_time = None
        self.end_time = None

        # 性能监控
        self.performance_monitor = SystemPerformanceMonitor()
        self.business_monitor = BusinessMetricsMonitor()

        # 业务引擎
        try:
            self.business_engine = CoreBusinessEngine()
            self.business_engine.initialize()
            self.risk_system = ComprehensiveRiskControlSystem()
            self.risk_system.start()
            self.portfolio_manager = PortfolioManager()
        except Exception as e:
            logger.warning(f"初始化业务引擎失败，使用模拟组件: {e}")
            self.business_engine = MockCoreBusinessEngine()
            self.risk_system = None
            self.portfolio_manager = None

        # 负载生成器
        self.load_generator = LoadGenerator(
            self.business_engine, self.risk_system, self.portfolio_manager)

        # 测试结果
        self.performance_history: List[PerformanceMetrics] = []
        self.business_history: List[BusinessMetrics] = []
        self.errors: List[Dict[str, Any]] = []

        logger.info("生产环境模拟器初始化完成")

    def start_simulation(self) -> bool:
        """启动生产环境模拟"""
        try:
            logger.info("🚀 开始生产环境模拟测试")
            logger.info(f"配置: 并发用户={self.load_profile.concurrent_users}, "
                        f"RPS={self.load_profile.requests_per_second}, "
                        f"持续时间={self.load_profile.duration_minutes}分钟")

            self.is_running = True
            self.start_time = datetime.now()

            # 启动监控线程
            monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitoring_thread.start()

            # 启动负载生成
            self.load_generator.start_load_test(
                concurrent_users=self.load_profile.concurrent_users,
                duration_minutes=self.load_profile.duration_minutes,
                ramp_up_seconds=self.load_profile.ramp_up_seconds
            )

            # 等待测试完成
            test_duration = self.load_profile.duration_minutes * 60
            time.sleep(test_duration)

            # 优雅停止
            self.stop_simulation()

            # 生成测试报告
            self._generate_simulation_report()

            return True

        except Exception as e:
            logger.error(f"生产环境模拟失败: {e}")
            self._record_error("simulation_failed", str(e))
            return False

    def stop_simulation(self):
        """停止模拟"""
        logger.info("🛑 停止生产环境模拟")
        self.is_running = False
        self.end_time = datetime.now()

        if self.risk_system:
            self.risk_system.stop()

        self.load_generator.stop_load_test()

    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集性能指标
                perf_metrics = self.performance_monitor.collect_metrics()
                self.performance_history.append(perf_metrics)

                # 收集业务指标
                business_metrics = self.business_monitor.collect_metrics()
                self.business_history.append(business_metrics)

                # 检查系统健康状态
                self._check_system_health(perf_metrics)

                time.sleep(5)  # 每5秒收集一次

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                self._record_error("monitoring_error", str(e))

    def _check_system_health(self, perf_metrics: PerformanceMetrics):
        """检查系统健康状态"""
        alerts = []

        # CPU使用率告警
        if perf_metrics.cpu_percent > 80:
            alerts.append(f"CPU使用率过高: {perf_metrics.cpu_percent:.1f}%")
            self._record_error("high_cpu", f"CPU使用率: {perf_metrics.cpu_percent:.1f}%")

        # 内存使用率告警
        if perf_metrics.memory_percent > 85:
            alerts.append(f"内存使用率过高: {perf_metrics.memory_percent:.1f}%")
            self._record_error("high_memory", f"内存使用率: {perf_metrics.memory_percent:.1f}%")

        # 响应时间告警
        if perf_metrics.response_times:
            avg_response_time = sum(perf_metrics.response_times) / len(perf_metrics.response_times)
            if avg_response_time > 2.0:  # 2秒
                alerts.append(f"平均响应时间过长: {avg_response_time:.2f}秒")
                self._record_error("slow_response", f"响应时间: {avg_response_time:.2f}秒")

        if alerts:
            logger.warning(f"系统健康告警: {', '.join(alerts)}")

    def _record_error(self, error_type: str, message: str):
        """记录错误"""
        error = {
            'timestamp': datetime.now(),
            'type': error_type,
            'message': message
        }
        self.errors.append(error)

    def _generate_simulation_report(self):
        """生成模拟测试报告"""
        report = {
            'simulation_summary': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_minutes': self.load_profile.duration_minutes,
                'concurrent_users': self.load_profile.concurrent_users,
                'requests_per_second': self.load_profile.requests_per_second
            },
            'performance_analysis': self._analyze_performance(),
            'business_analysis': self._analyze_business(),
            'error_analysis': self._analyze_errors(),
            'recommendations': self._generate_recommendations()
        }

        # 保存报告
        report_file = f'test_logs/phase5_production_simulation_{int(time.time())}.json'
        os.makedirs('test_logs', exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📊 生产环境模拟报告已保存: {report_file}")

        # 打印总结报告
        self._print_summary_report(report)

    def _analyze_performance(self) -> Dict[str, Any]:
        """分析性能数据"""
        if not self.performance_history:
            return {}

        cpu_usage = [m.cpu_percent for m in self.performance_history]
        memory_usage = [m.memory_percent for m in self.performance_history]
        response_times = []

        for metrics in self.performance_history:
            response_times.extend(metrics.response_times)

        return {
            'cpu_usage': {
                'avg': np.mean(cpu_usage) if cpu_usage else 0,
                'max': max(cpu_usage) if cpu_usage else 0,
                'p95': np.percentile(cpu_usage, 95) if cpu_usage else 0
            },
            'memory_usage': {
                'avg': np.mean(memory_usage) if memory_usage else 0,
                'max': max(memory_usage) if memory_usage else 0,
                'p95': np.percentile(memory_usage, 95) if memory_usage else 0
            },
            'response_time': {
                'avg': np.mean(response_times) if response_times else 0,
                'p95': np.percentile(response_times, 95) if response_times else 0,
                'p99': np.percentile(response_times, 99) if response_times else 0
            },
            'system_resources': {
                'avg_threads': np.mean([m.active_threads for m in self.performance_history]),
                'avg_open_files': np.mean([m.open_files for m in self.performance_history])
            }
        }

    def _analyze_business(self) -> Dict[str, Any]:
        """分析业务数据"""
        if not self.business_history:
            return {}

        total_orders = sum(m.orders_submitted for m in self.business_history)
        total_transactions = sum(m.total_transactions for m in self.business_history)

        return {
            'orders': {
                'total_submitted': total_orders,
                'avg_per_minute': total_orders / max(1, self.load_profile.duration_minutes)
            },
            'transactions': {
                'total': total_transactions,
                'avg_per_minute': total_transactions / max(1, self.load_profile.duration_minutes)
            },
            'throughput': {
                'orders_per_second': total_orders / max(1, self.load_profile.duration_minutes * 60),
                'transactions_per_second': total_transactions / max(1, self.load_profile.duration_minutes * 60)
            }
        }

    def _analyze_errors(self) -> Dict[str, Any]:
        """分析错误数据"""
        if not self.errors:
            return {'total_errors': 0, 'error_types': {}}

        error_types = {}
        for error in self.errors:
            error_type = error['type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            'total_errors': len(self.errors),
            'error_types': error_types,
            'error_rate': len(self.errors) / max(1, len(self.business_history))
        }

    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        perf_analysis = self._analyze_performance()
        error_analysis = self._analyze_errors()

        # CPU优化建议
        if perf_analysis.get('cpu_usage', {}).get('avg', 0) > 70:
            recommendations.append("🔧 CPU使用率过高，建议优化计算密集型操作，考虑使用异步处理")

        # 内存优化建议
        if perf_analysis.get('memory_usage', {}).get('avg', 0) > 80:
            recommendations.append("💾 内存使用率过高，建议优化内存管理，增加垃圾回收频率")

        # 响应时间优化建议
        if perf_analysis.get('response_time', {}).get('p95', 0) > 1.0:
            recommendations.append("⚡ 响应时间过长，建议优化数据库查询和缓存策略")

        # 错误率优化建议
        if error_analysis.get('error_rate', 0) > 0.05:
            recommendations.append("🚨 错误率较高，建议加强错误处理和系统稳定性")

        # 业务吞吐量建议
        business_analysis = self._analyze_business()
        if business_analysis.get('throughput', {}).get('orders_per_second', 0) < 10:
            recommendations.append("📈 订单处理吞吐量不足，建议优化并发处理和队列管理")

        if not recommendations:
            recommendations.append("✅ 系统表现良好，无明显性能问题")

        return recommendations

    def _print_summary_report(self, report: Dict[str, Any]):
        """打印总结报告"""
        print("\n" + "="*80)
        print("📊 生产环境模拟测试总结报告")
        print("="*80)

        summary = report['simulation_summary']
        print(f"🕐 测试时间: {summary['start_time']} - {summary['end_time']}")
        print(f"👥 并发用户: {summary['concurrent_users']}")
        print(f"📈 目标RPS: {summary['requests_per_second']}")
        print(f"⏱️  持续时间: {summary['duration_minutes']}分钟")

        perf = report['performance_analysis']
        if perf:
            print("\n🔧 性能指标:")
            print(
                f"  CPU使用率 - 平均: {perf['cpu_usage']['avg']:.1f}%, 峰值: {perf['cpu_usage']['max']:.1f}%")
            print(
                f"  内存使用率 - 平均: {perf['memory_usage']['avg']:.1f}%, 峰值: {perf['memory_usage']['max']:.1f}%")
            print(
                f"  响应时间 - 平均: {perf['response_time']['avg']:.2f}s, P95: {perf['response_time']['p95']:.2f}s")
            print(
                f"  系统资源 - 线程数: {perf['system_resources']['avg_threads']:.0f}, 文件句柄: {perf['system_resources']['avg_open_files']:.0f}")
        business = report['business_analysis']
        if business:
            print("\n💼 业务指标:")
            print(f"  总订单数: {business['orders']['total_submitted']}")
            print(f"  订单/分钟: {business['orders']['avg_per_minute']:.1f}")
            print(f"  交易/分钟: {business['transactions']['avg_per_minute']:.1f}")
        errors = report['error_analysis']
        print("\n🚨 错误统计:")
        print(f"  总错误数: {errors.get('total_errors', 0)}")
        print(f"  错误率: {errors.get('error_rate', 0):.3f}")
        print("\n💡 优化建议:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print("\n" + "="*80)


class SystemPerformanceMonitor:
    """系统性能监控器"""

    def __init__(self):
        self.initial_network = psutil.net_io_counters()
        self.initial_disk = psutil.disk_io_counters()

    def collect_metrics(self) -> PerformanceMetrics:
        """收集系统性能指标"""
        try:
            # CPU和内存
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # 磁盘IO
            disk_current = psutil.disk_io_counters()
            disk_read = disk_current.read_bytes - \
                self.initial_disk.read_bytes if disk_current and self.initial_disk else 0
            disk_write = disk_current.write_bytes - \
                self.initial_disk.write_bytes if disk_current and self.initial_disk else 0

            # 网络IO
            network_current = psutil.net_io_counters()
            network_sent = network_current.bytes_sent - self.initial_network.bytes_sent if network_current else 0
            network_recv = network_current.bytes_recv - self.initial_network.bytes_recv if network_current else 0

            # 进程信息
            process = psutil.Process()
            active_threads = process.num_threads()
            open_files = len(process.open_files())

            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_io_read=disk_read,
                disk_io_write=disk_write,
                network_bytes_sent=network_sent,
                network_bytes_recv=network_recv,
                active_threads=active_threads,
                open_files=open_files,
                response_times=[],  # 由业务层填充
                error_count=0
            )

        except Exception as e:
            logger.error(f"收集性能指标失败: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_io_read=0,
                disk_io_write=0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_threads=0,
                open_files=0
            )


class BusinessMetricsMonitor:
    """业务指标监控器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置指标"""
        self.metrics = BusinessMetrics(timestamp=datetime.now())

    def record_order_submitted(self):
        """记录订单提交"""
        self.metrics.orders_submitted += 1

    def record_order_filled(self):
        """记录订单成交"""
        self.metrics.orders_filled += 1

    def record_portfolio_update(self):
        """记录组合更新"""
        self.metrics.portfolio_updates += 1

    def record_risk_check(self):
        """记录风险检查"""
        self.metrics.risk_checks += 1

    def record_transaction(self):
        """记录交易"""
        self.metrics.total_transactions += 1

    def set_active_users(self, count: int):
        """设置活跃用户数"""
        self.metrics.active_users = count

    def collect_metrics(self) -> BusinessMetrics:
        """收集业务指标"""
        current_metrics = self.metrics
        current_metrics.timestamp = datetime.now()
        self.reset()  # 重置为下一个收集周期
        return current_metrics


class LoadGenerator:
    """负载生成器"""

    def __init__(self, business_engine, risk_system=None, portfolio_manager=None):
        self.business_engine = business_engine
        self.risk_system = risk_system
        self.portfolio_manager = portfolio_manager
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=50)

    def start_load_test(self, concurrent_users: int, duration_minutes: int, ramp_up_seconds: int):
        """启动负载测试"""
        logger.info(f"启动负载测试: {concurrent_users}并发用户, 持续{duration_minutes}分钟")

        self.is_running = True

        # 启动用户模拟线程
        for i in range(concurrent_users):
            delay = (i / concurrent_users) * ramp_up_seconds  # 均匀分布启动时间
            user_thread = threading.Thread(
                target=self._simulate_user,
                args=(i, duration_minutes * 60, delay),
                daemon=True
            )
            user_thread.start()

    def stop_load_test(self):
        """停止负载测试"""
        logger.info("停止负载测试")
        self.is_running = False
        self.executor.shutdown(wait=True)

    def _simulate_user(self, user_id: int, duration_seconds: int, start_delay: float):
        """模拟单个用户行为"""
        time.sleep(start_delay)

        end_time = time.time() + duration_seconds

        while self.is_running and time.time() < end_time:
            try:
                # 模拟用户操作
                self._perform_user_action(user_id)

                # 随机等待1-5秒
                time.sleep(np.random.uniform(1, 5))

            except Exception as e:
                logger.error(f"用户{user_id}操作失败: {e}")

    def _perform_user_action(self, user_id: int):
        """执行用户操作"""
        action_type = np.random.choice(['order', 'portfolio', 'risk_check', 'idle'],
                                       p=[0.3, 0.3, 0.2, 0.2])

        if action_type == 'order':
            # 模拟下单操作
            success, message, order_id = self.business_engine.submit_order(
                symbol=np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA']),
                transaction_type=np.random.choice([TransactionType.BUY, TransactionType.SELL]),
                quantity=np.random.randint(10, 100),
                order_type=OrderType.MARKET
            )

        elif action_type == 'portfolio' and self.portfolio_manager:
            # 模拟组合查询
            portfolios = self.portfolio_manager.get_all_portfolios()
            if portfolios:
                portfolio_id = portfolios[0]['portfolio_id']
                analysis = self.portfolio_manager.analyze_portfolio(portfolio_id)

        elif action_type == 'risk_check' and self.risk_system:
            # 模拟风险检查
            account = Account(
                account_id=f"user_{user_id}",
                balance=100000,
                positions={},
                total_value=100000
            )
            assessment = self.risk_system.assess_portfolio_risk(account, {})

        # 空闲操作 - 不做任何事


def run_production_simulation():
    """运行生产环境模拟测试"""
    logger.info("开始Phase 5: 生产环境模拟测试")

    # 配置负载参数（生产环境规模）
    load_profile = LoadProfile(
        concurrent_users=50,      # 50个并发用户
        requests_per_second=25,   # 每秒25个请求
        duration_minutes=10,      # 10分钟测试
        ramp_up_seconds=30,       # 30秒爬坡
        ramp_down_seconds=15      # 15秒降坡
    )

    # 创建模拟器并运行测试
    simulator = ProductionEnvironmentSimulator(load_profile)

    try:
        success = simulator.start_simulation()
        if success:
            logger.info("✅ 生产环境模拟测试完成")
        else:
            logger.error("❌ 生产环境模拟测试失败")
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止测试...")
        simulator.stop_simulation()
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")
        simulator.stop_simulation()


if __name__ == "__main__":
    run_production_simulation()
