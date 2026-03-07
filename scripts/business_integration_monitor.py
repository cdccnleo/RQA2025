#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
业务系统集成监控脚本
用于监控试点业务系统的集成状态、性能指标和问题反馈
"""
import os
import json
import requests
import time
import logging
import psutil
import threading
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class IntegrationMetrics:
    """集成指标"""
    timestamp: datetime
    system_name: str
    config_load_time: float
    config_update_time: float
    error_count: int
    success_count: int
    memory_usage: float
    cpu_usage: float
    response_time: float


@dataclass
class SystemHealth:
    """系统健康状态"""
    system_name: str
    status: str  # online, offline, warning
    last_check: datetime
    uptime: float
    error_rate: float
    performance_score: float


class BusinessIntegrationMonitor:
    """业务系统集成监控器"""

    def __init__(self, config_api_base: str = "http://localhost:8080"):
        self.config_api_base = config_api_base
        self.metrics_history = defaultdict(list)
        self.system_health = {}
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval: int = 60):
        """开始监控"""
        logger.info("开始业务系统集成监控...")
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        logger.info("停止业务系统集成监控...")
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring:
            try:
                # 监控配置管理服务
                self._monitor_config_service()

                # 监控业务系统集成
                self._monitor_business_systems()

                # 生成监控报告
                self._generate_monitoring_report()

                # 检查告警条件
                self._check_alerts()

                time.sleep(interval)

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(10)

    def _monitor_config_service(self):
        """监控配置管理服务"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.config_api_base}/api/health", timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ 配置管理服务正常，响应时间: {response_time:.2f}s")

                # 记录服务指标
                self._record_service_metrics("config_service", response_time, True)
            else:
                logger.error(f"❌ 配置管理服务异常: {response.status_code}")
                self._record_service_metrics("config_service", response_time, False)

        except Exception as e:
            logger.error(f"❌ 配置管理服务监控失败: {e}")
            self._record_service_metrics("config_service", 0, False)

    def _monitor_business_systems(self):
        """监控业务系统集成"""
        business_systems = [
            "trading_system",
            "risk_control_system",
            "data_analysis_system"
        ]

        for system in business_systems:
            try:
                self._monitor_single_system(system)
            except Exception as e:
                logger.error(f"监控业务系统 {system} 失败: {e}")

    def _monitor_single_system(self, system_name: str):
        """监控单个业务系统"""
        try:
            # 模拟业务系统监控
            # 实际环境中需要连接到真实的业务系统
            start_time = time.time()

            # 模拟配置加载测试
            config_load_time = self._simulate_config_load(system_name)

            # 模拟配置更新测试
            config_update_time = self._simulate_config_update(system_name)

            # 获取系统资源使用情况
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()

            # 记录指标
            metrics = IntegrationMetrics(
                timestamp=datetime.now(),
                system_name=system_name,
                config_load_time=config_load_time,
                config_update_time=config_update_time,
                error_count=self._get_error_count(system_name),
                success_count=self._get_success_count(system_name),
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                response_time=time.time() - start_time
            )

            self.metrics_history[system_name].append(metrics)

            # 更新系统健康状态
            self._update_system_health(system_name, metrics)

            logger.info(f"✅ {system_name} 监控完成")

        except Exception as e:
            logger.error(f"❌ {system_name} 监控失败: {e}")
            self._update_system_health(system_name, None, error=True)

    def _simulate_config_load(self, system_name: str) -> float:
        """模拟配置加载测试"""
        try:
            start_time = time.time()

            # 模拟API调用
            response = requests.get(f"{self.config_api_base}/api/config", timeout=5)

            if response.status_code == 200:
                return time.time() - start_time
            else:
                return 0.0

        except Exception:
            return 0.0

    def _simulate_config_update(self, system_name: str) -> float:
        """模拟配置更新测试"""
        try:
            start_time = time.time()

            # 模拟配置更新
            test_config = {
                "test_key": f"test_value_{int(time.time())}"
            }

            response = requests.put(f"{self.config_api_base}/api/config/test",
                                    json=test_config, timeout=5)

            if response.status_code == 200:
                return time.time() - start_time
            else:
                return 0.0

        except Exception:
            return 0.0

    def _get_error_count(self, system_name: str) -> int:
        """获取错误计数"""
        # 模拟错误计数，实际应从日志或监控系统获取
        return 0

    def _get_success_count(self, system_name: str) -> int:
        """获取成功计数"""
        # 模拟成功计数，实际应从日志或监控系统获取
        return 100

    def _record_service_metrics(self, service_name: str, response_time: float, success: bool):
        """记录服务指标"""
        metrics = IntegrationMetrics(
            timestamp=datetime.now(),
            system_name=service_name,
            config_load_time=response_time,
            config_update_time=0.0,
            error_count=0 if success else 1,
            success_count=1 if success else 0,
            memory_usage=psutil.virtual_memory().percent,
            cpu_usage=psutil.cpu_percent(),
            response_time=response_time
        )

        self.metrics_history[service_name].append(metrics)

    def _update_system_health(self, system_name: str, metrics: Optional[IntegrationMetrics], error: bool = False):
        """更新系统健康状态"""
        if error:
            status = "offline"
            performance_score = 0.0
            error_rate = 1.0
        elif metrics:
            # 计算性能评分
            performance_score = self._calculate_performance_score(metrics)
            error_rate = metrics.error_count / max(metrics.success_count + metrics.error_count, 1)

            if performance_score > 0.8 and error_rate < 0.1:
                status = "online"
            elif performance_score > 0.6 and error_rate < 0.3:
                status = "warning"
            else:
                status = "offline"
        else:
            status = "unknown"
            performance_score = 0.0
            error_rate = 0.0

        self.system_health[system_name] = SystemHealth(
            system_name=system_name,
            status=status,
            last_check=datetime.now(),
            uptime=self._calculate_uptime(system_name),
            error_rate=error_rate,
            performance_score=performance_score
        )

    def _calculate_performance_score(self, metrics: IntegrationMetrics) -> float:
        """计算性能评分"""
        # 基于响应时间、错误率、资源使用情况计算评分
        response_score = max(0, 1 - metrics.response_time / 5.0)  # 5秒内满分
        error_score = max(0, 1 - metrics.error_count /
                          max(metrics.success_count + metrics.error_count, 1))
        resource_score = max(0, 1 - (metrics.memory_usage + metrics.cpu_usage) / 200.0)  # 资源使用率

        return (response_score + error_score + resource_score) / 3.0

    def _calculate_uptime(self, system_name: str) -> float:
        """计算运行时间"""
        # 模拟运行时间计算，实际应从系统启动时间计算
        return 99.5  # 99.5% 可用性

    def _generate_monitoring_report(self):
        """生成监控报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self._get_overall_status(),
            "system_health": {name: asdict(health) for name, health in self.system_health.items()},
            "metrics_summary": self._get_metrics_summary(),
            "alerts": self._get_active_alerts()
        }

        # 保存报告
        self._save_report(report)

        # 打印摘要
        self._print_report_summary(report)

    def _get_overall_status(self) -> str:
        """获取整体状态"""
        if not self.system_health:
            return "unknown"

        online_count = sum(1 for health in self.system_health.values() if health.status == "online")
        total_count = len(self.system_health)

        if online_count == total_count:
            return "healthy"
        elif online_count > total_count * 0.5:
            return "warning"
        else:
            return "critical"

    def _get_metrics_summary(self) -> Dict:
        """获取指标摘要"""
        summary = {}

        for system_name, metrics_list in self.metrics_history.items():
            if not metrics_list:
                continue

            recent_metrics = metrics_list[-10:]  # 最近10次指标

            avg_load_time = sum(m.config_load_time for m in recent_metrics) / len(recent_metrics)
            avg_update_time = sum(
                m.config_update_time for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)

            total_errors = sum(m.error_count for m in recent_metrics)
            total_success = sum(m.success_count for m in recent_metrics)
            error_rate = total_errors / max(total_errors + total_success, 1)

            summary[system_name] = {
                "avg_load_time": avg_load_time,
                "avg_update_time": avg_update_time,
                "avg_response_time": avg_response_time,
                "error_rate": error_rate,
                "total_requests": total_errors + total_success
            }

        return summary

    def _get_active_alerts(self) -> List[Dict]:
        """获取活跃告警"""
        alerts = []

        for system_name, health in self.system_health.items():
            if health.status == "offline":
                alerts.append({
                    "system": system_name,
                    "type": "critical",
                    "message": f"系统 {system_name} 离线",
                    "timestamp": health.last_check.isoformat()
                })
            elif health.status == "warning":
                alerts.append({
                    "system": system_name,
                    "type": "warning",
                    "message": f"系统 {system_name} 性能下降",
                    "timestamp": health.last_check.isoformat()
                })
            elif health.error_rate > 0.1:
                alerts.append({
                    "system": system_name,
                    "type": "warning",
                    "message": f"系统 {system_name} 错误率过高: {health.error_rate:.1%}",
                    "timestamp": health.last_check.isoformat()
                })

        return alerts

    def _check_alerts(self):
        """检查告警条件"""
        alerts = self._get_active_alerts()

        if alerts:
            logger.warning(f"发现 {len(alerts)} 个告警:")
            for alert in alerts:
                logger.warning(f"  - {alert['type'].upper()}: {alert['message']}")

        # 这里可以添加告警通知逻辑（邮件、短信、webhook等）

    def _save_report(self, report: Dict):
        """保存监控报告"""
        try:
            report_dir = "reports/monitoring"
            os.makedirs(report_dir, exist_ok=True)

            filename = f"integration_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(report_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"监控报告已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存监控报告失败: {e}")

    def _print_report_summary(self, report: Dict):
        """打印报告摘要"""
        print("\n" + "="*60)
        print("📊 业务系统集成监控报告")
        print("="*60)
        print(f"⏰ 时间: {report['timestamp']}")
        print(f"📈 整体状态: {report['overall_status']}")

        print("\n🔧 系统健康状态:")
        for system_name, health in report['system_health'].items():
            status_icon = "✅" if health['status'] == "online" else "⚠️" if health['status'] == "warning" else "❌"
            print(
                f"  {status_icon} {system_name}: {health['status']} (性能评分: {health['performance_score']:.1%})")

        print("\n📊 性能指标摘要:")
        for system_name, metrics in report['metrics_summary'].items():
            print(f"  📈 {system_name}:")
            print(f"    平均加载时间: {metrics['avg_load_time']:.3f}s")
            print(f"    平均更新时间: {metrics['avg_update_time']:.3f}s")
            print(f"    错误率: {metrics['error_rate']:.1%}")

        if report['alerts']:
            print(f"\n🚨 活跃告警 ({len(report['alerts'])} 个):")
            for alert in report['alerts']:
                alert_icon = "🔴" if alert['type'] == "critical" else "🟡"
                print(f"  {alert_icon} {alert['message']}")

        print("="*60)


class FeedbackCollector:
    """反馈收集器"""

    def __init__(self):
        self.feedback_data = []

    def collect_user_feedback(self, system_name: str, user_id: str, feedback_type: str,
                              rating: int, comment: str = ""):
        """收集用户反馈"""
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "system_name": system_name,
            "user_id": user_id,
            "feedback_type": feedback_type,  # usability, performance, functionality, stability
            "rating": rating,  # 1-5
            "comment": comment
        }

        self.feedback_data.append(feedback)
        logger.info(f"收集到用户反馈: {system_name} - {feedback_type} - {rating}/5")

    def get_feedback_summary(self) -> Dict:
        """获取反馈摘要"""
        if not self.feedback_data:
            return {}

        summary = {}

        for feedback in self.feedback_data:
            system = feedback['system_name']
            feedback_type = feedback['feedback_type']

            if system not in summary:
                summary[system] = {}

            if feedback_type not in summary[system]:
                summary[system][feedback_type] = []

            summary[system][feedback_type].append(feedback['rating'])

        # 计算平均评分
        for system in summary:
            for feedback_type in summary[system]:
                ratings = summary[system][feedback_type]
                summary[system][feedback_type] = {
                    "avg_rating": sum(ratings) / len(ratings),
                    "count": len(ratings)
                }

        return summary

    def save_feedback_report(self):
        """保存反馈报告"""
        try:
            report_dir = "reports/feedback"
            os.makedirs(report_dir, exist_ok=True)

            filename = f"user_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(report_dir, filename)

            report = {
                "timestamp": datetime.now().isoformat(),
                "total_feedback": len(self.feedback_data),
                "feedback_summary": self.get_feedback_summary(),
                "detailed_feedback": self.feedback_data
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"反馈报告已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存反馈报告失败: {e}")


def main():
    """主函数"""
    print("🔍 业务系统集成监控工具")
    print("="*60)

    # 创建监控器
    monitor = BusinessIntegrationMonitor()

    # 创建反馈收集器
    feedback_collector = FeedbackCollector()

    try:
        # 启动监控
        monitor.start_monitoring(interval=30)  # 每30秒监控一次

        # 模拟收集用户反馈
        print("\n📝 模拟收集用户反馈...")
        feedback_collector.collect_user_feedback(
            "trading_system", "user1", "usability", 4, "界面友好，操作简单")
        feedback_collector.collect_user_feedback(
            "trading_system", "user2", "performance", 5, "响应速度快")
        feedback_collector.collect_user_feedback(
            "risk_control_system", "user3", "functionality", 4, "功能完整")
        feedback_collector.collect_user_feedback(
            "data_analysis_system", "user4", "stability", 3, "偶尔出现连接问题")

        # 运行一段时间
        print("\n⏱️ 监控运行中... (按 Ctrl+C 停止)")
        time.sleep(300)  # 运行5分钟

    except KeyboardInterrupt:
        print("\n🛑 停止监控...")
    finally:
        # 停止监控
        monitor.stop_monitoring()

        # 保存反馈报告
        feedback_collector.save_feedback_report()

        print("\n✅ 监控完成，报告已保存")


if __name__ == "__main__":
    main()
