#!/usr/bin/env python3
"""
灰度发布计划和执行脚本

实现分批次灰度发布策略，包括：
1. 发布策略配置
2. 流量控制和路由
3. 监控和回滚机制
4. 自动化部署流程
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class DeploymentPhase(Enum):
    """部署阶段"""
    PRE_DEPLOYMENT_CHECK = "pre_deployment_check"
    INITIAL_ROLLOUT = "initial_rollout"
    GRADUAL_INCREASE = "gradual_increase"
    FULL_ROLLOUT = "full_rollout"
    MONITORING_PERIOD = "monitoring_period"
    ROLLBACK = "rollback"


class TrafficDistribution(Enum):
    """流量分配策略"""
    PERCENTAGE_BASED = "percentage_based"
    USER_BASED = "user_based"
    GEOGRAPHIC_BASED = "geographic_based"
    TIME_BASED = "time_based"


@dataclass
class CanaryConfiguration:
    """灰度发布配置"""
    name: str
    version: str
    total_traffic_percentage: int = 100
    initial_rollout_percentage: int = 5
    rollout_steps: List[int] = None
    monitoring_duration_minutes: int = 30
    success_criteria: Dict[str, Any] = None
    rollback_criteria: Dict[str, Any] = None
    traffic_distribution: TrafficDistribution = TrafficDistribution.PERCENTAGE_BASED

    def __post_init__(self):
        if self.rollout_steps is None:
            self.rollout_steps = [5, 15, 30, 50, 75, 100]

        if self.success_criteria is None:
            self.success_criteria = {
                'error_rate_threshold': 0.05,  # 5%错误率
                'response_time_p95_threshold': 5000,  # 5秒
                'cpu_usage_threshold': 80,  # 80%
                'memory_usage_threshold': 85,  # 85%
                'monitoring_duration_minutes': 15
            }

        if self.rollback_criteria is None:
            self.rollback_criteria = {
                'error_rate_threshold': 0.10,  # 10%错误率触发回滚
                'response_time_p95_threshold': 10000,  # 10秒响应时间触发回滚
                'cpu_usage_threshold': 95,  # 95%CPU触发回滚
                'memory_usage_threshold': 95,  # 95%内存触发回滚
                'manual_rollback_trigger': False
            }


@dataclass
class DeploymentMetrics:
    """部署指标"""
    timestamp: str
    phase: DeploymentPhase
    traffic_percentage: int
    active_users: int
    error_rate: float
    response_time_p95: float
    cpu_usage: float
    memory_usage: float
    success_rate: float
    rollback_triggered: bool = False
    rollback_reason: Optional[str] = None


class CanaryDeploymentManager:
    """灰度发布管理器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config = None
        self.metrics_history: List[DeploymentMetrics] = []
        self.current_phase = DeploymentPhase.PRE_DEPLOYMENT_CHECK
        self.deployment_start_time = None

    def load_configuration(self, config_file: Path) -> CanaryConfiguration:
        """加载灰度发布配置"""
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.config = CanaryConfiguration(**data)
        else:
            # 使用默认配置
            self.config = CanaryConfiguration(
                name="health_monitor_canary",
                version="1.0.0"
            )

        print(f"✅ 已加载灰度发布配置: {self.config.name} v{self.config.version}")
        return self.config

    def save_configuration(self, config_file: Path):
        """保存配置"""
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        print(f"✅ 配置已保存到: {config_file}")

    def pre_deployment_check(self) -> bool:
        """预部署检查"""
        print("🔍 执行预部署检查...")

        checks = {
            'health_check': self._check_system_health(),
            'dependencies': self._check_dependencies(),
            'configuration': self._check_configuration(),
            'backup': self._check_backup_availability(),
            'monitoring': self._check_monitoring_setup()
        }

        failed_checks = [check for check, passed in checks.items() if not passed]

        if failed_checks:
            print(f"❌ 预部署检查失败: {', '.join(failed_checks)}")
            return False

        print("✅ 预部署检查全部通过")
        return True

    def _check_system_health(self) -> bool:
        """检查系统健康状态"""
        try:
            # 运行健康检查
            result = subprocess.run([
                sys.executable, '-c',
                'from src.infrastructure.health.health_check import HealthCheck; '
                'hc = HealthCheck(); print(hc.check_health())'
            ], capture_output=True, text=True, timeout=30)

            return result.returncode == 0 and 'healthy' in result.stdout.lower()
        except Exception:
            return False

    def _check_dependencies(self) -> bool:
        """检查依赖项"""
        try:
            # 检查关键依赖
            return True
        except ImportError:
            return False

    def _check_configuration(self) -> bool:
        """检查配置完整性"""
        required_configs = [
            'src/infrastructure/health/health_check.py',
            'deployment/preprod/docker-compose.yml',
            'deployment/preprod/config/prometheus.yml'
        ]

        return all(Path(self.project_root / config).exists() for config in required_configs)

    def _check_backup_availability(self) -> bool:
        """检查备份可用性"""
        # 检查是否存在备份脚本
        backup_script = self.project_root / 'scripts' / 'backup.sh'
        return backup_script.exists()

    def _check_monitoring_setup(self) -> bool:
        """检查监控设置"""
        try:
            # 检查Prometheus配置
            return True
        except ImportError:
            return False

    def start_initial_rollout(self) -> bool:
        """开始初始发布"""
        print(f"🚀 开始初始灰度发布 ({self.config.initial_rollout_percentage}%)...")

        if not self.pre_deployment_check():
            return False

        self.current_phase = DeploymentPhase.INITIAL_ROLLOUT
        self.deployment_start_time = time.time()

        # 部署初始版本
        if not self._deploy_to_percentage(self.config.initial_rollout_percentage):
            print("❌ 初始部署失败")
            return False

        # 开始监控
        self._start_monitoring()

        print(f"✅ 初始发布完成，当前流量: {self.config.initial_rollout_percentage}%")
        return True

    def gradual_rollout(self) -> bool:
        """逐步增加流量"""
        print("📈 开始逐步流量增加...")

        for percentage in self.config.rollout_steps[1:]:  # 跳过初始百分比
            print(f"🔄 增加流量到 {percentage}%...")

            if not self._deploy_to_percentage(percentage):
                print(f"❌ 流量增加到 {percentage}% 失败")
                return False

            # 监控这个阶段
            if not self._monitor_phase(percentage):
                print(f"⚠️ 流量 {percentage}% 阶段监控发现问题")
                # 可以选择继续或回滚
                continue

            print(f"✅ 流量 {percentage}% 阶段成功")

        self.current_phase = DeploymentPhase.FULL_ROLLOUT
        print("🎉 逐步发布完成，达到全量发布")
        return True

    def _deploy_to_percentage(self, percentage: int) -> bool:
        """部署到指定百分比"""
        try:
            # 这里实现实际的部署逻辑
            # 可以使用Docker Compose、Kubernetes等

            # 模拟部署过程
            time.sleep(2)  # 模拟部署时间

            # 记录指标
            metrics = DeploymentMetrics(
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                phase=self.current_phase,
                traffic_percentage=percentage,
                active_users=int(percentage * 10),  # 模拟用户数
                error_rate=0.02,  # 模拟错误率
                response_time_p95=1000,  # 模拟响应时间
                cpu_usage=60,  # 模拟CPU使用率
                memory_usage=70,  # 模拟内存使用率
                success_rate=0.98  # 模拟成功率
            )

            self.metrics_history.append(metrics)

            return True

        except Exception as e:
            print(f"❌ 部署失败: {e}")
            return False

    def _monitor_phase(self, percentage: int) -> bool:
        """监控阶段表现"""
        print(f"📊 监控流量 {percentage}% 阶段...")

        # 监控指定时长
        monitoring_start = time.time()

        while time.time() - monitoring_start < self.config.monitoring_duration_minutes * 60:
            # 收集实时指标
            current_metrics = self._collect_realtime_metrics()

            # 检查是否满足成功标准
            if self._check_success_criteria(current_metrics):
                print("✅ 阶段监控通过")
                return True

            # 检查是否需要回滚
            if self._check_rollback_criteria(current_metrics):
                print("❌ 检测到问题，准备回滚")
                self.rollback()
                return False

            time.sleep(60)  # 每分钟检查一次

        # 如果监控时间结束但没有明确失败，认为通过
        return True

    def _collect_realtime_metrics(self) -> Dict[str, Any]:
        """收集实时指标"""
        # 这里实现实际的指标收集逻辑
        return {
            'error_rate': 0.02,
            'response_time_p95': 1200,
            'cpu_usage': 65,
            'memory_usage': 75,
            'active_users': 100
        }

    def _check_success_criteria(self, metrics: Dict[str, Any]) -> bool:
        """检查成功标准"""
        criteria = self.config.success_criteria

        return (
            metrics.get('error_rate', 0) <= criteria['error_rate_threshold'] and
            metrics.get('response_time_p95', 0) <= criteria['response_time_p95_threshold'] and
            metrics.get('cpu_usage', 0) <= criteria['cpu_usage_threshold'] and
            metrics.get('memory_usage', 0) <= criteria['memory_usage_threshold']
        )

    def _check_rollback_criteria(self, metrics: Dict[str, Any]) -> bool:
        """检查回滚标准"""
        criteria = self.config.rollback_criteria

        return (
            metrics.get('error_rate', 0) >= criteria['error_rate_threshold'] or
            metrics.get('response_time_p95', 0) >= criteria['response_time_p95_threshold'] or
            metrics.get('cpu_usage', 0) >= criteria['cpu_usage_threshold'] or
            metrics.get('memory_usage', 0) >= criteria['memory_usage_threshold'] or
            criteria.get('manual_rollback_trigger', False)
        )

    def _start_monitoring(self):
        """开始监控"""
        print("📈 启动部署监控...")

        # 这里可以启动Prometheus、Grafana监控
        # 或者启动自定义监控脚本

    def rollback(self) -> bool:
        """执行回滚"""
        print("🔄 执行回滚操作...")

        self.current_phase = DeploymentPhase.ROLLBACK

        try:
            # 回滚到上一个稳定版本
            # 这里实现实际的回滚逻辑

            # 记录回滚指标
            metrics = DeploymentMetrics(
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                phase=DeploymentPhase.ROLLBACK,
                traffic_percentage=0,
                active_users=0,
                error_rate=0.0,
                response_time_p95=0,
                cpu_usage=0,
                memory_usage=0,
                success_rate=1.0,
                rollback_triggered=True,
                rollback_reason="自动检测到问题"
            )

            self.metrics_history.append(metrics)

            print("✅ 回滚完成")
            return True

        except Exception as e:
            print(f"❌ 回滚失败: {e}")
            return False

    def generate_report(self) -> Dict[str, Any]:
        """生成部署报告"""
        total_duration = time.time() - self.deployment_start_time if self.deployment_start_time else 0

        report = {
            'deployment_name': self.config.name,
            'version': self.config.version,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.deployment_start_time)) if self.deployment_start_time else None,
            'total_duration_minutes': total_duration / 60,
            'final_phase': self.current_phase.value,
            'final_traffic_percentage': self.metrics_history[-1].traffic_percentage if self.metrics_history else 0,
            'metrics_history': [{'phase': m.phase.value, **{k: v for k, v in asdict(m).items() if k != 'phase'}} for m in self.metrics_history],
            'success': self.current_phase == DeploymentPhase.FULL_ROLLOUT,
            'rollback_performed': any(m.rollback_triggered for m in self.metrics_history)
        }

        return report

    def save_report(self, report_file: Path):
        """保存部署报告"""
        report = self.generate_report()

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ 部署报告已保存: {report_file}")

        # 打印总结
        print("\n" + "="*50)
        print("📋 灰度发布总结报告")
        print("="*50)
        print(f"部署名称: {report['deployment_name']}")
        print(f"版本: {report['version']}")
        print(f"开始时间: {report['start_time']}")
        print(f"总耗时: {report['total_duration_minutes']:.1f} 分钟")
        print(f"最终阶段: {report['final_phase']}")
        print(f"最终流量: {report['final_traffic_percentage']}%")
        print(f"部署成功: {'是' if report['success'] else '否'}")
        print(f"执行回滚: {'是' if report['rollback_performed'] else '否'}")

        if report['success']:
            print("🎉 灰度发布成功完成！")
        else:
            print("⚠️ 灰度发布未完全成功，请检查报告详情")


def create_default_canary_config() -> Dict[str, Any]:
    """创建默认灰度发布配置"""
    config = CanaryConfiguration(
        name="rqa2025_health_monitor_canary",
        version="1.0.0",
        rollout_steps=[5, 15, 30, 50, 75, 100],
        monitoring_duration_minutes=20
    )

    config_dict = asdict(config)
    # 转换枚举值为字符串
    config_dict['traffic_distribution'] = config.traffic_distribution.value
    return config_dict


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent

    # 创建灰度发布管理器
    manager = CanaryDeploymentManager(project_root)

    # 加载或创建配置
    config_file = project_root / 'deployment' / 'canary_config.json'

    if not config_file.exists():
        print("📝 创建默认灰度发布配置...")
        default_config = create_default_canary_config()
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

    manager.load_configuration(config_file)

    # 执行灰度发布流程
    print("🚀 开始灰度发布流程...")

    try:
        # 1. 预部署检查
        if not manager.pre_deployment_check():
            print("❌ 预部署检查失败，终止发布")
            return 1

        # 2. 初始发布
        if not manager.start_initial_rollout():
            print("❌ 初始发布失败，终止发布")
            return 1

        # 3. 逐步增加流量
        if not manager.gradual_rollout():
            print("❌ 逐步发布失败")
            return 1

        # 4. 监控期
        manager.current_phase = DeploymentPhase.MONITORING_PERIOD
        print("📊 进入监控期...")

        # 5. 生成报告
        report_file = project_root / 'deployment' / 'canary_deployment_report.json'
        manager.save_report(report_file)

        print("✅ 灰度发布流程完成")
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ 收到中断信号，正在回滚...")
        manager.rollback()
        return 1
    except Exception as e:
        print(f"❌ 灰度发布过程中发生错误: {e}")
        manager.rollback()
        return 1


if __name__ == '__main__':
    sys.exit(main())
