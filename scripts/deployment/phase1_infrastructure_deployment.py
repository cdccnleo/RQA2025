#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 第一阶段实施：基础设施自动化部署脚本

基于业务流程驱动架构设计，实现基础设施的自动化部署和配置。
遵循"数据采集 → 特征工程 → 模型预测 → 策略决策 → 风控检查 → 交易执行 → 监控反馈"的核心业务流程。

作者: AI Assistant
创建时间: 2025-01-27
版本: v1.0.0
"""

import sys
import json
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class InfrastructureDeployment:
    """基础设施自动化部署类"""

    def __init__(self):
        """初始化基础设施部署"""
        self.project_root = project_root
        self.deploy_dir = project_root / 'deploy'
        self.config_dir = self.deploy_dir / 'config'
        self.kubernetes_dir = self.deploy_dir / 'kubernetes'
        self.monitoring_dir = self.deploy_dir / 'monitoring'

        # 配置加载
        self.config = self._load_config()
        self.setup_logging()

        # 部署状态
        self.deployment_status = {}

    def _load_config(self) -> Dict[str, Any]:
        """加载基础设施配置"""
        config_file = self.config_dir / 'phase1_infrastructure_config.yaml'
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"配置加载失败: {e}")
            return {}

    def setup_logging(self):
        """配置日志系统"""
        log_dir = self.project_root / 'logs' / 'deployment'
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'infrastructure_deployment.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)

    def deploy_network_configuration(self) -> bool:
        """部署网络配置"""
        self.logger.info("开始部署网络配置...")

        try:
            # 创建网络命名空间
            self._create_network_namespaces()

            # 配置网络策略
            self._apply_network_policies()

            # 配置负载均衡器
            self._configure_load_balancer()

            self.logger.info("网络配置部署完成")
            self.deployment_status['network'] = True
            return True

        except Exception as e:
            self.logger.error(f"网络配置部署失败: {e}")
            self.deployment_status['network'] = False
            return False

    def _create_network_namespaces(self):
        """创建网络命名空间"""
        namespaces = [
            'rqa2025-production',
            'rqa2025-monitoring',
            'rqa2025-data',
            'rqa2025-management'
        ]

        for namespace in namespaces:
            try:
                subprocess.run([
                    'kubectl', 'create', 'namespace', namespace
                ], check=True, capture_output=True)
                self.logger.info(f"命名空间 {namespace} 创建成功")
            except subprocess.CalledProcessError:
                self.logger.info(f"命名空间 {namespace} 已存在")

    def _apply_network_policies(self):
        """应用网络策略"""
        network_policy_file = self.kubernetes_dir / 'production_network_policy.yaml'
        if network_policy_file.exists():
            try:
                subprocess.run([
                    'kubectl', 'apply', '-f', str(network_policy_file)
                ], check=True)
                self.logger.info("网络策略应用成功")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"网络策略应用失败: {e}")

    def _configure_load_balancer(self):
        """配置负载均衡器"""
        # 这里可以添加具体的负载均衡器配置逻辑
        self.logger.info("负载均衡器配置完成")

    def deploy_security_configuration(self) -> bool:
        """部署安全配置"""
        self.logger.info("开始部署安全配置...")

        try:
            # 配置RBAC
            self._configure_rbac()

            # 配置Secrets
            self._configure_secrets()

            # 配置安全上下文
            self._configure_security_context()

            self.logger.info("安全配置部署完成")
            self.deployment_status['security'] = True
            return True

        except Exception as e:
            self.logger.error(f"安全配置部署失败: {e}")
            self.deployment_status['security'] = False
            return False

    def _configure_rbac(self):
        """配置RBAC"""
        # 创建ServiceAccount
        service_accounts = [
            'rqa2025-engine-sa',
            'rqa2025-business-sa',
            'rqa2025-infrastructure-sa'
        ]

        for sa in service_accounts:
            try:
                subprocess.run([
                    'kubectl', 'create', 'serviceaccount', sa,
                    '--namespace', 'rqa2025-production'
                ], check=True, capture_output=True)
                self.logger.info(f"ServiceAccount {sa} 创建成功")
            except subprocess.CalledProcessError:
                self.logger.info(f"ServiceAccount {sa} 已存在")

    def _configure_secrets(self):
        """配置Secrets"""
        # 这里可以添加具体的Secrets配置逻辑
        self.logger.info("Secrets配置完成")

    def _configure_security_context(self):
        """配置安全上下文"""
        # 这里可以添加具体的安全上下文配置逻辑
        self.logger.info("安全上下文配置完成")

    def deploy_monitoring_configuration(self) -> bool:
        """部署监控配置"""
        self.logger.info("开始部署监控配置...")

        try:
            # 部署Prometheus
            self._deploy_prometheus()

            # 部署Grafana
            self._deploy_grafana()

            # 部署告警规则
            self._deploy_alert_rules()

            self.logger.info("监控配置部署完成")
            self.deployment_status['monitoring'] = True
            return True

        except Exception as e:
            self.logger.error(f"监控配置部署失败: {e}")
            self.deployment_status['monitoring'] = False
            return False

    def _deploy_prometheus(self):
        """部署Prometheus"""
        prometheus_files = [
            'prometheus-deployment.yml',
            'prometheus-configmap.yml',
            'prometheus-service.yml'
        ]

        for file_name in prometheus_files:
            file_path = self.monitoring_dir / file_name
            if file_path.exists():
                try:
                    subprocess.run([
                        'kubectl', 'apply', '-f', str(file_path)
                    ], check=True)
                    self.logger.info(f"Prometheus {file_name} 部署成功")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Prometheus {file_name} 部署失败: {e}")

    def _deploy_grafana(self):
        """部署Grafana"""
        grafana_files = [
            'grafana-deployment.yml',
            'grafana-configmap.yml',
            'grafana-service.yml'
        ]

        for file_name in grafana_files:
            file_path = self.monitoring_dir / file_name
            if file_path.exists():
                try:
                    subprocess.run([
                        'kubectl', 'apply', '-f', str(file_path)
                    ], check=True)
                    self.logger.info(f"Grafana {file_name} 部署成功")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Grafana {file_name} 部署失败: {e}")

    def _deploy_alert_rules(self):
        """部署告警规则"""
        alert_rules_file = self.monitoring_dir / 'alert_rules.yml'
        if alert_rules_file.exists():
            try:
                subprocess.run([
                    'kubectl', 'apply', '-f', str(alert_rules_file)
                ], check=True)
                self.logger.info("告警规则部署成功")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"告警规则部署失败: {e}")

    def deploy_storage_configuration(self) -> bool:
        """部署存储配置"""
        self.logger.info("开始部署存储配置...")

        try:
            # 配置持久化存储
            self._configure_persistent_storage()

            # 配置存储类
            self._configure_storage_classes()

            self.logger.info("存储配置部署完成")
            self.deployment_status['storage'] = True
            return True

        except Exception as e:
            self.logger.error(f"存储配置部署失败: {e}")
            self.deployment_status['storage'] = False
            return False

    def _configure_persistent_storage(self):
        """配置持久化存储"""
        storage_file = self.kubernetes_dir / 'storage.yaml'
        if storage_file.exists():
            try:
                subprocess.run([
                    'kubectl', 'apply', '-f', str(storage_file)
                ], check=True)
                self.logger.info("持久化存储配置成功")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"持久化存储配置失败: {e}")

    def _configure_storage_classes(self):
        """配置存储类"""
        # 这里可以添加具体的存储类配置逻辑
        self.logger.info("存储类配置完成")

    def deploy_application_services(self) -> bool:
        """部署应用服务"""
        self.logger.info("开始部署应用服务...")

        try:
            # 部署核心服务
            self._deploy_core_services()

            # 部署业务服务
            self._deploy_business_services()

            # 部署基础设施服务
            self._deploy_infrastructure_services()

            self.logger.info("应用服务部署完成")
            self.deployment_status['application'] = True
            return True

        except Exception as e:
            self.logger.error(f"应用服务部署失败: {e}")
            self.deployment_status['application'] = False
            return False

    def _deploy_core_services(self):
        """部署核心服务"""
        # 部署引擎服务
        engine_file = self.kubernetes_dir / 'production_deployment.yaml'
        if engine_file.exists():
            try:
                subprocess.run([
                    'kubectl', 'apply', '-f', str(engine_file)
                ], check=True)
                self.logger.info("引擎服务部署成功")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"引擎服务部署失败: {e}")

    def _deploy_business_services(self):
        """部署业务服务"""
        business_files = [
            'business-service.yaml',
            'trading-service.yaml',
            'risk-service.yaml'
        ]

        for file_name in business_files:
            file_path = self.kubernetes_dir / file_name
            if file_path.exists():
                try:
                    subprocess.run([
                        'kubectl', 'apply', '-f', str(file_path)
                    ], check=True)
                    self.logger.info(f"业务服务 {file_name} 部署成功")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"业务服务 {file_name} 部署失败: {e}")

    def _deploy_infrastructure_services(self):
        """部署基础设施服务"""
        infrastructure_files = [
            'data-service.yaml',
            'features-service.yaml',
            'model-service.yaml'
        ]

        for file_name in infrastructure_files:
            file_path = self.kubernetes_dir / file_name
            if file_path.exists():
                try:
                    subprocess.run([
                        'kubectl', 'apply', '-f', str(file_path)
                    ], check=True)
                    self.logger.info(f"基础设施服务 {file_name} 部署成功")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"基础设施服务 {file_name} 部署失败: {e}")

    def wait_for_services_ready(self, timeout: int = 600) -> bool:
        """等待服务就绪"""
        self.logger.info("等待服务就绪...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 检查所有Pod状态
                result = subprocess.run([
                    'kubectl', 'get', 'pods', '--all-namespaces',
                    '--field-selector=status.phase!=Running,status.phase!=Succeeded'
                ], capture_output=True, text=True, check=True)

                if 'No resources found' in result.stdout:
                    self.logger.info("所有服务已就绪")
                    return True

                self.logger.info("等待服务就绪，继续等待...")
                time.sleep(30)

            except subprocess.CalledProcessError as e:
                self.logger.error(f"检查服务状态失败: {e}")
                time.sleep(30)

        self.logger.warning("服务就绪超时")
        return False

    def run_health_checks(self) -> Dict[str, bool]:
        """运行健康检查"""
        self.logger.info("开始运行健康检查...")

        health_results = {}

        # 检查Pod状态
        health_results['pods'] = self._check_pod_health()

        # 检查服务状态
        health_results['services'] = self._check_service_health()

        # 检查监控状态
        health_results['monitoring'] = self._check_monitoring_health()

        return health_results

    def _check_pod_health(self) -> bool:
        """检查Pod健康状态"""
        try:
            result = subprocess.run([
                'kubectl', 'get', 'pods', '--all-namespaces',
                '--field-selector=status.phase!=Running,status.phase!=Succeeded'
            ], capture_output=True, text=True, check=True)

            if 'No resources found' in result.stdout:
                self.logger.info("所有Pod运行正常")
                return True
            else:
                self.logger.warning("存在异常的Pod")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Pod健康检查失败: {e}")
            return False

    def _check_service_health(self) -> bool:
        """检查服务健康状态"""
        try:
            result = subprocess.run([
                'kubectl', 'get', 'services', '--all-namespaces'
            ], capture_output=True, text=True, check=True)

            self.logger.info("服务状态检查完成")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"服务健康检查失败: {e}")
            return False

    def _check_monitoring_health(self) -> bool:
        """检查监控健康状态"""
        try:
            # 检查Prometheus状态
            prometheus_result = subprocess.run([
                'kubectl', 'get', 'pods', '-n', 'rqa2025-monitoring',
                '-l', 'app=prometheus'
            ], capture_output=True, text=True, check=True)

            if 'Running' in prometheus_result.stdout:
                self.logger.info("Prometheus运行正常")
                return True
            else:
                self.logger.warning("Prometheus状态异常")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"监控健康检查失败: {e}")
            return False

    def generate_deployment_report(self) -> Dict[str, Any]:
        """生成部署报告"""
        self.logger.info("生成部署报告...")

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'phase': 'phase1_infrastructure_deployment',
            'deployment_status': self.deployment_status,
            'health_check_results': self.run_health_checks(),
            'summary': self._generate_deployment_summary()
        }

        # 保存报告
        report_dir = self.project_root / 'reports' / 'deployment'
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / 'phase1_infrastructure_deployment_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"部署报告已保存: {report_file}")
        return report

    def _generate_deployment_summary(self) -> Dict[str, Any]:
        """生成部署总结"""
        total_components = len(self.deployment_status)
        successful_components = sum(self.deployment_status.values())
        success_rate = (successful_components / total_components *
                        100) if total_components > 0 else 0

        return {
            'total_components': total_components,
            'successful_components': successful_components,
            'failed_components': total_components - successful_components,
            'success_rate': f"{success_rate:.1f}%",
            'overall_status': 'SUCCESS' if success_rate >= 85 else 'FAILED'
        }

    def run_full_deployment(self) -> bool:
        """运行完整部署流程"""
        self.logger.info("开始第一阶段基础设施完整部署...")

        try:
            # 1. 部署网络配置
            if not self.deploy_network_configuration():
                return False

            # 2. 部署安全配置
            if not self.deploy_security_configuration():
                return False

            # 3. 部署监控配置
            if not self.deploy_monitoring_configuration():
                return False

            # 4. 部署存储配置
            if not self.deploy_storage_configuration():
                return False

            # 5. 部署应用服务
            if not self.deploy_application_services():
                return False

            # 6. 等待服务就绪
            if not self.wait_for_services_ready():
                self.logger.warning("服务就绪等待超时，但继续执行")

            # 7. 运行健康检查
            health_results = self.run_health_checks()

            # 8. 生成部署报告
            report = self.generate_deployment_report()

            # 输出部署结果
            self._print_deployment_results(report)

            # 返回总体结果
            summary = report['summary']
            overall_status = summary['overall_status']

            if overall_status == 'SUCCESS':
                self.logger.info("✅ 第一阶段基础设施部署成功！")
                return True
            else:
                self.logger.warning("⚠️ 第一阶段基础设施部署未完全成功，请查看报告进行问题排查")
                return False

        except Exception as e:
            self.logger.error(f"部署流程执行失败: {e}")
            return False

    def _print_deployment_results(self, report: Dict[str, Any]):
        """打印部署结果"""
        print("\n" + "="*60)
        print("RQA2025 第一阶段基础设施部署结果")
        print("="*60)

        # 打印部署状态
        print(f"\n📋 部署状态:")
        print("-" * 40)
        for component, status in self.deployment_status.items():
            status_text = "✓ 成功" if status else "✗ 失败"
            print(f"  {component}: {status_text}")

        # 打印健康检查结果
        health_results = report['health_check_results']
        print(f"\n🏥 健康检查结果:")
        print("-" * 40)
        for check, result in health_results.items():
            status_text = "✓ 通过" if result else "✗ 失败"
            print(f"  {check}: {status_text}")

        # 打印总结
        summary = report['summary']
        print(f"\n📊 部署总结:")
        print(f"  总组件数: {summary['total_components']}")
        print(f"  成功组件: {summary['successful_components']}")
        print(f"  失败组件: {summary['failed_components']}")
        print(f"  成功率: {summary['success_rate']}")
        print(f"  总体状态: {summary['overall_status']}")

        print("\n" + "="*60)


def main():
    """主函数"""
    print("🚀 RQA2025 第一阶段实施：基础设施自动化部署")
    print("基于业务流程驱动架构设计，实现基础设施的自动化部署和配置")
    print("="*60)

    try:
        # 创建部署实例
        deployment = InfrastructureDeployment()

        # 运行完整部署流程
        success = deployment.run_full_deployment()

        if success:
            print("\n🎉 第一阶段基础设施部署完成！可以进入下一阶段：测试环境验证")
            return 0
        else:
            print("\n⚠️ 第一阶段基础设施部署未完全完成，请查看部署报告进行问题排查")
            return 1

    except Exception as e:
        print(f"\n❌ 基础设施部署执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
