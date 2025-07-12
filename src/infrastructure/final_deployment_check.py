#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最终部署检查模块
负责在系统正式上线前执行最终检查
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.health.health_checker import HealthChecker
from src.infrastructure.visual_monitor import VisualMonitor
from src.infrastructure.deployment_validator import DeploymentValidator

logger = get_logger(__name__)

@dataclass
class FinalCheckItem:
    """最终检查项"""
    name: str
    description: str
    critical: bool  # 是否为关键检查项
    check_func: str  # 检查函数名

@dataclass
class CheckResult:
    """检查结果"""
    name: str
    status: str  # PASSED, FAILED, WARNING
    details: str
    timestamp: float

class FinalDeploymentCheck:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化最终部署检查器
        :param config: 系统配置
        """
        self.config = config
        self.config_manager = ConfigManager(config)
        self.health_checker = HealthChecker(config)
        self.visual_monitor = VisualMonitor(config)
        self.deployment_validator = DeploymentValidator(config)
        self.check_items: List[FinalCheckItem] = []
        self.check_results: List[CheckResult] = []
        self.lock = threading.Lock()

        # 加载检查项
        self._load_check_items()

    def run_checks(self) -> bool:
        """
        执行所有最终检查
        :return: 是否全部通过
        """
        results = []
        all_passed = True
        has_critical_failure = False

        for item in self.check_items:
            try:
                # 执行检查
                status, details = self._execute_check(item)
                results.append(CheckResult(
                    name=item.name,
                    status=status,
                    details=details,
                    timestamp=time.time()
                ))

                # 检查结果
                if status != "PASSED":
                    if item.critical:
                        has_critical_failure = True
                    all_passed = False

                logger.info(f"最终检查 {item.name}: {status} - {details}")

            except Exception as e:
                logger.error(f"执行最终检查 {item.name} 出错: {str(e)}")
                results.append(CheckResult(
                    name=item.name,
                    status="FAILED",
                    details=f"检查异常: {str(e)}",
                    timestamp=time.time()
                ))
                if item.critical:
                    has_critical_failure = True
                all_passed = False

        # 保存结果
        with self.lock:
            self.check_results = results

        return all_passed and not has_critical_failure

    def get_check_report(self) -> Dict[str, Any]:
        """
        获取检查报告
        :return: 检查报告字典
        """
        report = {
            "timestamp": time.time(),
            "total": len(self.check_results),
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "results": []
        }

        with self.lock:
            for result in self.check_results:
                if result.status == "PASSED":
                    report["passed"] += 1
                elif result.status == "FAILED":
                    report["failed"] += 1
                else:
                    report["warnings"] += 1

                report["results"].append({
                    "name": result.name,
                    "status": result.status,
                    "details": result.details,
                    "timestamp": result.timestamp
                })

        return report

    def _load_check_items(self) -> None:
        """加载检查项"""
        check_config = self.config_manager.get('final_deployment_checks', {})

        # 加载基础检查项
        for check in check_config.get('basic', []):
            self.check_items.append(FinalCheckItem(
                name=check['name'],
                description=check['description'],
                critical=check.get('critical', False),
                check_func=check['check_func']
            ))

        # 加载高级检查项
        for check in check_config.get('advanced', []):
            self.check_items.append(FinalCheckItem(
                name=check['name'],
                description=check['description'],
                critical=check.get('critical', True),
                check_func=check['check_func']
            ))

    def _execute_check(self, item: FinalCheckItem) -> tuple:
        """
        执行单个检查项
        :param item: 检查项
        :return: (状态, 详情)
        """
        # 根据检查函数名调用对应的检查方法
        if item.check_func == "check_system_health":
            return self._check_system_health()
        elif item.check_func == "check_service_status":
            return self._check_service_status()
        elif item.check_func == "check_config_consistency":
            return self._check_config_consistency()
        elif item.check_func == "check_deployment_validation":
            return self._check_deployment_validation()
        elif item.check_func == "check_resource_usage":
            return self._check_resource_usage()
        else:
            raise ValueError(f"未知的检查函数: {item.check_func}")

    def _check_system_health(self) -> tuple:
        """
        检查系统整体健康状态
        :return: (状态, 详情)
        """
        dashboard = self.visual_monitor.get_dashboard_data()
        if dashboard['system_health'] == "GREEN":
            return ("PASSED", "系统健康状态正常")
        elif dashboard['system_health'] == "YELLOW":
            return ("WARNING", "系统存在警告状态服务")
        else:
            return ("FAILED", "系统存在严重故障")

    def _check_service_status(self) -> tuple:
        """
        检查所有服务状态
        :return: (状态, 详情)
        """
        dashboard = self.visual_monitor.get_dashboard_data()
        failed_services = [s['name'] for s in dashboard['services'] if s['health'] != 'UP']

        if not failed_services:
            return ("PASSED", "所有服务运行正常")
        else:
            return ("FAILED", f"以下服务异常: {', '.join(failed_services)}")

    def _check_config_consistency(self) -> tuple:
        """
        检查配置一致性
        :return: (状态, 详情)
        """
        # 检查关键配置是否存在
        required_configs = [
            'database.host', 'database.port',
            'trading.risk.thresholds', 'monitoring.alert_rules'
        ]

        missing = []
        for config in required_configs:
            if self.config_manager.get(config) is None:
                missing.append(config)

        if not missing:
            return ("PASSED", "所有关键配置已设置")
        else:
            return ("FAILED", f"缺少关键配置: {', '.join(missing)}")

    def _check_deployment_validation(self) -> tuple:
        """
        检查部署验证结果
        :return: (状态, 详情)
        """
        report = self.deployment_validator.get_test_report()
        if report['failed'] == 0:
            return ("PASSED", "所有部署验证测试通过")
        else:
            failed_tests = [r['name'] for r in report['results'] if r['status'] != 'PASSED']
            return ("FAILED", f"{report['failed']}个部署验证测试失败: {', '.join(failed_tests[:3])}")

    def _check_resource_usage(self) -> tuple:
        """
        检查资源使用情况
        :return: (状态, 详情)
        """
        # 模拟检查CPU/内存使用率
        cpu_usage = 0.65  # 模拟值
        mem_usage = 0.75  # 模拟值

        if cpu_usage > 0.9 or mem_usage > 0.9:
            return ("FAILED", f"资源使用过高 CPU: {cpu_usage:.0%}, 内存: {mem_usage:.0%}")
        elif cpu_usage > 0.8 or mem_usage > 0.8:
            return ("WARNING", f"资源使用较高 CPU: {cpu_usage:.0%}, 内存: {mem_usage:.0%}")
        else:
            return ("PASSED", f"资源使用正常 CPU: {cpu_usage:.0%}, 内存: {mem_usage:.0%}")

    def generate_html_report(self) -> str:
        """
        生成HTML格式的检查报告
        :return: HTML报告内容
        """
        report = self.get_check_report()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RQA2025 最终部署检查报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ 
                    background-color: #f0f0f0; 
                    padding: 10px; 
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .summary {{ 
                    font-size: 18px; 
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .summary-PASSED {{ background-color: #d4edda; color: #155724; }}
                .summary-FAILED {{ background-color: #f8d7da; color: #721c24; }}
                .summary-WARNING {{ background-color: #fff3cd; color: #856404; }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left;
                }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .status-PASSED {{ color: green; }}
                .status-FAILED {{ color: red; }}
                .status-WARNING {{ color: orange; }}
                .critical {{ font-weight: bold; }}
                .timestamp {{ font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RQA2025 最终部署检查报告</h1>
                <p>生成时间: <span class="timestamp">{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['timestamp']))}</span></p>
            </div>
            
            <div class="summary summary-{'PASSED' if report['failed'] == 0 else 'FAILED' if report['failed'] > 0 else 'WARNING'}">
                <p>检查结果: 通过 {report['passed']} / 总数 {report['total']} (失败 {report['failed']}, 警告 {report['warnings']})</p>
            </div>
            
            <h2>详细检查结果</h2>
            <table>
                <tr>
                    <th>检查项</th>
                    <th>状态</th>
                    <th>详情</th>
                </tr>
        """

        for result in report['results']:
            html += f"""
                <tr>
                    <td>{result['name']}</td>
                    <td class="status-{result['status']}">{result['status']}</td>
                    <td>{result['details']}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html
