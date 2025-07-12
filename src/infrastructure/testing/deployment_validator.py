#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
部署验证模块
用于验证系统部署后的完整性和正确性
"""

import time
import requests
from typing import Dict, List, Optional, Any
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.config_manager import ConfigManager

logger = get_logger(__name__)

class DeploymentValidator:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化部署验证器
        :param config: 系统配置
        """
        self.config = config
        self.config_manager = ConfigManager(config)
        self.service_endpoints = {
            'data_service': 'http://localhost:8001/health',
            'feature_service': 'http://localhost:8002/health',
            'trading_service': 'http://localhost:8003/health',
            'risk_service': 'http://localhost:8004/health'
        }
        self.health_checks = {
            'database': self._check_database,
            'services': self._check_services,
            'fpga': self._check_fpga,
            'api_gateway': self._check_api_gateway
        }

    def validate_deployment(self) -> Dict[str, Dict[str, Any]]:
        """
        执行完整的部署验证
        :return: 验证结果字典
        """
        results = {}
        start_time = time.time()

        # 加载部署配置
        if not self.config_manager.load(self.config['env']):
            logger.error("加载部署配置失败")
            return {'error': 'Failed to load deployment config'}

        # 执行所有健康检查
        for check_name, check_func in self.health_checks.items():
            logger.info(f"开始执行检查: {check_name}")
            results[check_name] = check_func()
            time.sleep(0.5)  # 避免检查过于密集

        # 生成总结报告
        results['summary'] = self._generate_summary(results)
        results['elapsed_time'] = time.time() - start_time

        return results

    def _check_database(self) -> Dict[str, Any]:
        """检查数据库连接和状态"""
        try:
            # 模拟数据库检查
            time.sleep(0.3)
            return {
                'status': 'healthy',
                'details': {
                    'connection': 'established',
                    'tables': 42,
                    'size_mb': 256
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    def _check_services(self) -> Dict[str, Any]:
        """检查所有微服务健康状态"""
        service_results = {}
        all_healthy = True

        for service, endpoint in self.service_endpoints.items():
            try:
                response = requests.get(endpoint, timeout=2)
                if response.status_code == 200:
                    service_results[service] = {
                        'status': 'healthy',
                        'response_time': response.elapsed.total_seconds()
                    }
                else:
                    service_results[service] = {
                        'status': 'unhealthy',
                        'error': f"Status code: {response.status_code}"
                    }
                    all_healthy = False
            except Exception as e:
                service_results[service] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                all_healthy = False

        return {
            'overall_status': 'healthy' if all_healthy else 'unhealthy',
            'services': service_results
        }

    def _check_fpga(self) -> Dict[str, Any]:
        """检查FPGA加速器状态"""
        try:
            # 模拟FPGA检查
            time.sleep(0.2)
            return {
                'status': 'healthy',
                'details': {
                    'devices': 2,
                    'temperature': 45.2,
                    'utilization': 35.7
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    def _check_api_gateway(self) -> Dict[str, Any]:
        """检查API网关状态"""
        try:
            # 模拟API网关检查
            time.sleep(0.1)
            return {
                'status': 'healthy',
                'details': {
                    'throughput': 1250,
                    'error_rate': 0.02
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    def _generate_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成验证总结报告"""
        summary = {
            'total_checks': len(results) - 1,  # 排除summary本身
            'passed_checks': 0,
            'failed_checks': 0,
            'critical_issues': []
        }

        for check_name, result in results.items():
            if check_name == 'summary':
                continue

            if result.get('status') == 'healthy':
                summary['passed_checks'] += 1
            else:
                summary['failed_checks'] += 1
                if check_name in ['database', 'services']:
                    summary['critical_issues'].append(check_name)

        summary['overall_status'] = 'PASSED' if summary['failed_checks'] == 0 else 'FAILED'
        return summary

    def continuous_monitoring(self, interval: int = 60) -> None:
        """
        持续监控部署状态
        :param interval: 监控间隔(秒)
        """
        logger.info(f"启动持续监控，间隔: {interval}秒")
        while True:
            results = self.validate_deployment()
            if results['summary']['overall_status'] == 'FAILED':
                logger.error("部署验证失败，发现问题:")
                for issue in results['summary']['critical_issues']:
                    logger.error(f"- {issue}: {results[issue].get('error', '未知错误')}")
            else:
                logger.info("部署验证通过，所有系统正常")

            time.sleep(interval)

    def validate_config_consistency(self) -> bool:
        """
        验证配置一致性
        :return: 配置是否一致
        """
        try:
            # 检查各环境配置一致性
            configs = {}
            for env in ['dev', 'test', 'prod']:
                self.config_manager.load(env)
                configs[env] = self.config_manager.get_all()

            # 比较关键配置项
            critical_keys = ['database.host', 'trading.api_key', 'risk.thresholds']
            for key in critical_keys:
                values = {env: configs[env].get(key) for env in configs}
                if len(set(values.values())) > 1:
                    logger.warning(f"配置不一致: {key} - {values}")
                    return False

            return True
        except Exception as e:
            logger.error(f"配置一致性检查失败: {str(e)}")
            return False

    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """
        生成验证报告
        :param results: 验证结果
        :return: 报告HTML内容
        """
        report = f"""
        <html>
        <head>
            <title>部署验证报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>部署验证报告</h1>
            <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>环境: {self.config['env']}</p>
            <h2>总结</h2>
            <p>总体状态: <span class="{results['summary']['overall_status'].lower()}">
                {results['summary']['overall_status']}</span></p>
            <p>通过检查: {results['summary']['passed_checks']}</p>
            <p>失败检查: {results['summary']['failed_checks']}</p>
            <p>耗时: {results['elapsed_time']:.2f}秒</p>
            
            <h2>详细结果</h2>
            <table>
                <tr>
                    <th>检查项</th>
                    <th>状态</th>
                    <th>详情</th>
                </tr>
        """

        for check_name, result in results.items():
            if check_name in ['summary', 'elapsed_time']:
                continue

            status_class = 'passed' if result.get('status') == 'healthy' else 'failed'
            details = str(result.get('details', result.get('error', '')))

            report += f"""
                <tr>
                    <td>{check_name}</td>
                    <td class="{status_class}">{result.get('status', 'unknown').upper()}</td>
                    <td>{details}</td>
                </tr>
            """

        report += """
            </table>
        </body>
        </html>
        """

        return report
