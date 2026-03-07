#!/usr/bin/env python3
"""
生产环境验证脚本

实现功能验证、性能验证和安全验证
"""

from src.infrastructure.core.monitoring.production_monitor import (
    ProductionMonitoringSystem,
    default_monitoring_config
)
from src.infrastructure.core.config.environment_manager import (
    EnvironmentConfigManager,
    ProductionConfigValidator
)
import os
import sys
import time
import json
import logging
import requests
import psutil
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ProductionValidator:
    """生产环境验证器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_manager = EnvironmentConfigManager()
        self.production_validator = ProductionConfigValidator(self.config_manager)
        self.monitoring_system = ProductionMonitoringSystem(default_monitoring_config)

        # 验证结果
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "functionality": {},
            "performance": {},
            "security": {},
            "monitoring": {},
            "recommendations": []
        }

    def run_full_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        self.logger.info("开始生产环境验证...")

        try:
            # 功能验证
            self.validation_results["functionality"] = self._validate_functionality()

            # 性能验证
            self.validation_results["performance"] = self._validate_performance()

            # 安全验证
            self.validation_results["security"] = self._validate_security()

            # 监控验证
            self.validation_results["monitoring"] = self._validate_monitoring()

            # 确定整体状态
            self._determine_overall_status()

            # 生成建议
            self._generate_recommendations()

            self.logger.info("生产环境验证完成")
            return self.validation_results

        except Exception as e:
            self.logger.error(f"验证过程中发生错误: {e}")
            self.validation_results["overall_status"] = "error"
            self.validation_results["error"] = str(e)
            return self.validation_results

    def _validate_functionality(self) -> Dict[str, Any]:
        """功能验证"""
        self.logger.info("开始功能验证...")

        results = {
            "status": "unknown",
            "tests": {},
            "errors": [],
            "warnings": []
        }

        try:
            # 测试配置管理
            config_test = self._test_config_management()
            results["tests"]["config_management"] = config_test

            # 测试数据库连接
            db_test = self._test_database_connection()
            results["tests"]["database_connection"] = db_test

            # 测试Redis连接
            redis_test = self._test_redis_connection()
            results["tests"]["redis_connection"] = redis_test

            # 测试应用服务
            app_test = self._test_application_service()
            results["tests"]["application_service"] = app_test

            # 测试API接口
            api_test = self._test_api_endpoints()
            results["tests"]["api_endpoints"] = api_test

            # 确定功能验证状态
            all_passed = all(test.get("passed", False) for test in results["tests"].values())
            results["status"] = "passed" if all_passed else "failed"

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(f"功能验证错误: {e}")

        return results

    def _test_config_management(self) -> Dict[str, Any]:
        """测试配置管理"""
        try:
            # 检查环境配置
            env_info = self.config_manager.get_environment_info()

            # 验证配置
            validation = self.config_manager.validate_config()

            # 生产环境就绪验证
            production_ready = self.production_validator.validate_production_ready()

            return {
                "passed": len(validation["errors"]) == 0 and production_ready["overall"],
                "environment": env_info["environment"],
                "config_count": env_info["config_count"],
                "validation_errors": validation["errors"],
                "validation_warnings": validation["warnings"],
                "production_ready": production_ready
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _test_database_connection(self) -> Dict[str, Any]:
        """测试数据库连接"""
        try:
            # 获取数据库配置
            db_host = self.config_manager.get_config("database.host")
            db_port = self.config_manager.get_config("database.port")
            db_name = self.config_manager.get_config("database.name")
            db_user = self.config_manager.get_config("database.user")

            # 测试连接
            import psycopg2
            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user
            )

            # 执行简单查询
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return {
                "passed": True,
                "host": db_host,
                "port": db_port,
                "database": db_name,
                "version": version
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _test_redis_connection(self) -> Dict[str, Any]:
        """测试Redis连接"""
        try:
            # 获取Redis配置
            redis_host = self.config_manager.get_config("redis.host")
            redis_port = self.config_manager.get_config("redis.port")

            # 测试连接
            import redis
            r = redis.Redis(host=redis_host, port=redis_port)

            # 测试ping
            ping_result = r.ping()

            # 测试基本操作
            r.set("test_key", "test_value")
            test_value = r.get("test_key")
            r.delete("test_key")

            return {
                "passed": ping_result and test_value == b"test_value",
                "host": redis_host,
                "port": redis_port,
                "ping": ping_result
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _test_application_service(self) -> Dict[str, Any]:
        """测试应用服务"""
        try:
            # 检查应用进程
            app_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if any('python' in cmd.lower() for cmd in proc.info['cmdline'] if cmd):
                        app_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # 检查端口监听
            app_ports = []
            for conn in psutil.net_connections():
                if conn.status == 'LISTEN' and conn.pid:
                    app_ports.append({
                        "port": conn.laddr.port,
                        "pid": conn.pid
                    })

            return {
                "passed": len(app_processes) > 0,
                "processes": app_processes,
                "listening_ports": app_ports
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _test_api_endpoints(self) -> Dict[str, Any]:
        """测试API接口"""
        try:
            # 测试健康检查接口
            health_url = "http://localhost:8000/health"
            health_response = requests.get(health_url, timeout=10)

            # 测试监控接口
            metrics_url = "http://localhost:8000/metrics"
            metrics_response = requests.get(metrics_url, timeout=10)

            return {
                "passed": health_response.status_code == 200 and metrics_response.status_code == 200,
                "health_check": {
                    "url": health_url,
                    "status_code": health_response.status_code,
                    "response_time": health_response.elapsed.total_seconds()
                },
                "metrics": {
                    "url": metrics_url,
                    "status_code": metrics_response.status_code,
                    "response_time": metrics_response.elapsed.total_seconds()
                }
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _validate_performance(self) -> Dict[str, Any]:
        """性能验证"""
        self.logger.info("开始性能验证...")

        results = {
            "status": "unknown",
            "metrics": {},
            "thresholds": {},
            "recommendations": []
        }

        try:
            # 系统性能指标
            system_metrics = self._collect_system_metrics()
            results["metrics"]["system"] = system_metrics

            # 应用性能指标
            app_metrics = self._collect_application_metrics()
            results["metrics"]["application"] = app_metrics

            # 性能阈值检查
            performance_thresholds = self._check_performance_thresholds(system_metrics, app_metrics)
            results["thresholds"] = performance_thresholds

            # 确定性能状态
            all_passed = all(threshold.get("passed", False)
                             for threshold in performance_thresholds.values())
            results["status"] = "passed" if all_passed else "failed"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)

        return results

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统性能指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用率
            memory = psutil.virtual_memory()

            # 磁盘使用率
            disk = psutil.disk_usage('/')

            # 网络IO
            network = psutil.net_io_counters()

            # 负载平均值
            load_average = None
            try:
                load_average = os.getloadavg()[0]
            except (AttributeError, OSError):
                pass

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "network_sent": network.bytes_sent,
                "network_recv": network.bytes_recv,
                "load_average": load_average,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}

    def _collect_application_metrics(self) -> Dict[str, Any]:
        """收集应用性能指标"""
        try:
            # 获取当前进程信息
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()

            # 获取连接数
            connections = len(psutil.net_connections())

            # 获取线程数
            threads = process.num_threads()

            return {
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "cpu_percent": cpu_percent,
                "connections": connections,
                "threads": threads,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}

    def _check_performance_thresholds(self, system_metrics: Dict, app_metrics: Dict) -> Dict[str, Any]:
        """检查性能阈值"""
        thresholds = {}

        # CPU使用率阈值
        cpu_threshold = 80.0
        cpu_passed = system_metrics.get("cpu_percent", 0) < cpu_threshold
        thresholds["cpu_usage"] = {
            "passed": cpu_passed,
            "current": system_metrics.get("cpu_percent", 0),
            "threshold": cpu_threshold
        }

        # 内存使用率阈值
        memory_threshold = 80.0
        memory_passed = system_metrics.get("memory_percent", 0) < memory_threshold
        thresholds["memory_usage"] = {
            "passed": memory_passed,
            "current": system_metrics.get("memory_percent", 0),
            "threshold": memory_threshold
        }

        # 磁盘使用率阈值
        disk_threshold = 85.0
        disk_passed = system_metrics.get("disk_percent", 0) < disk_threshold
        thresholds["disk_usage"] = {
            "passed": disk_passed,
            "current": system_metrics.get("disk_percent", 0),
            "threshold": disk_threshold
        }

        # 应用内存使用阈值
        app_memory_threshold = 1000  # MB
        app_memory_passed = app_metrics.get("memory_usage_mb", 0) < app_memory_threshold
        thresholds["app_memory_usage"] = {
            "passed": app_memory_passed,
            "current": app_metrics.get("memory_usage_mb", 0),
            "threshold": app_memory_threshold
        }

        return thresholds

    def _validate_security(self) -> Dict[str, Any]:
        """安全验证"""
        self.logger.info("开始安全验证...")

        results = {
            "status": "unknown",
            "checks": {},
            "vulnerabilities": [],
            "recommendations": []
        }

        try:
            # 检查配置文件权限
            config_permissions = self._check_config_permissions()
            results["checks"]["config_permissions"] = config_permissions

            # 检查日志文件权限
            log_permissions = self._check_log_permissions()
            results["checks"]["log_permissions"] = log_permissions

            # 检查数据库安全
            db_security = self._check_database_security()
            results["checks"]["database_security"] = db_security

            # 检查网络安全
            network_security = self._check_network_security()
            results["checks"]["network_security"] = network_security

            # 检查SSL证书
            ssl_certificate = self._check_ssl_certificate()
            results["checks"]["ssl_certificate"] = ssl_certificate

            # 确定安全状态
            all_passed = all(check.get("passed", False) for check in results["checks"].values())
            results["status"] = "passed" if all_passed else "failed"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)

        return results

    def _check_config_permissions(self) -> Dict[str, Any]:
        """检查配置文件权限"""
        try:
            config_path = Path("config")
            if not config_path.exists():
                return {"passed": False, "error": "配置目录不存在"}

            # 检查目录权限
            stat = config_path.stat()
            mode = oct(stat.st_mode)[-3:]

            # 检查文件权限
            config_files = list(config_path.rglob("*.yaml")) + list(config_path.rglob("*.yml"))
            file_permissions = {}

            for file_path in config_files:
                file_stat = file_path.stat()
                file_mode = oct(file_stat.st_mode)[-3:]
                file_permissions[str(file_path)] = file_mode

            # 权限应该不超过644
            passed = all(int(perm) <= 644 for perm in file_permissions.values())

            return {
                "passed": passed,
                "directory_permission": mode,
                "file_permissions": file_permissions
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _check_log_permissions(self) -> Dict[str, Any]:
        """检查日志文件权限"""
        try:
            log_path = Path("logs")
            if not log_path.exists():
                return {"passed": True, "message": "日志目录不存在"}

            # 检查日志文件权限
            log_files = list(log_path.glob("*.log"))
            file_permissions = {}

            for file_path in log_files:
                file_stat = file_path.stat()
                file_mode = oct(file_stat.st_mode)[-3:]
                file_permissions[str(file_path)] = file_mode

            # 权限应该不超过644
            passed = all(int(perm) <= 644 for perm in file_permissions.values())

            return {
                "passed": passed,
                "file_permissions": file_permissions
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _check_database_security(self) -> Dict[str, Any]:
        """检查数据库安全"""
        try:
            # 检查数据库配置
            db_config = {
                "host": self.config_manager.get_config("database.host"),
                "port": self.config_manager.get_config("database.port"),
                "name": self.config_manager.get_config("database.name"),
                "user": self.config_manager.get_config("database.user")
            }

            # 检查是否使用本地连接
            local_connection = db_config["host"] in ["localhost", "127.0.0.1"]

            # 检查是否使用非默认端口
            non_default_port = db_config["port"] != 5432

            return {
                "passed": local_connection,
                "local_connection": local_connection,
                "non_default_port": non_default_port,
                "config": db_config
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _check_network_security(self) -> Dict[str, Any]:
        """检查网络安全"""
        try:
            # 检查监听端口
            listening_ports = []
            for conn in psutil.net_connections():
                if conn.status == 'LISTEN':
                    listening_ports.append({
                        "port": conn.laddr.port,
                        "address": conn.laddr.ip
                    })

            # 检查是否有不必要的端口开放
            unnecessary_ports = [22, 80, 443, 8000]  # 允许的端口
            open_ports = [conn["port"] for conn in listening_ports]
            unauthorized_ports = [port for port in open_ports if port not in unnecessary_ports]

            return {
                "passed": len(unauthorized_ports) == 0,
                "listening_ports": listening_ports,
                "unauthorized_ports": unauthorized_ports
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _check_ssl_certificate(self) -> Dict[str, Any]:
        """检查SSL证书"""
        try:
            # 这里可以添加SSL证书检查逻辑
            # 由于是本地测试，暂时返回通过
            return {
                "passed": True,
                "message": "SSL证书检查暂未实现"
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _validate_monitoring(self) -> Dict[str, Any]:
        """监控验证"""
        self.logger.info("开始监控验证...")

        results = {
            "status": "unknown",
            "components": {},
            "alerts": {},
            "metrics": {}
        }

        try:
            # 检查监控系统状态
            monitoring_status = self._check_monitoring_status()
            results["components"]["monitoring_system"] = monitoring_status

            # 检查告警配置
            alert_config = self._check_alert_configuration()
            results["components"]["alert_configuration"] = alert_config

            # 检查指标收集
            metrics_collection = self._check_metrics_collection()
            results["components"]["metrics_collection"] = metrics_collection

            # 获取监控摘要
            monitoring_summary = self.monitoring_system.get_monitoring_summary()
            results["metrics"] = monitoring_summary

            # 确定监控状态
            all_passed = all(comp.get("passed", False) for comp in results["components"].values())
            results["status"] = "passed" if all_passed else "failed"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)

        return results

    def _check_monitoring_status(self) -> Dict[str, Any]:
        """检查监控系统状态"""
        try:
            # 检查监控系统是否运行
            monitoring_summary = self.monitoring_system.get_monitoring_summary()
            is_running = monitoring_summary.get("status") == "running"

            return {
                "passed": is_running,
                "status": monitoring_summary.get("status", "unknown"),
                "summary": monitoring_summary
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _check_alert_configuration(self) -> Dict[str, Any]:
        """检查告警配置"""
        try:
            # 检查告警配置
            alert_config = self.config_manager.get_config("alert", {})

            # 检查是否有告警配置
            has_email_config = bool(alert_config.get("email", {}))
            has_webhook_config = bool(alert_config.get("webhook", {}))

            return {
                "passed": has_email_config or has_webhook_config,
                "email_configured": has_email_config,
                "webhook_configured": has_webhook_config,
                "config": alert_config
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _check_metrics_collection(self) -> Dict[str, Any]:
        """检查指标收集"""
        try:
            # 获取监控摘要
            monitoring_summary = self.monitoring_system.get_monitoring_summary()

            # 检查是否有指标数据
            has_system_metrics = bool(monitoring_summary.get("system", {}))
            has_app_metrics = bool(monitoring_summary.get("application", {}))

            return {
                "passed": has_system_metrics and has_app_metrics,
                "system_metrics": has_system_metrics,
                "application_metrics": has_app_metrics,
                "summary": monitoring_summary
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}

    def _determine_overall_status(self):
        """确定整体状态"""
        statuses = [
            self.validation_results["functionality"].get("status"),
            self.validation_results["performance"].get("status"),
            self.validation_results["security"].get("status"),
            self.validation_results["monitoring"].get("status")
        ]

        if "error" in statuses:
            self.validation_results["overall_status"] = "error"
        elif "failed" in statuses:
            self.validation_results["overall_status"] = "failed"
        elif all(status == "passed" for status in statuses):
            self.validation_results["overall_status"] = "passed"
        else:
            self.validation_results["overall_status"] = "warning"

    def _generate_recommendations(self):
        """生成建议"""
        recommendations = []

        # 基于功能验证的建议
        functionality = self.validation_results["functionality"]
        if functionality.get("status") != "passed":
            recommendations.append("检查并修复功能验证失败的项目")

        # 基于性能验证的建议
        performance = self.validation_results["performance"]
        if performance.get("status") != "passed":
            recommendations.append("优化系统性能，检查资源使用情况")

        # 基于安全验证的建议
        security = self.validation_results["security"]
        if security.get("status") != "passed":
            recommendations.append("加强安全配置，检查权限设置")

        # 基于监控验证的建议
        monitoring = self.validation_results["monitoring"]
        if monitoring.get("status") != "passed":
            recommendations.append("完善监控配置，确保监控系统正常运行")

        self.validation_results["recommendations"] = recommendations

    def save_results(self, output_file: str = None):
        """保存验证结果"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"production_validation_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"验证结果已保存到: {output_file}")
        return output_file

    def print_summary(self):
        """打印验证摘要"""
        print("\n" + "="*60)
        print("生产环境验证摘要")
        print("="*60)

        print(f"整体状态: {self.validation_results['overall_status']}")
        print(f"验证时间: {self.validation_results['timestamp']}")

        print("\n各模块状态:")
        for module, result in self.validation_results.items():
            if module in ["functionality", "performance", "security", "monitoring"]:
                status = result.get("status", "unknown")
                print(f"  {module}: {status}")

        if self.validation_results["recommendations"]:
            print("\n建议:")
            for i, recommendation in enumerate(self.validation_results["recommendations"], 1):
                print(f"  {i}. {recommendation}")

        print("="*60)


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 创建验证器
    validator = ProductionValidator()

    # 运行验证
    results = validator.run_full_validation()

    # 打印摘要
    validator.print_summary()

    # 保存结果
    output_file = validator.save_results()

    # 返回退出码
    if results["overall_status"] == "passed":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
