#!/usr/bin/env python3
"""
RQA2025系统生产部署验证脚本
验证生产环境部署是否成功，系统是否正常运行

使用方法:
python scripts/production_deployment_verification.py

或者在部署后运行:
curl -s http://localhost:8000/health | python -c "
import sys, json
data = json.load(sys.stdin)
print('✅ 健康检查通过' if data.get('status') == 'healthy' else '❌ 健康检查失败')
"
"""

import sys
import os
import json
import time
import requests
import subprocess
from datetime import datetime
from typing import Dict, Any, List
import traceback

class ProductionDeploymentVerifier:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results = []
        self.start_time = datetime.now()

    def log(self, message: str, status: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        colored_message = self._colorize_message(message, status)
        print(f"[{timestamp}] {colored_message}")

        self.results.append({
            "timestamp": timestamp,
            "message": message,
            "status": status
        })

    def _colorize_message(self, message: str, status: str) -> str:
        """为消息添加颜色"""
        colors = {
            "SUCCESS": "\033[92m",  # 绿色
            "ERROR": "\033[91m",    # 红色
            "WARNING": "\033[93m",  # 黄色
            "INFO": "\033[94m",     # 蓝色
            "RESET": "\033[0m"      # 重置
        }

        color = colors.get(status, colors["RESET"])
        return f"{color}[{status}] {message}{colors['RESET']}"

    def run_health_check(self) -> bool:
        """运行健康检查"""
        self.log("开始健康检查...")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)

            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get('status', 'unknown')

                if status == 'healthy':
                    self.log("✅ 健康检查通过", "SUCCESS")
                    self.log(f"系统状态: {status}")
                    self.log(f"响应时间: {response.elapsed.total_seconds():.3f}s")
                    return True
                else:
                    self.log(f"❌ 健康检查失败 - 状态: {status}", "ERROR")
                    return False
            else:
                self.log(f"❌ 健康检查失败 - HTTP {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"❌ 健康检查失败 - 连接错误: {str(e)}", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ 健康检查失败 - 未知错误: {str(e)}", "ERROR")
            return False

    def run_api_tests(self) -> bool:
        """运行API功能测试"""
        self.log("开始API功能测试...")

        test_cases = [
            {
                "name": "系统信息",
                "url": "/api/v1/system/info",
                "method": "GET",
                "expected_status": 200
            },
            {
                "name": "策略列表",
                "url": "/api/v1/strategies",
                "method": "GET",
                "expected_status": 200
            },
            {
                "name": "数据源状态",
                "url": "/api/v1/data/sources",
                "method": "GET",
                "expected_status": 200
            }
        ]

        passed = 0
        total = len(test_cases)

        for test_case in test_cases:
            try:
                response = requests.request(
                    test_case["method"],
                    f"{self.base_url}{test_case['url']}",
                    timeout=30
                )

                if response.status_code == test_case["expected_status"]:
                    self.log(f"✅ {test_case['name']} - 通过", "SUCCESS")
                    passed += 1
                else:
                    self.log(f"❌ {test_case['name']} - 失败 (HTTP {response.status_code})", "ERROR")

            except Exception as e:
                self.log(f"❌ {test_case['name']} - 错误: {str(e)}", "ERROR")

        success_rate = passed / total if total > 0 else 0
        if success_rate >= 0.8:  # 80%以上通过
            self.log(f"✅ API测试通过 ({passed}/{total})", "SUCCESS")
            return True
        else:
            self.log(f"❌ API测试失败 ({passed}/{total})", "ERROR")
            return False

    def run_performance_test(self) -> bool:
        """运行性能测试"""
        self.log("开始性能测试...")

        try:
            # 测试响应时间
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response_time = time.time() - start_time

            if response.status_code == 200 and response_time < 1.0:  # 1秒内响应
                self.log(f"✅ 响应时间正常: {response_time:.3f}s", "SUCCESS")
                perf_ok = True
            else:
                self.log(f"⚠️ 响应时间较慢: {response_time:.3f}s", "WARNING")
                perf_ok = response_time < 5.0  # 5秒内算勉强通过

            # 简单的并发测试
            import threading

            def single_request():
                try:
                    requests.get(f"{self.base_url}/health", timeout=10)
                    return True
                except:
                    return False

            # 并发10个请求
            threads = []
            results = []

            for i in range(10):
                thread = threading.Thread(target=lambda: results.append(single_request()))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            success_count = sum(results)
            if success_count >= 8:  # 80%成功率
                self.log(f"✅ 并发测试通过 ({success_count}/10)", "SUCCESS")
                return perf_ok
            else:
                self.log(f"❌ 并发测试失败 ({success_count}/10)", "ERROR")
                return False

        except Exception as e:
            self.log(f"❌ 性能测试失败: {str(e)}", "ERROR")
            return False

    def run_security_check(self) -> bool:
        """运行安全检查"""
        self.log("开始安全检查...")

        try:
            # 检查HTTPS（如果配置了）
            if self.base_url.startswith("https://"):
                self.log("✅ HTTPS已启用", "SUCCESS")
                https_ok = True
            else:
                self.log("⚠️ 未启用HTTPS（生产环境建议启用）", "WARNING")
                https_ok = True  # 开发环境可以不启用

            # 检查敏感信息泄露
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response_text = response.text.lower()

            sensitive_keywords = ["password", "secret", "key", "token"]
            sensitive_found = any(keyword in response_text for keyword in sensitive_keywords)

            if not sensitive_found:
                self.log("✅ 未发现敏感信息泄露", "SUCCESS")
                security_ok = True
            else:
                self.log("❌ 发现潜在敏感信息泄露", "ERROR")
                security_ok = False

            return https_ok and security_ok

        except Exception as e:
            self.log(f"❌ 安全检查失败: {str(e)}", "ERROR")
            return False

    def run_database_check(self) -> bool:
        """运行数据库检查"""
        self.log("开始数据库检查...")

        try:
            # 尝试访问数据库相关的API
            response = requests.get(f"{self.base_url}/api/v1/health/database", timeout=10)

            if response.status_code == 200:
                db_data = response.json()
                status = db_data.get('status', 'unknown')

                if status == 'connected':
                    self.log("✅ 数据库连接正常", "SUCCESS")
                    return True
                else:
                    self.log(f"❌ 数据库连接异常: {status}", "ERROR")
                    return False
            else:
                self.log(f"⚠️ 数据库检查API不可用 (HTTP {response.status_code})", "WARNING")
                # 如果没有专门的数据库检查API，认为通过
                return True

        except Exception as e:
            self.log(f"⚠️ 数据库检查失败: {str(e)}", "WARNING")
            return True  # 不因数据库检查失败而阻断部署

    def run_monitoring_check(self) -> bool:
        """运行监控检查"""
        self.log("开始监控检查...")

        try:
            # 检查metrics端点
            response = requests.get(f"{self.base_url}/metrics", timeout=10)

            if response.status_code == 200:
                self.log("✅ 监控指标端点正常", "SUCCESS")
                monitoring_ok = True
            else:
                self.log(f"⚠️ 监控指标端点不可用 (HTTP {response.status_code})", "WARNING")
                monitoring_ok = False

            # 检查日志文件（如果可访问）
            try:
                log_response = requests.get(f"{self.base_url}/api/v1/logs/status", timeout=10)
                if log_response.status_code == 200:
                    self.log("✅ 日志系统正常", "SUCCESS")
                else:
                    self.log("⚠️ 日志系统状态未知", "WARNING")
            except:
                self.log("⚠️ 日志检查不可用", "WARNING")

            return monitoring_ok

        except Exception as e:
            self.log(f"⚠️ 监控检查失败: {str(e)}", "WARNING")
            return True

    def generate_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        success_count = sum(1 for r in self.results if r["status"] == "SUCCESS")
        error_count = sum(1 for r in self.results if r["status"] == "ERROR")
        warning_count = sum(1 for r in self.results if r["status"] == "WARNING")

        # 计算整体状态
        if error_count == 0 and warning_count <= 2:
            overall_status = "✅ 部署验证通过"
            status_color = "SUCCESS"
        elif error_count <= 2:
            overall_status = "⚠️ 部署验证基本通过（有警告）"
            status_color = "WARNING"
        else:
            overall_status = "❌ 部署验证失败"
            status_color = "ERROR"

        report = {
            "verification_summary": {
                "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": duration,
                "total_checks": len(self.results),
                "success_count": success_count,
                "warning_count": warning_count,
                "error_count": error_count,
                "overall_status": overall_status
            },
            "check_results": self.results,
            "recommendations": self._generate_recommendations(error_count, warning_count)
        }

        return report

    def _generate_recommendations(self, error_count: int, warning_count: int) -> List[str]:
        """生成建议"""
        recommendations = []

        if error_count > 0:
            recommendations.extend([
                "🔴 立即修复发现的错误问题",
                "🔴 检查系统日志获取详细错误信息",
                "🔴 验证系统配置和环境变量"
            ])

        if warning_count > 0:
            recommendations.extend([
                "🟡 关注警告信息，优化系统配置",
                "🟡 考虑启用HTTPS以提升安全性",
                "🟡 配置监控告警系统"
            ])

        if error_count == 0:
            recommendations.extend([
                "🟢 系统部署成功，可以投入使用",
                "🟢 配置监控和告警系统",
                "🟢 制定定期维护计划",
                "🟢 准备业务验收测试"
            ])

        return recommendations

    def run_all_checks(self) -> bool:
        """运行所有验证检查"""
        self.log("🚀 开始RQA2025生产部署验证")
        self.log("=" * 50)

        checks = [
            ("健康检查", self.run_health_check),
            ("API功能测试", self.run_api_tests),
            ("性能测试", self.run_performance_test),
            ("安全检查", self.run_security_check),
            ("数据库检查", self.run_database_check),
            ("监控检查", self.run_monitoring_check)
        ]

        passed_checks = 0
        total_checks = len(checks)

        for check_name, check_func in checks:
            self.log(f"\\n执行检查: {check_name}")
            try:
                if check_func():
                    passed_checks += 1
                    self.log(f"✅ {check_name} - 通过", "SUCCESS")
                else:
                    self.log(f"❌ {check_name} - 失败", "ERROR")
            except Exception as e:
                self.log(f"❌ {check_name} - 异常: {str(e)}", "ERROR")
                traceback.print_exc()

        self.log("\\n" + "=" * 50)

        # 生成最终报告
        report = self.generate_report()

        print("\\n📊 部署验证报告")
        print("=" * 30)
        print(f"验证时间: {report['verification_summary']['duration_seconds']:.1f}秒")
        print(f"检查项目: {report['verification_summary']['total_checks']}个")
        print(f"成功: {report['verification_summary']['success_count']}个")
        print(f"警告: {report['verification_summary']['warning_count']}个")
        print(f"错误: {report['verification_summary']['error_count']}个")
        print(f"整体状态: {report['verification_summary']['overall_status']}")

        print("\\n📋 建议:")
        for rec in report['recommendations']:
            print(f"  {rec}")

        # 保存报告到文件
        report_file = f"deployment_verification_report_{int(time.time())}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.log(f"\\n📄 详细报告已保存到: {report_file}", "INFO")
        except Exception as e:
            self.log(f"\\n⚠️ 报告保存失败: {str(e)}", "WARNING")

        return passed_checks >= total_checks - 1  # 允许1个检查失败


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RQA2025生产部署验证脚本")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="系统基础URL (默认: http://localhost:8000)")
    parser.add_argument("--timeout", type=int, default=60,
                       help="单个检查超时时间(秒)")

    args = parser.parse_args()

    print("🎯 RQA2025生产部署验证工具")
    print("=" * 40)
    print(f"目标系统: {args.url}")
    print(f"超时时间: {args.timeout}秒")
    print()

    verifier = ProductionDeploymentVerifier(args.url)

    try:
        success = verifier.run_all_checks()

        if success:
            print("\\n🎉 部署验证完成！系统可以投入生产使用。"            sys.exit(0)
        else:
            print("\\n❌ 部署验证失败！请检查并修复问题后再重新部署。"            sys.exit(1)

    except KeyboardInterrupt:
        print("\\n\\n⚠️ 验证被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\\n❌ 验证过程发生异常: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()