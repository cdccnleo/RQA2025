#!/usr/bin/env python3
"""
RQA2025 API安全测试脚本

测试API的安全性，包括SQL注入、XSS、身份验证等安全问题
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
import aiohttp
import json
from typing import Dict, List, Any
import requests
from concurrent.futures import ThreadPoolExecutor
import time


class APISecurityTester:
    """API安全测试器"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def test_sql_injection_protection(self) -> Dict[str, Any]:
        """测试SQL注入防护"""
        print("🧪 测试SQL注入防护...")

        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT password FROM users --",
            "' OR 1=1; --",
            "admin' --",
            "1' OR '1' = '1",
            "' OR '' = '",
            "'; EXEC xp_cmdshell 'dir'; --"
        ]

        results = []

        for payload in sql_payloads:
            try:
                # 测试登录接口
                response = self.session.post(
                    f"{self.base_url}/api/auth/login",
                    json={
                        "username": payload,
                        "password": "test123"
                    },
                    timeout=5
                )

                # 应该被拒绝或返回错误
                if response.status_code in [400, 401, 403]:
                    result = "✅ BLOCKED"
                elif response.status_code == 200:
                    result = "❌ VULNERABLE"
                else:
                    result = f"⚠️ UNEXPECTED ({response.status_code})"

                results.append({
                    'payload': payload,
                    'result': result,
                    'status_code': response.status_code
                })

            except Exception as e:
                results.append({
                    'payload': payload,
                    'result': f"❌ ERROR: {str(e)}",
                    'status_code': None
                })

        passed = sum(1 for r in results if "BLOCKED" in r['result'])
        total = len(results)

        return {
            'test_name': 'SQL Injection Protection',
            'passed': passed,
            'total': total,
            'success_rate': passed / total if total > 0 else 0,
            'results': results
        }

    def test_xss_protection(self) -> Dict[str, Any]:
        """测试XSS防护"""
        print("🧪 测试XSS防护...")

        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "'><script>alert('xss')</script>",
            "<iframe src='javascript:alert(\"xss\")'>",
            "<body onload=alert('xss')>",
            "<input onfocus=alert('xss') autofocus>"
        ]

        results = []

        for payload in xss_payloads:
            try:
                # 测试用户注册接口
                response = self.session.post(
                    f"{self.base_url}/api/user/register",
                    json={
                        "username": f"test_user_{hash(payload) % 1000}",
                        "email": f"test_{hash(payload) % 1000}@example.com",
                        "password": "Test123456!",
                        "full_name": payload
                    },
                    timeout=5
                )

                # 检查响应是否包含未转义的脚本
                response_text = response.text.lower()
                has_script = any(tag in response_text for tag in [
                    '<script', 'javascript:', 'onerror=', 'onload=', 'onfocus='
                ])

                if has_script and response.status_code == 200:
                    result = "❌ VULNERABLE"
                elif response.status_code in [200, 201, 400]:
                    result = "✅ SAFE"
                else:
                    result = f"⚠️ UNEXPECTED ({response.status_code})"

                results.append({
                    'payload': payload,
                    'result': result,
                    'status_code': response.status_code,
                    'has_script_in_response': has_script
                })

            except Exception as e:
                results.append({
                    'payload': payload,
                    'result': f"❌ ERROR: {str(e)}",
                    'status_code': None,
                    'has_script_in_response': False
                })

        passed = sum(1 for r in results if "SAFE" in r['result'])
        total = len(results)

        return {
            'test_name': 'XSS Protection',
            'passed': passed,
            'total': total,
            'success_rate': passed / total if total > 0 else 0,
            'results': results
        }

    def test_brute_force_protection(self) -> Dict[str, Any]:
        """测试暴力破解防护"""
        print("🧪 测试暴力破解防护...")

        # 模拟暴力破解
        failed_attempts = 0
        blocked = False

        for i in range(10):  # 尝试10次
            try:
                response = self.session.post(
                    f"{self.base_url}/api/auth/login",
                    json={
                        "username": "admin",
                        "password": f"wrong_password_{i}"
                    },
                    timeout=5
                )

                if response.status_code == 429:  # Too Many Requests
                    blocked = True
                    break
                elif response.status_code in [401, 403]:
                    failed_attempts += 1
                else:
                    # 意外的状态码
                    pass

            except Exception as e:
                print(f"暴力破解测试第{i+1}次失败: {e}")
                continue

            time.sleep(0.1)  # 小延迟避免过快请求

        if blocked:
            result = "✅ PROTECTED"
        elif failed_attempts >= 5:
            result = "⚠️ NO PROTECTION DETECTED"
        else:
            result = "❓ INSUFFICIENT ATTEMPTS"

        return {
            'test_name': 'Brute Force Protection',
            'result': result,
            'failed_attempts': failed_attempts,
            'blocked': blocked
        }

    def test_authentication_security(self) -> Dict[str, Any]:
        """测试身份验证安全"""
        print("🧪 测试身份验证安全...")

        results = []

        # 测试1: 无效令牌
        try:
            response = self.session.get(
                f"{self.base_url}/api/user/profile",
                headers={"Authorization": "Bearer invalid_token"},
                timeout=5
            )
            results.append({
                'test': 'Invalid Token',
                'result': "✅ BLOCKED" if response.status_code == 401 else "❌ ALLOWED",
                'status_code': response.status_code
            })
        except Exception as e:
            results.append({
                'test': 'Invalid Token',
                'result': f"❌ ERROR: {str(e)}",
                'status_code': None
            })

        # 测试2: 缺失认证头
        try:
            response = self.session.get(
                f"{self.base_url}/api/user/profile",
                timeout=5
            )
            results.append({
                'test': 'Missing Auth Header',
                'result': "✅ BLOCKED" if response.status_code == 401 else "❌ ALLOWED",
                'status_code': response.status_code
            })
        except Exception as e:
            results.append({
                'test': 'Missing Auth Header',
                'result': f"❌ ERROR: {str(e)}",
                'status_code': None
            })

        # 测试3: 尝试访问受保护资源
        try:
            response = self.session.get(
                f"{self.base_url}/api/trading/orders",
                timeout=5
            )
            results.append({
                'test': 'Protected Resource Access',
                'result': "✅ BLOCKED" if response.status_code == 401 else "❌ ALLOWED",
                'status_code': response.status_code
            })
        except Exception as e:
            results.append({
                'test': 'Protected Resource Access',
                'result': f"❌ ERROR: {str(e)}",
                'status_code': None
            })

        passed = sum(1 for r in results if "BLOCKED" in r['result'])
        total = len(results)

        return {
            'test_name': 'Authentication Security',
            'passed': passed,
            'total': total,
            'success_rate': passed / total if total > 0 else 0,
            'results': results
        }

    def test_rate_limiting(self) -> Dict[str, Any]:
        """测试速率限制"""
        print("🧪 测试速率限制...")

        # 快速发送多个请求
        responses = []
        for i in range(20):
            try:
                response = self.session.get(
                    f"{self.base_url}/api/health",
                    timeout=2
                )
                responses.append(response.status_code)
            except Exception as e:
                responses.append(f"ERROR: {str(e)}")

            time.sleep(0.05)  # 50ms间隔

        # 检查是否有限制
        rate_limited = any(
            status == 429 or (isinstance(status, str) and "timeout" in status.lower())
            for status in responses
        )

        success_count = sum(1 for r in responses if r == 200)
        rate_limit_count = sum(1 for r in responses if r == 429)

        if rate_limit_count > 0:
            result = "✅ RATE LIMITED"
        elif success_count < len(responses) * 0.8:  # 少于80%成功
            result = "⚠️ POSSIBLE LIMITING"
        else:
            result = "❌ NO LIMITING DETECTED"

        return {
            'test_name': 'Rate Limiting',
            'result': result,
            'total_requests': len(responses),
            'successful_requests': success_count,
            'rate_limited_requests': rate_limit_count,
            'responses': responses[:10]  # 只显示前10个
        }

    def test_https_enforcement(self) -> Dict[str, Any]:
        """测试HTTPS强制使用"""
        print("🧪 测试HTTPS强制使用...")

        # 在开发环境中通常不强制HTTPS，但应该有配置
        try:
            response = self.session.get(
                f"http://{self.base_url.replace('http://', '').replace('https://', '')}/api/health",
                timeout=5,
                allow_redirects=False
            )

            if response.status_code in [301, 302] and 'https://' in response.headers.get('Location', ''):
                result = "✅ HTTPS ENFORCED"
            elif self.base_url.startswith('https://'):
                result = "✅ ALREADY HTTPS"
            else:
                result = "⚠️ NO HTTPS ENFORCEMENT"

        except Exception as e:
            result = f"❌ ERROR: {str(e)}"

        return {
            'test_name': 'HTTPS Enforcement',
            'result': result
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有安全测试"""
        print("🚀 开始API安全测试套件...")

        tests = [
            self.test_sql_injection_protection,
            self.test_xss_protection,
            self.test_brute_force_protection,
            self.test_authentication_security,
            self.test_rate_limiting,
            self.test_https_enforcement
        ]

        results = {}
        for test_func in tests:
            try:
                test_result = test_func()
                results[test_result['test_name']] = test_result
                print(f"✅ {test_result['test_name']}: 完成")
            except Exception as e:
                print(f"❌ {test_func.__name__}: 失败 - {e}")
                results[test_func.__name__] = {
                    'test_name': test_func.__name__,
                    'error': str(e)
                }

        # 生成汇总报告
        summary = {
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results.values() if 'success_rate' in r and r['success_rate'] == 1.0),
            'failed_tests': sum(1 for r in results.values() if 'success_rate' in r and r['success_rate'] < 1.0),
            'error_tests': sum(1 for r in results.values() if 'error' in r),
            'overall_score': self._calculate_overall_score(results)
        }

        return {
            'summary': summary,
            'detailed_results': results,
            'timestamp': time.time(),
            'target_url': self.base_url
        }

    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """计算总体安全评分"""
        scores = []

        for test_result in results.values():
            if 'success_rate' in test_result:
                scores.append(test_result['success_rate'])
            elif 'result' in test_result:
                # 简单的二进制评分
                if "✅" in test_result['result']:
                    scores.append(1.0)
                elif "⚠️" in test_result['result']:
                    scores.append(0.5)
                else:
                    scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025 API安全测试')
    parser.add_argument('--url', default='http://localhost:8001',
                       help='目标API URL')
    parser.add_argument('--output', default='api_security_test_results.json',
                       help='输出文件')

    args = parser.parse_args()

    print(f"🎯 目标URL: {args.url}")
    print(f"📄 输出文件: {args.output}")

    tester = APISecurityTester(args.url)
    results = tester.run_all_tests()

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 输出汇总
    summary = results['summary']
    score = summary['overall_score']

    print("\n" + "="*60)
    print("📊 API安全测试结果汇总")
    print("="*60)
    print(f"总测试数: {summary['total_tests']}")
    print(f"通过测试: {summary['passed_tests']}")
    print(f"失败测试: {summary['failed_tests']}")
    print(f"错误测试: {summary['error_tests']}")
    print(f"总体评分: {score:.1f}")
    if score >= 0.9:
        print("🎉 安全状况优秀！")
    elif score >= 0.7:
        print("✅ 安全状况良好")
    elif score >= 0.5:
        print("⚠️ 安全状况一般，需要改进")
    else:
        print("❌ 安全状况较差，需要紧急修复")

    print(f"\n📄 详细报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
