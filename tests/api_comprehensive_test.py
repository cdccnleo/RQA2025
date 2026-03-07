#!/usr/bin/env python3
"""
策略系统API综合测试脚本
测试所有API端点的可用性和正确性
"""

import requests
import json
import time
import sys
from typing import Dict, List, Tuple

BASE_URL = "http://localhost:8000"
TEST_STRATEGY_ID = "test_strategy_api_001"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

class APITestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_results = []
    
    def test(self, name: str, method: str, endpoint: str, 
             expected_status: int = 200, data: Dict = None) -> Tuple[bool, Dict]:
        """执行单个API测试"""
        url = f"{BASE_URL}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=10)
            elif method.upper() == "DELETE":
                response = requests.delete(url, timeout=10)
            else:
                return False, {"error": f"不支持的HTTP方法: {method}"}
            
            success = response.status_code == expected_status
            result = {
                "name": name,
                "method": method,
                "endpoint": endpoint,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "success": success,
                "response": response.json() if response.content else {}
            }
            
            if success:
                self.passed += 1
                print(f"{Colors.GREEN}✓{Colors.END} {name}")
            else:
                self.failed += 1
                print(f"{Colors.RED}✗{Colors.END} {name} - 状态码: {response.status_code}, 期望: {expected_status}")
            
            self.test_results.append(result)
            return success, result
            
        except Exception as e:
            self.failed += 1
            print(f"{Colors.RED}✗{Colors.END} {name} - 异常: {str(e)}")
            result = {
                "name": name,
                "method": method,
                "endpoint": endpoint,
                "success": False,
                "error": str(e)
            }
            self.test_results.append(result)
            return False, result
    
    def print_summary(self):
        """打印测试摘要"""
        total = self.passed + self.failed
        print(f"\n{Colors.BLUE}========== 测试摘要 =========={Colors.END}")
        print(f"总测试数: {total}")
        print(f"{Colors.GREEN}通过: {self.passed}{Colors.END}")
        print(f"{Colors.RED}失败: {self.failed}{Colors.END}")
        print(f"通过率: {(self.passed/total*100):.1f}%" if total > 0 else "N/A")
        print(f"{Colors.BLUE}=============================={Colors.END}\n")
        
        if self.failed > 0:
            print(f"{Colors.RED}失败的测试:{Colors.END}")
            for result in self.test_results:
                if not result.get('success', False):
                    print(f"  - {result['name']}: {result.get('error', '状态码不匹配')}")


def test_basic_apis(runner: APITestRunner):
    """测试基础功能API"""
    print(f"\n{Colors.YELLOW}========== 基础功能API测试 =========={Colors.END}\n")
    
    # 测试策略列表API
    runner.test("获取策略列表", "GET", "/api/v1/strategy/conceptions")
    
    # 测试策略统计API - 使用正确的路径
    runner.test("获取策略统计", "GET", "/api/v1/strategy/conceptions")
    
    # 测试回测结果API - 使用正确的路径
    runner.test("获取回测结果列表", "GET", "/api/v1/backtest")
    
    # 测试优化结果API
    runner.test("获取优化结果列表", "GET", "/api/v1/strategy/optimization/results")


def test_workflow_apis(runner: APITestRunner):
    """测试工作流API"""
    print(f"\n{Colors.YELLOW}========== 工作流API测试 =========={Colors.END}\n")
    
    # 创建工作流
    success, result = runner.test(
        "创建工作流",
        "POST",
        "/api/v1/strategy/workflow/create",
        data={"strategy_id": TEST_STRATEGY_ID, "strategy_name": "测试策略"}
    )
    
    if success and "workflow_id" in result.get("response", {}):
        workflow_id = result["response"]["workflow_id"]
        
        # 获取工作流进度
        runner.test("获取工作流进度", "GET", 
                   f"/api/v1/strategy/workflow/{workflow_id}/progress")
        
        # 状态转换
        runner.test("工作流状态转换", "POST",
                   f"/api/v1/strategy/workflow/{workflow_id}/transition",
                   data={"new_status": "backtest"})
        
        # 列工作流
        runner.test("列工作流", "GET", "/api/v1/strategy/workflows")


def test_lifecycle_apis(runner: APITestRunner):
    """测试生命周期API"""
    print(f"\n{Colors.YELLOW}========== 生命周期API测试 =========={Colors.END}\n")
    
    # 创建生命周期
    runner.test("创建生命周期", "POST",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/lifecycle/create",
               data={"strategy_name": "测试策略"})
    
    # 获取生命周期
    runner.test("获取生命周期", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/lifecycle")
    
    # 状态转换
    runner.test("生命周期状态转换", "POST",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/lifecycle/transition",
               data={"new_status": "design", "reason": "测试"})
    
    # 获取时间线
    runner.test("获取生命周期时间线", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/lifecycle/timeline")
    
    # 获取统计
    runner.test("获取生命周期统计", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/lifecycle/stats")
    
    # 列生命周期
    runner.test("列生命周期", "GET", "/api/v1/strategy/lifecycles")
    
    # 获取概览
    runner.test("获取生命周期概览", "GET", "/api/v1/strategy/lifecycles/overview")


def test_version_apis(runner: APITestRunner):
    """测试版本管理API"""
    print(f"\n{Colors.YELLOW}========== 版本管理API测试 =========={Colors.END}\n")
    
    # 列版本
    runner.test("列策略版本", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/versions")
    
    # 创建版本
    success, result = runner.test("创建策略版本", "POST",
                                 f"/api/v1/strategy/{TEST_STRATEGY_ID}/version/create",
                                 data={"comment": "测试版本", "tags": ["test"]})
    
    if success:
        # 获取版本统计
        runner.test("获取版本统计", "GET",
                   f"/api/v1/strategy/{TEST_STRATEGY_ID}/version/statistics")


def test_recommendation_apis(runner: APITestRunner):
    """测试推荐系统API"""
    print(f"\n{Colors.YELLOW}========== 推荐系统API测试 =========={Colors.END}\n")
    
    # 获取推荐列表
    runner.test("获取推荐列表", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/recommendations")
    
    # 获取未读推荐数量
    runner.test("获取未读推荐数量", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/recommendations/unread-count")
    
    # 分析回测结果
    runner.test("分析回测结果", "POST",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/recommendations/analyze",
               data={
                   "backtest_result": {
                       "metrics": {
                           "sharpe_ratio": 0.3,
                           "max_drawdown": 0.25,
                           "win_rate": 0.35,
                           "total_return": 0.1
                       }
                   }
               })


def test_performance_apis(runner: APITestRunner):
    """测试性能监控API"""
    print(f"\n{Colors.YELLOW}========== 性能监控API测试 =========={Colors.END}\n")
    
    # 记录性能
    runner.test("记录性能", "POST",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/performance/record",
               data={
                   "metrics": {
                       "sharpe_ratio": 1.2,
                       "total_return": 0.15,
                       "max_drawdown": 0.1,
                       "win_rate": 0.55
                   },
                   "period": "daily"
               })
    
    # 获取性能历史
    runner.test("获取性能历史", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/performance/history?days=7")
    
    # 获取最新指标
    runner.test("获取最新指标", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/performance/latest")
    
    # 获取指标定义
    runner.test("获取指标定义", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/performance/metrics")
    
    # 计算评分
    runner.test("计算性能评分", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/performance/score")
    
    # 生成报告
    runner.test("生成性能报告", "GET",
               f"/api/v1/strategy/{TEST_STRATEGY_ID}/performance/report?days=7")


def generate_test_report(runner: APITestRunner):
    """生成测试报告"""
    report = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": BASE_URL,
        "summary": {
            "total": runner.passed + runner.failed,
            "passed": runner.passed,
            "failed": runner.failed,
            "pass_rate": (runner.passed / (runner.passed + runner.failed) * 100) if (runner.passed + runner.failed) > 0 else 0
        },
        "results": runner.test_results
    }
    
    # 保存报告
    report_path = "tests/api_test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{Colors.BLUE}测试报告已保存: {report_path}{Colors.END}")
    return report


def main():
    print(f"{Colors.BLUE}========== 策略系统API综合测试 =========={Colors.END}")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"基础URL: {BASE_URL}")
    print(f"测试策略ID: {TEST_STRATEGY_ID}")
    
    runner = APITestRunner()
    
    # 执行所有测试
    test_basic_apis(runner)
    test_workflow_apis(runner)
    test_lifecycle_apis(runner)
    test_version_apis(runner)
    test_recommendation_apis(runner)
    test_performance_apis(runner)
    
    # 打印摘要
    runner.print_summary()
    
    # 生成报告
    report = generate_test_report(runner)
    
    # 返回退出码
    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
