"""
仪表盘测试验证主程序
按照业务流程顺序执行所有测试
"""

import subprocess
import sys
import time
from datetime import datetime


def print_section(title: str):
    """打印测试章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_tests(test_file: str, description: str):
    """运行测试文件"""
    print_section(description)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "-s"],
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutError:
        print(f"❌ {description} 测试超时")
        return False
    except Exception as e:
        print(f"❌ {description} 测试失败: {e}")
        return False


def main():
    """主测试流程"""
    print("\n" + "=" * 80)
    print("  RQA2025 仪表盘测试验证")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    test_results = {}
    
    # Phase 1: 量化策略开发流程测试
    print_section("Phase 1: 量化策略开发流程测试")
    
    # 1.1 数据收集阶段验证
    test_results["数据收集API"] = run_tests(
        "test_api_endpoints.py::TestDataCollectionAPIs",
        "1.1 数据收集阶段API测试"
    )
    
    # 1.2 特征工程监控验证
    test_results["特征工程API"] = run_tests(
        "test_api_endpoints.py::TestFeatureEngineeringAPIs",
        "1.2 特征工程监控API测试"
    )
    
    # 1.3 模型训练监控验证
    test_results["模型训练API"] = run_tests(
        "test_api_endpoints.py::TestModelTrainingAPIs",
        "1.3 模型训练监控API测试"
    )
    
    # 1.4 策略性能评估验证
    test_results["策略性能API"] = run_tests(
        "test_api_endpoints.py::TestStrategyPerformanceAPIs",
        "1.4 策略性能评估API测试"
    )
    
    # Phase 2: 交易执行流程测试
    print_section("Phase 2: 交易执行流程测试")
    
    # 2.2 交易信号生成监控验证
    test_results["交易信号API"] = run_tests(
        "test_api_endpoints.py::TestTradingSignalAPIs",
        "2.2 交易信号生成监控API测试"
    )
    
    # 2.3 订单智能路由监控验证
    test_results["订单路由API"] = run_tests(
        "test_api_endpoints.py::TestOrderRoutingAPIs",
        "2.3 订单智能路由监控API测试"
    )
    
    # Phase 3: 风险控制流程测试
    print_section("Phase 3: 风险控制流程测试")
    
    # 3.1 风险报告生成验证
    test_results["风险报告API"] = run_tests(
        "test_api_endpoints.py::TestRiskReportingAPIs",
        "3.1 风险报告生成API测试"
    )
    
    # WebSocket测试
    print_section("WebSocket实时数据推送测试")
    test_results["WebSocket"] = run_tests(
        "test_websocket_connections.py",
        "WebSocket连接测试"
    )
    
    # 仪表盘页面测试
    print_section("仪表盘页面加载测试")
    test_results["仪表盘页面"] = run_tests(
        "test_dashboard_pages.py",
        "仪表盘页面加载测试"
    )
    
    # 业务流程数据流测试
    print_section("业务流程数据流测试")
    test_results["业务流程数据流"] = run_tests(
        "test_business_process_flow.py",
        "业务流程数据流连通性测试"
    )
    
    # 测试结果汇总
    print_section("测试结果汇总")
    print(f"测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests} ✅")
    print(f"失败: {failed_tests} ❌")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%\n")
    
    print("详细结果:")
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 80)
    
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

