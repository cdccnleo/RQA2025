#!/usr/bin/env python3
"""
基础设施层测试运行脚本
用于验证修复效果和生成测试报告
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_tests(test_path, verbose=True):
    """运行指定路径的测试"""
    print(f"\n{'='*60}")
    print(f"运行测试: {test_path}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "-m", "pytest", 
        test_path,
        "-v" if verbose else "",
        "--tb=short",
        "--no-header"
    ]
    
    # 过滤空字符串
    cmd = [arg for arg in cmd if arg]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding='utf-8', errors='ignore')
        return result
    except subprocess.TimeoutExpired:
        print(f"测试超时: {test_path}")
        return None
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return None

def analyze_test_results(result):
    """分析测试结果"""
    if result is None:
        return {"status": "error", "message": "测试运行失败"}
    
    output = result.stdout or ""
    error_output = result.stderr or ""
    
    # 解析测试结果
    if "FAILED" in output:
        failed_count = output.count("FAILED")
        passed_count = output.count("PASSED") if "PASSED" in output else 0
        return {
            "status": "partial_success",
            "passed": passed_count,
            "failed": failed_count,
            "output": output,
            "error": error_output
        }
    elif "PASSED" in output:
        passed_count = output.count("PASSED")
        return {
            "status": "success",
            "passed": passed_count,
            "failed": 0,
            "output": output
        }
    else:
        return {
            "status": "unknown",
            "output": output,
            "error": error_output
        }

def generate_test_report(results):
    """生成测试报告"""
    report = []
    report.append("# 基础设施层测试修复验证报告")
    report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    total_passed = 0
    total_failed = 0
    
    for test_path, result in results.items():
        report.append(f"## {test_path}")
        report.append("")
        
        if result["status"] == "success":
            report.append(f"✅ **通过** - {result['passed']} 个测试通过")
            total_passed += result['passed']
        elif result["status"] == "partial_success":
            report.append(f"⚠️ **部分通过** - {result['passed']} 个通过, {result['failed']} 个失败")
            total_passed += result['passed']
            total_failed += result['failed']
        elif result["status"] == "error":
            report.append(f"❌ **运行错误** - {result['message']}")
            total_failed += 1
        else:
            report.append(f"❓ **未知状态**")
        
        report.append("")
        
        if "output" in result and result["output"]:
            report.append("### 测试输出")
            report.append("```")
            report.append(result["output"][:1000])  # 限制输出长度
            if len(result["output"]) > 1000:
                report.append("... (输出已截断)")
            report.append("```")
            report.append("")
        
        if "error" in result and result["error"]:
            report.append("### 错误输出")
            report.append("```")
            report.append(result["error"][:1000])  # 限制输出长度
            if len(result["error"]) > 1000:
                report.append("... (错误输出已截断)")
            report.append("```")
            report.append("")
    
    report.append("## 总结")
    report.append(f"- 总通过测试: {total_passed}")
    report.append(f"- 总失败测试: {total_failed}")
    report.append(f"- 成功率: {total_passed/(total_passed+total_failed)*100:.1f}%" if total_passed + total_failed > 0 else "- 成功率: 0%")
    
    return "\n".join(report)

def main():
    """主函数"""
    print("开始运行基础设施层测试验证...")
    
    # 测试路径列表
    test_paths = [
        "tests/unit/infrastructure/database/test_influxdb_error_handler.py",
        "tests/unit/infrastructure/m_logging/test_log_manager.py",
        "tests/unit/infrastructure/monitoring/test_application_monitor.py",
        "tests/unit/infrastructure/health/test_health_checker.py",
        "tests/unit/infrastructure/m_logging/test_log_sampler.py",
        "tests/unit/infrastructure/m_logging/test_log_aggregator.py",
        "tests/unit/infrastructure/m_logging/test_resource_manager.py",
        "tests/unit/infrastructure/m_logging/test_log_compressor.py",
        "tests/unit/infrastructure/m_logging/test_security_filter.py",
        "tests/unit/infrastructure/m_logging/test_quant_filter.py",
        "tests/unit/infrastructure/monitoring/test_backtest_monitor.py",
        "tests/unit/infrastructure/web/test_app_factory.py",
        "tests/unit/infrastructure/error/test_error_handler.py",
        "tests/unit/infrastructure/m_logging/test_log_metrics.py",
        "tests/unit/infrastructure/config/test_config_manager.py",
        "tests/unit/infrastructure/database/test_database_manager.py"
    ]
    
    results = {}
    
    for test_path in test_paths:
        if os.path.exists(test_path):
            result = run_tests(test_path)
            analysis = analyze_test_results(result)
            results[test_path] = analysis
        else:
            results[test_path] = {
                "status": "error",
                "message": f"测试文件不存在: {test_path}"
            }
    
    # 生成报告
    report = generate_test_report(results)
    
    # 保存报告
    report_file = "docs/infrastructure_test_verification_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n测试验证完成！")
    print(f"报告已保存到: {report_file}")
    
    # 打印简要结果
    total_passed = sum(r.get("passed", 0) for r in results.values())
    total_failed = sum(r.get("failed", 0) for r in results.values())
    error_count = sum(1 for r in results.values() if r["status"] == "error")
    
    print(f"\n简要结果:")
    print(f"- 通过测试: {total_passed}")
    print(f"- 失败测试: {total_failed}")
    print(f"- 运行错误: {error_count}")
    
    if total_passed + total_failed > 0:
        success_rate = total_passed / (total_passed + total_failed) * 100
        print(f"- 成功率: {success_rate:.1f}%")

if __name__ == "__main__":
    main() 