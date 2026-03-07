#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 业务流程测试执行脚本

执行所有业务流程测试用例，验证系统业务逻辑完整性
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import json


def run_business_process_tests():
    """执行业务流程测试"""
    print("🚀 RQA2025 业务流程测试执行")
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    # 测试文件列表
    test_files = [
        "tests/business_process/test_strategy_management_flow.py",
        "tests/business_process/test_portfolio_management_flow.py",
        "tests/business_process/test_user_service_management_flow.py",
        "tests/business_process/test_system_monitoring_flow.py"
    ]

    # 测试结果
    test_results = {
        "start_time": datetime.now().isoformat(),
        "test_files": [],
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "errors": [],
        "coverage": {}
    }

    # 执行每个测试文件
    for test_file in test_files:
        test_path = project_root / test_file
        if not test_path.exists():
            print(f"❌ 测试文件不存在: {test_file}")
            test_results["errors"].append(f"测试文件不存在: {test_file}")
            continue

        print(f"\n📋 执行测试文件: {test_file}")
        print("-" * 40)

        # 执行pytest
        try:
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v",
                "--tb=short",
                "--durations=10",
                "-x"  # 遇到第一个失败就停止
            ]

            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )

            # 解析测试结果
            test_file_result = {
                "file": test_file,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "passed": False,
                "test_count": 0
            }

            # 分析输出
            stdout_lines = result.stdout.split('\n')
            for line in stdout_lines:
                if "PASSED" in line or "FAILED" in line or "ERROR" in line:
                    print(line)
                if "passed" in line and "failed" in line:
                    # 解析测试统计
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            test_file_result["test_count"] = int(part)
                            break

            if result.returncode == 0:
                test_file_result["passed"] = True
                test_results["passed_tests"] += test_file_result["test_count"]
                print(f"✅ {test_file} 测试通过")
            else:
                test_results["failed_tests"] += 1
                test_file_result["passed"] = False
                print(f"❌ {test_file} 测试失败")
                if result.stderr:
                    print("错误信息:")
                    print(result.stderr)

            test_results["test_files"].append(test_file_result)
            test_results["total_tests"] += test_file_result["test_count"]

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} 测试超时")
            test_results["errors"].append(f"{test_file} 测试超时")
        except Exception as e:
            print(f"💥 {test_file} 测试执行异常: {str(e)}")
            test_results["errors"].append(f"{test_file} 执行异常: {str(e)}")

    # 生成测试报告
    generate_test_report(test_results, project_root)

    # 输出总结
    print("\n" + "=" * 50)
    print("📊 业务流程测试执行总结")
    print("=" * 50)
    print(f"总测试文件数: {len(test_files)}")
    print(f"成功执行文件数: {len([f for f in test_results['test_files'] if f['passed']])}")
    print(f"失败执行文件数: {len([f for f in test_results['test_files'] if not f['passed']])}")
    print(f"总测试用例数: {test_results['total_tests']}")
    print(f"通过测试用例数: {test_results['passed_tests']}")
    print(f"失败测试用例数: {test_results['failed_tests']}")
    print(f"错误数量: {len(test_results['errors'])}")

    if test_results["errors"]:
        print("\n❌ 发现的错误:")
        for error in test_results["errors"]:
            print(f"  - {error}")

    # 计算通过率
    if test_results["total_tests"] > 0:
        pass_rate = (test_results["passed_tests"] / test_results["total_tests"]) * 100
        print(".2f"
        if pass_rate >= 90:
            print("🎉 业务流程测试覆盖目标达成!")
        elif pass_rate >= 70:
            print("⚠️ 业务流程测试覆盖接近目标")
        else:
            print("❌ 业务流程测试覆盖需要改进")

    test_results["end_time"]=datetime.now().isoformat()

    return test_results

def generate_test_report(test_results, project_root):
    """生成测试报告"""
    print("\n📝 生成测试报告...")

    # 创建报告目录
    report_dir=project_root / "reports" / "business_process_tests"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 生成HTML报告
    html_report=generate_html_report(test_results)

    # 保存HTML报告
    html_path=report_dir /
        f"business_process_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_report)

    # 生成JSON报告
    json_path=report_dir /
        f"business_process_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    print(f"✅ HTML报告已保存: {html_path}")
    print(f"✅ JSON报告已保存: {json_path}")

def generate_html_report(test_results):
    """生成HTML测试报告"""
    html_template="""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQA2025 业务流程测试报告</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }
        .metric h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 2em;
        }
        .metric p {
            margin: 0;
            color: #666;
            font-size: 0.9em;
        }
        .passed { color: #28a745; }
        .failed { color: #dc3545; }
        .warning { color: #ffc107; }
        .test-file {
            margin-bottom: 20px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            overflow: hidden;
        }
        .test-file-header {
            background: #f8f9fa;
            padding: 15px;
            font-weight: bold;
            border-bottom: 1px solid #dee2e6;
        }
        .test-file-content {
            padding: 15px;
        }
        .status-passed { background-color: #d4edda; border-color: #c3e6cb; }
        .status-failed { background-color: #f8d7da; border-color: #f5c6cb; }
        .errors {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin-top: 20px;
        }
        .error-item {
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }
        pre {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.9em;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #666;
            border-top: 1px solid #dee2e6;
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RQA2025 业务流程测试报告</h1>
            <p>测试执行时间: {start_time} - {end_time}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <h3 class="{total_class}">{total_files}</h3>
                <p>总测试文件</p>
            </div>
            <div class="metric">
                <h3 class="{passed_class}">{passed_files}</h3>
                <p>通过文件</p>
            </div>
            <div class="metric">
                <h3 class="{failed_class}">{failed_files}</h3>
                <p>失败文件</p>
            </div>
            <div class="metric">
                <h3 class="{pass_rate_class}">{pass_rate:.1f}%</h3>
                <p>通过率</p>
            </div>
        </div>

        <h2>测试文件详情</h2>
        {test_files_html}

        {errors_html}

        <div class="footer">
            <p>RQA2025 业务流程测试执行完成</p>
            <p>报告生成时间: {report_time}</p>
        </div>
    </div>
</body>
</html>
"""

    # 计算统计数据
    total_files=len(test_results["test_files"])
    passed_files=len([f for f in test_results["test_files"] if f["passed"]])
    failed_files=total_files - passed_files

    if test_results["total_tests"] > 0:
        pass_rate=(test_results["passed_tests"] / test_results["total_tests"]) * 100
    else:
        pass_rate=0

    # 确定样式类
    total_class="passed" if total_files > 0 else "warning"
    passed_class="passed" if passed_files > 0 else "warning"
    failed_class="failed" if failed_files > 0 else "passed"
    pass_rate_class="passed" if pass_rate >= 90 else "warning" if pass_rate >= 70 else "failed"

    # 生成测试文件详情HTML
    test_files_html=""
    for test_file in test_results["test_files"]:
        status_class="status-passed" if test_file["passed"] else "status-failed"
        status_text="✅ 通过" if test_file["passed"] else "❌ 失败"

        test_files_html += f"""
        <div class="test-file {status_class}">
            <div class="test-file-header">
                {test_file['file']} - {status_text}
            </div>
            <div class="test-file-content">
                <p><strong>测试用例数:</strong> {test_file['test_count']}</p>
                <p><strong>返回码:</strong> {test_file['return_code']}</p>
                {f'<pre>{test_file["stdout"][-1000:]}</pre>' if test_file["stdout"] else ""}
                {f'<pre style="color: red;">{test_file["stderr"][-1000:]}</pre>' if test_file["stderr"] else ""}
            </div>
        </div>
        """

    # 生成错误信息HTML
    errors_html=""
    if test_results["errors"]:
        errors_html='<div class="errors"><h3>❌ 执行错误</h3>'
        for error in test_results["errors"]:
            errors_html += f'<div class="error-item">{error}</div>'
        errors_html += '</div>'

    # 格式化时间
    start_time=test_results.get("start_time", "N/A")
    end_time=test_results.get("end_time", datetime.now().isoformat())
    report_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return html_template.format(
        start_time=start_time,
        end_time=end_time,
        total_files=total_files,
        passed_files=passed_files,
        failed_files=failed_files,
        pass_rate=pass_rate,
        total_class=total_class,
        passed_class=passed_class,
        failed_class=failed_class,
        pass_rate_class=pass_rate_class,
        test_files_html=test_files_html,
        errors_html=errors_html,
        report_time=report_time
    )

def main():
    """主函数"""
    try:
        results=run_business_process_tests()

        # 根据测试结果返回退出码
        if results["failed_tests"] > 0 or results["errors"]:
            print("\n❌ 业务流程测试发现问题")
            return 1
        else:
            print("\n✅ 业务流程测试全部通过")
            return 0

    except Exception as e:
        print(f"💥 测试执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
