#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成基础设施层最终覆盖率报告
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime


def generate_final_report():
    """生成最终覆盖率报告"""
    base_dir = Path(__file__).parent.parent
    test_logs_dir = base_dir / "test_logs"
    test_logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("🎯 基础设施层最终覆盖率验证")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 关键模块列表
    key_modules = ["core", "versioning", "config", "cache", "logging"]
    
    results = {}
    
    for module in key_modules:
        print(f"\n{'='*80}")
        print(f"验证模块: {module}")
        print(f"{'='*80}")
        
        # 查找测试路径
        test_paths = []
        
        infra_test = base_dir / "tests" / "infrastructure"
        if infra_test.exists():
            module_tests = list(infra_test.glob(f"*{module}*.py"))
            test_paths.extend([str(p) for p in module_tests if p.is_file()])
        
        unit_test = base_dir / "tests" / "unit" / "infrastructure" / module
        if unit_test.exists():
            test_paths.append(str(unit_test))
        
        if not test_paths:
            print(f"⚠️  未找到 {module} 的测试")
            continue
        
        # 运行测试
        coverage_json = test_logs_dir / f"coverage_{module}_final_{timestamp}.json"
        
        cmd = [
            "pytest",
            *test_paths,
            "-v",
            "-n", "auto",
            "-q",
            f"--cov=src/infrastructure/{module}",
            "--cov-report=term-missing",
            f"--cov-report=json:{coverage_json}",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(base_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            # 解析结果
            output = result.stdout + result.stderr
            
            # 解析覆盖率
            coverage = 0.0
            if coverage_json.exists():
                with open(coverage_json, 'r', encoding='utf-8') as f:
                    cov_data = json.load(f)
                    coverage = cov_data.get('totals', {}).get('percent_covered', 0.0)
            
            # 解析测试数
            import re
            passed_match = re.search(r'(\d+) passed', output)
            failed_match = re.search(r'(\d+) failed', output)
            
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            total = passed + failed
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            results[module] = {
                "coverage": coverage,
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": pass_rate
            }
            
            print(f"✅ {module}")
            print(f"   覆盖率: {coverage:.2f}%")
            print(f"   测试: {passed}/{total} 通过")
            print(f"   通过率: {pass_rate:.2f}%")
            
        except Exception as e:
            print(f"❌ {module} 出错: {e}")
            continue
    
    # 生成汇总报告
    print("\n" + "="*80)
    print("📊 最终汇总")
    print("="*80)
    
    if results:
        avg_coverage = sum(r['coverage'] for r in results.values()) / len(results)
        total_tests = sum(r['total'] for r in results.values())
        total_passed = sum(r['passed'] for r in results.values())
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n已验证模块: {len(results)}/{len(key_modules)}")
        print(f"平均覆盖率: {avg_coverage:.2f}%")
        print(f"总测试数: {total_tests}")
        print(f"总通过数: {total_passed}")
        print(f"整体通过率: {overall_pass_rate:.2f}%")
        
        # 保存结果
        report_file = test_logs_dir / f"final_coverage_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "modules": results,
                "summary": {
                    "avg_coverage": avg_coverage,
                    "total_tests": total_tests,
                    "total_passed": total_passed,
                    "overall_pass_rate": overall_pass_rate
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 报告已保存: {report_file}")
    
    print("\n" + "="*80)
    print("✅ 验证完成")
    print("="*80)


if __name__ == "__main__":
    generate_final_report()

