#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证大型基础设施模块（config, cache, logging, security）
解决Windows编码问题
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict


def validate_module(module_name: str, base_dir: Path, timeout: int = 180) -> Dict:
    """验证单个模块"""
    print(f"\n{'='*80}")
    print(f"🔍 验证模块: {module_name}")
    print(f"{'='*80}")
    
    test_logs_dir = base_dir / "test_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 查找测试路径
    test_paths = []
    
    infra_test_dir = base_dir / "tests" / "infrastructure"
    if infra_test_dir.exists():
        module_tests = list(infra_test_dir.glob(f"*{module_name}*.py"))
        test_paths.extend([str(p) for p in module_tests if p.is_file()])
    
    unit_test_dir = base_dir / "tests" / "unit" / "infrastructure" / module_name
    if unit_test_dir.exists():
        test_paths.append(str(unit_test_dir))
    
    if not test_paths:
        print(f"⚠️  未找到测试")
        return {"module": module_name, "status": "no_tests", "coverage": 0}
    
    print(f"📁 找到 {len(test_paths)} 个测试路径")
    
    # 构建pytest命令
    coverage_json = test_logs_dir / f"coverage_{module_name}_{timestamp}.json"
    coverage_html = test_logs_dir / f"coverage_{module_name}"
    
    cmd = [
        "pytest",
        *test_paths,
        "-v",
        "-n", "auto",
        "-q",
        "--tb=no",
        f"--cov=src/infrastructure/{module_name}",
        "--cov-report=term",
        f"--cov-report=json:{coverage_json}",
        f"--cov-report=html:{coverage_html}",
    ]
    
    print(f"🚀 开始测试...")
    start_time = datetime.now()
    
    try:
        # 关键修复：使用encoding='utf-8'和errors='ignore'
        result = subprocess.run(
            cmd,
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='ignore'  # 忽略编码错误
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # 解析覆盖率
        coverage = 0.0
        if coverage_json.exists():
            with open(coverage_json, 'r', encoding='utf-8') as f:
                cov_data = json.load(f)
                coverage = cov_data.get('totals', {}).get('percent_covered', 0.0)
        
        # 解析测试结果
        output = result.stdout + result.stderr
        import re
        
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)
        error_match = re.search(r'(\d+) error', output)
        skipped_match = re.search(r'(\d+) skipped', output)
        
        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        errors = int(error_match.group(1)) if error_match else 0
        skipped = int(skipped_match.group(1)) if skipped_match else 0
        total = passed + failed + errors
        
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        result_data = {
            "module": module_name,
            "status": "completed",
            "coverage": coverage,
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "pass_rate": pass_rate,
            "duration": duration,
        }
        
        # 判断状态
        if coverage >= 90:
            status = "🌟 优秀"
        elif coverage >= 80:
            status = "✅ 良好"
        elif coverage >= 70:
            status = "🟡 一般"
        else:
            status = "⚠️ 需改进"
        
        print(f"\n{status}")
        print(f"   覆盖率: {coverage:.2f}%")
        print(f"   测试: {passed}/{total} 通过")
        print(f"   失败: {failed}个")
        print(f"   通过率: {pass_rate:.2f}%")
        print(f"   耗时: {duration:.1f}秒")
        
        return result_data
        
    except subprocess.TimeoutExpired:
        print(f"⏰ 超时（{timeout}秒）")
        return {"module": module_name, "status": "timeout", "coverage": 0}
    except Exception as e:
        print(f"❌ 出错: {type(e).__name__}: {e}")
        return {"module": module_name, "status": "error", "coverage": 0}


def main():
    """主函数"""
    base_dir = Path(__file__).parent.parent
    
    # 要验证的大型模块
    modules = ["config", "cache", "logging", "security", "monitoring", "resource"]
    
    print("\n" + "="*80)
    print("🎯 大型基础设施模块覆盖率验证")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"待验证模块: {len(modules)}个")
    print("="*80)
    
    results = []
    
    for module in modules:
        result = validate_module(module, base_dir, timeout=180)
        results.append(result)
    
    # 生成汇总
    print("\n" + "="*80)
    print("📊 验证汇总")
    print("="*80)
    
    completed = [r for r in results if r.get('status') == 'completed']
    
    if completed:
        avg_coverage = sum(r['coverage'] for r in completed) / len(completed)
        total_tests = sum(r['total'] for r in completed)
        total_passed = sum(r['passed'] for r in completed)
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n验证成功模块: {len(completed)}/{len(modules)}")
        print(f"平均覆盖率: {avg_coverage:.2f}%")
        print(f"总测试数: {total_tests}")
        print(f"总通过数: {total_passed}")
        print(f"整体通过率: {overall_pass_rate:.2f}%")
        
        print(f"\n详细结果:")
        for r in completed:
            达标 = "✅" if r['coverage'] >= 80 else "⚠️"
            print(f"  {达标} {r['module']:15} | 覆盖率: {r['coverage']:5.1f}% | "
                  f"测试: {r['passed']:4}/{r['total']:4} | 通过率: {r['pass_rate']:5.1f}%")
        
        # 保存结果
        test_logs_dir = base_dir / "test_logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = test_logs_dir / f"large_modules_validation_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "completed": len(completed),
                    "avg_coverage": avg_coverage,
                    "total_tests": total_tests,
                    "overall_pass_rate": overall_pass_rate
                },
                "modules": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 报告已保存: {report_file}")
    else:
        print("\n⚠️  没有成功完成的验证")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

