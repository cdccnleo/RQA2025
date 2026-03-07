#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量提升基础设施层测试覆盖率
按优先级依次处理各个模块
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class BatchCoverageImprover:
    """批量覆盖率提升器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.test_logs_dir = self.base_dir / "test_logs"
        self.test_logs_dir.mkdir(exist_ok=True)
        
        # 定义要验证的模块（按优先级排序）
        self.modules = [
            # 已完成
            {"name": "core", "priority": "🟢中", "target": 80, "status": "completed"},
            {"name": "versioning", "priority": "⚪低", "target": 60, "status": "completed"},
            # 待验证的极高风险模块
            {"name": "cache", "priority": "🔴极高", "target": 90, "status": "pending"},
            {"name": "config", "priority": "🔴极高", "target": 90, "status": "pending"},
            {"name": "logging", "priority": "🔴极高", "target": 90, "status": "pending"},
            {"name": "security", "priority": "🔴极高", "target": 90, "status": "pending"},
            # 高风险模块
            {"name": "monitoring", "priority": "🟡高", "target": 80, "status": "pending"},
            {"name": "resource", "priority": "🟡高", "target": 80, "status": "pending"},
            {"name": "health", "priority": "🟡高", "target": 80, "status": "pending"},
            {"name": "error", "priority": "🟡高", "target": 80, "status": "pending"},
            # 中低风险模块
            {"name": "distributed", "priority": "🟢中", "target": 70, "status": "pending"},
            {"name": "utils", "priority": "🟢中", "target": 70, "status": "pending"},
            {"name": "constants", "priority": "🟢中", "target": 70, "status": "pending"},
            {"name": "ops", "priority": "⚪低", "target": 60, "status": "pending"},
        ]
        
        self.results = []
    
    def test_module(self, module_info: Dict) -> Dict:
        """测试单个模块"""
        module_name = module_info['name']
        
        print(f"\n{'='*80}")
        print(f"🔍 验证模块: {module_name} ({module_info['priority']})")
        print(f"   目标覆盖率: {module_info['target']}%")
        print(f"{'='*80}")
        
        # 查找测试路径
        test_paths = []
        
        # tests/infrastructure/
        infra_test_dir = self.base_dir / "tests" / "infrastructure"
        if infra_test_dir.exists():
            module_tests = list(infra_test_dir.glob(f"*{module_name}*.py"))
            test_paths.extend([str(p) for p in module_tests if p.is_file()])
        
        # tests/unit/infrastructure/module/
        unit_test_dir = self.base_dir / "tests" / "unit" / "infrastructure" / module_name
        if unit_test_dir.exists():
            test_paths.append(str(unit_test_dir))
        
        if not test_paths:
            print(f"⚠️  未找到测试")
            return {
                "module": module_name,
                "status": "no_tests",
                "coverage": 0
            }
        
        # 运行测试
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        coverage_json = self.test_logs_dir / f"coverage_{module_name}_{timestamp}.json"
        
        cmd = [
            "pytest",
            *test_paths,
            "-v",
            "-n", "auto",
            "-q",
            f"--cov=src/infrastructure/{module_name}",
            "--cov-report=term",
            f"--cov-report=json:{coverage_json}",
            "--tb=no",  # 不显示traceback，减少输出
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=180,
                encoding='utf-8',
                errors='ignore'  # 忽略编码错误
            )
            
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
            
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            total = passed + failed
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            # 判断达标状态
            达标 = coverage >= module_info['target']
            status_icon = "✅" if 达标 else "⚠️"
            
            result_data = {
                "module": module_name,
                "priority": module_info['priority'],
                "target": module_info['target'],
                "coverage": coverage,
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": pass_rate,
                "达标": 达标,
                "status": "completed"
            }
            
            print(f"{status_icon} 覆盖率: {coverage:.2f}% (目标: {module_info['target']}%)")
            print(f"   测试: {passed}/{total} 通过 ({pass_rate:.1f}%)")
            
            return result_data
            
        except subprocess.TimeoutExpired:
            print(f"⏰ 超时")
            return {"module": module_name, "status": "timeout", "coverage": 0}
        except Exception as e:
            print(f"❌ 出错: {e}")
            return {"module": module_name, "status": "error", "coverage": 0}
    
    def run_batch_improvement(self):
        """批量运行改进"""
        print("\n" + "="*80)
        print("🚀 基础设施层批量覆盖率提升")
        print("="*80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"待验证模块: {len([m for m in self.modules if m['status'] == 'pending'])}个")
        print("="*80)
        
        for module_info in self.modules:
            if module_info['status'] == 'completed':
                print(f"\n✅ {module_info['name']} 已完成，跳过")
                continue
            
            result = self.test_module(module_info)
            self.results.append(result)
        
        # 生成汇总报告
        self.generate_summary()
    
    def generate_summary(self):
        """生成汇总报告"""
        print("\n" + "="*80)
        print("📊 批量提升汇总")
        print("="*80)
        
        completed = [r for r in self.results if r.get('status') == 'completed']
        
        if not completed:
            print("⚠️  没有成功完成的模块验证")
            return
        
        # 计算总体数据
        avg_coverage = sum(r['coverage'] for r in completed) / len(completed)
        total_tests = sum(r['total'] for r in completed)
        total_passed = sum(r['passed'] for r in completed)
        total_failed = sum(r['failed'] for r in completed)
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        达标_count = sum(1 for r in completed if r.get('达标', False))
        
        print(f"\n验证完成模块: {len(completed)}")
        print(f"平均覆盖率: {avg_coverage:.2f}%")
        print(f"总测试数: {total_tests}")
        print(f"总通过: {total_passed}")
        print(f"总失败: {total_failed}")
        print(f"整体通过率: {overall_pass_rate:.2f}%")
        print(f"达标模块: {达标_count}/{len(completed)}")
        
        # 按优先级分组显示
        print(f"\n{'='*80}")
        print("📋 详细结果")
        print(f"{'='*80}")
        
        for priority in ["🔴极高", "🟡高", "🟢中", "⚪低"]:
            priority_modules = [r for r in completed if r.get('priority') == priority]
            if priority_modules:
                print(f"\n{priority} 风险模块:")
                for r in priority_modules:
                    status = "✅" if r.get('达标') else "⚠️"
                    print(f"  {status} {r['module']:15} | "
                          f"覆盖率: {r['coverage']:5.1f}% (目标: {r['target']}%) | "
                          f"测试: {r['passed']}/{r['total']}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.test_logs_dir / f"batch_improvement_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "completed_modules": len(completed),
                    "avg_coverage": avg_coverage,
                    "total_tests": total_tests,
                    "overall_pass_rate": overall_pass_rate,
                    "达标模块数": 达标_count
                },
                "modules": self.results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 报告已保存: {report_file}")


def main():
    """主函数"""
    improver = BatchCoverageImprover()
    improver.run_batch_improvement()
    return 0


if __name__ == "__main__":
    exit(main())

