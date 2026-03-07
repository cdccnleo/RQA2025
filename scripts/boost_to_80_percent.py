#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提升基础设施层覆盖率至80%
计算需要补充的测试并生成行动计划
"""

import json
from pathlib import Path
from datetime import datetime


class CoverageBooster:
    """覆盖率提升器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        
        # 已验证模块的实际数据
        self.verified_modules = {
            "core": {
                "statements": 977,
                "covered": 869,
                "coverage": 89.0,
                "status": "✅ 达标"
            },
            "config": {
                "statements": 12201,
                "covered": 9753,
                "coverage": 79.94,
                "status": "🟡 补充10%"
            },
            "cache": {
                "statements": 4763,
                "covered": 3773,
                "coverage": 79.21,
                "status": "🟡 补充11%"
            },
            "logging": {
                "statements": 7143,
                "covered": 4972,
                "coverage": 69.61,
                "status": "⚠️ 补充20%"
            },
            "versioning": {
                "statements": 1249,
                "covered": 514,
                "coverage": 41.0,
                "status": "⚠️ 补充39%"
            },
        }
        
        # 待验证模块（基于测试充分度推测）
        self.unverified_modules = {
            "health": {"statements": 28544, "estimated_coverage": 80, "priority": "🟡高"},
            "security": {"statements": 13947, "estimated_coverage": 75, "priority": "🔴极高"},
            "utils": {"statements": 16100, "estimated_coverage": 75, "priority": "🟢中"},
            "error": {"statements": 3273, "estimated_coverage": 75, "priority": "🟡高"},
            "distributed": {"statements": 3102, "estimated_coverage": 65, "priority": "🟢中"},
            "resource": {"statements": 13407, "estimated_coverage": 55, "priority": "🟡高"},
            "monitoring": {"statements": 16209, "estimated_coverage": 40, "priority": "🟡高"},
            "constants": {"statements": 308, "estimated_coverage": 85, "priority": "🟢中"},
            "ops": {"statements": 370, "estimated_coverage": 40, "priority": "⚪低"},
        }
    
    def calculate_current_overall_coverage(self) -> float:
        """计算当前整体覆盖率"""
        # 已验证模块
        total_stmts = sum(m['statements'] for m in self.verified_modules.values())
        total_covered = sum(m['covered'] for m in self.verified_modules.values())
        
        # 待验证模块（使用估算值）
        for module, data in self.unverified_modules.items():
            total_stmts += data['statements']
            estimated_covered = int(data['statements'] * data['estimated_coverage'] / 100)
            total_covered += estimated_covered
        
        overall_coverage = (total_covered / total_stmts * 100) if total_stmts > 0 else 0
        return overall_coverage
    
    def calculate_boost_plan(self, target_coverage: float = 80.0) -> dict:
        """计算提升计划"""
        print("\n" + "="*80)
        print(f"🎯 基础设施层覆盖率提升至{target_coverage}%计划")
        print("="*80)
        
        # 计算当前覆盖率
        current_coverage = self.calculate_current_overall_coverage()
        gap = target_coverage - current_coverage
        
        print(f"\n当前推算覆盖率: {current_coverage:.2f}%")
        print(f"目标覆盖率: {target_coverage}%")
        print(f"差距: {gap:.2f}%")
        
        # 计算总语句数
        total_stmts = sum(m['statements'] for m in self.verified_modules.values())
        total_stmts += sum(m['statements'] for m in self.unverified_modules.values())
        
        # 需要额外覆盖的语句数
        additional_coverage_needed = int(total_stmts * gap / 100)
        
        print(f"\n基础设施层总语句数: {total_stmts:,}")
        print(f"需要额外覆盖语句数: {additional_coverage_needed:,}")
        
        # 生成模块级别的提升计划
        print(f"\n{'='*80}")
        print("📋 模块提升计划")
        print(f"{'='*80}")
        
        boost_plan = []
        
        # 已验证模块的提升需求
        print(f"\n📌 已验证模块提升需求:")
        for module, data in self.verified_modules.items():
            module_target = self._get_module_target(module)
            if data['coverage'] < module_target:
                gap_pct = module_target - data['coverage']
                gap_stmts = int(data['statements'] * gap_pct / 100)
                estimated_tests = max(int(gap_stmts / 10), 5)
                
                boost_plan.append({
                    "module": module,
                    "current": data['coverage'],
                    "target": module_target,
                    "gap_percent": gap_pct,
                    "gap_statements": gap_stmts,
                    "estimated_tests": estimated_tests,
                    "priority": self._get_module_priority(module),
                    "type": "verified"
                })
                
                print(f"  {module:15} | {data['coverage']:5.1f}% → {module_target}% | "
                      f"差距: {gap_pct:5.1f}% | 需新增: ~{estimated_tests}个测试")
        
        # 待验证模块的提升需求
        print(f"\n📌 待验证模块提升需求:")
        for module, data in self.unverified_modules.items():
            module_target = self._get_module_target(module)
            est_coverage = data['estimated_coverage']
            
            if est_coverage < module_target:
                gap_pct = module_target - est_coverage
                gap_stmts = int(data['statements'] * gap_pct / 100)
                estimated_tests = max(int(gap_stmts / 10), 5)
                
                boost_plan.append({
                    "module": module,
                    "current": est_coverage,
                    "target": module_target,
                    "gap_percent": gap_pct,
                    "gap_statements": gap_stmts,
                    "estimated_tests": estimated_tests,
                    "priority": data['priority'],
                    "type": "unverified"
                })
                
                print(f"  {module:15} | {est_coverage:5}% → {module_target}% | "
                      f"差距: {gap_pct:5.1f}% | 需新增: ~{estimated_tests}个测试")
        
        # 按优先级和差距排序
        priority_order = {"🔴极高": 0, "🟡高": 1, "🟢中": 2, "⚪低": 3}
        boost_plan.sort(key=lambda x: (priority_order.get(x['priority'], 9), -x['gap_statements']))
        
        # 分阶段计划
        total_tests_needed = sum(p['estimated_tests'] for p in boost_plan)
        total_gap_stmts = sum(p['gap_statements'] for p in boost_plan)
        
        print(f"\n{'='*80}")
        print("📊 总体需求")
        print(f"{'='*80}")
        print(f"需要提升的模块数: {len(boost_plan)}个")
        print(f"需要新增测试用例: ~{total_tests_needed}个")
        print(f"需要覆盖语句数: ~{total_gap_stmts:,}行")
        
        # 生成分阶段计划
        self.generate_phased_plan(boost_plan, target_coverage)
        
        # 保存计划
        self.save_boost_plan(boost_plan, current_coverage, target_coverage)
        
        return boost_plan
    
    def _get_module_target(self, module: str) -> float:
        """获取模块目标覆盖率"""
        high_risk = ["config", "cache", "logging", "security"]
        medium_risk = ["health", "error", "resource", "monitoring"]
        low_risk = ["core", "utils", "distributed", "constants"]
        very_low = ["versioning", "ops"]
        
        if module in high_risk:
            return 90.0
        elif module in medium_risk:
            return 80.0
        elif module in low_risk:
            return 70.0
        else:
            return 60.0
    
    def _get_module_priority(self, module: str) -> str:
        """获取模块优先级"""
        high_risk = ["config", "cache", "logging", "security"]
        medium_risk = ["health", "error", "resource", "monitoring"]
        
        if module in high_risk:
            return "🔴极高"
        elif module in medium_risk:
            return "🟡高"
        else:
            return "🟢中"
    
    def generate_phased_plan(self, boost_plan: list, target: float):
        """生成分阶段计划"""
        print(f"\n{'='*80}")
        print("📅 分阶段执行计划")
        print(f"{'='*80}")
        
        # 阶段1：快速提升（高优先级小差距）
        phase1 = [p for p in boost_plan if p['gap_statements'] < 500]
        if phase1:
            print(f"\n🔴 阶段1：快速补齐（1-2天）")
            print(f"   目标: 小差距模块快速达标")
            print(f"   模块数: {len(phase1)}个")
            print(f"   预计新增测试: ~{sum(p['estimated_tests'] for p in phase1)}个")
            for p in phase1[:5]:
                print(f"   - {p['module']:15} ({p['current']}% → {p['target']}%, 需{p['estimated_tests']}个测试)")
        
        # 阶段2：重点提升（中型差距）
        phase2 = [p for p in boost_plan if 500 <= p['gap_statements'] < 1500]
        if phase2:
            print(f"\n🟡 阶段2：重点提升（3-5天）")
            print(f"   目标: 中型差距模块达标")
            print(f"   模块数: {len(phase2)}个")
            print(f"   预计新增测试: ~{sum(p['estimated_tests'] for p in phase2)}个")
            for p in phase2[:5]:
                print(f"   - {p['module']:15} ({p['current']}% → {p['target']}%, 需{p['estimated_tests']}个测试)")
        
        # 阶段3：攻坚提升（大型差距）
        phase3 = [p for p in boost_plan if p['gap_statements'] >= 1500]
        if phase3:
            print(f"\n⚠️  阶段3：攻坚提升（1-2周）")
            print(f"   目标: 大型差距模块达标")
            print(f"   模块数: {len(phase3)}个")
            print(f"   预计新增测试: ~{sum(p['estimated_tests'] for p in phase3)}个")
            for p in phase3:
                print(f"   - {p['module']:15} ({p['current']}% → {p['target']}%, 需{p['estimated_tests']}个测试)")
    
    def save_boost_plan(self, boost_plan: list, current: float, target: float):
        """保存提升计划"""
        test_logs_dir = self.base_dir / "test_logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plan_file = test_logs_dir / f"boost_to_80_plan_{timestamp}.json"
        
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "current_coverage": current,
                "target_coverage": target,
                "gap": target - current,
                "boost_plan": boost_plan,
                "summary": {
                    "modules_to_boost": len(boost_plan),
                    "total_tests_needed": sum(p['estimated_tests'] for p in boost_plan),
                    "total_statements_to_cover": sum(p['gap_statements'] for p in boost_plan),
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 提升计划已保存: {plan_file}")


def main():
    """主函数"""
    booster = CoverageBooster()
    boost_plan = booster.calculate_boost_plan(target_coverage=80.0)
    return 0


if __name__ == "__main__":
    exit(main())

