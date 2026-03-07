#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
识别基础设施层低覆盖率模块
生成优先级列表和改进计划
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class LowCoverageIdentifier:
    """低覆盖率模块识别器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        
        # 基于之前的评估结果
        self.module_data = {
            "cache": {"src_lines": 7653, "test_lines": 22481, "test_ratio": 293.8, "coverage": None, "priority": "🔴极高"},
            "config": {"src_lines": 20838, "test_lines": 44957, "test_ratio": 215.7, "coverage": None, "priority": "🔴极高"},
            "constants": {"src_lines": 308, "test_lines": 1924, "test_ratio": 624.7, "coverage": None, "priority": "🟢中"},
            "core": {"src_lines": 1613, "test_lines": 902, "test_ratio": 55.9, "coverage": 74.82, "priority": "🟢中"},
            "distributed": {"src_lines": 3102, "test_lines": 4502, "test_ratio": 145.1, "coverage": None, "priority": "🟢中"},
            "error": {"src_lines": 3273, "test_lines": 8219, "test_ratio": 251.1, "coverage": None, "priority": "🟡高"},
            "health": {"src_lines": 28544, "test_lines": 66679, "test_ratio": 233.6, "coverage": None, "priority": "🟡高"},
            "logging": {"src_lines": 13138, "test_lines": 20977, "test_ratio": 159.7, "coverage": None, "priority": "🔴极高"},
            "monitoring": {"src_lines": 16209, "test_lines": 7417, "test_ratio": 45.8, "coverage": None, "priority": "🟡高"},
            "ops": {"src_lines": 370, "test_lines": 160, "test_ratio": 43.2, "coverage": None, "priority": "⚪低"},
            "resource": {"src_lines": 13407, "test_lines": 12274, "test_ratio": 91.5, "coverage": None, "priority": "🟡高"},
            "security": {"src_lines": 13947, "test_lines": 23527, "test_ratio": 168.7, "coverage": None, "priority": "🔴极高"},
            "utils": {"src_lines": 16100, "test_lines": 29791, "test_ratio": 185.0, "coverage": None, "priority": "🟢中"},
            "versioning": {"src_lines": 2435, "test_lines": 379, "test_ratio": 15.6, "coverage": 31.96, "priority": "⚪低"},
        }
        
        # 覆盖率目标
        self.coverage_targets = {
            "🔴极高": 90.0,  # 极高风险模块
            "🟡高": 80.0,    # 高风险模块
            "🟢中": 70.0,    # 中风险模块
            "⚪低": 60.0,    # 低风险模块
        }
        
    def identify_low_coverage_modules(self) -> List[Dict]:
        """识别低覆盖率模块"""
        low_coverage_modules = []
        
        for module, data in self.module_data.items():
            priority = data['priority']
            target = self.coverage_targets[priority]
            current_coverage = data.get('coverage')
            
            # 推测覆盖率（基于测试充分度）
            if current_coverage is None:
                if data['test_ratio'] >= 200:
                    estimated_coverage = 85.0
                elif data['test_ratio'] >= 150:
                    estimated_coverage = 75.0
                elif data['test_ratio'] >= 100:
                    estimated_coverage = 65.0
                elif data['test_ratio'] >= 60:
                    estimated_coverage = 55.0
                else:
                    estimated_coverage = 40.0
            else:
                estimated_coverage = current_coverage
            
            gap = target - estimated_coverage
            
            if gap > 0:
                low_coverage_modules.append({
                    "module": module,
                    "priority": priority,
                    "src_lines": data['src_lines'],
                    "test_lines": data['test_lines'],
                    "test_ratio": data['test_ratio'],
                    "current_coverage": estimated_coverage,
                    "target_coverage": target,
                    "gap": gap,
                    "status": "verified" if current_coverage is not None else "estimated",
                    "improvement_level": self._calculate_improvement_level(gap, data['src_lines']),
                })
        
        # 按优先级和差距排序
        priority_order = {"🔴极高": 0, "🟡高": 1, "🟢中": 2, "⚪低": 3}
        low_coverage_modules.sort(key=lambda x: (priority_order[x['priority']], -x['gap']))
        
        return low_coverage_modules
    
    def _calculate_improvement_level(self, gap: float, src_lines: int) -> str:
        """计算改进难度等级"""
        effort = gap * src_lines / 100  # 需要测试的代码行数
        
        if effort > 3000:
            return "🔴 高难度（大型模块+大差距）"
        elif effort > 1500:
            return "🟡 中等难度"
        elif effort > 500:
            return "🟢 较低难度"
        else:
            return "✅ 低难度"
    
    def generate_improvement_plan(self, low_coverage_modules: List[Dict]) -> Dict:
        """生成改进计划"""
        # 分阶段计划
        phase1 = []  # 紧急修复
        phase2 = []  # 重点提升
        phase3 = []  # 全面达标
        
        for module in low_coverage_modules:
            if module['priority'] == "🔴极高" and module['gap'] > 5:
                phase1.append(module)
            elif module['gap'] > 10 or module['priority'] in ["🔴极高", "🟡高"]:
                phase2.append(module)
            else:
                phase3.append(module)
        
        return {
            "phase1_urgent": {
                "name": "阶段一：紧急修复（1周）",
                "target": "极高风险模块达标",
                "modules": phase1,
                "expected_coverage_gain": sum(m['gap'] * m['src_lines'] / sum(self.module_data[x]['src_lines'] for x in self.module_data) for m in phase1),
            },
            "phase2_important": {
                "name": "阶段二：重点提升（2周）",
                "target": "高风险模块达标",
                "modules": phase2,
                "expected_coverage_gain": sum(m['gap'] * m['src_lines'] / sum(self.module_data[x]['src_lines'] for x in self.module_data) for m in phase2),
            },
            "phase3_complete": {
                "name": "阶段三：全面达标（1个月）",
                "target": "所有模块达到目标",
                "modules": phase3,
                "expected_coverage_gain": sum(m['gap'] * m['src_lines'] / sum(self.module_data[x]['src_lines'] for x in self.module_data) for m in phase3),
            }
        }
    
    def calculate_test_requirements(self, low_coverage_modules: List[Dict]) -> Dict:
        """计算测试需求"""
        total_src_lines = sum(self.module_data[m]['src_lines'] for m in self.module_data)
        
        requirements = {}
        for module_info in low_coverage_modules:
            module = module_info['module']
            gap = module_info['gap']
            src_lines = module_info['src_lines']
            
            # 估算需要新增的测试用例数
            # 假设每个测试用例覆盖约10-20行代码
            lines_to_cover = src_lines * gap / 100
            estimated_tests_needed = int(lines_to_cover / 15)  # 平均15行/测试
            
            # 估算需要新增的测试代码行数
            # 假设测试代码与源代码的比率为1:1（覆盖新代码）
            estimated_test_lines = int(lines_to_cover * 1.2)  # 多20%用于setup/teardown
            
            requirements[module] = {
                "lines_to_cover": int(lines_to_cover),
                "estimated_tests_needed": max(estimated_tests_needed, 5),  # 至少5个测试
                "estimated_test_lines": estimated_test_lines,
                "current_test_lines": module_info['test_lines'],
                "target_test_lines": module_info['test_lines'] + estimated_test_lines,
            }
        
        return requirements
    
    def generate_report(self):
        """生成识别报告"""
        print("\n" + "="*80)
        print("🔍 基础设施层低覆盖率模块识别报告")
        print("="*80)
        print(f"识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 识别低覆盖模块
        low_coverage_modules = self.identify_low_coverage_modules()
        
        print(f"📊 识别结果：共 {len(low_coverage_modules)} 个模块需要改进\n")
        
        # 按优先级分组显示
        priority_groups = {}
        for module in low_coverage_modules:
            priority = module['priority']
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(module)
        
        for priority in ["🔴极高", "🟡高", "🟢中", "⚪低"]:
            if priority in priority_groups:
                modules = priority_groups[priority]
                print(f"\n{priority} 风险模块（目标覆盖率：{self.coverage_targets[priority]}%）")
                print("-" * 80)
                for m in modules:
                    status_icon = "✓" if m['status'] == 'verified' else "?"
                    print(f"  {status_icon} {m['module']:15} | "
                          f"当前: {m['current_coverage']:5.1f}% | "
                          f"目标: {m['target_coverage']:5.1f}% | "
                          f"差距: {m['gap']:5.1f}% | "
                          f"{m['improvement_level']}")
        
        # 生成改进计划
        print("\n" + "="*80)
        print("📋 改进计划")
        print("="*80)
        
        improvement_plan = self.generate_improvement_plan(low_coverage_modules)
        
        total_expected_gain = 0
        for phase_key, phase in improvement_plan.items():
            print(f"\n{phase['name']}")
            print(f"目标: {phase['target']}")
            print(f"模块数: {len(phase['modules'])}")
            print(f"预期覆盖率提升: +{phase['expected_coverage_gain']:.2f}%")
            
            if phase['modules']:
                print("  包含模块:")
                for m in phase['modules']:
                    print(f"    - {m['module']:15} (差距: {m['gap']:5.1f}%)")
            
            total_expected_gain += phase['expected_coverage_gain']
        
        # 计算测试需求
        print("\n" + "="*80)
        print("📝 测试需求评估")
        print("="*80)
        
        test_requirements = self.calculate_test_requirements(low_coverage_modules)
        
        total_tests_needed = sum(r['estimated_tests_needed'] for r in test_requirements.values())
        total_test_lines_needed = sum(r['estimated_test_lines'] for r in test_requirements.values())
        
        print(f"\n预计新增测试用例总数: ~{total_tests_needed}个")
        print(f"预计新增测试代码行数: ~{total_test_lines_needed:,}行")
        print(f"当前基础设施层整体覆盖率: 53.39%")
        print(f"预期提升后覆盖率: ~{53.39 + total_expected_gain:.1f}%")
        
        print(f"\n各模块测试需求:")
        for module in sorted(test_requirements.keys(), key=lambda x: test_requirements[x]['estimated_tests_needed'], reverse=True):
            req = test_requirements[module]
            print(f"  {module:15} | "
                  f"需新增测试: {req['estimated_tests_needed']:3}个 | "
                  f"需新增代码: {req['estimated_test_lines']:5}行 | "
                  f"覆盖代码: {req['lines_to_cover']:5}行")
        
        # 保存报告
        self.save_report(low_coverage_modules, improvement_plan, test_requirements, total_expected_gain)
        
        return {
            "low_coverage_modules": low_coverage_modules,
            "improvement_plan": improvement_plan,
            "test_requirements": test_requirements,
            "expected_total_gain": total_expected_gain,
        }
    
    def save_report(self, low_coverage_modules, improvement_plan, test_requirements, total_gain):
        """保存报告"""
        test_logs_dir = self.base_dir / "test_logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON格式
        json_file = test_logs_dir / f"low_coverage_identification_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "identification_time": datetime.now().isoformat(),
                "low_coverage_modules": low_coverage_modules,
                "improvement_plan": improvement_plan,
                "test_requirements": test_requirements,
                "summary": {
                    "modules_need_improvement": len(low_coverage_modules),
                    "total_tests_needed": sum(r['estimated_tests_needed'] for r in test_requirements.values()),
                    "expected_coverage_gain": total_gain,
                    "current_coverage": 53.39,
                    "expected_coverage": 53.39 + total_gain,
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 详细报告已保存: {json_file}")


def main():
    """主函数"""
    identifier = LowCoverageIdentifier()
    result = identifier.generate_report()
    return 0


if __name__ == "__main__":
    exit(main())

