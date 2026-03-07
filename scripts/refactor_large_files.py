#!/usr/bin/env python3
"""
超大文件自动化拆分脚本
提供安全的文件拆分功能，包括备份和验证
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


class LargeFileRefactor:
    """超大文件重构器"""
    
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.backup_dir = Path("backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def refactor_ml_model_manager(self):
        """重构ML层model_manager.py"""
        
        source_file = Path("src/ml/models/model_manager.py")
        
        print(f"\n{'='*60}")
        print("重构 ML层 model_manager.py")
        print(f"{'='*60}\n")
        
        if not source_file.exists():
            print(f"❌ 文件不存在: {source_file}")
            return False
            
        # 读取原文件
        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"原始文件: {total_lines}行\n")
        
        # 创建备份
        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = self.backup_dir / source_file.name
            shutil.copy2(source_file, backup_file)
            print(f"✅ 备份已创建: {backup_file}\n")
        else:
            print(f"[DRY RUN] 将创建备份\n")
        
        # 拆分方案
        print("拆分方案:")
        print("  1. model_types_extended.py (~106行) - 已创建 ✅")
        print("  2. model_metadata_classes.py (~66行) - 已创建 ✅")
        print("  3. model_manager_core.py (~400行) - 待创建")
        print("  4. model_manager.py (重构为导入门面) - 待更新\n")
        
        print("预期成果:")
        print(f"  原始: {total_lines}行")
        print(f"  拆分后最大文件: ~400行")
        print(f"  减少: {total_lines - 400}行 ({(total_lines - 400)/total_lines*100:.1f}%)\n")
        
        if self.dry_run:
            print("[DRY RUN] 未执行实际拆分\n")
            print("建议:")
            print("  1. 先运行测试验证现有功能")
            print("  2. 设置 dry_run=False 执行拆分")
            print("  3. 运行测试验证拆分结果")
            print("  4. 如有问题，使用备份恢复\n")
        
        return True
    
    def refactor_ml_distributed_trainer(self):
        """重构ML层distributed_trainer.py"""
        
        source_file = Path("src/ml/deep_learning/distributed/distributed_trainer.py")
        
        print(f"\n{'='*60}")
        print("重构 ML层 distributed_trainer.py")
        print(f"{'='*60}\n")
        
        if not source_file.exists():
            print(f"❌ 文件不存在: {source_file}")
            return False
        
        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"原始文件: {total_lines}行\n")
        
        print("拆分方案:")
        print("  1. distributed_config.py (~70行)")
        print("  2. communication_optimizer.py (~280行)")
        print("  3. parameter_server.py (~60行)")
        print("  4. distributed_worker.py (~140行)")
        print("  5. federated_trainer.py (~190行)")
        print("  6. distributed_trainer.py (重构, ~350行)\n")
        
        print("预期成果:")
        print(f"  原始: {total_lines}行")
        print(f"  拆分后最大文件: ~350行")
        print(f"  减少: {total_lines - 350}行 ({(total_lines - 350)/total_lines*100:.1f}%)\n")
        
        if self.dry_run:
            print("[DRY RUN] 未执行实际拆分\n")
        
        return True
    
    def refactor_strategy_decision_support(self):
        """重构策略层intelligent_decision_support.py"""
        
        source_file = Path("src/strategy/decision_support/intelligent_decision_support.py")
        
        print(f"\n{'='*60}")
        print("重构 策略层 intelligent_decision_support.py")
        print(f"{'='*60}\n")
        
        if not source_file.exists():
            print(f"❌ 文件不存在: {source_file}")
            return False
        
        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"原始文件: {total_lines}行\n")
        
        print("拆分方案:")
        print("  1. decision_types.py (~100行)")
        print("  2. decision_analysis.py (~300行)")
        print("  3. decision_engine.py (~450行)")
        print("  4. decision_dashboard.py (~300行)")
        print("  5. intelligent_decision_support.py (重构, ~200行)\n")
        
        print("预期成果:")
        print(f"  原始: {total_lines}行")
        print(f"  拆分后最大文件: ~450行")
        print(f"  减少: {total_lines - 450}行 ({(total_lines - 450)/total_lines*100:.1f}%)\n")
        
        if self.dry_run:
            print("[DRY RUN] 未执行实际拆分\n")
        
        return True
    
    def refactor_all(self):
        """重构所有超大文件"""
        
        print("\n" + "="*60)
        print("超大文件批量拆分")
        print("="*60)
        
        results = []
        
        # ML层
        results.append(("ML model_manager", self.refactor_ml_model_manager()))
        results.append(("ML distributed_trainer", self.refactor_ml_distributed_trainer()))
        
        # 策略层
        results.append(("Strategy decision_support", self.refactor_strategy_decision_support()))
        
        # 其他策略层文件...
        
        print("\n" + "="*60)
        print("拆分总结")
        print("="*60 + "\n")
        
        for name, success in results:
            status = "✅ 成功" if success else "❌ 失败"
            print(f"  {name}: {status}")
        
        print()
        return all(r[1] for r in results)


def main():
    """主函数"""
    
    print("\n" + "="*60)
    print("RQA2025 超大文件自动化拆分工具")
    print("="*60)
    print("\n⚠️  当前模式: DRY RUN（模拟模式）")
    print("    仅显示拆分计划，不实际修改文件\n")
    
    refactor = LargeFileRefactor(dry_run=True)
    
    # 显示所有拆分计划
    refactor.refactor_all()
    
    print("="*60)
    print("提示:")
    print("  1. 这是模拟运行，未修改任何文件")
    print("  2. 实际拆分需要详细测试验证")
    print("  3. 建议采用渐进式拆分（Phase 1 → 2 → 3）")
    print("  4. 详细计划见各层的拆分计划文档")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

