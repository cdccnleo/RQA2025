#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略框架迁移验证脚本
Strategy Framework Migration Validation Script

验证策略框架迁移合并工作的完成情况和正确性。
"""

import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MigrationValidator:
    """迁移验证器"""

    def __init__(self):
        """初始化验证器"""
        self.project_root = project_root
        self.src_dir = project_root / "src"

    def validate_migration(self) -> Dict[str, Any]:
        """
        验证迁移结果

        Returns:
            Dict[str, Any]: 验证结果
        """
        print("🔍 开始验证策略框架迁移结果...")

        results = {
            "strategy_structure": self._validate_strategy_structure(),
            "backtest_migration": self._validate_backtest_migration(),
            "strategies_migration": self._validate_strategies_migration(),
            "workspace_migration": self._validate_workspace_migration(),
            "optimization_migration": self._validate_optimization_migration(),
            "monitoring_migration": self._validate_monitoring_migration(),
            "code_adaptation": self._validate_code_adaptation()
        }

        self._print_validation_summary(results)
        return results

    def _validate_strategy_structure(self) -> Dict[str, Any]:
        """验证策略服务层结构"""
        strategy_dir = self.src_dir / "strategy"

        expected_structure = {
            "interfaces/": ["strategy_interfaces.py", "backtest_interfaces.py", "optimization_interfaces.py", "monitoring_interfaces.py"],
            "core/": ["strategy_service.py", "business_process_orchestrator.py", "dependency_config.py"],
            "strategies/": ["base_strategy.py", "strategy_factory.py", "momentum_strategy.py", "mean_reversion_strategy.py"],
            "backtest/": ["backtest_engine.py", "backtest_service.py", "backtest_persistence.py"],
            "optimization/": ["optimization_service.py", "parameter_optimizer.py"],
            "monitoring/": ["monitoring_service.py", "alert_service.py"],
            "workspace/": ["web_api.py", "web_server.py", "visualization_service.py"],
            "lifecycle/": ["strategy_lifecycle_manager.py"]
        }

        structure_results = {}
        for dir_path, expected_files in expected_structure.items():
            full_dir = strategy_dir / dir_path
            if full_dir.exists():
                actual_files = [f.name for f in full_dir.glob(
                    "*.py") if not f.name.startswith("__")]
                missing_files = [f for f in expected_files if f not in actual_files]
                structure_results[dir_path] = {
                    "exists": True,
                    "expected_files": expected_files,
                    "actual_files": actual_files,
                    "missing_files": missing_files,
                    "complete": len(missing_files) == 0
                }
            else:
                structure_results[dir_path] = {
                    "exists": False,
                    "complete": False
                }

        return structure_results

    def _validate_backtest_migration(self) -> Dict[str, Any]:
        """验证回测功能迁移"""
        backtest_dir = self.src_dir / "strategy" / "backtest"
        results = {
            "total_files": 0,
            "analysis_files": 0,
            "engine_files": 0,
            "evaluation_files": 0,
            "optimization_files": 0
        }

        if backtest_dir.exists():
            for file_path in backtest_dir.rglob("*.py"):
                if file_path.is_file():
                    results["total_files"] += 1
                    file_name = file_path.name

                    # 分类统计
                    if "analysis" in str(file_path):
                        results["analysis_files"] += 1
                    elif "engine" in str(file_path):
                        results["engine_files"] += 1
                    elif "evaluation" in str(file_path):
                        results["evaluation_files"] += 1
                    elif "optimization" in str(file_path):
                        results["optimization_files"] += 1

        results["migration_complete"] = results["total_files"] > 50  # 预期至少50个文件
        return results

    def _validate_strategies_migration(self) -> Dict[str, Any]:
        """验证策略实现迁移"""
        strategies_dir = self.src_dir / "strategy" / "strategies"
        results = {
            "total_strategies": 0,
            "basic_strategies": 0,
            "china_strategies": 0,
            "advanced_strategies": 0
        }

        if strategies_dir.exists():
            for file_path in strategies_dir.rglob("*.py"):
                if file_path.is_file() and not file_path.name.startswith("__"):
                    results["total_strategies"] += 1
                    file_name = file_path.name

                    # 分类统计
                    if "basic" in str(file_path):
                        results["basic_strategies"] += 1
                    elif "china" in str(file_path):
                        results["china_strategies"] += 1
                    elif any(keyword in file_name for keyword in ["reinforcement", "arbitrage", "ml", "enhanced"]):
                        results["advanced_strategies"] += 1

        results["migration_complete"] = results["total_strategies"] > 20  # 预期至少20个策略文件
        return results

    def _validate_workspace_migration(self) -> Dict[str, Any]:
        """验证工作空间迁移"""
        workspace_dir = self.src_dir / "strategy" / "workspace"
        expected_components = [
            "web_api.py", "web_server.py", "visualization_service.py",
            "debug_service.py", "auth_service.py", "static/index.html"
        ]

        results = {
            "expected_components": expected_components,
            "found_components": [],
            "missing_components": []
        }

        if workspace_dir.exists():
            for component in expected_components:
                component_path = workspace_dir / component
                if component_path.exists():
                    results["found_components"].append(component)
                else:
                    results["missing_components"].append(component)

        results["migration_complete"] = len(results["missing_components"]) == 0
        return results

    def _validate_optimization_migration(self) -> Dict[str, Any]:
        """验证优化功能迁移"""
        optimization_dir = self.src_dir / "strategy" / "optimization"
        expected_files = [
            "optimization_service.py", "parameter_optimizer.py",
            "walk_forward_optimizer.py", "genetic_optimizer.py"
        ]

        results = {
            "expected_files": expected_files,
            "found_files": [],
            "missing_files": []
        }

        if optimization_dir.exists():
            for file in expected_files:
                file_path = optimization_dir / file
                if file_path.exists():
                    results["found_files"].append(file)
                else:
                    results["missing_files"].append(file)

        results["migration_complete"] = len(results["missing_files"]) == 0
        return results

    def _validate_monitoring_migration(self) -> Dict[str, Any]:
        """验证监控功能迁移"""
        monitoring_dir = self.src_dir / "strategy" / "monitoring"
        results = {
            "total_files": 0,
            "has_service": False,
            "has_alert": False,
            "has_analysis": False
        }

        if monitoring_dir.exists():
            for file_path in monitoring_dir.rglob("*.py"):
                if file_path.is_file():
                    results["total_files"] += 1
                    file_name = file_path.name

                    if "monitoring_service" in file_name:
                        results["has_service"] = True
                    elif "alert" in file_name:
                        results["has_alert"] = True
                    elif "analysis" in str(file_path):
                        results["has_analysis"] = True

        results["migration_complete"] = (
            results["total_files"] > 10 and
            results["has_service"] and
            results["has_alert"]
        )
        return results

    def _validate_code_adaptation(self) -> Dict[str, Any]:
        """验证代码适配"""
        results = {
            "adapted_files": 0,
            "total_checked": 0,
            "adaptation_patterns": []
        }

        # 检查关键文件的适配情况
        key_files = [
            "src/strategy/strategies/base_strategy.py",
            "src/strategy/strategies/mean_reversion_strategy.py",
            "src/strategy/strategies/cross_market_arbitrage.py"
        ]

        for file_path_str in key_files:
            file_path = self.project_root / file_path_str
            if file_path.exists():
                results["total_checked"] += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查适配模式
                    if "from src.strategy" in content:
                        results["adapted_files"] += 1
                        results["adaptation_patterns"].append(f"{file_path_str}: unified imports")

                except Exception as e:
                    results["adaptation_patterns"].append(f"{file_path_str}: error - {e}")

        results["adaptation_complete"] = results["adapted_files"] == results["total_checked"]
        return results

    def _print_validation_summary(self, results: Dict[str, Any]):
        """打印验证摘要"""
        print("\n" + "="*80)
        print("🎯 策略框架迁移验证结果")
        print("="*80)

        success_count = 0
        total_checks = len(results)

        for check_name, check_result in results.items():
            if isinstance(check_result, dict) and "migration_complete" in check_result:
                if check_result["migration_complete"]:
                    success_count += 1
                    status = "✅"
                else:
                    status = "❌"
            else:
                status = "ℹ️"

            print(f"{status} {check_name}")

        success_rate = (success_count / total_checks * 100) if total_checks > 0 else 0

        print(
            f"\n📊 总体状态: {'✅ 迁移成功' if success_rate >= 90 else '⚠️ 需要检查' if success_rate >= 70 else '❌ 迁移失败'}")
        print(f"📋 检查项目: {total_checks}")
        print(f"✅ 通过项目: {success_count}")
        print(f"📈 成功率: {success_rate:.1f}%")

        # 详细结果
        print("\n📋 详细验证结果:")
        for check_name, check_result in results.items():
            if isinstance(check_result, dict):
                print(f"\n🔍 {check_name}:")
                for key, value in check_result.items():
                    if key != "migration_complete":
                        print(f"  • {key}: {value}")

        print("\n" + "="*80)
        print("✅ 策略框架迁移验证完成！")
        print("="*80)


def main():
    """主函数"""
    try:
        validator = MigrationValidator()
        results = validator.validate_migration()

        # 根据验证结果设置退出码
        success_count = sum(1 for r in results.values()
                            if isinstance(r, dict) and r.get("migration_complete", False))

        if success_count >= len(results) * 0.9:  # 90%成功率
            print("\n🎉 策略框架迁移验证通过！所有功能已成功迁移。")
            sys.exit(0)
        else:
            print(f"\n⚠️ 策略框架迁移验证完成，但发现 {len(results) - success_count} 个问题需要处理。")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 验证过程异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
