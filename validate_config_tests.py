#!/usr/bin/env python3
"""
配置管理测试验证脚本
逐个验证核心组件的测试覆盖情况
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class ConfigTestValidator:
    """配置管理测试验证器"""

    def __init__(self):
        self.results = {
            'imports': {},
            'functionality': {},
            'coverage': {},
            'recommendations': []
        }

    def validate_imports(self):
        """验证核心模块导入"""

        print("🔍 验证核心模块导入...")

        core_modules = {
            'factory': 'src.infrastructure.config.core.factory',
            'manager': 'src.infrastructure.config.core.unified_manager',
            'storage': 'src.infrastructure.config.core.config_storage',
            'validators': 'src.infrastructure.config.core.validators',
            'service': 'src.infrastructure.config.core.config_service',
            'strategy': 'src.infrastructure.config.core.config_strategy'
        }

        for module_name, module_path in core_modules.items():
            try:
                __import__(module_path)
                self.results['imports'][module_name] = True
                print(f"  ✅ {module_name}: 导入成功")
            except ImportError as e:
                self.results['imports'][module_name] = False
                print(f"  ❌ {module_name}: 导入失败 - {e}")
            except Exception as e:
                self.results['imports'][module_name] = False
                print(f"  ⚠️  {module_name}: 导入异常 - {e}")

    def validate_functionality(self):
        """验证核心功能"""

        print("\n🔧 验证核心功能...")

        # 测试工厂功能
        try:
            from src.infrastructure.config.core.factory import ConfigFactory
            factory = ConfigFactory()
            self.results['functionality']['factory'] = True
            print("  ✅ 工厂功能正常")
        except Exception as e:
            self.results['functionality']['factory'] = False
            print(f"  ❌ 工厂功能异常: {e}")

        # 测试管理器功能
        try:
            from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
            manager = UnifiedConfigManager()
            self.results['functionality']['manager'] = True
            print("  ✅ 管理器功能正常")
        except Exception as e:
            self.results['functionality']['manager'] = False
            print(f"  ❌ 管理器功能异常: {e}")

        # 测试存储功能
        try:
            from src.infrastructure.config.core.config_storage import StorageConfig
            config = StorageConfig()
            self.results['functionality']['storage'] = True
            print("  ✅ 存储功能正常")
        except Exception as e:
            self.results['functionality']['storage'] = False
            print(f"  ❌ 存储功能异常: {e}")

    def analyze_test_coverage(self):
        """分析测试覆盖情况"""

        print("\n📊 分析测试覆盖...")

        test_dir = project_root / "tests" / "unit" / "infrastructure" / "config"
        if not test_dir.exists():
            print("  ❌ 测试目录不存在")
            return

        test_files = list(test_dir.glob("test_*.py"))
        print(f"  发现 {len(test_files)} 个测试文件")

        # 分析覆盖的核心模块
        core_modules = {
            'factory': ['factory', 'config_factory'],
            'manager': ['unified_manager', 'config_manager'],
            'storage': ['storage', 'config_storage'],
            'validators': ['validator', 'config_validator'],
            'service': ['service', 'config_service'],
            'strategy': ['strategy', 'config_strategy']
        }

        coverage_results = {}
        for module, patterns in core_modules.items():
            covered = False
            for pattern in patterns:
                for test_file in test_files:
                    if pattern in test_file.name:
                        covered = True
                        break
                if covered:
                    break

            coverage_results[module] = covered
            status = "✅" if covered else "❌"
            print(f"  {status} {module} 模块测试覆盖")

        self.results['coverage'] = coverage_results

        # 计算覆盖率
        total_modules = len(core_modules)
        covered_modules = sum(1 for covered in coverage_results.values() if covered)
        coverage_rate = covered_modules / total_modules * 100

        print(f"测试覆盖率: {coverage_rate:.1f}%")
        if coverage_rate >= 90:
            print("  ✅ 测试覆盖率优秀")
        elif coverage_rate >= 75:
            print("  ⚠️ 测试覆盖率良好")
        else:
            print("  ❌ 测试覆盖率需要改进")

    def generate_recommendations(self):
        """生成改进建议"""

        print("\n💡 生成改进建议...")

        recommendations = []

        # 检查导入问题
        failed_imports = [m for m, success in self.results['imports'].items() if not success]
        if failed_imports:
            recommendations.append(f"修复导入问题: {', '.join(failed_imports)}")

        # 检查功能问题
        failed_functions = [f for f, success in self.results['functionality'].items()
                            if not success]
        if failed_functions:
            recommendations.append(f"修复功能问题: {', '.join(failed_functions)}")

        # 检查覆盖问题
        if 'coverage' in self.results:
            uncovered = [m for m, covered in self.results['coverage'].items() if not covered]
            if uncovered:
                recommendations.append(f"补充测试覆盖: {', '.join(uncovered)}")

        # 一般性建议
        recommendations.extend([
            "运行完整的单元测试套件",
            "生成详细的覆盖率报告",
            "检查边界条件和错误处理",
            "验证向后兼容性"
        ])

        self.results['recommendations'] = recommendations

        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    def generate_report(self):
        """生成完整报告"""

        print("\n" + "="*60)
        print("配置管理测试验证报告")
        print("="*60)

        # 导入状态
        print("\n📦 模块导入状态:")
        for module, success in self.results['imports'].items():
            status = "✅" if success else "❌"
            print(f"  {status} {module}")

        # 功能状态
        print("\n🔧 核心功能状态:")
        for function, success in self.results['functionality'].items():
            status = "✅" if success else "❌"
            print(f"  {status} {function}")

        # 覆盖状态
        if 'coverage' in self.results:
            print("\n📊 测试覆盖状态:")
            for module, covered in self.results['coverage'].items():
                status = "✅" if covered else "❌"
                print(f"  {status} {module}")

        # 建议
        if self.results['recommendations']:
            print("\n💡 改进建议:")
            for rec in self.results['recommendations']:
                print(f"  • {rec}")

        # 总体评估
        print("\n🏆 总体评估:")

        import_success = sum(1 for s in self.results['imports'].values() if s)
        function_success = sum(1 for s in self.results['functionality'].values() if s)

        if import_success >= 5 and function_success >= 2:
            print("  ✅ 配置管理模块测试基础良好")
        elif import_success >= 3:
            print("  ⚠️ 配置管理模块需要改进")
        else:
            print("  ❌ 配置管理模块测试基础薄弱")

        print("\n报告生成完成")


def main():
    """主函数"""

    print("配置管理测试验证器")
    print("-" * 40)

    validator = ConfigTestValidator()

    try:
        validator.validate_imports()
        validator.validate_functionality()
        validator.analyze_test_coverage()
        validator.generate_recommendations()
        validator.generate_report()

    except Exception as e:
        print(f"验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
