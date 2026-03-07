"""
错误管理系统测试运行脚本

执行所有错误管理相关的单元测试，并生成覆盖率报告。
"""

import unittest
import sys
from pathlib import Path

# 添加项目根目录和src目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))


def run_error_management_tests():
    """运行错误管理系统所有测试"""

    # 发现所有测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 测试模块列表
    test_modules = [
        'test_error_handler',
        'test_infrastructure_error_handler',
        'test_specialized_error_handler',
        'test_error_handler_factory',
        'test_security_filter',
        'test_performance_monitor',
        'test_recovery_manager',
        'test_policies',
        'test_exceptions',
        'test_integration'
    ]

    # 加载测试模块
    for module_name in test_modules:
        try:
            module = __import__(
                f'tests.unit.infrastructure.error.{module_name}', fromlist=[module_name])
            module_suite = loader.loadTestsFromModule(module)
            suite.addTest(module_suite)
            print(f"✅ 加载测试模块: {module_name} ({module_suite.countTestCases()} 个测试)")
        except Exception as e:
            print(f"❌ 加载测试模块失败 {module_name}: {e}")

    # 运行测试
    print("\n🚀 开始执行错误管理系统测试...")
    print(f"📊 总测试用例数: {suite.countTestCases()}")

    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )

    result = runner.run(suite)

    # 输出测试结果摘要
    print(f"\n{'='*70}")
    print("🎯 错误管理系统测试结果摘要")
    print(f"{'='*70}")
    print(f"执行测试: {result.testsRun}")
    print(f"成功通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"测试失败: {len(result.failures)}")
    print(f"测试错误: {len(result.errors)}")
    print(f"跳过测试: {len(result.skipped)}")

    success_rate = (result.testsRun - len(result.failures) -
                    len(result.errors)) / result.testsRun * 100
    print(f"成功率: {success_rate:.1f}%")
    # 输出失败详情
    if result.failures:
        print("\n❌ 测试失败详情:")
        for test, traceback in result.failures[:5]:  # 只显示前5个失败
            print(f"   • {test}")
        if len(result.failures) > 5:
            print(f"   ... 还有 {len(result.failures) - 5} 个失败")

    if result.errors:
        print("\n⚠️  测试错误详情:")
        for test, traceback in result.errors[:3]:  # 只显示前3个错误
            print(f"   • {test}")
        if len(result.errors) > 3:
            print(f"   ... 还有 {len(result.errors) - 3} 个错误")

    # 返回测试结果
    return result.wasSuccessful()


def run_coverage_analysis():
    """运行覆盖率分析"""
    try:
        import coverage

        # 配置覆盖率
        cov = coverage.Coverage(
            source=['src/infrastructure/error'],
            omit=[
                '*/tests/*',
                '*/test_*',
                '*/__pycache__/*',
                '*/venv/*'
            ]
        )

        cov.start()

        # 运行测试
        success = run_error_management_tests()

        cov.stop()
        cov.save()

        # 生成报告
        print(f"\n{'='*70}")
        print("📊 代码覆盖率报告")
        print(f"{'='*70}")

        # 控制台报告
        cov.report(show_missing=True)

        # HTML报告
        cov.html_report(directory='tests/unit/infrastructure/error/coverage_report')
        print("📄 HTML覆盖率报告已生成: tests/unit/infrastructure/error/coverage_report/index.html")

        # JSON报告用于CI/CD
        cov.json_report(outfile='tests/unit/infrastructure/error/coverage.json')

        return success

    except ImportError:
        print("⚠️  coverage 模块未安装，使用基本测试运行")
        return run_error_management_tests()


if __name__ == '__main__':
    print("🔧 RQA2025 错误管理系统单元测试")
    print("=" * 50)

    # 检查是否安装了必要的依赖
    try:
        print("✅ 错误管理系统模块加载成功")
    except ImportError as e:
        print(f"❌ 错误管理系统模块加载失败: {e}")
        sys.exit(1)

    # 运行测试
    try:
        if '--coverage' in sys.argv:
            success = run_coverage_analysis()
        else:
            success = run_error_management_tests()

        if success:
            print("\n🎉 所有测试通过！错误管理系统达到投产质量标准")
            sys.exit(0)
        else:
            print("\n❌ 部分测试失败，需要修复后重新运行")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 测试执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
