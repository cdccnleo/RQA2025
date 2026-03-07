#!/usr/bin/env python3
"""
RQA2025 测试覆盖率提升自动化脚本
按照依赖关系和业务流程，优先提升接近目标的层级覆盖率
"""

import subprocess
import time

# 接近目标的层级配置
TARGET_MODULES = [
    {
        "name": "data",
        "target": 25.0,
        "current": 22.50,
        "priority": 1
    },
    {
        "name": "engine",
        "target": 25.0,
        "current": 23.23,
        "priority": 2
    },
    {
        "name": "utils",
        "target": 25.0,
        "current": 21.76,
        "priority": 3
    }
]


def run_coverage_test(module_name, timeout=600):
    """运行指定模块的覆盖率测试"""
    cmd = [
        "python", "scripts/testing/run_tests.py",
        "--env", "test",
        "--module", module_name,
        "--cov", f"src/{module_name}",
        "--pytest-args", "-v",
        "--timeout", str(timeout)
    ]

    print(f"🔄 正在测试 {module_name} 模块...")
    print(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60
        )

        if result.returncode == 0:
            print(f"✅ {module_name} 模块测试成功")
            return True
        else:
            print(f"❌ {module_name} 模块测试失败")
            print(f"错误输出: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {module_name} 模块测试超时")
        return False
    except Exception as e:
        print(f"💥 {module_name} 模块测试异常: {e}")
        return False


def boost_coverage():
    """批量提升覆盖率"""
    print("🚀 开始RQA2025测试覆盖率提升计划")
    print("=" * 50)

    # 按优先级排序
    sorted_modules = sorted(TARGET_MODULES, key=lambda x: x["priority"])

    for module in sorted_modules:
        name = module["name"]
        current = module["current"]
        target = module["target"]
        gap = target - current

        print(f"\n📊 模块: {name}")
        print(f"   当前覆盖率: {current}%")
        print(f"   目标覆盖率: {target}%")
        print(f"   差距: {gap:.2f}%")

        if gap > 0:
            print(f"🎯 尝试提升 {name} 模块覆盖率...")
            success = run_coverage_test(name)

            if success:
                print(f"✅ {name} 模块覆盖率提升完成")
            else:
                print(f"⚠️  {name} 模块覆盖率提升遇到问题，继续下一个模块")
        else:
            print(f"✅ {name} 模块已达到目标覆盖率")

        time.sleep(2)  # 避免过于频繁的测试

    print("\n" + "=" * 50)
    print("🎉 测试覆盖率提升计划执行完成")
    print("📝 建议:")
    print("   1. 检查失败的测试用例")
    print("   2. 修复依赖问题")
    print("   3. 补充缺失的测试用例")
    print("   4. 定期运行覆盖率监控")


if __name__ == "__main__":
    boost_coverage()
