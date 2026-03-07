"""快速测试验证脚本"""
import subprocess


def run_quick_validation():
    """运行快速验证测试"""
    print("=" * 60)
    print("RQA2025 快速测试验证")
    print("=" * 60)

    # 测试Infrastructure层的高优先级模块
    test_cases = [
        {
            "name": "Infrastructure Database模块",
            "command": ["python", "scripts/run_focused_tests.py", "--layer", "infrastructure", "--target", "module:database", "--timeout", "120"]
        },
        {
            "name": "Infrastructure Monitoring模块",
            "command": ["python", "scripts/run_focused_tests.py", "--layer", "infrastructure", "--target", "module:monitoring", "--timeout", "120"]
        },
        {
            "name": "Infrastructure Cache模块",
            "command": ["python", "scripts/run_focused_tests.py", "--layer", "infrastructure", "--target", "module:cache", "--timeout", "120"]
        }
    ]

    results = []

    for test_case in test_cases:
        print(f"\n🔍 测试: {test_case['name']}")
        print(f"命令: {' '.join(test_case['command'])}")

        try:
            result = subprocess.run(test_case['command'],
                                    capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ 通过")
                results.append({"name": test_case['name'], "status": "PASS", "details": "所有测试通过"})
            else:
                # 解析失败信息
                failed_count = 0
                passed_count = 0
                for line in result.stdout.split('\n'):
                    if "FAILED" in line:
                        failed_count += 1
                    elif "PASSED" in line:
                        passed_count += 1

                print(f"⚠️  部分失败 - 通过: {passed_count}, 失败: {failed_count}")
                results.append({
                    "name": test_case['name'],
                    "status": "PARTIAL",
                    "details": f"通过: {passed_count}, 失败: {failed_count}"
                })

        except subprocess.TimeoutExpired:
            print("⏰ 超时")
            results.append({"name": test_case['name'], "status": "TIMEOUT", "details": "测试超时"})
        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append({"name": test_case['name'], "status": "ERROR", "details": str(e)})

    # 生成总结报告
    print("\n" + "=" * 60)
    print("测试验证总结")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['status'] == 'PASS')
    partial_tests = sum(1 for r in results if r['status'] == 'PARTIAL')
    failed_tests = sum(1 for r in results if r['status'] in ['ERROR', 'TIMEOUT'])

    print(f"总测试模块: {total_tests}")
    print(f"完全通过: {passed_tests}")
    print(f"部分通过: {partial_tests}")
    print(f"完全失败: {failed_tests}")

    if passed_tests > 0:
        print(f"\n✅ 成功修复的模块:")
        for result in results:
            if result['status'] == 'PASS':
                print(f"  - {result['name']}")

    if partial_tests > 0:
        print(f"\n⚠️  需要进一步修复的模块:")
        for result in results:
            if result['status'] == 'PARTIAL':
                print(f"  - {result['name']}: {result['details']}")

    if failed_tests > 0:
        print(f"\n❌ 需要重点关注的模块:")
        for result in results:
            if result['status'] in ['ERROR', 'TIMEOUT']:
                print(f"  - {result['name']}: {result['details']}")

    print("\n" + "=" * 60)
    return results


if __name__ == "__main__":
    run_quick_validation()
