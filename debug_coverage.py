#!/usr/bin/env python3
"""
调试覆盖率问题的脚本
"""

import os
import sys
import subprocess

os.chdir('C:/PythonProject/RQA2025')
sys.path.insert(0, 'src')

print('=== 覆盖率问题诊断 ===')

# 1. 直接运行测试代码，检查是否执行了实际代码
print('1. 直接执行测试代码:')

try:
    from infrastructure.cache.core.cache_components import CacheComponent
    from infrastructure.cache.interfaces import ICacheComponent

    print("开始执行测试代码...")

    # 创建组件 - 这应该执行实际代码
    comp = CacheComponent(component_id=1, component_type="memory")
    print(f"组件创建成功: {comp.component_name()}")

    # 检查是否是接口实例
    is_instance = isinstance(comp, ICacheComponent)
    print(f"接口检查: {is_instance}")

    # 调用方法 - 这应该增加覆盖率
    name = comp.component_name()
    comp_type = comp.component_type()
    status = comp.get_component_status()

    print(f"方法调用成功: name={name}, type={comp_type}")
    print("测试代码执行完成 - 如果这里有输出，说明代码被执行了")

except Exception as e:
    print(f"测试执行失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 2. 检查覆盖率配置
print('2. 检查覆盖率配置:')

if os.path.exists('.coveragerc'):
    with open('.coveragerc', 'r') as f:
        content = f.read()
    print('.coveragerc内容:')
    print(content)
else:
    print('未找到.coveragerc文件')

# 3. 检查pytest-cov是否能正常工作
print()
print('3. 检查pytest-cov状态:')

try:
    import coverage
    print(f"✅ coverage库版本: {coverage.__version__}")

    # 检查是否有coverage数据文件
    if os.path.exists('.coverage'):
        size = os.path.getsize('.coverage')
        print(f"✅ 找到覆盖率数据文件: .coverage ({size} bytes)")
    else:
        print("❌ 未找到覆盖率数据文件")

except ImportError:
    print("❌ coverage库未安装")

# 4. 尝试简单的覆盖率测试
print()
print('4. 运行简单的覆盖率测试:')

# 创建一个简单的测试函数
test_code = '''
import os
import sys
sys.path.insert(0, "src")

def test_simple():
    from infrastructure.cache.core.cache_components import CacheComponent
    comp = CacheComponent(component_id=1, component_type="memory")
    name = comp.component_name()
    return name

if __name__ == "__main__":
    result = test_simple()
    print(f"测试结果: {result}")
'''

# 写入临时测试文件
with open('temp_test.py', 'w') as f:
    f.write(test_code)

try:
    # 使用coverage运行临时测试
    result = subprocess.run([
        sys.executable, '-m', 'coverage', 'run', '--source=src/infrastructure/cache',
        'temp_test.py'
    ], capture_output=True, text=True, cwd='.')

    print('Coverage run结果:')
    print(result.stdout)
    if result.stderr:
        print('错误:')
        print(result.stderr)

    # 生成报告
    report_result = subprocess.run([
        sys.executable, '-m', 'coverage', 'report', '--include=src/infrastructure/cache/*'
    ], capture_output=True, text=True, cwd='.')

    print()
    print('覆盖率报告:')
    print(report_result.stdout)
    if report_result.stderr:
        print('报告错误:')
        print(report_result.stderr)

except Exception as e:
    print(f'覆盖率测试失败: {e}')

# 清理临时文件
if os.path.exists('temp_test.py'):
    os.remove('temp_test.py')

print()
print('=== 诊断完成 ===')
