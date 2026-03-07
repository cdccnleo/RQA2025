#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import sys

def test_basic_loading():
    """测试页面基本加载功能"""
    print("🔍 开始页面基本加载测试...")

    try:
        with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return False

    # 1. 检查HTML结构完整性
    checks = [
        ('DOCTYPE', '<!DOCTYPE html>' in content),
        ('html标签', bool(re.search(r'<html[^>]*>', content))),
        ('head标签', bool(re.search(r'<head[^>]*>', content))),
        ('/head标签', '</head>' in content),
        ('body标签', bool(re.search(r'<body[^>]*>', content))),
        ('/body标签', '</body>' in content),
        ('/html标签', '</html>' in content)
    ]

    print("\n📋 HTML结构检查:")
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False

    # 2. 检查Script标签平衡
    script_open = len(re.findall(r'<script[^>]*>', content))
    script_close = len(re.findall(r'</script>', content))
    print(f"\n📜 Script标签: {script_open} 个开始, {script_close} 个结束")
    script_balanced = script_open == script_close
    print("✅ Script标签平衡" if script_balanced else "❌ Script标签不平衡")

    # 3. 检查JavaScript语法
    script_regex = r'<script[^>]*>(.*?)</script>'
    scripts = re.findall(script_regex, content, re.DOTALL)
    syntax_errors = 0

    for i, script in enumerate(scripts, 1):
        try:
            # 在Python中无法直接测试JS语法，但我们可以检查基本的结构问题
            if script.strip():
                print(f"✅ Script {i}: 找到内容 ({len(script)} 字符)")
            else:
                print(f"⚠️ Script {i}: 空脚本")
        except Exception as e:
            print(f"❌ Script {i}: 处理错误 - {e}")
            syntax_errors += 1

    # 4. 检查关键元素存在
    critical_elements = [
        'data-sources-table',
        'dataSourceModal',
        'dataSourceForm',
        'modalTitle'
    ]

    print("\n🎯 关键元素检查:")
    elements_found = 0
    for element_id in critical_elements:
        if f'id="{element_id}"' in content or f"id='{element_id}'" in content:
            print(f"✅ {element_id}")
            elements_found += 1
        else:
            print(f"❌ {element_id}")

    # 5. 检查函数定义
    functions = re.findall(r'function\s+\w+\s*\(', content)
    async_functions = re.findall(r'async\s+function\s+\w+\s*\(', content)
    total_functions = len(functions) + len(async_functions)

    print(f"\n🔧 函数定义: {total_functions} 个 ({len(async_functions)} 个异步函数)")

    # 总结
    print("\n🎯 测试总结:")
    print(f"- HTML结构: {'✅ 完整' if all_passed else '❌ 有问题'}")
    print(f"- Script标签: {'✅ 平衡' if script_balanced else '❌ 不平衡'}")
    print(f"- 关键元素: {elements_found}/{len(critical_elements)} 个找到")
    print(f"- 函数定义: {total_functions} 个")

    success = all_passed and script_balanced and elements_found == len(critical_elements)

    if success:
        print("\n🎉 页面基本加载测试通过！")
        return True
    else:
        print("\n⚠️ 页面基本加载测试失败！")
        return False

if __name__ == "__main__":
    success = test_basic_loading()
    sys.exit(0 if success else 1)
