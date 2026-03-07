#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import requests
import time

def test_chartjs_loading():
    """测试Chart.js CDN加载机制"""
    print("📊 开始Chart.js CDN加载测试...")

    try:
        with open('web-static/data-sources-config.html', 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return False

    # 提取Chart.js加载逻辑
    script_pattern = r'<script[^>]*>(.*?)</script>'
    scripts = re.findall(script_pattern, content, re.DOTALL)

    chartjs_script = None
    for script in scripts:
        if 'chartJSLoaded' in script or 'loadChartJS' in script:
            chartjs_script = script
            break

    if not chartjs_script:
        print("❌ 未找到Chart.js加载脚本")
        return False

    # 提取CDN URL
    cdn_urls = re.findall(r"script\d*\.src\s*=\s*['\"]([^'\"]+)['\"]", chartjs_script)

    print(f"\n🔗 发现 {len(cdn_urls)} 个CDN URL:")
    for i, url in enumerate(cdn_urls, 1):
        print(f"  {i}. {url}")

    # 检查备用机制
    has_fallback = 'onerror' in chartjs_script
    has_mock = 'window.Chart = function()' in chartjs_script

    print(f"\n🛡️ 备用机制检查:")
    print(f"  CDN失败处理: {'✅' if has_fallback else '❌'}")
    print(f"  Mock Chart对象: {'✅' if has_mock else '❌'}")

    # 测试CDN连接性（可选，不强制要求网络连接）
    print(f"\n🌐 CDN可用性测试 (可选):")
    for url in cdn_urls[:2]:  # 只测试前2个
        try:
            response = requests.head(url, timeout=5)
            status = f"✅ {response.status_code}"
        except requests.exceptions.RequestException as e:
            status = f"⚠️ 连接失败: {str(e)[:50]}"
        except Exception as e:
            status = f"❌ 错误: {str(e)[:50]}"

        print(f"  {url}: {status}")

    # 检查Chart.js初始化逻辑
    init_checks = [
        ('window.chartJSLoaded = false', '初始状态设置'),
        ('window.pendingChartsInit', '待初始化回调'),
        ('initCharts()', '图表初始化调用'),
        ('updateCharts()', '图表更新调用')
    ]

    print(f"\n🎨 Chart.js初始化逻辑检查:")
    for pattern, description in init_checks:
        found = pattern in chartjs_script or pattern in content
        status = "✅" if found else "❌"
        print(f"  {status} {description}")

    # 检查图表canvas元素
    canvas_elements = re.findall(r'<canvas[^>]*id="[^"]*"[^>]*>', content)
    print(f"\n📈 Canvas元素检查:")
    print(f"  发现 {len(canvas_elements)} 个canvas元素:")
    for canvas in canvas_elements:
        canvas_id = re.search(r'id="([^"]*)"', canvas)
        if canvas_id:
            print(f"    ✅ {canvas_id.group(1)}")

    # 检查图表初始化函数
    init_charts_found = 'function initCharts()' in content
    print(f"\n🔧 图表初始化函数: {'✅ 找到' if init_charts_found else '❌ 未找到'}")

    # 总结
    success_criteria = [
        len(cdn_urls) >= 2,  # 至少2个CDN
        has_fallback,        # 有错误处理
        has_mock,           # 有mock对象
        len(canvas_elements) >= 3,  # 至少3个图表
        init_charts_found    # 有初始化函数
    ]

    success = all(success_criteria)

    print(f"\n🎯 测试总结:")
    print(f"- CDN数量: {len(cdn_urls)} 个")
    print(f"- 备用机制: {'✅ 完整' if has_fallback and has_mock else '❌ 不完整'}")
    print(f"- Canvas元素: {len(canvas_elements)} 个")
    print(f"- 初始化函数: {'✅' if init_charts_found else '❌'}")

    if success:
        print("\n🎉 Chart.js加载机制测试通过！")
        return True
    else:
        print("\n⚠️ Chart.js加载机制测试失败！")
        return False

if __name__ == "__main__":
    import sys
    success = test_chartjs_loading()
    sys.exit(0 if success else 1)
